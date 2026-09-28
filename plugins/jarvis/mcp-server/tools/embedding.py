"""Embedding service for Jarvis v3.0 — explicit vector generation.

Supports four backends:
  - onnx: Local ONNX Runtime (default, ~400MB RAM, no API calls)
  - torch: Local PyTorch via sentence-transformers (fallback)
  - bedrock: Amazon Bedrock API (zero local model, requires AWS credentials)
  - host: A host-native Jarvis model service (zero model memory in container)

When backend is "onnx" or "torch", uses a local model (e.g. Granite 384d).
When backend is "bedrock", calls the Bedrock InvokeModel API (e.g. Titan
Embed v2) — no local model is loaded, saving ~550MB RAM.
When backend is "host", calls a trusted service running on the Docker host.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton state
_service: EmbeddingService | None = None
_service_cache_key: tuple | None = None
# Hooks and MCP tools call in from worker threads: one service per config.
_service_lock = threading.Lock()


class EmbeddingService:
    """Encode text into dense vector embeddings.

    Backends:
        onnx: Raw ONNX Runtime for minimal memory. Mean pooling +
              L2 normalization post-inference. Falls back to torch.
        torch: sentence-transformers + PyTorch.
        bedrock: Amazon Bedrock InvokeModel API. Model ID is passed
                 as model_name (e.g. "amazon.titan-embed-text-v2:0").

    Args:
        model_name: HuggingFace model ID, local path, or Bedrock model ID.
        dimensions: Expected embedding dimensions (validated on encode).
        device: Torch device string ("cpu", "cuda", etc.). Ignored for bedrock.
        backend: "onnx" (preferred), "torch", "bedrock", or "host".
        bedrock_region: AWS region for Bedrock API. Ignored for local backends.
        host_url: Base URL for the host model service.
        host_token: Optional bearer token for the host model service.
        host_timeout_ms: Per-request timeout for host inference.
    """

    # Reset the ONNX session every N encode calls to prevent C++ heap
    # fragmentation from accumulating over very long indexing runs.
    # Conservative interval: 50 calls × 8 texts/call = ~400 embeddings.
    # Under QEMU/emulated CPU, ONNX accumulates internal state faster.
    _ONNX_RESET_INTERVAL = 50

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        device: str = "cpu",
        backend: str = "onnx",
        bedrock_region: str = "eu-central-1",
        host_url: str = "http://host.docker.internal:8751",
        host_token: str = "",
        host_timeout_ms: int = 2000,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._device = device
        self._backend = backend
        self._bedrock_region = bedrock_region
        self._host_url = host_url
        self._host_token = host_token
        self._host_timeout_ms = host_timeout_ms
        # Raw ONNX mode
        self._ort_session = None
        self._tokenizer = None
        self._onnx_call_count = 0
        # Hooks and MCP tools encode from worker threads. The periodic reset
        # sets _ort_session/_tokenizer to None under any concurrent encode,
        # and HF fast tokenizers are not safe to share across threads either,
        # so local-model loading, reset and inference run one at a time.
        self._local_lock = threading.RLock()
        # Sentence-transformers fallback
        self._model: SentenceTransformer | None = None
        # Bedrock client (lazy-initialized)
        self._bedrock_client = None
        self._host_client = None

    # ── Host service backend ───────────────────────────────────────

    def _init_host(self):
        """Lazy-initialize the host model service client."""
        if self._host_client is not None:
            return
        from .model_host_client import ModelHostClient
        self._host_client = ModelHostClient(
            base_url=self._host_url,
            model_name=self._model_name,
            dimensions=self._dimensions,
            token=self._host_token,
            timeout_ms=self._host_timeout_ms,
        )

    def _host_encode(self, texts: list[str]) -> list[list[float]]:
        self._init_host()
        return self._host_client.embed(texts)

    def tokenize(self, text: str, *, with_pieces: bool = False) -> list:
        """Return host tokenizer IDs/pieces for exact bounded windowing."""
        if self._backend != "host":
            raise RuntimeError("Tokenizer pieces are only available for host inference")
        self._init_host()
        return self._host_client.tokenize(text, with_pieces=with_pieces)

    # ── Bedrock backend ─────────────────────────────────────────────

    def _init_bedrock(self):
        """Lazy-initialize the Bedrock Runtime client."""
        if self._bedrock_client is not None:
            return

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for the 'bedrock' embedding backend. "
                "Install with: pip install boto3"
            )

        self._bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=self._bedrock_region,
        )
        logger.info(
            "Initialized Bedrock client (region=%s, model=%s)",
            self._bedrock_region,
            self._model_name,
        )

    def _bedrock_encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts via Amazon Bedrock InvokeModel API.

        Calls the API once per text (Titan Embed has no native batch).
        Each call returns a normalized embedding vector.
        """
        self._init_bedrock()

        results = []
        for text in texts:
            response = self._bedrock_client.invoke_model(
                modelId=self._model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": text,
                    "dimensions": self._dimensions,
                    "normalize": True,
                }),
            )
            body = json.loads(response["body"].read())
            embedding = body["embedding"]
            if len(embedding) != self._dimensions:
                raise ValueError(
                    f"Dimension mismatch: Bedrock returned {len(embedding)}d, "
                    f"expected {self._dimensions}d"
                )
            results.append(embedding)
        return results

    # ── ONNX backend ────────────────────────────────────────────────

    def _load_onnx(self):
        """Load ONNX Runtime session + tokenizer directly."""
        with self._local_lock:
            self._load_onnx_locked()

    def _load_onnx_locked(self):
        if self._ort_session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1

        # Find the ONNX model file
        model_path = self._model_name
        onnx_path = os.path.join(model_path, "onnx", "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self._ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        self._onnx_call_count = 0
        logger.info(
            "Loaded %s with raw ONNX Runtime (CPU, mem_arena=on)",
            self._model_name,
        )

    def reset_onnx_session(self) -> None:
        """Release and recreate the ONNX session to reclaim C++ heap memory.

        Called periodically during long indexing runs to prevent heap
        fragmentation in the ONNX Runtime C++ allocator. Uses aggressive
        cleanup (multiple gc passes) because ONNX C++ objects may hold
        indirect references that require multiple collection cycles.
        """
        with self._local_lock:
            self._reset_onnx_session_locked()

    def _reset_onnx_session_locked(self) -> None:
        if self._ort_session is not None:
            import gc
            count = self._onnx_call_count
            # Release session and tokenizer
            self._ort_session = None
            self._tokenizer = None
            # Multiple gc passes to clean up C++ → Python reference chains
            gc.collect()
            gc.collect()
            self._onnx_call_count = 0
            logger.info("ONNX session reset after %d encode calls", count)

    def _onnx_encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts using raw ONNX Runtime + mean pooling.

        Steps:
        1. Tokenize with HuggingFace tokenizer
        2. Run ONNX session (returns last_hidden_state + sentence_embedding)
        3. If model outputs sentence_embedding, use it directly
        4. Otherwise, apply attention-masked mean pooling
        5. L2-normalize
        """
        with self._local_lock:
            return self._onnx_encode_locked(texts, normalize)

    def _onnx_encode_locked(self, texts: list[str], normalize: bool) -> np.ndarray:
        # Periodic session reset to prevent C++ heap fragmentation
        self._onnx_call_count += 1
        if self._onnx_call_count >= self._ONNX_RESET_INTERVAL:
            self.reset_onnx_session()

        self._load_onnx()

        inputs = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        outputs = self._ort_session.run(None, dict(inputs))

        # Check if model provides sentence_embedding output directly
        output_names = [o.name for o in self._ort_session.get_outputs()]
        if "sentence_embedding" in output_names:
            idx = output_names.index("sentence_embedding")
            embeddings = outputs[idx]
        else:
            # Mean pooling over last_hidden_state with attention mask
            last_hidden = outputs[0]  # (batch, seq_len, hidden_dim)
            mask = inputs["attention_mask"]  # (batch, seq_len)
            mask_expanded = np.expand_dims(mask, -1)  # (batch, seq_len, 1)
            summed = np.sum(last_hidden * mask_expanded, axis=1)
            counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = summed / counts

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings = embeddings / norms

        return embeddings

    # ── PyTorch backend ─────────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model (PyTorch fallback)."""
        with self._local_lock:
            if self._model is not None:
                return self._model

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self._model_name,
                device=self._device,
            )
            logger.info(
                "Loaded %s with PyTorch backend on %s",
                self._model_name,
                self._device,
            )
            return self._model

    # ── Public API ──────────────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        """Encode a single text into a vector. Used for queries."""
        if self._backend == "host":
            return self._host_encode([text])[0]

        if self._backend == "bedrock":
            return self._bedrock_encode([text])[0]

        if self._backend == "onnx":
            try:
                embeddings = self._onnx_encode([text])
                vec = embeddings[0].tolist()
                if len(vec) != self._dimensions:
                    raise ValueError(
                        f"Dimension mismatch: model produced {len(vec)}d, "
                        f"expected {self._dimensions}d"
                    )
                return vec
            except FileNotFoundError:
                logger.warning("ONNX model not found, falling back to PyTorch")
                self._backend = "torch"
            except ImportError:
                logger.warning("onnxruntime not available, falling back to PyTorch")
                self._backend = "torch"

        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        vec = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        if len(vec) != self._dimensions:
            raise ValueError(
                f"Dimension mismatch: model produced {len(vec)}d, "
                f"expected {self._dimensions}d"
            )
        return vec

    def encode_batch(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """Encode multiple texts into vectors. Used for indexing.

        Args:
            texts: List of strings to encode.
            batch_size: Internal batch size for the model.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        if self._backend == "host":
            # Keep every HTTP payload bounded even when an arbitrarily large
            # canonical document expands into many inference windows.
            result = []
            for index in range(0, len(texts), batch_size):
                result.extend(self._host_encode(texts[index : index + batch_size]))
            return result

        if self._backend == "bedrock":
            return self._bedrock_encode(texts)

        if self._backend == "onnx":
            try:
                # Process in sub-batches for memory efficiency
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    embs = self._onnx_encode(batch)
                    all_embeddings.append(embs)

                embeddings = np.concatenate(all_embeddings, axis=0)
                result = []
                for emb in embeddings:
                    vec = emb.tolist()
                    if len(vec) != self._dimensions:
                        raise ValueError(
                            f"Dimension mismatch: model produced {len(vec)}d, "
                            f"expected {self._dimensions}d"
                        )
                    result.append(vec)
                return result
            except FileNotFoundError:
                logger.warning("ONNX model not found, falling back to PyTorch")
                self._backend = "torch"
            except ImportError:
                logger.warning("onnxruntime not available, falling back to PyTorch")
                self._backend = "torch"

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = []
        for emb in embeddings:
            vec = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
            if len(vec) != self._dimensions:
                raise ValueError(
                    f"Dimension mismatch: model produced {len(vec)}d, "
                    f"expected {self._dimensions}d"
                )
            result.append(vec)
        return result

    @property
    def dimensions(self) -> int:
        """Return configured embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def backend(self) -> str:
        """Return the backend in use."""
        return self._backend

    @property
    def is_loaded(self) -> bool:
        """Whether the backend has been initialized."""
        return (
            self._ort_session is not None
            or self._model is not None
            or self._bedrock_client is not None
            or self._host_client is not None
        )


def _effective_model_name(config: dict) -> str:
    """Resolve the runtime model without leaking baked paths to host APIs."""
    if config["backend"] == "host":
        return config.get("host_model", config["model"])
    return config["model"]


def get_embedding_model_identity(config: dict | None = None) -> str:
    """Return the stable model identity stored alongside database vectors.

    Runtime locators are backend-specific: ONNX commonly receives a baked
    filesystem path while host inference receives an API alias.  Consistency
    checks must compare the underlying model identity, not those locators.
    """
    if config is None:
        from .config import get_embedding_config
        config = get_embedding_config()
    return config.get("model_id") or _effective_model_name(config)


def get_embedding_service() -> EmbeddingService:
    """Get or create singleton EmbeddingService with config-based invalidation.

    Recreates the service if embedding config changes (model, dimensions, etc.).
    Same singleton + cache-key pattern as the PG pool.
    """
    global _service, _service_cache_key
    from .config import get_embedding_config

    cfg = get_embedding_config()
    effective_model = _effective_model_name(cfg)
    key = (
        effective_model, cfg["dimensions"], cfg["device"], cfg["backend"],
        cfg.get("host_url"), cfg.get("host_token"), cfg.get("host_timeout_ms"),
    )

    service = _service
    if service is not None and _service_cache_key == key:
        return service

    with _service_lock:
        if _service is not None and _service_cache_key == key:
            return _service
        _service = EmbeddingService(**_service_kwargs(cfg, effective_model))
        _service_cache_key = key
        return _service


def _service_kwargs(cfg: dict, effective_model: str) -> dict:
    kwargs = {
        "model_name": effective_model,
        "dimensions": cfg["dimensions"],
        "device": cfg["device"],
        "backend": cfg["backend"],
    }
    if cfg["backend"] == "bedrock":
        kwargs["bedrock_region"] = cfg.get("bedrock_region", "eu-central-1")
    if cfg["backend"] == "host":
        kwargs.update(
            host_url=cfg["host_url"],
            host_token=cfg.get("host_token", ""),
            host_timeout_ms=cfg.get("host_timeout_ms", 2000),
        )
    return kwargs


def warm_embedding_service() -> float | None:
    """Load and probe the local embedding backend before serving hook traffic.

    Returns elapsed milliseconds for local and host backends. Bedrock is deliberately
    skipped: warming a remote backend would add a billable network request and
    cannot eliminate network cold starts.
    """
    service = get_embedding_service()
    if service.backend == "bedrock":
        logger.info("Skipping startup warmup for remote Bedrock embeddings")
        return None

    started = time.perf_counter()
    vector = service.encode("Jarvis embedding startup warmup")
    if len(vector) != service.dimensions:
        raise ValueError(
            f"Embedding warmup produced {len(vector)}d, expected {service.dimensions}d"
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("Embedding startup warmup completed in %.1f ms", elapsed_ms)
    return elapsed_ms


def reset_embedding_service() -> None:
    """Reset the singleton. Used in tests and config changes."""
    global _service, _service_cache_key
    _service = None
    _service_cache_key = None
