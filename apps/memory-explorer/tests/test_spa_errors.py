"""SPA error display for failed API calls.

During the outage the search status bar showed the raw response body:
`Error: {"detail":"couldn't get a connection after 30.00 sec"}`. apiError()
now reads the body once, prefers FastAPI's JSON detail, and callers render
the message via textContent / esc() only.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


def _script() -> str:
    html = TestClient(app_module.app).get("/").text
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert scripts
    return "\n".join(scripts)


def _function(src: str, name: str) -> str:
    match = re.search(r"^function " + name + r"\(.*?^}$", src, re.DOTALL | re.MULTILINE)
    assert match, f"function {name} not found in the SPA script"
    return match.group(0)


_HARNESS = """
function resp(status, body) {
  var reads = 0;
  return {
    status: status, ok: status >= 200 && status < 300,
    text: function() { reads++; if (reads > 1) throw new Error('body already used'); return Promise.resolve(body); },
    json: function() { throw new Error('apiError must read the body once, as text'); },
  };
}
var cases = JSON.parse(process.argv[2]);
Promise.all(cases.map(function(c) {
  return apiError(resp(c[0], c[1])).then(function() { return 'RESOLVED'; }, function(m) { return m; });
})).then(function(out) { console.log(JSON.stringify(out)); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_api_error_messages(tmp_path):
    cases = [
        [503, json.dumps({"detail": "Database unavailable: the database system is in recovery mode"})],
        [503, "Service Unavailable"],
        [504, json.dumps({"detail": "query timed out"})],
        [500, ""],
        [422, json.dumps({"detail": [{"loc": ["body", "source"], "msg": "Field required"}]})],
        [500, "<img src=x onerror=alert(1)>"],
    ]
    path = tmp_path / "api_error.js"
    path.write_text(_function(_script(), "apiError") + "\n" + _HARNESS)
    proc = subprocess.run(
        ["node", str(path), json.dumps(cases)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [
        "Database unavailable: the database system is in recovery mode",  # no double prefix
        "Database unavailable: Service Unavailable",
        "query timed out",
        "HTTP 500",
        '[{"loc":["body","source"],"msg":"Field required"}]',
        "<img src=x onerror=alert(1)>",  # returned as text; callers never innerHTML it raw
    ]


def test_search_and_detail_errors_render_as_text():
    src = _script()
    run = _function(src, "run")
    assert "apiError(r)" in run
    assert "r.text().then(t => Promise.reject(t))" not in run
    assert "getElementById('statusbar').textContent = 'Error: ' + err" in run

    detail = _function(src, "showDetail")
    assert "apiError(r)" in detail
    assert "esc(String(err))" in detail

    delete = _function(src, "deleteMemory")
    assert "apiError(r)" in delete
    assert "r.json().catch" not in delete  # the old path re-read a consumed body
