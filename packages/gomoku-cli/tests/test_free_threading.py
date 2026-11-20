"""The project must not switch the GIL back on under a free-threaded Python.

On ``python3.14t`` importing a C extension that has not declared it can run
without the GIL turns the GIL on for the whole process, and threaded brain
code (``probe_threads``) then runs one thread at a time again.  This checks
in a fresh interpreter that importing every module of every package, and
what the API server loads, keeps the GIL off.
"""

import os
import subprocess
import sys
import sysconfig

import pytest

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs a free-threaded Python (python3.14t)"
)

_SCRIPT = """
import importlib, pkgutil, sys
import gomoku, gomoku_brain, gomoku_cli, gomoku_fastapi
for pkg in (gomoku, gomoku_brain, gomoku_cli, gomoku_fastapi):
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(info.name)
# What gomoku-api starts: uvicorn with the h11 parser (see gomoku_fastapi.main.run).
import uvicorn, uvicorn.protocols.http.h11_impl  # noqa: F401
from gomoku_brain.evaluate_jit import check_winner_grid  # numba kernels  # noqa: F401
print("GIL", sys._is_gil_enabled())
"""


def test_imports_keep_the_gil_off():
    out = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        # Without PYTHON_GIL, which would force the GIL either way.
        env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("GIL False"), out.stdout
