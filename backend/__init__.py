"""backend package init.

Sets the OMP threading-conflict env vars BEFORE any submodule import.
Python evaluates this file before any `from backend.X import ...` runs,
so by the time `backend.expression` pulls torch and `backend.graph`
pulls faiss, both libraries see the env and respect single-threaded
mode. This prevents the silent kill we hit when torch's libomp and
faiss-cpu's libomp tried to initialise their thread pools concurrently
(no Python traceback — just the resource_tracker leaked-semaphore
warning at process shutdown).

The same env vars are also exported by start.sh and set at the top of
run_curriculum.py — three layers of belt-and-braces, but the package
__init__ is the load-bearing one for any process that imports backend.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
