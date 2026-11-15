"""Where a dashboard session keeps its registry, models and streamed batches.

Locally everything lives in the project directory, as before, so the scripts
and the dashboard share one registry.

On the live demo every visitor gets a private workspace (a temp directory) with
its own MLflow store, so one visitor's Retrain or Reset never changes what
another visitor sees. Model v1 (linked, not copied), the data and the 10
history batches are shared and read-only.

Sandbox mode is on when OBSERVATORY_SANDBOX is "1"/"true", off when it is
"0"/"false", and otherwise on only when running on Streamlit Community Cloud
(which checks the app out under /mount/src).
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

SANDBOX_BASE = Path(tempfile.gettempdir()) / "observatory-sessions"
MAX_AGE_SECONDS = 24 * 3600


def sandbox_enabled(env=None, cwd=None):
    env = os.environ if env is None else env
    flag = env.get("OBSERVATORY_SANDBOX", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    cwd = Path.cwd() if cwd is None else Path(cwd)
    return cwd.as_posix().startswith("/mount/src/")


def cleanup_old_sessions(base=None, max_age=MAX_AGE_SECONDS, now=None):
    """Deletes workspaces untouched for longer than max_age seconds."""
    base = SANDBOX_BASE if base is None else Path(base)
    if not base.exists():
        return
    now = time.time() if now is None else now
    for d in base.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def new_session_root(base=None):
    base = SANDBOX_BASE if base is None else Path(base)
    base.mkdir(parents=True, exist_ok=True)
    cleanup_old_sessions(base)
    return Path(tempfile.mkdtemp(prefix="session-", dir=base))


def session_root(state):
    """The workspace for one dashboard session. state is st.session_state (a dict works too)."""
    if not sandbox_enabled():
        return Path(".")
    root = state.get("workspace")
    if root is None or not Path(root).exists():
        root = str(new_session_root())
        state["workspace"] = root
    os.utime(root)  # mark as in use so cleanup keeps it
    return Path(root)
