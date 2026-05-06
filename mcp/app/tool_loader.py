import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

JSON = Dict[str, Any]
ToolFn = Callable[[JSON], JSON]

_CACHE: Dict[str, ToolFn] = {}


def _find_toolstore_dir() -> Path:
    env_dir = os.getenv("TOOLSTORE_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    start = Path(__file__).resolve().parent
    for parent in [start] + list(start.parents):
        cand = parent / "toolstore" / "index.json"
        if cand.exists():
            return parent / "toolstore"

    return start / "toolstore"


STORE_DIR = _find_toolstore_dir()
TOOLS_DIR = STORE_DIR / "tools"
INDEX_FILE = STORE_DIR / "index.json"


def _read_index() -> Dict[str, str]:
    if not INDEX_FILE.exists():
        return {}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_index(index: Dict[str, str]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def save_tool_code(name: str, code: str) -> None:
    name = (name or "").strip()
    if not name:
        raise ValueError("tool name is empty")

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    index = _read_index()
    filename = f"{name}.py"
    (TOOLS_DIR / filename).write_text(code, encoding="utf-8")

    index[name] = filename
    _write_index(index)

    _CACHE.pop(name, None)


def delete_tool(name: str) -> bool:
    name = (name or "").strip()
    if not name:
        return False

    index = _read_index()
    filename = index.get(name)
    if not filename:
        return False

    path = TOOLS_DIR / filename
    if path.exists():
        path.unlink()

    index.pop(name, None)
    _write_index(index)

    _CACHE.pop(name, None)
    return True


def list_tools() -> Dict[str, str]:
    return _read_index()


def get_tool(name: str) -> Optional[ToolFn]:
    name = (name or "").strip()
    if not name:
        return None

    if name in _CACHE:
        return _CACHE[name]

    index = _read_index()
    filename = index.get(name)
    if not filename:
        return None

    path = TOOLS_DIR / filename
    if not path.exists():
        return None

    code = path.read_text(encoding="utf-8")

    ns: Dict[str, Any] = {}
    try:
        exec(compile(code, filename, "exec"), ns, ns)
    except Exception as e:
        # Tool code is broken; return None so server can report TOOL_NOT_FOUND or TOOL_CRASH.
        # Better: raise and let MCP server return TOOL_CRASH with details.
        raise RuntimeError(f"Failed to load tool '{name}' from {filename}: {e}")

    # Primary contract: function name equals tool name
    candidate = ns.get(name)

    # allow `run(args)` function
    if not callable(candidate):
        candidate = ns.get("run")

    if not callable(candidate):
        return None

    _CACHE[name] = candidate  
    return candidate
