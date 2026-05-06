# tool_runtime.py
from typing import Any, Dict

from tool_loader import get_tool

JSON = Dict[str, Any]


def run_tool_request(request: JSON) -> JSON:
    tool_name = request.get("tool")
    args = request.get("args") or {}

    if not tool_name:
        return {"error": "missing tool"}

    if not isinstance(args, dict):
        return {"error": "args must be an object/dict"}

    fn = get_tool(tool_name)
    if fn is None:
        return {"error": f"unknown tool: {tool_name}"}

    try:
        out = fn(args)
    except Exception as e:
        return {"error": "tool crashed", "tool": tool_name, "detail": str(e)}

    # enforce JSON output shape
    if not isinstance(out, dict):
        return {"error": "tool must return JSON object", "tool": tool_name, "raw": str(out)}

    return out
