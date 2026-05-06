import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from mcp.app.tool_loader import get_tool, list_tools as store_list_tools, save_tool_code, delete_tool

log = logging.getLogger("mcp")

JSON = Dict[str, Any]

app = FastAPI(title="Kai MCP Tool Server")

class ToolCall(BaseModel):
    tool: str
    args: Optional[Any] = None

class ToolUpload(BaseModel):
    name: str
    code: str

class BulkToolUpload(BaseModel):
    tools: list[ToolUpload]

# Helpers
def _ok(result: Any) -> JSON:
    return {"ok": True, "result": result, "served_by": "mcp"}


def _err(code: str, message: str, detail: Any = None) -> JSON:
    out: JSON = {"ok": False, "error": {"code": code, "message": message}}
    if detail is not None:
        out["error"]["detail"] = detail
    return out


def _normalize_args(args: Any) -> Optional[JSON]:
    if args is None:
        args = {}

    if not isinstance(args, dict):
        return None

    # Backward compatible: allow {"args": {...}} only
    if "args" in args and isinstance(args["args"], dict) and len(args) == 1:
        return args["args"]

    return args

# MCP Standard APIs
@app.post("/tools/list")
def tools_list() -> JSON:
    log.info("tools.list requested")
    tools = store_list_tools()
    names = sorted(tools.keys()) if isinstance(tools, dict) else []
    log.info(f"tools.list returning {len(names)} tools")
    return {"tools": names}

@app.post("/tools/call")
def tools_call(req: ToolCall) -> JSON:
    log.info(f"tools.call received | tool={req.tool} args={req.args}")
    tool_name = (req.tool or "").strip()
    if not tool_name:
        return _err("BAD_TOOL", "Tool name is required")

    args = _normalize_args(req.args)
    if args is None:
        return _err("BAD_ARGS", "Args must be a JSON object")

    log.info(f"looking up tool: {tool_name}")
    fn = get_tool(tool_name)

    if not fn:
        return _err("TOOL_NOT_FOUND", f"Unknown tool: {tool_name}")

    log.info(f"executing tool: {tool_name}")
    try:
        result = fn(args)
    except Exception as e:
        return _err("TOOL_CRASH", f"Tool '{tool_name}' crashed", str(e))

    if not isinstance(result, dict):
        return _err(
            "BAD_TOOL_OUTPUT",
            f"Tool '{tool_name}' must return a JSON object",
            str(result),
        )
    log.info(f"tool executed successfully: {tool_name} | result={result}")
    return _ok(result)

@app.get("/health")
def health() -> JSON:
    return {"ok": True, "service": "mcp", "ts": int(time.time())}

@app.post("/tools/add")
def add_tool(payload: ToolUpload) -> JSON:
    name = (payload.name or "").strip()
    if not name:
        return _err("BAD_NAME", "Tool name is required")

    code = payload.code or ""
    if not code.strip():
        return _err("BAD_CODE", "Tool code is empty")

    log.info(f"adding tool: {name}")
    save_tool_code(name, code)
    log.info(f"tool added successfully: {name}")

    return _ok({"added": name})


@app.delete("/tools/remove/{name}")
def remove_tool(name: str) -> JSON:
    tool_name = (name or "").strip()
    if not tool_name:
        return _err("BAD_NAME", "Tool name is required")

    log.info(f"removing tool: {tool_name}")
    ok = delete_tool(tool_name)

    if not ok:
        return _err("TOOL_NOT_FOUND", f"Tool not found: {tool_name}")

    log.info(f"tool removed successfully: {tool_name}")
    return _ok({"removed": tool_name})

@app.post("/tools/add/bulk")
def add_tools_bulk(payload: BulkToolUpload) -> JSON:
    log.info(f"bulk add requested | count={len(payload.tools)}")
    added = []
    for tool in payload.tools:
        name = (tool.name or "").strip()
        code = tool.code or ""

        if not name or not code.strip():
            log.warning("skipping invalid tool entry in bulk upload")
            continue 

        log.info(f"bulk add tool: {name}")
        save_tool_code(name, code)
        added.append(name)
    log.info(f"bulk add completed | added={added}")
    return _ok({"added": added, "count": len(added)})

if __name__ == "__main__":
    import os
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        "mcp.app.main:app",
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "9200")),
        log_config=LOGGING_CONFIG,
        log_level=os.getenv("MCP_LOG_LEVEL", os.getenv("DEMO_LOG_LEVEL", "info")).lower(),
        access_log=True,
        lifespan="on",
    )

