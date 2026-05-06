#toolregistryapi.PythonFinalizationError
from typing import Any, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel

from tool_loader import save_tool_code, delete_tool, list_tools

app = FastAPI(title="Tool Registry")

JSON = Dict[str, Any]


class ToolUpload(BaseModel):
    name: str
    code: str


class BulkUpload(BaseModel):
    tools: List[ToolUpload]


@app.get("/tools")
def get_tools() -> JSON:
    # returns persisted tools stored on disk
    return {"tools": list(list_tools().keys())}


@app.post("/tools")
def add_tool(payload: ToolUpload) -> JSON:
    save_tool_code(payload.name, payload.code)
    return {"ok": True, "added": payload.name}


@app.post("/tools/bulk")
def add_tools_bulk(payload: BulkUpload) -> JSON:
    for t in payload.tools:
        save_tool_code(t.name, t.code)
    return {"ok": True, "count": len(payload.tools)}


@app.delete("/tools/{name}")
def remove_tool(name: str) -> JSON:
    ok = delete_tool(name)
    if not ok:
        return {"ok": False, "error": "tool not found"}
    return {"ok": True, "removed": name}
