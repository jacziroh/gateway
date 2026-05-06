import os
from typing import Any, Dict, List

import httpx

JSON = Dict[str, Any]


class MCPHttpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def list_tools(self, http: httpx.AsyncClient) -> List[str]:
        """
        Calls MCP /tools/list and returns a list of tool names.
        Expected response:
          {"tools": ["square", "add", ...]}
        """
        r = await http.post(f"{self.base_url}/tools/list", json={})
        r.raise_for_status()
        data = r.json()

        tools = data.get("tools", [])
        if not isinstance(tools, list):
            return []

        out: List[str] = []
        for t in tools:
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
        return out

    async def call_tool(self, http: httpx.AsyncClient, tool: str, args: JSON) -> JSON:
        """
        Calls MCP /tools/call.
        Request:
          {"tool": "square", "args": {"n": 23}}

        Response (recommended):
          {"ok": true, "result": {...}}  OR {"ok": false, "error": {...}}
        """
        payload = {"tool": tool, "args": args}
        r = await http.post(f"{self.base_url}/tools/call", json=payload)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {"ok": True, "result": data}


def build_mcp_client() -> MCPHttpClient:
    """
    Single place to construct MCP client from env.
    """
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9200")
    return MCPHttpClient(base_url)
