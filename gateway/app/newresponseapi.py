import json
import os
import re
import time
import uuid
import logging
import httpx
import jwt
from jwt import InvalidTokenError

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from gateway.app.mcp_client import build_mcp_client

JSON = Dict[str, Any]
logger = logging.getLogger("kai-gateway")

# Config
DEPLOY_MAP_PATH = os.getenv("DEPLOY_MAP_PATH", "./deploy-map.json")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "120"))
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
UPSTREAM_CHAT_PATH = os.getenv("UPSTREAM_CHAT_PATH", "/chat/completions")
#JWT
JWT_SECRET = os.getenv("KAI_JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("KAI_JWT_SECRET is not set")
JWT_ALGOS = ["HS256"] 
JWT_REQUIRED = os.getenv("JWT_REQUIRED", "true").lower() == "true"

CONVERSATIONS: dict[str, list[dict[str, Any]]] = {}
MAX_TURNS = 20  # keep last 20 messages 

# Small helpers
def now_ms() -> float:
    return time.perf_counter() * 1000.0

def respond(payload: Any, status_code: int = 200) -> JSONResponse:
    r = JSONResponse(payload, status_code=status_code)
    r.headers["X-Served-By"] = "kai-tools-gateway"
    return r

def forbidden() -> Response:
    r = Response(status_code=403)
    r.headers["X-Served-By"] = "kai-tools-gateway"
    return r

def error_response(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    payload = {"error": {"message": message, "type": err_type}}
    return respond(payload, status_code=status)


def upstream_error_response(e: httpx.HTTPStatusError) -> JSONResponse:
    status = getattr(e.response, "status_code", 502) or 502
    body = ""
    try:
        body = e.response.text
    except Exception:
        pass
    return respond(
        {"error": {"message": f"{str(e)} | upstream_body={body}", "type": "upstream_error"}},
        status_code=status,
    )


def extract_assistant_content(resp: JSON) -> str:
    try:
        return resp["choices"][0]["message"].get("content", "") or ""
    except Exception:
        return ""

def make_public_stats(*, completion_time_ms: float) -> JSON:
    return {
        "completion_time": completion_time_ms,
        "memory": False,
    }

def extract_usage(resp: JSON) -> JSON:
    u = resp.get("usage")
    return u if isinstance(u, dict) else {}


def merge_usage(u1: JSON, u2: JSON) -> JSON:
    out: JSON = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = 0
        if isinstance(u1.get(k), int):
            v += int(u1[k])
        if isinstance(u2.get(k), int):
            v += int(u2[k])
        if v:
            out[k] = v
    return out


def build_chat_completion_response(
    *,
    model: str,
    content: str,
    created: Optional[int] = None,
    usage: Optional[JSON] = None,
    upstream_id: Optional[str] = None,
    service_tier: str = "default",
    stats: Optional[JSON] = None,
) -> JSON:
    created_ts = created if isinstance(created, int) else int(time.time())
    resp_id = upstream_id or f"kai-chat-comp-{uuid.uuid4().int % 10**12}"

    out: JSON = {}

    out["choices"] = [{
        "finish_reason": "stop",
        "index": 0,
        "message": {"content": content, "role": "assistant"},
    }]
    out["created"] = created_ts
    out["id"] = resp_id
    out["model"] = model
    out["object"] = "chat.completion"
    out["service_tier"] = service_tier

    if isinstance(stats, dict):
        out["stats"] = stats
    if isinstance(usage, dict):
        out["usage"] = usage

    return out

# DeployMap
class DeployMap:
    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime: float = 0.0
        self._data: Dict[str, List[str]] = {}
        self._rr: Dict[str, int] = {}

    def _load_if_needed(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"deploy-map.json not found at: {self.path}")

        mtime = self.path.stat().st_mtime
        if self._data and mtime == self._mtime:
            return

        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        if not isinstance(raw, dict):
            raise ValueError("deploy-map.json must be a JSON object: {model: url(s)}")

        cleaned: Dict[str, List[str]] = {}
        for model, urls in raw.items():
            if not isinstance(model, str) or not model.strip():
                continue

            if isinstance(urls, str) and urls.strip():
                cleaned[model.strip()] = [urls.strip()]
            elif isinstance(urls, list):
                cleaned[model.strip()] = [u.strip() for u in urls if isinstance(u, str) and u.strip()]

        self._data = cleaned
        self._mtime = mtime

    def pick_url(self, model: str) -> str:
        self._load_if_needed()
        urls = self._data.get(model)
        if not urls:
            raise KeyError(f"Model '{model}' not found in deploy-map.json")

        i = self._rr.get(model, 0)
        url = urls[i % len(urls)]
        self._rr[model] = (i + 1) % len(urls)
        return url

    def pick_any(self) -> str:
        """
        Pick a default upstream base when we don't have a model (infra APIs like /health, /deploy, /models).
        Strategy: take the first URL found anywhere in deploy-map.json.
        """
        self._load_if_needed()
        for urls in self._data.values():
            if urls:
                return urls[0]
        raise KeyError("No upstream URLs found in deploy-map.json")

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, limits=httpx.Limits(max_keepalive_connections=0, max_connections=20))
    app.state.deploy_map = DeployMap(DEPLOY_MAP_PATH)
    app.state.mcp = build_mcp_client()
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _clean_headers(req: Request) -> dict:
    out = {}
    for k, v in req.headers.items():
        if k.lower() not in HOP_HEADERS:
            out[k] = v
    return out


async def forward_raw(req: Request, path: str) -> Response:
    base = app.state.deploy_map.pick_any()
    url = base.rstrip("/") + "/" + path.lstrip("/")

    body = await req.body()

    r = await app.state.http.request(
        method=req.method,
        url=url,
        params=dict(req.query_params),
        headers=_clean_headers(req),
        content=body if body else None,
    )

    resp = Response(content=r.content, status_code=r.status_code)

    # copy upstream headers (so output looks same)
    for k, v in r.headers.items():
        if k.lower() in HOP_HEADERS:
            continue
        resp.headers[k] = v

    resp.headers["X-Served-By"] = "kai-tools-gateway"
    return resp

# Upstream helpers
def ensure_upstream_defaults(payload: JSON) -> JSON:
    p = dict(payload)
    p.pop("store", None)
    p.setdefault("max_completion_tokens", 1024)
    p.setdefault("stream", False)
    p.setdefault("top_p", 1)
    p.setdefault("n", 1)
    p.setdefault("store", False)
    p.setdefault("temperature", 0.0)
    return p


def strip_tool_fields_for_upstream(payload: JSON, *, drop_response_format: bool = False) -> JSON:
    p = dict(payload)
    p.pop("tools", None)
    p.pop("tool_choice", None)
    p.pop("parallel_tool_calls", None)
    if drop_response_format:
        p.pop("response_format", None)
    return p


async def call_upstream(payload: JSON) -> JSON:
    payload = ensure_upstream_defaults(payload)

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise httpx.HTTPStatusError("Missing 'model' in request", request=None, response=None)

    base = app.state.deploy_map.pick_url(model.strip())
    url = base.rstrip("/") + "/" + UPSTREAM_CHAT_PATH.lstrip("/")

    print("UPSTREAM base=", base, "path=", UPSTREAM_CHAT_PATH, "url=", url, "model=", model.strip(), flush=True)
    r = await app.state.http.post(url, json=payload)
    if r.status_code >= 400:
        print("UPSTREAM:", r.status_code, r.text, flush=True)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}

# Tool extraction + validation
def extract_tool_names_from_openai_tools(tools: Any) -> List[str]:
    if not isinstance(tools, list):
        return []
    names: List[str] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    # dedupe preserve order
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_tool_schema_map(client_tools: Any) -> Dict[str, JSON]:
    if not isinstance(client_tools, list):
        return {}
    out: Dict[str, JSON] = {}
    for t in client_tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str) and isinstance(params, dict):
            out[name] = params
    return out


DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$")
DT_EXTRACT = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})")


def _json_type_matches(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def validate_args_against_schema(args: JSON, schema: JSON) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args must be an object"

    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [k for k in required if k not in args]
        if missing:
            return False, f"missing required fields: {missing}"

    props = schema.get("properties", {})
    if isinstance(props, dict):
        for k, p in props.items():
            if k not in args or not isinstance(p, dict):
                continue
            t = p.get("type")
            if isinstance(t, str) and not _json_type_matches(args[k], t):
                return False, f"field '{k}' has wrong type (expected {t})"

    if "meeting_time" in args:
        mt = args.get("meeting_time")
        if not isinstance(mt, str):
            return False, "meeting_time must be a string"
        mt2 = mt.strip()
        if not DT_RE.match(mt2):
            m = DT_EXTRACT.search(mt2)
            if m:
                args["meeting_time"] = m.group(1)
            else:
                return False, "meeting_time must match 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DDTHH:MM'"

    return True, "ok"


# Router parsing
def parse_json_object(text: str) -> Optional[JSON]:
    if not isinstance(text, str):
        return None
    s = text.strip()

    # tolerate ```json ... ```
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:] if lines else []
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        obj = json.loads(s)
    except Exception:
        return None

    return obj if isinstance(obj, dict) else None


def is_tool_call(obj: JSON) -> bool:
    return isinstance(obj.get("tool"), str) and isinstance(obj.get("args"), dict)


def normalize_router_output(obj: Optional[JSON], allowed_tools: List[str]) -> Optional[JSON]:
    if not obj or not isinstance(obj, dict):
        return None

    if is_tool_call(obj):
        return obj

    # allow {"toolname": {...}}
    if len(obj) == 1:
        k = next(iter(obj.keys()))
        v = obj.get(k)
        if isinstance(k, str) and k in allowed_tools and isinstance(v, dict):
            return {"tool": k, "args": v}

    return obj


def make_router_system(allowed_tools: List[str], tool_schema_map: Dict[str, JSON]) -> JSON:
    schemas = {t: tool_schema_map.get(t) for t in allowed_tools if t in tool_schema_map}
    return {
        "role": "system",
        "content": (
            "You are a routing assistant.\n"
            "Return ONLY valid JSON. No markdown. No explanation.\n\n"
            f"Available tools: {', '.join(allowed_tools)}\n\n"
            "If a tool is required, output exactly:\n"
            '  {"tool":"<tool_name>","args":{...}}\n'
            "Args must match the tool JSON schema.\n\n"
            "Tool schemas:\n"
            f"{json.dumps(schemas, ensure_ascii=False)}\n\n"
            'If no tool needed: {"answer":"..."}\n'
        ),
    }


def make_final_system() -> JSON:
    return {
        "role": "system",
        "content": (
            "You are an assistant.\n"
            "Use ONLY the tool_result JSON provided.\n"
            "If tool_result contains an error, reply exactly with that error.\n"
            "Do NOT guess, invent, or add any extra customer details.\n"
            "Return plain text only.\n"
        ),
    }


def make_tool_context_msg(tool_name: str, tool_args: JSON, tool_result: Any) -> JSON:
    return {
        "role": "user",
        "content": json.dumps(
            {"tool_used": tool_name, "tool_args": tool_args, "tool_result": tool_result},
            ensure_ascii=False,
        ),
    }

# Mode handlers
def request_wants_tools(client_payload: JSON) -> bool:
    return bool(extract_tool_names_from_openai_tools(client_payload.get("tools")))


async def handle_proxy_mode(client_payload: JSON, *, req_id: str, t_start_ms: float) -> JSONResponse:
    # proxy = single upstream call, but we still wrap the response to add stats consistently
    upstream_payload = strip_tool_fields_for_upstream(client_payload)

    t0 = now_ms()
    try:
        llm = await call_upstream(upstream_payload)
    except httpx.HTTPStatusError as e:
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return error_response(400, str(e))
    t1 = now_ms()

    content = extract_assistant_content(llm).strip()
    usage = extract_usage(llm)

    stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))


    resp = build_chat_completion_response(
        model=str(llm.get("model") or client_payload.get("model") or ""),
        content=content,
        created=llm.get("created"),
        usage=usage,
        upstream_id=llm.get("id"),
        stats=stats,
    )
    return respond(resp)


async def run_router_llm(
    client_payload: JSON,
    messages: List[JSON],
    allowed_tools: List[str],
    tool_schema_map: Dict[str, JSON],
) -> Tuple[Optional[JSON], JSON, str, float]:
    payload = strip_tool_fields_for_upstream(client_payload, drop_response_format=True)
    payload["messages"] = [make_router_system(allowed_tools, tool_schema_map)] + messages

    t0 = now_ms()
    llm = await call_upstream(payload)
    t1 = now_ms()

    text = extract_assistant_content(llm)
    obj = normalize_router_output(parse_json_object(text), allowed_tools)
    usage = extract_usage(llm)
    return obj, usage, text, (t1 - t0)


async def execute_tool(tool_name: str, tool_args: JSON) -> Any:
    return await app.state.mcp.call_tool(app.state.http, tool_name, dict(tool_args))


async def run_final_llm(
    client_payload: JSON,
    messages: List[JSON],
    tool_name: str,
    tool_args: JSON,
    tool_result: Any,
) -> Tuple[JSON, float]:
    payload = strip_tool_fields_for_upstream(client_payload)
    payload["messages"] = [make_final_system()] + messages + [make_tool_context_msg(tool_name, tool_args, tool_result)]

    t0 = now_ms()
    llm = await call_upstream(payload)
    t1 = now_ms()
    return llm, (t1 - t0)


async def handle_tool_mode(client_payload: JSON, *, req_id: str, t_start_ms: float) -> JSONResponse:
    messages = client_payload.get("messages", [])
    if not isinstance(messages, list):
        return error_response(400, "messages must be a list")

    client_tools = client_payload.get("tools")
    client_tool_names = extract_tool_names_from_openai_tools(client_tools)

    # list tools from MCP
    try:
        mcp_tools = await app.state.mcp.list_tools(app.state.http)
    except Exception as e:
        return error_response(502, f"MCP list_tools failed: {e}", err_type="mcp_error")

    mcp_tool_names = {t.strip() for t in mcp_tools if isinstance(t, str) and t.strip()}
    allowed_tools = [t for t in client_tool_names if t in mcp_tool_names]

    tool_schema_map = build_tool_schema_map(client_tools)

    # If no tools
    if not allowed_tools:
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))

        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content="No requested tools are available in MCP toolstore.",
            usage={},
            stats=stats,
        )
        return respond(resp)

    # router call
    try:
        router_obj, usage1, raw_text, router_ms = await run_router_llm(
            client_payload=client_payload,
            messages=messages,
            allowed_tools=allowed_tools,
            tool_schema_map=tool_schema_map,
        )
    except httpx.HTTPStatusError as e:
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return error_response(400, str(e))

    # If router failed to produce JSON, treat router output as answer
    if not router_obj:
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))

        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content=raw_text.strip(),
            usage=usage1,
            stats=stats,
        )
        return respond(resp)

    # If router says {"answer": "..."}
    if isinstance(router_obj, dict) and "answer" in router_obj and not is_tool_call(router_obj):
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))

        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content=str(router_obj.get("answer") or "").strip(),
            usage=usage1,
            stats=stats,
        )
        return respond(resp)

    # If router returns some other JSON, return it as string
    if not is_tool_call(router_obj):
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))

        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content=json.dumps(router_obj, ensure_ascii=False),
            usage=usage1,
            stats=stats,
        )
        return respond(resp)

    tool_name = router_obj["tool"]
    tool_args: JSON = router_obj["args"]

    # timezone override
    if tool_name == "create_googlemeet":
        tool_args["timezone"] = os.getenv("GOOGLE_MEET_TIMEZONE", DEFAULT_TIMEZONE)

    # validate tool args against schema
    schema = tool_schema_map.get(tool_name) or {}
    if isinstance(schema, dict) and schema:
        ok, reason = validate_args_against_schema(tool_args, schema)
        if not ok:
            return error_response(400, f"Invalid tool args: {reason}")

    # apply some defaults if schema hints exist
    if isinstance(schema, dict):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        if "timezone" in props:
            tool_args.setdefault("timezone", DEFAULT_TIMEZONE)
        if "duration_minutes" in props:
            tool_args.setdefault("duration_minutes", 60)
        if "meeting_title" in props:
            tool_args.setdefault("meeting_title", "Google Meet")

    # execute tool
    t_tool0 = now_ms()
    try:
        tool_result = await execute_tool(tool_name, tool_args)
    except Exception as e:
        return error_response(502, f"MCP tool call failed: {e}", err_type="mcp_error")
    t_tool1 = now_ms()

    # final call
    try:
        llm2, final_ms = await run_final_llm(
            client_payload=client_payload,
            messages=messages,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )
    except httpx.HTTPStatusError as e:
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return error_response(400, str(e))

    content2 = extract_assistant_content(llm2).strip()
    usage2 = extract_usage(llm2)
    usage = merge_usage(usage1, usage2)

    stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))


    resp = build_chat_completion_response(
        model=str(llm2.get("model") or client_payload.get("model") or ""),
        content=content2,
        created=llm2.get("created"),
        usage=usage,
        upstream_id=llm2.get("id"),
        stats=stats,
    )
    return respond(resp)

def build_strict_chat_payload(model: str, user_text: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Follow instructions strictly."},
            {"role": "user", "content": user_text},
        ],
        "max_completion_tokens": 1024,
        "stream": False,
        "stream_options": {"include_usage": False},
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "store": False,
    }

def build_strict_chat_payload_with_memory(model: str, user_text: str, history: list) -> dict:
    system_prompt = "Follow instructions strictly."

    # inject last user facts into system prompt
    if history:
        memory_lines = []
        for m in history:
            if m["role"] == "user":
                memory_lines.append(m["content"])

        if memory_lines:
            memory_text = "Previous conversation:\n" + "\n".join(memory_lines[-5:])
            system_prompt = system_prompt + "\n\n" + memory_text

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_completion_tokens": 1024,
        "stream": False,
        "stream_options": {"include_usage": False},
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "store": False,
    }
def _extract_bearer_token(req: Request) -> Optional[str]:
    auth = req.headers.get("Authorization") or req.headers.get("authorization")
    if not auth or not isinstance(auth, str):
        return None
    parts = auth.strip().split()
    if len(parts) != 2:
        return None
    if parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _decode_and_verify_jwt(token: str) -> JSON:
    payload = jwt.decode(
        token,
        JWT_SECRET,
        algorithms=JWT_ALGOS,
        options={"verify_signature": True, "verify_exp": False},  # No exp in your tokens, so keep verify_exp False
    )
    return payload if isinstance(payload, dict) else {}


def _allowed_models_from_payload(payload: JSON) -> List[str]:
    raw = payload.get("models")
    if not isinstance(raw, str):
        return []
    # comma-separated: "a,b,c"
    allowed = [m.strip() for m in raw.split(",") if m.strip()]
    return allowed


def _normalize_model_for_match(model: str) -> str:
    # Keep strict, only trim spaces.
    return (model or "").strip()


def enforce_jwt_model_access(req: Request, requested_model: str) -> JSON:
    if not JWT_REQUIRED:
        return {}

    token = _extract_bearer_token(req)
    if not token:
        # 401: authentication failed (missing token)
        raise PermissionError("Missing Authorization: Bearer <token>")

    try:
        payload = _decode_and_verify_jwt(token)
    except InvalidTokenError:
        raise PermissionError("Invalid JWT")
    except Exception:
        raise PermissionError("JWT decode failed")

    allowed_models = _allowed_models_from_payload(payload)
    if not allowed_models:
        # token valid but does not grant any models
        raise PermissionError("Token has no allowed models")

    req_model = _normalize_model_for_match(requested_model)
    if req_model not in allowed_models:
        # 403: authenticated but not authorized
        raise LookupError(f"Access denied for model '{req_model}'")

    return payload

def responses_input_to_messages(inp: Any) -> List[Dict[str, Any]]:
    # If input is list of messages, convert directly
    if isinstance(inp, list):
        out = []
        for m in inp:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str):
                out.append({"role": role, "content": content})
        if out:
            return out

    # If input is string, treat as user message
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]

    # fallback: dump anything else as user text
    return [{"role": "user", "content": json.dumps(inp, ensure_ascii=False)}]

# Endpoints
@app.get("/health")
async def health(req: Request):
    return await forward_raw(req, "/health")


@app.get("/deploy")
async def deploy(req: Request):
    return await forward_raw(req, "/deploy")


@app.get("/models")
async def models(req: Request):
    return await forward_raw(req, "/models")


@app.get("/models/{provider}/{model_name}")
async def model_details(provider: str, model_name: str, req: Request):
    return await forward_raw(req, f"/models/{provider}/{model_name}")


@app.get("/lscpu")
async def lscpu(req: Request):
    return await forward_raw(req, "/lscpu")

@app.post("/chat/completions")
async def chat_completions(req: Request):
    t_start = now_ms()
    req_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        client_payload: JSON = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    if client_payload.get("stream") is True:
        return error_response(400, "stream=true not supported by this gateway")
    
    #JWT model access check
    requested_model = client_payload.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        return error_response(400, "Missing 'model' in request")
    
    try:
        enforce_jwt_model_access(req, requested_model)
    except Exception:
        return forbidden()
    
    mode = "tool" if request_wants_tools(client_payload) else "proxy"
    logger.info("request mode=%s req_id=%s", mode, req_id)

    if mode == "proxy":
        return await handle_proxy_mode(client_payload, req_id=req_id, t_start_ms=t_start)
    else:
        return await handle_tool_mode(client_payload, req_id=req_id, t_start_ms=t_start)

@app.post("/responses")
async def responses(req: Request):
    t_start = now_ms()
    req_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        body: JSON = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    if body.get("stream") is True:
        return error_response(400, "stream=true not supported by this gateway")

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response(400, "Missing 'model' in request")

    inp = body.get("input", "")
    messages_from_input = responses_input_to_messages(inp)

    conv_id = body.get("conversation_id")
    history: list[dict[str, Any]] = []

    if isinstance(conv_id, str) and conv_id.strip():
        conv_id = conv_id.strip()
        history = CONVERSATIONS.get(conv_id, []).copy()
    else:
        conv_id = None

    # add user message
    history.extend(messages_from_input)
    if len(history) > MAX_TURNS:
        history = history[-MAX_TURNS:]

    client_payload = {
    "model": model,
    "messages": history,
    "max_completion_tokens": 1024,
    "stream": False,
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "store": False,
    }


    #Allow tools only if it is consists in request body
    for k in ("tools", "tool_choice", "parallel_tool_calls", "response_format"):
        if k in body:
            client_payload[k] = body[k]
    
    # optional pass-through
    for k in ("tools", "tool_choice", "parallel_tool_calls", "response_format",
              "temperature", "top_p", "n", "max_completion_tokens"):
        if k in body:
            client_payload[k] = body[k]
    try:
        enforce_jwt_model_access(req, model)
    except PermissionError as e:
        return error_response(401, str(e), err_type="auth_error")
    except LookupError as e:
        return error_response(403, str(e), err_type="access_denied")

    mode = "tool" if request_wants_tools(client_payload) else "proxy"
    logger.info("responses mode=%s req_id=%s", mode, req_id)

    if mode == "proxy":
        chat_resp = await handle_proxy_mode(client_payload, req_id=req_id, t_start_ms=t_start)
    else:
        chat_resp = await handle_tool_mode(client_payload, req_id=req_id, t_start_ms=t_start)

    # parse chat completion result
    try:
        chat_data = json.loads(chat_resp.body.decode("utf-8"))
    except Exception:
        return chat_resp

    # pass-through errors
    if isinstance(chat_data, dict) and "error" in chat_data:
        return respond(chat_data, status_code=chat_resp.status_code)

    # extract assistant text
    text = ""
    try:
        text = chat_data["choices"][0]["message"].get("content", "") or ""
    except Exception:
        text = ""

    # save conversation memory (only if conv_id provided)
    if conv_id:
        new_history = history + [{"role": "assistant", "content": text}]
        CONVERSATIONS[conv_id] = new_history[-MAX_TURNS:]

    now_s = int(time.time())
    resp_out: JSON = {
        "id": chat_data.get("id") or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": now_s,
        "status": "completed",
        "completed_at": now_s,
        "error": None,
        "model": chat_data.get("model", model),
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }

    if isinstance(chat_data.get("usage"), dict):
        resp_out["usage"] = chat_data["usage"]
    if isinstance(chat_data.get("stats"), dict):
        resp_out["stats"] = chat_data["stats"]

    return respond(resp_out, status_code=chat_resp.status_code)

@app.post("/conversations/reset")
async def reset_conversation(req: Request):
    try:
        body: JSON = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    conv_id = body.get("conversation_id")
    if not isinstance(conv_id, str) or not conv_id.strip():
        return error_response(400, "conversation_id is required")

    conv_id = conv_id.strip()
    CONVERSATIONS.pop(conv_id, None)
    return respond({"ok": True, "conversation_id": conv_id, "reset": True})


if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        "gateway.app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "9000")),
        log_config=LOGGING_CONFIG,
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True,
        lifespan="on",
    )
