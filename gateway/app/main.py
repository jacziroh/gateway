import csv
import json
import logging
import os
import re
import sqlite3
import time
import uuid

import torch
import torch.nn.functional as F

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jwt import ExpiredSignatureError, InvalidTokenError
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration


from gateway.app.mcp_client import build_mcp_client

JSON = Dict[str, Any]
logger = logging.getLogger("kai-gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# Config
DEPLOY_MAP_PATH = os.getenv("DEPLOY_MAP_PATH", "./deploy-map.json")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "120"))
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
UPSTREAM_CHAT_PATH = os.getenv("UPSTREAM_CHAT_PATH", "/chat/completions")
UPSTREAM_CHAT_STREAM_PATH = os.getenv("UPSTREAM_CHAT_STREAM_PATH", UPSTREAM_CHAT_PATH)
ELEPHANT_DB_PATH = "/home/jacob/Documents/dev/gitelephant/elephant/bin/bin/elephant_routing.db"

# Multimodal upstream paths
UPSTREAM_IMAGES_GENERATE_PATH = os.getenv("UPSTREAM_IMAGES_GENERATE_PATH", "/images/generations")
UPSTREAM_IMAGES_EDIT_PATH = os.getenv("UPSTREAM_IMAGES_EDIT_PATH", "/images/edits")
UPSTREAM_AUDIO_TRANSCRIPTIONS_PATH = os.getenv("UPSTREAM_AUDIO_TRANSCRIPTIONS_PATH", "/audio/transcriptions")
UPSTREAM_AUDIO_TRANSLATIONS_PATH = os.getenv("UPSTREAM_AUDIO_TRANSLATIONS_PATH", "/audio/translations")
UPSTREAM_AUDIO_SPEECH_PATH = os.getenv("UPSTREAM_AUDIO_SPEECH_PATH", "/audio/speech")
UPSTREAM_EMBEDDINGS_PATH = os.getenv("UPSTREAM_EMBEDDINGS_PATH", "/embeddings")
UPSTREAM_SENTIMENT_PATH = os.getenv("UPSTREAM_SENTIMENT_PATH", "/sentiment")
UPSTREAM_TEXT_TRANSLATIONS_PATH = os.getenv("UPSTREAM_TEXT_TRANSLATIONS_PATH","/text/translations")
UPSTREAM_IMAGE_CLASSIFY_PATH = os.getenv("UPSTREAM_IMAGE_CLASSIFY_PATH","/image/classify")
UPSTREAM_FILL_MASK_PATH = os.getenv("UPSTREAM_FILL_MASK_PATH","/fill-mask")
# JWT
JWT_SECRET = os.getenv("KAI_JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("KAI_JWT_SECRET is not set")

JWT_ALGOS = ["HS256"]
JWT_REQUIRED = os.getenv("JWT_REQUIRED", "true").lower() == "true"


# Small helpers
def now_ms() -> float:
    return time.perf_counter() * 1000.0


def log_latency_csv(
    req_id: str,
    tool: str,
    model: str,
    mcp_list_ms: float,
    llm1_ms: float,
    mcp_exec_ms: float,
    llm2_ms: float,
    total_ms: float,
    csv_path: str = "latency_log.csv",
):
    path = Path(csv_path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "timestamp",
                    "req_id",
                    "tool",
                    "model",
                    "mcp_list_ms",
                    "llm1_ms",
                    "mcp_exec_ms",
                    "llm2_ms",
                    "total_ms",
                ]
            )
        w.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                req_id,
                tool,
                model,
                f"{mcp_list_ms:.1f}",
                f"{llm1_ms:.1f}",
                f"{mcp_exec_ms:.1f}",
                f"{llm2_ms:.1f}",
                f"{total_ms:.1f}",
            ]
        )


def respond(payload: Any, status_code: int = 200) -> JSONResponse:
    r = JSONResponse(payload, status_code=status_code)
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


def _sse_data(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def make_public_stats(*, completion_time_ms: float) -> JSON:
    return {"completion_time": completion_time_ms, "memory": False}


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

    out: JSON = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": content, "role": "assistant"},
            }
        ],
        "created": created_ts,
        "id": resp_id,
        "model": model,
        "object": "chat.completion",
        "service_tier": service_tier,
    }

    if isinstance(stats, dict):
        out["stats"] = stats
    if isinstance(usage, dict):
        out["usage"] = usage

    return out


def _openai_chunk(
    *,
    model: str,
    content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    created: Optional[int] = None,
    resp_id: Optional[str] = None,
    usage: Optional[dict] = None,
) -> dict:
    return {
        "id": resp_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {k: v for k, v in (("role", role), ("content", content)) if v is not None},
                "finish_reason": finish_reason,
            }
        ],
        **({"usage": usage} if isinstance(usage, dict) else {}),
    }


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
        idx = i % len(urls)
        url = urls[idx]
        self._rr[model] = (i + 1) % len(urls)

        logger.info("rr_pick model=%s idx=%s selected=%s next_idx=%s pid=%s", model, idx, url, self._rr[model], os.getpid())
        return url

    def pick_any(self) -> str:
        self._load_if_needed()
        for urls in self._data.values():
            if urls:
                return urls[0]
        raise KeyError("No upstream URLs found in deploy-map.json")


# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_S,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=20),
    )
    app.state.deploy_map = DeployMap(DEPLOY_MAP_PATH)
    app.state.mcp = build_mcp_client()
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
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

    for k, v in r.headers.items():
        if k.lower() in HOP_HEADERS:
            continue
        resp.headers[k] = v

    resp.headers["X-Served-By"] = "kai-tools-gateway"
    return resp


# Upstream helpers
def ensure_upstream_defaults(payload: JSON) -> JSON:
    p = dict(payload)
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


async def call_upstream(payload: JSON, *, req_id: str = "-") -> JSON:
    payload = ensure_upstream_defaults(payload)

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Missing 'model' in request")

    model = model.strip()
    base = app.state.deploy_map.pick_url(model)
    url = base.rstrip("/") + "/" + UPSTREAM_CHAT_PATH.lstrip("/")

    logger.info("req_id=%s model=%s selected_upstream=%s", req_id, model, url)
    r = await app.state.http.post(url, json=payload)
    logger.info("req_id=%s model=%s upstream_status=%s upstream=%s", req_id, model, r.status_code, url)

    if r.status_code >= 400:
        logger.error("req_id=%s upstream_error_body=%s", req_id, r.text)

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
        "content": json.dumps({"tool_used": tool_name, "tool_args": tool_args, "tool_result": tool_result}, ensure_ascii=False),
    }


def get_client_ip(req: Request) -> str:
    if req.client and req.client.host:
        return req.client.host
    return "unknown"


def get_user(req: Request) -> str:
    token = _extract_bearer_token(req)
    if not token:
        return "unknown"
    try:
        payload = _decode_and_verify_jwt(token)
        return payload.get("sub", "unknown")
    except Exception:
        return "unknown"


def print_audit(req: Request, model: str, usage: dict, status: str):
    user = get_user(req)
    ip = get_client_ip(req)

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    print(
        f"[{timestamp}]"
        f"[{user}]"
        f"[inference]"
        f"[{model}] "
        f"input_tokens={input_tokens} "
        f"output_tokens={output_tokens} "
        f"ip={ip} "
        f"status={status}",
        flush=True,
    )


def get_elephant_ip_for_user(user_id):
    conn = sqlite3.connect(ELEPHANT_DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT e.elephant_ip
        FROM user_ele_mapping m
        JOIN elephant_info e ON m.elephant_id = e.elephant_id
        WHERE m.user_id = ?
    """,
        (user_id,),
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return row[0]


async def forward_to_user_elephant(payload: dict, jwt_payload: dict):
    user_id = jwt_payload.get("sub")
    if not user_id:
        raise ValueError("Token missing user.")

    elephant_ip = get_elephant_ip_for_user(user_id)
    if not elephant_ip:
        raise ValueError(f"No elephant mapping found for user: {user_id}")

    guid = uuid.uuid4().hex[:8]
    target_url = f"{elephant_ip.rstrip('/')}/memory/match/{guid}"

    response = await app.state.http.post(target_url, json=payload)
    response.raise_for_status()

    data = response.json()
    return data if isinstance(data, dict) else {"result": data}


# Mode handlers
def request_wants_tools(client_payload: JSON) -> bool:
    return bool(extract_tool_names_from_openai_tools(client_payload.get("tools")))


async def handle_proxy_mode(req: Request, client_payload: JSON, *, req_id: str, t_start_ms: float) -> JSONResponse:
    upstream_payload = strip_tool_fields_for_upstream(client_payload)
    model_name = str(client_payload.get("model") or "")

    try:
        llm = await call_upstream(upstream_payload, req_id=req_id)
    except httpx.HTTPStatusError as e:
        print_audit(req, model_name, {}, "failure")
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, model_name, {}, "failure")
        return error_response(400, str(e))

    content = extract_assistant_content(llm).strip()
    usage = extract_usage(llm)
    stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))

    print_audit(req, model_name, usage, "success")

    resp = build_chat_completion_response(
        model=str(llm.get("model") or model_name),
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
    *,
    req_id: str,
) -> Tuple[Optional[JSON], JSON, str]:
    payload = strip_tool_fields_for_upstream(client_payload, drop_response_format=True)
    payload["messages"] = [make_router_system(allowed_tools, tool_schema_map)] + messages

    llm = await call_upstream(payload, req_id=req_id)
    text = extract_assistant_content(llm)
    obj = normalize_router_output(parse_json_object(text), allowed_tools)
    usage = extract_usage(llm)
    return obj, usage, text


async def execute_tool(tool_name: str, tool_args: JSON) -> Any:
    return await app.state.mcp.call_tool(app.state.http, tool_name, dict(tool_args))


async def run_final_llm(
    client_payload: JSON,
    messages: List[JSON],
    tool_name: str,
    tool_args: JSON,
    tool_result: Any,
    *,
    req_id: str,
) -> JSON:
    payload = strip_tool_fields_for_upstream(client_payload)
    payload["messages"] = [make_final_system()] + messages + [make_tool_context_msg(tool_name, tool_args, tool_result)]
    return await call_upstream(payload, req_id=req_id)


async def handle_tool_mode(req: Request, client_payload: JSON, *, req_id: str, t_start_ms: float) -> JSONResponse:
    messages = client_payload.get("messages", [])
    if not isinstance(messages, list):
        print_audit(req, str(client_payload.get("model") or ""), {}, "failure")
        return error_response(400, "messages must be a list")

    client_tools = client_payload.get("tools")
    client_tool_names = extract_tool_names_from_openai_tools(client_tools)

    t_mcp_list_start = now_ms()
    try:
        mcp_tools = await app.state.mcp.list_tools(app.state.http)
    except Exception as e:
        print_audit(req, str(client_payload.get("model") or ""), {}, "failure")
        return error_response(502, f"MCP list_tools failed: {e}", err_type="mcp_error")
    t_mcp_list_ms = now_ms() - t_mcp_list_start

    mcp_tool_names = {t.strip() for t in mcp_tools if isinstance(t, str) and t.strip()}
    allowed_tools = [t for t in client_tool_names if t in mcp_tool_names]
    tool_schema_map = build_tool_schema_map(client_tools)

    if not allowed_tools:
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))
        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content="No requested tools are available in MCP toolstore.",
            usage={},
            stats=stats,
        )
        return respond(resp)

    t_llm1_start = now_ms()
    try:
        router_obj, usage1, raw_text = await run_router_llm(
            client_payload=client_payload,
            messages=messages,
            allowed_tools=allowed_tools,
            tool_schema_map=tool_schema_map,
            req_id=req_id,
        )
    except httpx.HTTPStatusError as e:
        print_audit(req, str(client_payload.get("model") or ""), {}, "failure")
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, str(client_payload.get("model") or ""), {}, "failure")
        return error_response(400, str(e))
    t_llm1_ms = now_ms() - t_llm1_start

    if not router_obj:
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))
        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content=raw_text.strip(),
            usage=usage1,
            stats=stats,
        )
        return respond(resp)

    if isinstance(router_obj, dict) and "answer" in router_obj and not is_tool_call(router_obj):
        stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))
        resp = build_chat_completion_response(
            model=str(client_payload.get("model") or ""),
            content=str(router_obj.get("answer") or "").strip(),
            usage=usage1,
            stats=stats,
        )
        return respond(resp)

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

    if tool_name == "create_googlemeet":
        tool_args["timezone"] = os.getenv("GOOGLE_MEET_TIMEZONE", DEFAULT_TIMEZONE)

    schema = tool_schema_map.get(tool_name) or {}
    if isinstance(schema, dict) and schema:
        ok, reason = validate_args_against_schema(tool_args, schema)
        if not ok:
            return error_response(400, f"Invalid tool args: {reason}")

    if isinstance(schema, dict):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        if "timezone" in props:
            tool_args.setdefault("timezone", DEFAULT_TIMEZONE)
        if "duration_minutes" in props:
            tool_args.setdefault("duration_minutes", 60)
        if "meeting_title" in props:
            tool_args.setdefault("meeting_title", "Google Meet")

    t_mcp_exec_start = now_ms()
    try:
        tool_result = await execute_tool(tool_name, tool_args)
    except Exception as e:
        print_audit(req, str(client_payload.get("model") or ""), usage1, "failure")
        return error_response(502, f"MCP tool call failed: {e}", err_type="mcp_error")
    t_mcp_exec_ms = now_ms() - t_mcp_exec_start

    t_llm2_start = now_ms()
    try:
        llm2 = await run_final_llm(
            client_payload=client_payload,
            messages=messages,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            req_id=req_id,
        )
    except httpx.HTTPStatusError as e:
        print_audit(req, str(client_payload.get("model") or ""), usage1, "failure")
        return upstream_error_response(e)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, str(client_payload.get("model") or ""), usage1, "failure")
        return error_response(400, str(e))
    t_llm2_ms = now_ms() - t_llm2_start

    t_total_ms = now_ms() - t_start_ms
    print(
        f"LATENCY req_id={req_id} tool={tool_name} "
        f"mcp_list={t_mcp_list_ms:.1f}ms llm1={t_llm1_ms:.1f}ms "
        f"mcp_exec={t_mcp_exec_ms:.1f}ms llm2={t_llm2_ms:.1f}ms "
        f"total={t_total_ms:.1f}ms",
        flush=True,
    )
    log_latency_csv(
        req_id=req_id,
        tool=tool_name,
        model=str(llm2.get("model") or client_payload.get("model") or ""),
        mcp_list_ms=t_mcp_list_ms,
        llm1_ms=t_llm1_ms,
        mcp_exec_ms=t_mcp_exec_ms,
        llm2_ms=t_llm2_ms,
        total_ms=t_total_ms,
    )

    content2 = extract_assistant_content(llm2).strip()
    usage2 = extract_usage(llm2)
    usage = merge_usage(usage1, usage2)
    stats = make_public_stats(completion_time_ms=(now_ms() - t_start_ms))
    print_audit(req, str(llm2.get("model") or client_payload.get("model") or ""), usage, "success")
    resp = build_chat_completion_response(
        model=str(llm2.get("model") or client_payload.get("model") or ""),
        content=content2,
        created=llm2.get("created"),
        usage=usage,
        upstream_id=llm2.get("id"),
        stats=stats,
    )
    return respond(resp)


# JWT helpers
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
        # options={"verify_signature": True, "verify_exp": False},
    )
    return payload if isinstance(payload, dict) else {}


def _allowed_models_from_payload(payload: JSON) -> List[str]:
    models = payload.get("models")

    if isinstance(models, list):
        return [m.strip() for m in models if isinstance(m, str) and m.strip()]

    if isinstance(models, str):
        return [models.strip()] if models.strip() else []

    return []


def enforce_jwt_model_access(req: Request, requested_model: str) -> JSON:
    if not JWT_REQUIRED:
        return {}

    token = _extract_bearer_token(req)
    if not token:
        raise PermissionError("Missing token")

    try:
        payload = _decode_and_verify_jwt(token)
    except ExpiredSignatureError:
        raise PermissionError("Token expired")
    except InvalidTokenError:
        raise PermissionError("Invalid token")

    allowed_models = _allowed_models_from_payload(payload)
    if not allowed_models:
        raise PermissionError("No models found in token")

    requested_model = requested_model.strip()
    if requested_model not in allowed_models:
        raise LookupError(f"Access denied for model '{requested_model}'")

    return payload


async def stream_chat_completion(req: Request, client_payload: JSON, *, req_id: str):
    requested_model = str(client_payload.get("model") or "").strip()
    if not requested_model:
        return error_response(400, "Missing 'model' in request")

    try:
        enforce_jwt_model_access(req, requested_model)
    except PermissionError:
        print_audit(req, requested_model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, requested_model, {}, "failure")
        return error_response(403, "Forbidden")

    base = app.state.deploy_map.pick_url(requested_model)
    url = base.rstrip("/") + "/" + UPSTREAM_CHAT_STREAM_PATH.lstrip("/")

    payload = ensure_upstream_defaults(dict(client_payload))
    payload["stream"] = True

    stream_id = f"kai-chat-comp-{uuid.uuid4().hex[:12]}"

    async def event_gen() -> AsyncGenerator[bytes, None]:
        try:
            async with app.state.http.stream(
                "POST",
                url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    print_audit(req, requested_model, {}, "failure")
                    yield _sse_data(
                        {
                            "error": {
                                "message": f"Upstream error {r.status_code}: {body.decode('utf-8', 'ignore')}",
                                "type": "upstream_error",
                            }
                        }
                    )
                    yield b"data: [DONE]\n\n"
                    return

                print_audit(req, requested_model, {}, "success")

                # OpenAI-like first chunk: role announcement
                yield _sse_data(
                    _openai_chunk(
                        model=requested_model,
                        role="assistant",
                        resp_id=stream_id,
                    )
                )

                async for line in r.aiter_lines():
                    if not line:
                        continue

                    text = line.strip()
                    if not text:
                        continue

                    if text.startswith("data:"):
                        text = text[5:].strip()

                    if text == "[DONE]":
                        break

                    token = None
                    finish_reason = None
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            try:
                                token = obj["choices"][0]["delta"].get("content")
                            except Exception:
                                try:
                                    token = obj["choices"][0]["message"].get("content")
                                except Exception:
                                    token = obj.get("content")

                            try:
                                finish_reason = obj["choices"][0].get("finish_reason")
                            except Exception:
                                finish_reason = None
                    except Exception:
                        token = text

                    if token is not None and token != "":
                        yield _sse_data(
                            _openai_chunk(
                                model=requested_model,
                                content=str(token),
                                resp_id=stream_id,
                            )
                        )

                    if finish_reason:
                        yield _sse_data(
                            _openai_chunk(
                                model=requested_model,
                                finish_reason=finish_reason,
                                resp_id=stream_id,
                            )
                        )

                usage = None
                if isinstance(client_payload.get("stream_options"), dict) and client_payload["stream_options"].get("include_usage") is True:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                yield _sse_data(
                    _openai_chunk(
                        model=requested_model,
                        finish_reason="stop",
                        resp_id=stream_id,
                        usage=usage,
                    )
                )
                yield b"data: [DONE]\n\n"

        except Exception as e:
            print_audit(req, requested_model, {}, "failure")
            yield _sse_data({"error": {"message": str(e), "type": "server_error"}})
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Served-By": "kai-tools-gateway",
        },
    )


# Generic proxy helpers for image/audio/embeddings routes
async def proxy_json_request(req: Request, upstream_path: str) -> JSONResponse:
    """Proxy a JSON-body request: parse JSON, auth, route, forward, audit, respond."""
    t_start = now_ms()

    try:
        payload: JSON = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response(400, "Missing 'model'")
    model = model.strip()

    try:
        enforce_jwt_model_access(req, model)
    except PermissionError:
        print_audit(req, model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, model, {}, "failure")
        return error_response(403, "Forbidden")

    try:
        base = app.state.deploy_map.pick_url(model)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, model, {}, "failure")
        return error_response(400, str(e))

    url = base.rstrip("/") + "/" + upstream_path.lstrip("/")

    try:
        r = await app.state.http.post(url, json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print_audit(req, model, {}, "failure")
        return upstream_error_response(e)
    except httpx.RequestError as e:
        print_audit(req, model, {}, "failure")
        return error_response(502, f"Upstream unreachable: {e}", err_type="upstream_error")

    data = r.json() if r.content else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    print_audit(req, model, usage, "success")
    data["stats"] = make_public_stats(completion_time_ms=(now_ms() - t_start))
    return respond(data)


async def proxy_multipart_request(req: Request, file_key: str, upstream_path: str) -> JSONResponse:
    """Proxy a multipart/form-data request: parse form, auth, route, forward file + fields, audit, respond."""
    t_start = now_ms()

    try:
        form = await req.form()
    except Exception:
        return error_response(400, "Invalid multipart form body")

    model = form.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response(400, "Missing 'model'")
    model = model.strip()

    upload = form.get(file_key)
    if upload is None or not hasattr(upload, "read"):
        return error_response(400, f"Missing '{file_key}' file")
    file_bytes = await upload.read()
    if not file_bytes:
        return error_response(400, f"Uploaded {file_key} is empty")

    try:
        enforce_jwt_model_access(req, model)
    except PermissionError:
        print_audit(req, model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, model, {}, "failure")
        return error_response(403, "Forbidden")

    try:
        base = app.state.deploy_map.pick_url(model)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, model, {}, "failure")
        return error_response(400, str(e))

    url = base.rstrip("/") + "/" + upstream_path.lstrip("/")

    files = {
        file_key: (
            getattr(upload, "filename", "file"),
            file_bytes,
            getattr(upload, "content_type", "application/octet-stream"),
        )
    }
    data_fields = {k: v for k, v in form.multi_items() if k != file_key and isinstance(v, str)}

    try:
        r = await app.state.http.post(url, data=data_fields, files=files)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print_audit(req, model, {}, "failure")
        return upstream_error_response(e)
    except httpx.RequestError as e:
        print_audit(req, model, {}, "failure")
        return error_response(502, f"Upstream unreachable: {e}", err_type="upstream_error")

    result = r.json() if r.content else {}
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    print_audit(req, model, usage, "success")
    result["stats"] = make_public_stats(completion_time_ms=(now_ms() - t_start))
    return respond(result)


async def proxy_binary_response(req: Request, upstream_path: str) -> Response:
    """Proxy a JSON request that returns binary data (e.g. TTS audio)."""
    t_start = now_ms()

    try:
        payload: JSON = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response(400, "Missing 'model'")
    model = model.strip()

    try:
        enforce_jwt_model_access(req, model)
    except PermissionError:
        print_audit(req, model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, model, {}, "failure")
        return error_response(403, "Forbidden")

    try:
        base = app.state.deploy_map.pick_url(model)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, model, {}, "failure")
        return error_response(400, str(e))

    url = base.rstrip("/") + "/" + upstream_path.lstrip("/")

    try:
        r = await app.state.http.post(url, json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print_audit(req, model, {}, "failure")
        return upstream_error_response(e)
    except httpx.RequestError as e:
        print_audit(req, model, {}, "failure")
        return error_response(502, f"Upstream unreachable: {e}", err_type="upstream_error")

    print_audit(req, model, {}, "success")
    resp = Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "audio/mpeg"),
    )
    resp.headers["X-Served-By"] = "kai-tools-gateway"
    return resp

async def proxy_json_payload(req: Request, payload: JSON, upstream_path: str) -> JSONResponse:
    t_start = now_ms()

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response(400, "Missing 'model'")
    model = model.strip()

    try:
        enforce_jwt_model_access(req, model)
    except PermissionError:
        print_audit(req, model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, model, {}, "failure")
        return error_response(403, "Forbidden")

    try:
        base = app.state.deploy_map.pick_url(model)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print_audit(req, model, {}, "failure")
        return error_response(400, str(e))

    url = base.rstrip("/") + "/" + upstream_path.lstrip("/")

    try:
        r = await app.state.http.post(url, json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print_audit(req, model, {}, "failure")
        return upstream_error_response(e)
    except httpx.RequestError as e:
        print_audit(req, model, {}, "failure")
        return error_response(502, f"Upstream unreachable: {e}", err_type="upstream_error")

    data = r.json() if r.content else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    print_audit(req, model, usage, "success")
    data["stats"] = make_public_stats(completion_time_ms=(now_ms() - t_start))
    return respond(data)


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

    requested_model = client_payload.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        return error_response(400, "Missing 'model' in request")

    if client_payload.get("stream") is True:
        return await stream_chat_completion(req, client_payload, req_id=req_id)

    try:
        enforce_jwt_model_access(req, requested_model)
    except PermissionError:
        print_audit(req, requested_model, {}, "failure")
        return error_response(401, "Unauthorized")
    except LookupError:
        print_audit(req, requested_model, {}, "failure")
        return error_response(403, "Forbidden")
    except Exception:
        print_audit(req, requested_model, {}, "failure")
        return error_response(401, "Unauthorized")

    mode = "tool" if request_wants_tools(client_payload) else "proxy"
    logger.info("request mode=%s req_id=%s", mode, req_id)

    if mode == "proxy":
        return await handle_proxy_mode(req, client_payload, req_id=req_id, t_start_ms=t_start)
    return await handle_tool_mode(req, client_payload, req_id=req_id, t_start_ms=t_start)


@app.post("/memory/match")
async def memory_match(req: Request):
    try:
        payload = await req.json()
    except Exception:
        return error_response(400, "Invalid JSON body")

    token = _extract_bearer_token(req)
    if not token:
        return error_response(401, "Missing token")

    try:
        jwt_payload = _decode_and_verify_jwt(token)
    except ExpiredSignatureError:
        return error_response(401, "Token expired")
    except InvalidTokenError:
        return error_response(401, "Invalid token")

    try:
        result = await forward_to_user_elephant(payload, jwt_payload)
        return respond(result)
    except httpx.HTTPStatusError as e:
        return upstream_error_response(e)
    except Exception as e:
        return error_response(400, str(e))


@app.post("/images/generations")
async def images_gen(req: Request):
    return await proxy_json_request(req, UPSTREAM_IMAGES_GENERATE_PATH)


@app.post("/images/edits")
async def images_edit(req: Request):
    return await proxy_multipart_request(req, "image", UPSTREAM_IMAGES_EDIT_PATH)


@app.post("/audio/transcriptions")
async def audio_transcribe(req: Request):
    return await proxy_multipart_request(req, "file", UPSTREAM_AUDIO_TRANSCRIPTIONS_PATH)


@app.post("/audio/translations")
async def audio_translate(req: Request):
    return await proxy_multipart_request(req, "file", UPSTREAM_AUDIO_TRANSLATIONS_PATH)


@app.post("/audio/speech")
async def audio_speech(req: Request):
    return await proxy_binary_response(req, UPSTREAM_AUDIO_SPEECH_PATH)


@app.post("/embeddings")
async def embeddings(req: Request):
    return await proxy_json_request(req, UPSTREAM_EMBEDDINGS_PATH)

@app.post("/sentiment")
async def sentiment(req: Request):
    return await proxy_json_request(req, UPSTREAM_SENTIMENT_PATH)


@app.post("/text/translations")
async def text_generations(req: Request):
    return await proxy_json_request(req, UPSTREAM_TEXT_TRANSLATIONS_PATH)


@app.post("/image/classify")
async def image_classify(req: Request):
    return await proxy_multipart_request(req, "image", UPSTREAM_IMAGE_CLASSIFY_PATH)

@app.post("/fill-mask")
async def fill_mask(req: Request):
    return await proxy_json_request(req, UPSTREAM_FILL_MASK_PATH)


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
