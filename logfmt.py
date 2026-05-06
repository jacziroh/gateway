from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

JSON = Dict[str, Any]

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _short(s: Any, n: int = 120) -> str:
    if s is None:
        return ""
    t = str(s)
    return t if len(t) <= n else t[: n - 1] + "…"


def _compact_json(obj: Any, n: int = 140) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return _short(s, n)


class DemoLogger:
    """
    Human-friendly, one-line-per-event logger.
    - Prints to stderr
    - Also writes per-request logs to LOG_DIR/requests/<rid>.log
    """

    def __init__(self, name: str = "gateway", level: str = "INFO"):
        self.name = name
        self.level = LEVELS.get(level.upper(), 20)

        # # ---- file logging ----
        # self.log_dir = os.getenv("LOG_DIR", "./logs")
        # self.req_dir = os.path.join(self.log_dir, "requests")

        # # Create directories once at startup
        # os.makedirs(self.req_dir, exist_ok=True)

    # def _write_req_file(self, rid: str, line: str) -> None:
    #     # One file per request id
    #     path = os.path.join(self.req_dir, f"{rid}.log")
    #     try:
    #         with open(path, "a", encoding="utf-8") as f:
    #             f.write(line + "\n")
    #     except Exception:
    #         # Never break request flow because of logging
    #         pass

    def log(self, level: str, rid: str, tag: str, msg: str, extra: Any = None) -> None:
        lvl = LEVELS.get(level.upper(), 20)
        if lvl < self.level:
            return

        extra_s = ""
        if extra is not None:
            extra_s = "  " + _compact_json(extra)

        line = f"{_now()}  {rid:<8}  {tag:<5} {msg}{extra_s}"

        # stdout/stderr log (what you already see)
        print(line, file=sys.stderr, flush=True)

        # # per-request file log
        # self._write_req_file(rid, line)

    def info(self, rid: str, tag: str, msg: str, extra: Any = None) -> None:
        self.log("INFO", rid, tag, msg, extra)

    def debug(self, rid: str, tag: str, msg: str, extra: Any = None) -> None:
        self.log("DEBUG", rid, tag, msg, extra)

    def warn(self, rid: str, tag: str, msg: str, extra: Any = None) -> None:
        self.log("WARN", rid, tag, msg, extra)

    def error(self, rid: str, tag: str, msg: str, extra: Any = None) -> None:
        self.log("ERROR", rid, tag, msg, extra)


def get_demo_logger() -> DemoLogger:
    level = os.getenv("DEMO_LOG_LEVEL", "INFO")
    name = os.getenv("DEMO_LOG_NAME", "gateway")
    return DemoLogger(name=name, level=level)
