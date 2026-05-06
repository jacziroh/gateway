from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


JSON = Dict[str, Any]


SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

DEFAULT_TIMEZONE_ID = os.getenv("GOOGLE_MEET_TIMEZONE", "Asia/Kolkata")
DEFAULT_DURATION_MINUTES = int(os.getenv("GOOGLE_MEET_DEFAULT_DURATION_MINUTES", "60"))
DEFAULT_MEETING_TITLE = os.getenv("GOOGLE_MEET_DEFAULT_TITLE", "Meeting")
DEFAULT_CALENDAR_ID = os.getenv("GOOGLE_MEET_CALENDAR_ID", "primary")
DEFAULT_HOST_EMAIL = os.getenv("GOOGLE_MEET_HOST_EMAIL", "")  # optional


SECRET_DIR = Path(os.getenv("GOOGLE_MEET_SECRET_DIR", "./mcp/app/secret")).resolve()
CREDENTIALS_FILE = Path(os.getenv("GOOGLE_MEET_CREDENTIALS_FILE", str(SECRET_DIR / "credentials.json")))
TOKEN_FILE = Path(os.getenv("GOOGLE_MEET_TOKEN_FILE", str(SECRET_DIR / "token.json")))


MEETING_TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")


def _parse_meeting_time(meeting_time: str, timezone_id: str) -> dt.datetime:
    tz = ZoneInfo(timezone_id)
    for fmt in MEETING_TIME_FORMATS:
        try:
            naive_dt = dt.datetime.strptime(meeting_time, fmt)
            return naive_dt.replace(tzinfo=tz)
        except ValueError:
            continue
    raise ValueError(
        f"meeting_time '{meeting_time}' must be like 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DDTHH:MM'"
    )


def _get_credentials() -> Credentials:
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Missing credentials.json at {CREDENTIALS_FILE}. "
            f"Set GOOGLE_MEET_CREDENTIALS_FILE or place it there."
        )

    creds: Optional[Credentials] = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # NOTE: this opens a local browser flow ON THE MCP MACHINE
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _build_event_body(
    *,
    meeting_title: str,
    meeting_start: dt.datetime,
    duration_minutes: int,
    timezone_id: str,
    host_email: str,
    other_email: str,
    meeting_notes: str,
) -> dict:
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be > 0")

    meeting_end = meeting_start + dt.timedelta(minutes=duration_minutes)

    attendees = []
    if host_email:
        attendees.append({"email": host_email})
    if other_email and other_email != host_email:
        attendees.append({"email": other_email})

    body = {
        "summary": meeting_title,
        "description": meeting_notes or "",
        "start": {"dateTime": meeting_start.isoformat(), "timeZone": timezone_id},
        "end": {"dateTime": meeting_end.isoformat(), "timeZone": timezone_id},
        "attendees": attendees,
        "reminders": {"useDefault": True},
        "conferenceData": {
            "createRequest": {
                "requestId": f"meet-{uuid.uuid4().hex}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }
    return body


def create_googlemeet(args: JSON) -> JSON:
    """
    Toolstore tool entrypoint.

    Expected args:
      - meeting_time (required): "YYYY-MM-DD HH:MM" or "YYYY-MM-DDTHH:MM"
      - other_email (required): attendee email
      - meeting_title (optional)
      - meeting_notes (optional)
      - duration_minutes (optional int)
      - timezone (optional, default Asia/Kolkata)
      - host_email (optional, default env GOOGLE_MEET_HOST_EMAIL)
    """
    meeting_time = str(args.get("meeting_time") or "").strip()
    other_email = str(args.get("other_email") or "").strip()

    if not meeting_time:
        raise ValueError("missing required arg: meeting_time")
    if not other_email:
        raise ValueError("missing required arg: other_email")

    meeting_title = str(args.get("meeting_title") or DEFAULT_MEETING_TITLE).strip() or DEFAULT_MEETING_TITLE
    meeting_notes = str(args.get("meeting_notes") or "").strip()
    timezone_id = DEFAULT_TIMEZONE_ID

    duration_minutes_raw = args.get("duration_minutes", DEFAULT_DURATION_MINUTES)
    duration_minutes = int(duration_minutes_raw)

    host_email = str(args.get("host_email") or DEFAULT_HOST_EMAIL).strip()

    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    start_dt = _parse_meeting_time(meeting_time, timezone_id)
    event_body = _build_event_body(
        meeting_title=meeting_title,
        meeting_start=start_dt,
        duration_minutes=duration_minutes,
        timezone_id=timezone_id,
        host_email=host_email,
        other_email=other_email,
        meeting_notes=meeting_notes,
    )

    event = (
        service.events()
        .insert(
            calendarId=DEFAULT_CALENDAR_ID,
            body=event_body,
            sendUpdates="all",
            conferenceDataVersion=1,
        )
        .execute()
    )
    print("DEBUG ARGS =", args)
    print("DEBUG timezone_id =", timezone_id)
    print("DEBUG meeting_time =", meeting_time)
    return {
        "summary": event.get("summary"),
        "status": event.get("status"),
        "timezone": timezone_id,
        "start": (event.get("start") or {}).get("dateTime"),
        "end": (event.get("end") or {}).get("dateTime"),
        "hangoutLink": event.get("hangoutLink"),
        "htmlLink": event.get("htmlLink"),
        "eventId": event.get("id"),
    }
