import asyncio
import json
import os
import socket
import sys
import time
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import machineid
import requests

from keygen_sh import validate
from keygen_sh.config import set_config, KeygenConfig
from keygen_sh.errors import KeygenError, Error


KEYGEN_ACCOUNT_ID = os.getenv("KEYGEN_ACCOUNT_ID")
KEYGEN_PRODUCT_ID = os.getenv("KEYGEN_PRODUCT_ID")
KEYGEN_TRIAL_POLICY_ID = os.getenv("KEYGEN_TRIAL_POLICY_ID")
KEYGEN_ADMIN_TOKEN = os.getenv("KEYGEN_ADMIN_TOKEN")
KEYGEN_PUBLIC_KEY = os.getenv("KEYGEN_PUBLIC_KEY", "")

KEYGEN_API_BASE = f"https://api.keygen.sh/v1/accounts/{KEYGEN_ACCOUNT_ID}"

SBOX_LICENSE_FILE = os.getenv(
    "SBOX_LICENSE_FILE",
    os.path.expanduser("~/.sbox/license.json")
)

MACHINE_FINGERPRINT_SECRET = os.getenv(
    "SBOX_MACHINE_FINGERPRINT_SECRET",
    "ziroh-sbox-machine-fingerprint-secret"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        value = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)

        # If Keygen/SDK gives datetime without timezone,
        # force it to UTC so subtraction works.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    except Exception:
        return None


def calculate_time_left(expiry: Optional[str]) -> Optional[str]:
    expiry_dt = parse_datetime(expiry)

    if not expiry_dt:
        return None

    now = datetime.now(timezone.utc)

    if expiry_dt.tzinfo is None:
        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)

    remaining = expiry_dt - now

    if remaining.total_seconds() <= 0:
        if time_left == "Expired":
            status = "expired"

    total_seconds = int(remaining.total_seconds())

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days} day(s), {hours} hour(s), {minutes} minute(s) remaining"

    if hours > 0:
        return f"{hours} hour(s), {minutes} minute(s) remaining"

    return f"{minutes} minute(s) remaining"

def get_machine_fingerprint() -> str:
    raw_machine_id = machineid.id()

    return hmac.new(
        MACHINE_FINGERPRINT_SECRET.encode("utf-8"),
        raw_machine_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def format_human_datetime(value: Optional[str]) -> Optional[str]:
    """
    Converts technical datetime into human-readable format.

    Example:
    2026-05-15T09:57:17.575870+00:00
    becomes:
    15 May 2026, 03:27 PM IST
    """
    dt = parse_datetime(value)

    if not dt:
        return None

    # Convert UTC time to IST.
    # If you want system local timezone instead, tell me.
    ist = timezone(timedelta(hours=5, minutes=30))
    dt = dt.astimezone(ist)

    return dt.strftime("%d %B %Y, %I:%M %p IST")


def load_cached_license() -> Optional[Dict[str, Any]]:
    path = Path(SBOX_LICENSE_FILE)

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_license(data: Dict[str, Any]) -> None:
    path = Path(SBOX_LICENSE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    os.chmod(path, 0o600)


def delete_cached_license() -> None:
    path = Path(SBOX_LICENSE_FILE)

    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def admin_headers() -> Dict[str, str]:
    if not KEYGEN_ADMIN_TOKEN:
        raise RuntimeError("KEYGEN_ADMIN_TOKEN is not set.")

    return {
        "Authorization": f"Bearer {KEYGEN_ADMIN_TOKEN}",
        "Content-Type": "application/vnd.api+json",
        "Accept": "application/vnd.api+json",
    }


def collect_customer_details() -> Dict[str, str]:
    print("\nEnter customer details for license registration.")

    name = input("Name: ").strip()
    email = input("Email: ").strip()
    phone = input("Phone number: ").strip()
    company = input("Company name: ").strip()

    if not name:
        name = "Unknown"

    if not email:
        email = "unknown@example.com"

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "company": company,
    }


def configure_keygen_sdk(license_key: str) -> None:
    if not KEYGEN_ACCOUNT_ID:
        raise RuntimeError("KEYGEN_ACCOUNT_ID is not set.")

    if not KEYGEN_PRODUCT_ID:
        raise RuntimeError("KEYGEN_PRODUCT_ID is not set.")

    set_config(KeygenConfig(
        api_url="https://api.keygen.sh",
        api_prefix="v1",
        api_version="v1.8",
        account=KEYGEN_ACCOUNT_ID,
        product=KEYGEN_PRODUCT_ID,
        license_key=license_key,
        public_key=KEYGEN_PUBLIC_KEY
    ))


async def sdk_validate_license(license_key: str, fingerprint: str):
    configure_keygen_sdk(license_key)

    # keygen-py validates using fingerprint list.
    return await validate([fingerprint], [])


def validate_license_sync(license_key: str, fingerprint: str):
    return asyncio.run(sdk_validate_license(license_key, fingerprint))


def create_trial_license(customer: Dict[str, str], fingerprint: str) -> Dict[str, Any]:
    """
    Creates 30-day trial license using Keygen API.

    Important:
    This needs KEYGEN_ADMIN_TOKEN.
    For production, this should ideally be done from company backend,
    not directly from customer machine.
    """
    if not KEYGEN_ACCOUNT_ID:
        raise RuntimeError("KEYGEN_ACCOUNT_ID is not set.")

    if not KEYGEN_TRIAL_POLICY_ID:
        raise RuntimeError("KEYGEN_TRIAL_POLICY_ID is not set.")

    url = f"{KEYGEN_API_BASE}/licenses"

    payload = {
        "data": {
            "type": "licenses",
            "attributes": {
                "name": f"SBox Trial - {customer.get('name')}",
                "metadata": {
                    "customer_name": customer.get("name"),
                    "customer_email": customer.get("email"),
                    "customer_phone": customer.get("phone"),
                    "customer_company": customer.get("company"),
                    "license_type": "trial",
                    "trial_days": 30,
                    "created_from": "sbox_cli",
                    "requested_at": utc_now_iso(),
                    "sbox_machine_fingerprint": fingerprint,
                    "sbox_hostname": socket.gethostname()
                }
            },
            "relationships": {
                "policy": {
                    "data": {
                        "type": "policies",
                        "id": KEYGEN_TRIAL_POLICY_ID
                    }
                }
            }
        }
    }

    response = requests.post(
        url,
        headers=admin_headers(),
        json=payload,
        timeout=30
    )

    body = response.json()

    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create trial license: {json.dumps(body, indent=2)}")

    return body


def get_license_key_from_create_response(body: Dict[str, Any]) -> str:
    data = body.get("data") or {}
    attrs = data.get("attributes") or {}

    key = attrs.get("key")

    if not key:
        raise RuntimeError("License created but key not found in Keygen response.")

    return key


def get_license_id_from_create_response(body: Dict[str, Any]) -> Optional[str]:
    data = body.get("data") or {}
    return data.get("id")


def get_license_id_from_sdk_data(license_data: Any) -> Optional[str]:
    return getattr(license_data, "id", None)


def get_license_expiry_from_sdk_data(license_data: Any) -> Optional[str]:
    expiry = getattr(license_data, "expiry", None)

    if expiry is None:
        return None

    return str(expiry)


def create_machine_for_license(
    license_id: str,
    fingerprint: str,
    customer: Dict[str, str]
) -> Optional[str]:
    """
    Creates a machine under the license so Keygen dashboard shows Machine ID.

    Demo-safe:
    - Always prints request step
    - Always has timeout
    - Always prints Keygen response
    - Does not silently hang/fail
    """
    if not license_id:
        print("Cannot create machine: license_id is empty.")
        return None

    if not KEYGEN_ACCOUNT_ID:
        print("Cannot create machine: KEYGEN_ACCOUNT_ID is not set.")
        return None

    if not KEYGEN_ADMIN_TOKEN:
        print("Cannot create machine: KEYGEN_ADMIN_TOKEN is not set.")
        return None

    url = f"{KEYGEN_API_BASE}/machines"

    payload = {
        "data": {
            "type": "machines",
            "attributes": {
                "fingerprint": fingerprint,
                "name": socket.gethostname(),
                "metadata": {
                    "app": "sbox",
                    "customer_name": customer.get("name"),
                    "customer_email": customer.get("email"),
                    "customer_phone": customer.get("phone"),
                    "customer_company": customer.get("company"),
                    "python": sys.version.split()[0],
                    "platform": sys.platform,
                    "created_from": "sbox_cli"
                }
            },
            "relationships": {
                "license": {
                    "data": {
                        "type": "licenses",
                        "id": license_id
                    }
                }
            }
        }
    }

    print("Calling Keygen machine creation API...")
    # print(f"Account ID: {KEYGEN_ACCOUNT_ID}")
    # print(f"License ID: {license_id}")
    # print(f"Fingerprint: {fingerprint}")

    try:
        response = requests.post(
            url,
            headers=admin_headers(),
            json=payload,
            timeout=20
        )
    except requests.exceptions.Timeout:
        print("Keygen machine creation timed out after 20 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Keygen machine creation request failed: {e}")
        return None

    try:
        body = response.json()
    except Exception:
        print(f"Keygen returned non-JSON response. HTTP {response.status_code}")
        print(response.text)
        return None

    # print(f"Create machine HTTP status: {response.status_code}")
    # print("Create machine response:")
    # print(json.dumps(body, indent=2))

    if response.status_code in (200, 201):
        machine_id = body.get("data", {}).get("id")
        if machine_id:
            return machine_id

        print("Machine created, but machine ID was not found in response.")
        return None

    error_text = json.dumps(body)

    if "FINGERPRINT_TAKEN" in error_text or "MACHINE_LIMIT_EXCEEDED" in error_text:
        print("\nThis fingerprint is already used by another machine/license in Keygen.")
        print("For demo, do one of these:")
        print("1. Open Keygen -> Machines -> delete the old machine with same fingerprint")
        print("2. Or open the existing machine and paste its Machine ID below")
        machine_id = input("Paste existing Machine ID, or press Enter to skip: ").strip()

        if machine_id:
            return machine_id

        return None

    if "Unauthorized" in error_text or "Forbidden" in error_text or response.status_code in (401, 403):
        print("\nYour KEYGEN_ADMIN_TOKEN does not have permission to create machines.")
        print("Create/use a token with machine create/update permission.")
        return None

    return None


def update_license_metadata(
    license_id: str,
    customer: Dict[str, str],
    machine_id: Optional[str],
    fingerprint: str,
    status: str,
    activated_at: str,
    expires_at: Optional[str]
) -> None:
    """
    Updates Keygen license metadata so manager can see who is using this license.
    """
    if not license_id:
        return

    url = f"{KEYGEN_API_BASE}/licenses/{license_id}"

    payload = {
        "data": {
            "type": "licenses",
            "id": license_id,
            "attributes": {
                "metadata": {
                    "customer_name": customer.get("name"),
                    "customer_email": customer.get("email"),
                    "customer_phone": customer.get("phone"),
                    "customer_company": customer.get("company"),
                    "sbox_license_status": status,
                    "sbox_machine_id": machine_id,
                    "sbox_machine_fingerprint": fingerprint,
                    "sbox_hostname": socket.gethostname(),
                    "sbox_activated_at": activated_at,
                    "sbox_expires_at": expires_at,
                    "last_seen_at": utc_now_iso()
                }
            }
        }
    }

    response = requests.patch(
        url,
        headers=admin_headers(),
        json=payload,
        timeout=30
    )

    if response.status_code not in (200, 204):
        print("Warning: failed to update license metadata.")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)


def update_machine_metadata(
    machine_id: Optional[str],
    customer: Dict[str, str],
    license_id: Optional[str]
) -> None:
    if not machine_id:
        return

    url = f"{KEYGEN_API_BASE}/machines/{machine_id}"

    payload = {
        "data": {
            "type": "machines",
            "id": machine_id,
            "attributes": {
                "metadata": {
                    "customer_name": customer.get("name"),
                    "customer_email": customer.get("email"),
                    "customer_phone": customer.get("phone"),
                    "customer_company": customer.get("company"),
                    "sbox_license_id": license_id,
                    "hostname": socket.gethostname(),
                    "last_seen_at": utc_now_iso()
                }
            }
        }
    }

    response = requests.patch(
        url,
        headers=admin_headers(),
        json=payload,
        timeout=30
    )

    if response.status_code not in (200, 204):
        print("Warning: failed to update machine metadata.")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)


def request_or_generate_license_key() -> bool:
    """
    First-time flow.

    User can:
    1. Paste existing license key
    2. Press Enter and generate 30-day trial license
    """
    fingerprint = get_machine_fingerprint()

    print("\nSBox license is not activated.")
    license_key = input("Enter license key, or press Enter to generate trial license: ").strip()

    customer = collect_customer_details()

    license_id = None

    if not license_key:
        choice = input("Do you want to generate a free trial license key for 30 days? (y/n): ").strip().lower()

        if choice != "y":
            print("License key is required to use SBox.")
            return False

        print("\nGenerating 30-day trial license in Keygen...")

        create_body = create_trial_license(customer, fingerprint)
        license_key = get_license_key_from_create_response(create_body)
        license_id = get_license_id_from_create_response(create_body)

        print("\nTrial license key generated successfully:")
        print(license_key)

    save_cached_license({
        "license_key": license_key,
        "license_id": license_id,
        "machine_id": None,
        "machine_fingerprint": fingerprint,
        "customer": customer,
        "status": "pending",
        "created_at": utc_now_iso(),
        "last_validated_at": None
    })

    return ensure_sbox_license()


def ask_new_license_key_and_retry() -> bool:
    fingerprint = get_machine_fingerprint()

    new_key = input("Enter new SBox license key: ").strip()

    if not new_key:
        print("License key cannot be empty.")
        return False

    customer = collect_customer_details()

    save_cached_license({
        "license_key": new_key,
        "license_id": None,
        "machine_id": None,
        "machine_fingerprint": fingerprint,
        "customer": customer,
        "status": "pending",
        "created_at": utc_now_iso(),
        "last_validated_at": None
    })

    return ensure_sbox_license()

def get_license_id_by_key(license_key: str) -> Optional[str]:
    """
    Donkey meaning:
    User pasted only license key.
    But to create/link a machine, Keygen needs license ID.

    So this function asks Keygen:
    'For this license key, what is the license ID?'
    """
    if not KEYGEN_ACCOUNT_ID:
        raise RuntimeError("KEYGEN_ACCOUNT_ID is not set.")

    url = f"{KEYGEN_API_BASE}/licenses/actions/validate-key"

    payload = {
        "meta": {
            "key": license_key
        }
    }

    response = requests.post(
        url,
        headers={
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/vnd.api+json"
        },
        json=payload,
        timeout=30
    )

    try:
        body = response.json()
    except Exception:
        print("Could not parse Keygen response while fetching license ID.")
        return None

    data = body.get("data")

    if isinstance(data, dict):
        return data.get("id")

    print("Could not find license ID from Keygen response.")
    print(json.dumps(body, indent=2))
    return None

def ensure_sbox_license() -> bool:
    """
    Main gatekeeper.

    After EULA:
    - If no license: ask key or generate trial
    - If key exists: validate
    - Save customer details to Keygen metadata
    - Save machine ID/fingerprint locally
    - Allow SBox only if valid
    """
    fingerprint = get_machine_fingerprint()

    cached = load_cached_license()

    if not cached or not cached.get("license_key"):
        return request_or_generate_license_key()

    license_key = cached.get("license_key")
    license_id = cached.get("license_id")
    machine_id = cached.get("machine_id")
    customer = cached.get("customer")

    if not customer:
        customer = collect_customer_details()

    try:
        license_data = validate_license_sync(license_key, fingerprint)

        license_id = get_license_id_from_sdk_data(license_data) or license_id
        expires_at = get_license_expiry_from_sdk_data(license_data)

        activated_at = cached.get("activated_at")
        if not activated_at:
            activated_at = utc_now_iso()

        if not machine_id:
            machine_id = create_machine_for_license(
                license_id=license_id,
                fingerprint=fingerprint,
                customer=customer
            )

        update_license_metadata(
            license_id=license_id,
            customer=customer,
            machine_id=machine_id,
            fingerprint=fingerprint,
            status="active",
            activated_at=activated_at,
            expires_at=expires_at
        )

        update_machine_metadata(
            machine_id=machine_id,
            customer=customer,
            license_id=license_id
        )

        save_cached_license({
            "license_key": license_key,
            "license_id": license_id,
            "machine_id": machine_id,
            "machine_fingerprint": fingerprint,
            "customer": customer,
            "status": "active",
            "activated_at": activated_at,
            "expires_at": expires_at,
            "time_left": calculate_time_left(expires_at),
            "last_validated_at": int(time.time())
        })

        print("SBox license is active.")
        return True

    except KeygenError as ex:
        try:
            error = Error.from_error(ex)
            code = getattr(error, "code", error.__class__.__name__)
        except Exception:
            code = ex.__class__.__name__

        machine_missing_codes = {
            "NO_MACHINE",
            "NO_MACHINES",
            "NOT_ACTIVATED",
            "LICENSE_NOT_ACTIVATED",
            "LicenseNotActivated",
            "ErrLicenseNotActivated",
        }

        expired_or_invalid_codes = {
            "EXPIRED",
            "LICENSE_EXPIRED",
            "LicenseExpired",
            "ErrLicenseExpired",
            "REVOKED",
            "SUSPENDED",
            "INVALID",
            "NOT_FOUND",
            "NO_LICENSE",
        }

        if code in machine_missing_codes:
            print("License is valid, but this machine is not activated yet.")
            print("Creating/linking machine in Keygen now...")

            # If license key was pasted manually, license_id may be missing.
            # In that case, fetch license_id from Keygen using license key.
            if not license_id:
                print("License ID is missing locally. Fetching license ID from Keygen...")

                license_id = get_license_id_by_key(license_key)

                if not license_id:
                    print("License ID is missing. Cannot create machine.")
                    print("Please check whether this license key exists in Keygen.")
                    cached["status"] = "machine_activation_failed"
                    cached["last_failed_validation_at"] = utc_now_iso()
                    save_cached_license(cached)
                    return False

                cached["license_id"] = license_id
                save_cached_license(cached)

                print(f"License ID found: {license_id}")

            # IMPORTANT:
            # This must be outside `if not license_id`.
            # For generated trial license, license_id already exists,
            # but machine still needs to be created.
            machine_id = create_machine_for_license(
                license_id=license_id,
                fingerprint=fingerprint,
                customer=customer
            )

            if not machine_id:
                print("Machine could not be created or linked.")
                cached["status"] = "machine_activation_failed"
                cached["last_failed_validation_at"] = utc_now_iso()
                save_cached_license(cached)
                return False

            cached["license_id"] = license_id
            cached["machine_id"] = machine_id
            cached["machine_fingerprint"] = fingerprint
            cached["status"] = "machine_linked"
            cached["last_validated_at"] = int(time.time())
            save_cached_license(cached)

            print(f"Machine linked successfully. Machine ID: {machine_id}")
            print("Validating license again...")

            return ensure_sbox_license()

        if code in expired_or_invalid_codes:
            print(f"SBox license validation failed: {code}")

            cached["status"] = "expired_or_invalid"
            cached["last_failed_validation_at"] = utc_now_iso()
            save_cached_license(cached)

            choice = input("Do you want to enter a new permanent license key? (y/n): ").strip().lower()

            if choice == "y":
                delete_cached_license()
                return ask_new_license_key_and_retry()

            return False

        print(f"SBox license validation failed: {code}")

        cached["status"] = "validation_failed"
        cached["last_failed_validation_at"] = utc_now_iso()
        cached["last_error_code"] = code
        save_cached_license(cached)

        choice = input("Do you want to enter a new license key? (y/n): ").strip().lower()

        if choice == "y":
            delete_cached_license()
            return ask_new_license_key_and_retry()

        return False

    except Exception as e:
        print(f"SBox license validation failed: {e}")

        cached["status"] = "inactive"
        cached["last_failed_validation_at"] = utc_now_iso()
        save_cached_license(cached)

        choice = input("Do you want to enter a new license key? (y/n): ").strip().lower()

        if choice == "y":
            delete_cached_license()
            return ask_new_license_key_and_retry()

        return False


def get_sbox_license_status() -> Dict[str, Any]:
    """
    Used by check_system_status() in sbox.py.
    """
    cached = load_cached_license()

    if not cached:
        return {
            "status": "inactive",
            "license_id": None,
            "machine_id": None,
            "machine_fingerprint": None,
            "activated_at": None,
            "expires_at": None,
            "time_left": None,
            "customer": None
        }

    expires_at = cached.get("expires_at")
    time_left = calculate_time_left(expires_at)

    status = cached.get("status", "unknown")

    if time_left == "expired":
        status = "expired"

    return {
        "status": status,
        "license_id": cached.get("license_id"),
        "machine_id": cached.get("machine_id"),
        "machine_fingerprint": cached.get("machine_fingerprint"),
        "activated_at": format_human_datetime(cached.get("activated_at")),
        "expires_at": format_human_datetime(expires_at),
        "time_left": time_left,
        "customer": cached.get("customer")
    }

