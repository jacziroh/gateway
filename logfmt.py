import sys
import json
import logging
import os
import re
import subprocess
import time
import shutil
import tempfile
import pyfiglet
import jwt

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from huggingface_hub import snapshot_download, login
from sbox_license import ensure_sbox_license, get_sbox_license_status

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sbox.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
try:
    if os.path.exists("sbox.log"):
        os.chmod("sbox.log", 0o600)
except Exception as exc:
    print(f"Warning: could not protect sbox.log permissions: {exc}")

# Constants
DEPLOY_MAP_TEMPLATE_PATH = "/opt/kai/kai-tools/gateway/deploy-map-template.json"
DEPLOY_MAP_PATH = "/opt/kai/kai-tools/gateway/deploy-map.json"
GATEWAY_SERVICE = "kai-tools-gateway.service"
WOLFMIND_SERVICE = "kai-wolfmind-svc.service"
MCP_SERVICE = "mcp.service"
ELEPHANT_SERVICE = "kai-elephant-svc.service"

LOCAL_URL = "http://127.0.0.1:8080/"

KAI_JWT_SECRET: Optional[str] = os.getenv("KAI_JWT_SECRET")
JWT_ALG: str = os.getenv("JWT_ALG", "HS256")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
SBOX_CONFIG_PATH: str = os.getenv("SBOX_CONFIG_PATH", "bin/sbox.json")

# Minimum secret length to reduce brute-force risk
_MIN_SECRET_LEN = 32

# Allowed characters for usernames used as file-name components
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Dataclasses
@dataclass
class Model:
    id: int
    name: str
    family: str
    owner: str
    license: str
    created_at: int
    path: str


@dataclass
class ModelConfigured:
    id: int
    model_id: int


@dataclass
class NumaNode:
    id: int
    cpus: List[int]
    gpus: List[int]


@dataclass
class SboxConfig:
    numa: bool
    mcp: bool
    gateway: bool
    elephant: bool
    models: Dict[int, Model]
    model_configured: Dict[int, ModelConfigured]
    numa_nodes: Dict[int, NumaNode]

# Config helpers
def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def list_to_dict(items, cls):
    return {item["id"]: cls(**item) for item in items}


def load_config(json_path: str) -> SboxConfig:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SboxConfig(
        numa=to_bool(data["numa"]),
        mcp=to_bool(data["mcp"]),
        gateway=to_bool(data["gateway"]),
        elephant=to_bool(data["elephant"]),
        models=list_to_dict(data["models"], Model),
        model_configured=list_to_dict(data["model_configured"], ModelConfigured),
        numa_nodes=list_to_dict(data["numa_nodes"], NumaNode),
    )


def dict_to_list(d) -> list:
    return [asdict(v) for v in d.values()]


def config_to_json_dict(config: SboxConfig) -> dict:
    return {
        "numa": bool(config.numa),
        "mcp": bool(config.mcp),
        "gateway": bool(config.gateway),
        "elephant": bool(config.elephant),
        "models": dict_to_list(config.models),
        "model_configured": dict_to_list(config.model_configured),
        "numa_nodes": dict_to_list(config.numa_nodes),
    }


def save_config(config: SboxConfig, path: str) -> None:
    """Atomically write config so a crash never leaves a half-written file."""
    data = config_to_json_dict(config)
    _atomic_write_json(path, data)


# Generic file helpers
def _load_json_object_or_empty(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return obj


def _atomic_write_json(path: str, data: dict) -> None:
    """Write *data* to *path* atomically (temp file + rename)."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=4, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name

    os.replace(tmp_name, path)


def backup_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    log.info("Backed up %s -> %s", path, backup_path)
    return backup_path


# Deploy-map validation
def validate_deploy_map(deploy_map: dict) -> None:
    if not isinstance(deploy_map, dict):
        raise ValueError("deploy-map.json must be a JSON object.")

    for model_name, urls in deploy_map.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Invalid model name in deploy-map.json.")
        if not isinstance(urls, list):
            raise ValueError(f"URLs for model '{model_name}' must be a list.")
        for url in urls:
            if not isinstance(url, str):
                raise ValueError(f"Invalid URL under model '{model_name}'.")
            if not url.startswith("http://") and not url.startswith("https://"):
                raise ValueError(f"Invalid URL format: {url}")

def check_runtime_environment() -> None:
    systemctl_path = shutil.which("systemctl")
    sudo_path = shutil.which("sudo")

    if systemctl_path:
        print(f"systemctl available : yes ({systemctl_path})")
    else:
        print("systemctl available : no")
        print("Warning: systemd commands will fail in this environment.")

    if sudo_path:
        print(f"sudo available      : yes ({sudo_path})")
    else:
        print("sudo available      : no")
        print("Warning: sudo commands will fail in this environment.")


# systemd helpers
def run_command(command: List[str], action_name: str, check: bool = True) -> bool:
    try:
        subprocess.run(command, check=check)
        return True
    except subprocess.CalledProcessError as exc:
        log.error("%s failed: %s", action_name, exc)
        print(f"\nError: {action_name} failed.")
        print(f"Command: {' '.join(command)}")
        print(f"Reason : {exc}")
        return False
    except FileNotFoundError:
        log.error("%s failed because command was not found: %s", action_name, command[0])
        print(f"\nError: {action_name} failed.")
        print(f"Command not found: {command[0]}")
        return False
    except Exception as exc:
        log.error("%s failed: %s", action_name, exc)
        print(f"\nError: {action_name} failed: {exc}")
        return False

def systemd_status(service_name: str) -> str:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "").strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def wait_for_service_active(service_name: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if systemd_status(service_name) == "active":
            return True
        time.sleep(2)
    return False


def _restart_gateway() -> None:
    if not run_command(
        ["sudo", "systemctl", "restart", GATEWAY_SERVICE],
        "Restart gateway",
        check=True,
    ):
        raise RuntimeError(f"Failed to restart {GATEWAY_SERVICE}.")

    if not wait_for_service_active(GATEWAY_SERVICE, timeout_seconds=30):
        raise RuntimeError(
            f"{GATEWAY_SERVICE} did not become active within 30 s after restart."
        )

def _ensure_wolfmind_active() -> bool:
    if wait_for_service_active(WOLFMIND_SERVICE, timeout_seconds=30):
        return True

    print(f"\nWarning: {WOLFMIND_SERVICE} did not become active within 30 seconds.")
    print("Please check logs using:")
    print(f"  journalctl -u {WOLFMIND_SERVICE} -n 100 --no-pager")
    return False


# Config mutation
def mark_configured(flag_name: str) -> None:
    setattr(sbox_config, flag_name, True)
    save_config(sbox_config, SBOX_CONFIG_PATH)


# Input helpers
def _normalize_base_url(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("SBox IP cannot be empty.")
    if not s.startswith("http://") and not s.startswith("https://"):
        s = "http://" + s
    return s.rstrip("/") + "/"


def _is_local_url(url: str) -> bool:
    return "127.0.0.1:" in url or "localhost:" in url


def _safe_username(value: str) -> str:
    """Validate a username that will be used as a file-name component."""
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(
            "Username must be 1-64 characters and contain only "
            "letters, digits, dots, hyphens, or underscores."
        )
    return value


def _get_configured_model() -> Optional[Model]:
    """Return the currently configured Model, or None if none is set."""
    if not sbox_config.model_configured:
        return None
    first_entry = next(iter(sbox_config.model_configured.values()))
    if first_entry.model_id == 0:
        return None
    return sbox_config.models.get(first_entry.model_id)


# UI helpers
def _pause() -> None:
    input("\nPress Enter to return to menu...")

# Menu
def main_menu() -> None:
    title = "Kompact AI  S-BOX"
    banner = pyfiglet.figlet_format(title, font="small")
    print(banner)
    print("1. Download and Configure Model")
    print("2. Remove Existing Model")
    print("3. Configure MCP server")
    print("4. Configure Access Gateway")
    print("5. Configure REST API")
    print("6. Configure Elephant Semantic memory")
    print("7. Create Access Token")
    print("8. Add Remote SBox")
    print("9. Remove Remote SBox")
    print("s. Check System Status")
    print("a. About SBox")
    print("l. EULA")
    print("h. Help")
    print("0. Exit")
    print("------------------------")


# Model selection
def select_model() -> Optional[Model]:
    while True:
        search_text = input(
            "\nEnter model name to search or 0 to go back: "
        ).strip().lower()

        if search_text == "0":
            return None

        if not search_text:
            print("Search text cannot be empty.")
            continue

        matched_models = [
            model
            for model in sbox_config.models.values()
            if search_text in model.name.lower()
        ]

        if not matched_models:
            print("No matching models found.")
            continue

        print("\n--- Matching Models ---")
        for model in matched_models:
            print(f"{model.id}. {model.name}")
        print("0. Back")
        print("------------------------")

        model_id_raw = input("Enter model ID to download (0 to search again): ").strip()
        if model_id_raw == "0":
            continue

        try:
            model_id = int(model_id_raw)
        except ValueError:
            print("Invalid model ID.")
            continue

        allowed_ids = {model.id for model in matched_models}
        if model_id not in allowed_ids:
            print("Invalid selection. Please choose from the list.")
            continue

        model = sbox_config.models.get(model_id)
        if model is None:
            print("Model not found. Please try again.")
            continue

        return model

# Weight download
def download_weights(model_name: str, weights_path: str) -> None:
    if os.path.exists(weights_path):
        shutil.rmtree(weights_path)
        log.info("Removed existing directory at %s.", weights_path)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Please export HF_TOKEN before running.")

    login(token=HF_TOKEN)
    snapshot_download(
        repo_id=model_name,
        local_dir=weights_path,
        ignore_patterns=["original/**"],
    )

# Feature: Download & configure model
def download_and_configure_model() -> None:
    model = select_model()
    if model is None:
        return

    log.info("Downloading and configuring model: '%s'", model.name)
    weights_path = "dist/" + model.path

    # Stop service before making changes
    if not run_command(
        ["sudo", "systemctl", "stop", WOLFMIND_SERVICE],
        "Stop Wolfmind service",
        check=True,
    ):
        _pause()
        return

    # Back up the start script if present
    start_script = "ops/start-wolfmind-svc.sh"
    if os.path.exists(start_script):
        os.replace(start_script, start_script + ".sbox-bak")

    # Download — roll back service startup on failure
    try:
        download_weights(model.name, weights_path)
    except Exception as exc:
        log.error("Download failed: %s", exc)
        print(f"\nDownload failed: {exc}")
        print("Attempting to restart the service in its previous state...")
        run_command(
            ["sudo", "systemctl", "start", WOLFMIND_SERVICE],
            "Restart previous Wolfmind service",
            check=False,
        )
        _pause()
        return

    if not run_command(
        [
            "python3",
            "bin/py_replace_wolfmind_sbox_vars.py",
            "--friendly-name", model.path,
            "--name", model.name,
            "--family", model.family,
            "--owner", model.owner,
            "--license", model.license,
            "--created-at", str(model.created_at),
        ],
        "Update Wolfmind start script",
        check=True,
    ):
        _pause()
        return
    
    if not os.path.exists(DEPLOY_MAP_TEMPLATE_PATH):
        print(f"\nError: deploy-map template not found: {DEPLOY_MAP_TEMPLATE_PATH}")
        _pause()
        return
    
    if not run_command(
        [
            "python3",
            "bin/py_replace_deploy_vars.py",
            "--name", model.name,
            "--input", DEPLOY_MAP_TEMPLATE_PATH,
            "--output", DEPLOY_MAP_PATH,
        ],
        "Update deploy-map.json",
        check=True,
    ):
        _pause()
        return

    sbox_config.model_configured[0] = ModelConfigured(id=0, model_id=model.id)
    save_config(sbox_config, SBOX_CONFIG_PATH)

    if not run_command(
        ["sudo", "systemctl", "start", WOLFMIND_SERVICE],
        "Start Wolfmind service",
        check=True,
    ):
        _pause()
        return

    _ensure_wolfmind_active()

    log.info("Restarting gateway...")
    try:
        _restart_gateway()
        log.info("Gateway restarted successfully.")
    except RuntimeError as exc:
        log.error("Gateway restart failed: %s", exc)
        print(f"\nWarning: {exc}")

    print(f"\nModel '{model.name}' downloaded and configured successfully.")
    _pause()


# Remove existing model
def remove_existing_model() -> None:
    model = _get_configured_model()
    if model is None:
        print("Model configured: None")
        _pause()
        return

    log.info("Removing configured model: %s", model.name)

    if not run_command(
        ["sudo", "systemctl", "stop", WOLFMIND_SERVICE],
        "Stop Wolfmind service",
        check=True,
    ):
        _pause()
        return

    start_script = "ops/start-wolfmind-svc.sh"
    if os.path.exists(start_script):
        os.replace(start_script, start_script + ".sbox-bak")

    weights_path = "dist/" + model.path
    if os.path.exists(weights_path):
        shutil.rmtree(weights_path)

    sbox_config.model_configured[0] = ModelConfigured(id=0, model_id=0)
    save_config(sbox_config, SBOX_CONFIG_PATH)

    try:
        deploy_map = _load_json_object_or_empty(DEPLOY_MAP_PATH)
        backup_file(DEPLOY_MAP_PATH)

        for model_name, urls in list(deploy_map.items()):
            if isinstance(urls, list):
                deploy_map[model_name] = [u for u in urls if u != LOCAL_URL]
                if not deploy_map[model_name]:
                    deploy_map.pop(model_name, None)

        validate_deploy_map(deploy_map)
        _atomic_write_json(DEPLOY_MAP_PATH, deploy_map)

        log.info("Restarting gateway...")
        _restart_gateway()
        log.info("Gateway restarted.")
    except RuntimeError as exc:
        log.error("Gateway restart failed: %s", exc)
        print(f"\nWarning: {exc}")
    except Exception as exc:
        log.warning("Failed to update deploy-map.json: %s", exc)
        print(f"Warning: failed to update deploy-map.json: {exc}")

    print(f"\nModel '{model.name}' removed successfully.")
    _pause()


# check system status
def check_system_status() -> None:
    print("\nChecking system status...")
    print(f"Models available : {len(sbox_config.models)}")

    model = _get_configured_model()
    print(f"Model configured : {model.name if model else 'None'}")
    print(f"Numa enabled     : {sbox_config.numa}")

    print(f"MCP configured   : {sbox_config.mcp}")
    if sbox_config.mcp:
        print(f"  MCP service status      : {systemd_status('mcp.service')}")

    print(f"Gateway configured : {sbox_config.gateway}")
    if sbox_config.gateway:
        print(
            f"  Gateway service status  : {systemd_status('kai-tools-gateway.service')}"
        )

    print(f"REST API service status : {systemd_status(WOLFMIND_SERVICE)}")

    print("\nDeploy Map:")
    try:
        deploy_map = _load_json_object_or_empty(DEPLOY_MAP_PATH)
        if not deploy_map:
            print("  No deploy-map entries found.")
        else:
            for model_name, urls in deploy_map.items():
                print(f"  {model_name}:")
                if isinstance(urls, list):
                    for url in urls:
                        print(f"    - {url}")
                else:
                    print("    Invalid URL format in deploy-map.")
    except Exception as exc:
        print(f"  Could not read deploy-map.json: {exc}")

    print(f"Elephant configured : {sbox_config.elephant}")
    if sbox_config.elephant:
        print(
            f"  Elephant service status : {systemd_status('kai-elephant-svc.service')}"
        )

    license_status = get_sbox_license_status()
    print("\nLicense Status:")
    print(f"  Status       : {license_status.get('status')}")
    print(f"  License ID   : {license_status.get('license_id')}")
    print(f"  Machine ID   : {license_status.get('machine_id')}")
    print(f"  Activated At : {license_status.get('activated_at')}")
    print(f"  Expires At   : {license_status.get('expires_at')}")
    print(f"  Time Left    : {license_status.get('time_left')}")

    _pause()


#Create access token
def create_access_token() -> None:
    if not KAI_JWT_SECRET:
        print("Error: KAI_JWT_SECRET is not set.")
        _pause()
        return

    if len(KAI_JWT_SECRET) < _MIN_SECRET_LEN:
        print(
            f"Error: KAI_JWT_SECRET must be at least {_MIN_SECRET_LEN} characters long."
        )
        _pause()
        return

    # --- user ---
    user_raw = input("Enter User: ").strip()
    try:
        user = _safe_username(user_raw)
    except ValueError as exc:
        print(f"Invalid username: {exc}")
        _pause()
        return

    # --- models ---
    models_raw = input("Enter Model Names (comma-separated): ").strip()
    if not models_raw:
        print("Model names cannot be empty.")
        _pause()
        return
    model_names = [m.strip() for m in models_raw.split(",") if m.strip()]
    if not model_names:
        print("Please enter at least one valid model name.")
        _pause()
        return
    available_model_names = {model.name for model in sbox_config.models.values()}
    unknown_models = [m for m in model_names if m not in available_model_names]

    if unknown_models:
        print("\nWarning: These model names are not present in sbox.json:")
        for m in unknown_models:
            print(f"  - {m}")
        print("Token will still be created, but API calls may fail if the gateway does not know these models.")

    # --- ip ---
    ip = input("Enter allowed IP: ").strip()
    if not ip:
        print("IP cannot be empty.")
        _pause()
        return

    # --- expiry ---
    days_raw = input("Enter expiry in days [7]: ").strip()
    expiry_days = 7
    if days_raw:
        try:
            expiry_days = int(days_raw)
            if expiry_days <= 0:
                raise ValueError
        except ValueError:
            print("Invalid number of days. Must be a positive integer.")
            _pause()
            return

    now = int(time.time())
    exp = now + (expiry_days * 24 * 60 * 60)

    payload = {
        "sub": user,
        "models": model_names,
        "ip": ip,
        "iat": now,
        "exp": exp,
    }

    token = jwt.encode(payload, KAI_JWT_SECRET, algorithm=JWT_ALG)
    _save_token_to_file(user, model_names, ip, token, now, exp)

    print("\nGenerated Token:")
    print("Warning: keep this token secret. Anyone with this token can access the allowed models.")
    print(token)
    print(f"Expires in {expiry_days} day(s).")

    _pause()


def _save_token_to_file(
    user: str,
    model_names: List[str],
    ip: str,
    token: str,
    created_at: int,
    expires_at: int,
) -> None:
    """Atomically append a token record to the per-user JSON file."""
    tokens_dir = "tokens"
    os.makedirs(tokens_dir, exist_ok=True)

    user_file = os.path.join(tokens_dir, f"{user}.json")

    # Load existing records (best-effort)
    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Corrupt token file — resetting.")
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Token file %s is corrupt (%s). Starting fresh.", user_file, exc)
            backup_file(user_file)
            data = {"user": user, "details": []}
    else:
        data = {"user": user, "details": []}

    data["details"].append(
        {
            "models": model_names,
            "ip": ip,
            "token": token,   # stored for retrieval; protect the tokens/ dir in prod
            "created_at": created_at,
            "expires_at": expires_at,
        }
    )

    # Atomic write
    dir_name = os.path.dirname(user_file) or "."
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=4)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name

    os.replace(tmp_name, user_file)
    os.chmod(user_file, 0o600)
    log.info("Token saved to %s", user_file)


def manage_service(service_name: str, display_name: str) -> None:
    try:
        status = systemd_status(service_name)
        print(f"\n{display_name} status: {status}")

        if status == "active":
            choice = input("Service is running. Stop it? (y/n): ").strip().lower()
            if choice == "y":
                if run_command(
                    ["sudo", "systemctl", "stop", service_name],
                    f"Stop {display_name}",
                    check=True,
                ):
                    print(f"{display_name} stopped.")
        else:
            choice = input("Service is stopped. Start it? (y/n): ").strip().lower()
            if choice == "y":
                if run_command(
                    ["sudo", "systemctl", "start", service_name],
                    f"Start {display_name}",
                    check=True,
                ):
                    print(f"{display_name} started.")

    except Exception as exc:
        log.error("Failed to manage %s: %s", display_name, exc)
        print(f"Failed to manage {display_name}. Check the service name or sudo permissions.")


def configure_mcp() -> None:
    if not sbox_config.mcp:
        print("MCP not configured. Configuring for the first time...")
        mark_configured("mcp")
    manage_service("mcp.service", "MCP Server")


def configure_elephant() -> None:
    if not sbox_config.elephant:
        print("Elephant not configured. Configuring for the first time...")
        mark_configured("elephant")
    manage_service("kai-elephant-svc.service", "Elephant Semantic Memory")


def configure_gateway() -> None:
    if not sbox_config.gateway:
        print("Gateway not configured. Configuring for the first time...")
        mark_configured("gateway")
    manage_service("kai-tools-gateway.service", "Gateway")


#Add remote SBox
def add_remote_sbox() -> None:
    print("\n--- Add Remote SBox ---")

    try:
        deploy_map = _load_json_object_or_empty(DEPLOY_MAP_PATH)

        ip = input("Give SBox IP: ").strip()
        model_name = input("Give model name: ").strip()

        if not model_name:
            print("Model name cannot be empty.")
            _pause()
            return

        base_url = _normalize_base_url(ip)

        deploy_map.setdefault(model_name, [])
        if not isinstance(deploy_map[model_name], list):
            deploy_map[model_name] = []

        if base_url not in deploy_map[model_name]:
            deploy_map[model_name].append(base_url)

        deploy_map[model_name] = sorted(set(deploy_map[model_name]))

        validate_deploy_map(deploy_map)
        backup_file(DEPLOY_MAP_PATH)
        _atomic_write_json(DEPLOY_MAP_PATH, deploy_map)

        print("Added new model mapping:")
        print(f"  Model : {model_name}")
        print(f"  URLs  : {deploy_map[model_name]}")

        print("\nRestarting gateway...")
        _restart_gateway()
        print("Gateway restarted.")

    except RuntimeError as exc:
        log.error("Gateway restart failed: %s", exc)
        print(f"\nWarning: {exc}")
    except Exception as exc:
        log.error("Error adding remote SBox: %s", exc)
        print(f"Error adding remote SBox: {exc}")

    _pause()



# Remove remote SBox
def remove_remote_sbox() -> None:
    print("\n--- Remove Remote SBox ---")

    try:
        deploy_map = _load_json_object_or_empty(DEPLOY_MAP_PATH)
        if not deploy_map:
            print("deploy-map.json is empty or missing. Nothing to remove.")
            _pause()
            return

        remote_entries = [
            (model_name, url)
            for model_name, urls in deploy_map.items()
            if isinstance(urls, list)
            for url in urls
            if isinstance(url, str) and not _is_local_url(url)
        ]

        if not remote_entries:
            print("No remote SBox entries found.")
            _pause()
            return

        print("\nRemote mappings:")
        for i, (model_name, url) in enumerate(remote_entries, start=1):
            print(f"{i}. {model_name} -> {url}")
        print("a. Remove ALL remote mappings")
        print("0. Cancel")

        choice = input("Select what to remove: ").strip().lower()

        if choice == "0":
            _pause()
            return

        if choice == "a":
            for model_name, url in remote_entries:
                if isinstance(deploy_map.get(model_name), list):
                    deploy_map[model_name] = [
                        u for u in deploy_map[model_name] if u != url
                    ]
            for k in [k for k, v in deploy_map.items() if not v]:
                deploy_map.pop(k, None)
            print("Removed all remote mappings.")

        elif choice.isdigit():
            idx = int(choice)
            if idx < 1 or idx > len(remote_entries):
                print("Invalid selection.")
                _pause()
                return
            model_name, url = remote_entries[idx - 1]
            deploy_map[model_name] = [
                u for u in deploy_map[model_name] if u != url
            ]
            if not deploy_map[model_name]:
                deploy_map.pop(model_name, None)
            print(f"Removed remote mapping for: {model_name} -> {url}")

        else:
            print("Invalid choice.")
            _pause()
            return

        validate_deploy_map(deploy_map)
        backup_file(DEPLOY_MAP_PATH)
        _atomic_write_json(DEPLOY_MAP_PATH, deploy_map)

        print("Restarting gateway...")
        _restart_gateway()
        print("Gateway restarted.")

    except RuntimeError as exc:
        log.error("Gateway restart failed: %s", exc)
        print(f"\nWarning: {exc}")
    except Exception as exc:
        log.error("Error removing remote SBox: %s", exc)
        print(f"Error removing remote SBox: {exc}")

    _pause()


# About / Help / EULA
def show_about() -> None:
    print("\nAbout S-BOX\n")
    print(
        "S-Box (Sovereign AI Box) is a software system optimised to run AI models directly on Intel "
        "processors and deploy them locally. Installed on a standard Intel machine, whether 12, 24, or 48 "
        "cores, it transforms the device into a fully capable AI system, providing both an efficient model "
        "runtime and the operational features required to run AI where it is needed."
    )
    _pause()


def show_help() -> None:
    print("\nHelp\n")
    print("For any issues or support, please contact:")
    print("contact@ziroh.com")
    _pause()


def _eula_file_path() -> str:
    return os.path.expanduser("~/.sbox/eula_accepted.json")


def _save_eula_acceptance() -> None:
    path = _eula_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"accepted": True, "timestamp": int(time.time())}
    _atomic_write_json(path, data)


def check_eula() -> bool:
    path = _eula_file_path()
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("accepted", False) is True
    except Exception:
        return False


def show_eula() -> bool:
    print("""
1. DEFINITION:

"You" and "Your" mean the individual, legal entity, or Licensee of the Software under this EULA.
"Use" or "Using" means to download, install, activate, access, or otherwise use the application
"SBox" developed by Ziroh Labs. "Software" means the "SBox" developed by Ziroh Labs and any
Upgrades made available by Ziroh Labs.

2. ACCEPTANCE OF TERMS

By using the Software, you agree to be bound by the terms of the EULA. If you enter this EULA on
behalf of an entity, you represent that you have the authority to bind that entity.

3. LICENSE RESTRICTIONS

Ziroh Labs grants you a limited, non-exclusive, and non-transferable license to use object code
versions of the Software and the Documentation solely for your internal operations and per the
Entitlement and the Documentation, but not for the following purpose:

(i) reproduce or copy the Software, except that You may make one (1) copy of the Software solely
for archival purposes, provided that You agree to reproduce all copyright and other proprietary
right notices on the archival copy made available by Ziroh Labs; or

(ii) use, cause, or permit the use of the Software in whole or in part for any purpose other than
as permitted under this EULA; or

(iii) distribute, sell, lease, license, or otherwise make the Software available to any third
party without the prior written consent of Ziroh Labs; or

(iv) decompile, disassemble, decrypt, extract or otherwise reverse-engineer the Software.

The Software "SBox" is owned by Ziroh Labs, and Ziroh Labs retains all rights, title, and interest
in and to the Software. You have the right only to use the Software in the manner and to the extent
expressly licensed within this EULA, and nothing in this EULA shall be construed as conferring any
license to you of Ziroh Labs's intellectual property rights, whether by estoppel, implication, or
otherwise.

4. SUPPORT

During the EULA, Ziroh Labs shall support the Software in such manner, including providing updates,
bug fixes, builds, or error corrections, as Ziroh Labs deems fit and proper (collectively "Software
Updates"). If Ziroh Labs, in its sole discretion, provides Software Updates to you, the Software
Updates will be considered part of the Software and will be subject to the terms and conditions of
this EULA.

5. DISCLAIMER OF WARRANTIES

You acknowledge that the Software is provided "as is, with all faults," without any maintenance,
debugging, support, or improvement. Ziroh Labs MAKES NO REPRESENTATIONS AND EXTENDS NO WARRANTIES
OF ANY KIND, EITHER EXPRESS OR IMPLIED, IN RESPECT OF THE SOFTWARE. FURTHERMORE, ZIROH LABS
DISCLAIMS ALL EXPRESS OR IMPLIED CONDITIONS, REPRESENTATIONS, AND WARRANTIES, INCLUDING WITHOUT
LIMITATION ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
NON-INFRINGEMENT IN RESPECT OF THE SOFTWARE.

6. LIMITATION OF LIABILITY

You assume the entire risk as to the quality, results, performance, and/or non-performance of the
Software. In no event shall Ziroh Labs be responsible or liable for any damages whatsoever,
including lost profits, business, revenue, use, or data.

7. NO OBLIGATION

This EULA does not represent any commitment by you to purchase or use other products or services
of Ziroh Labs.

8. CONFIDENTIALITY

You acknowledge that the Software is proprietary and confidential to Ziroh Labs and must be
protected from unauthorized disclosure.

9. TERM AND TERMINATION

This Agreement is effective upon acceptance and remains in effect until termination. Upon
termination, you must cease use and destroy all copies of the Software.

10. ASSIGNMENT

You may not transfer any rights under this EULA without written consent from Ziroh Labs.

11. ENTIRE AGREEMENT

This EULA constitutes the entire agreement between the parties.

12. ENFORCEMENT

This EULA shall be governed by the laws of the Republic of India.

13. WAIVER

Failure to enforce any provision shall not constitute a waiver.
""")

    if check_eula():
        print("(EULA already accepted previously.)")
        _pause()
        return True

    choice = input("Type 'agree' to accept the terms and continue: ").strip().lower()
    if choice == "agree":
        _save_eula_acceptance()
        print("\nEULA accepted.\n")
        return True

    print("\nYou must accept the EULA to use S-BOX.\n")
    return False


def run_app() -> None:
    check_runtime_environment()
    while True:
        main_menu()
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            download_and_configure_model()
        elif choice == "2":
            remove_existing_model()
        elif choice == "3":
            print("Configuring MCP server...")
            configure_mcp()
        elif choice == "4":
            print("Configuring Gateway...")
            configure_gateway()
        elif choice == "5":
            print("Configuring API...")
            manage_service("kai-wolfmind-svc.service", "REST API")
        elif choice == "6":
            print("Configuring Elephant Semantic Memory...")
            configure_elephant()
        elif choice == "7":
            create_access_token()
        elif choice == "8":
            add_remote_sbox()
        elif choice == "9":
            remove_remote_sbox()
        elif choice == "s":
            check_system_status()
        elif choice == "a":
            show_about()
        elif choice == "l":
            show_eula()
        elif choice == "h":
            show_help()
        elif choice == "0":
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid selection. Please try again.")


# Entry point
try:
    sbox_config = load_config(SBOX_CONFIG_PATH)
except FileNotFoundError:
    log.critical("Config file not found: %s", SBOX_CONFIG_PATH)
    sys.exit(1)
except (KeyError, TypeError, json.JSONDecodeError) as exc:
    log.critical("Failed to parse config file %s: %s", SBOX_CONFIG_PATH, exc)
    sys.exit(1)

if __name__ == "__main__":
    if not check_eula():
        if not show_eula():
            sys.exit(0)

    if not ensure_sbox_license():
        log.critical("SBox license activation failed. Exiting.")
        sys.exit(1)

    run_app()
