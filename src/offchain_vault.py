"""
Prototype "controlled off-chain vault" for split-layout canonical data under outputs/records/_datasets/.

Enable with environment variables (disabled by default so existing scripts keep working):

  BLOCKCHAIN_OFFCHAIN_VAULT_ENABLED=1
  BLOCKCHAIN_OFFCHAIN_VAULT_KEY=<shared secret string>

The secret may live **only** in the environment (good for collaborators who cloned GitHub without
``.offchain_vault_key``), or in a local file (override path optional):

  BLOCKCHAIN_OFFCHAIN_VAULT_KEY_FILE  (default: <project_root>/.offchain_vault_key)

When a key **file** exists, ``BLOCKCHAIN_OFFCHAIN_VAULT_KEY`` must match its first line
(so one canonical secret — whoever ran first can save it to the file; others paste the same
string into env or into their own local file — never commit it).

Auditing: append-only JSON lines to outputs/records/_vault/access_audit.jsonl when paths under
outputs/records/_datasets/ are read/written through hooked code paths.

This is an engineering / paper-experiment wrapper, not a production KMS or RBAC system.

At-rest encryption (Fernet) for files under ``outputs/records/_datasets/`` when
``BLOCKCHAIN_OFFCHAIN_VAULT_ENABLED=1``: without the key, on-disk files are not intelligible
as CSV/JSONL (prototype — still protect key file and OS permissions for real deployments).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FILE_MAGIC = b"BCVAULT1\n"

_ENABLED_ENV = "BLOCKCHAIN_OFFCHAIN_VAULT_ENABLED"
_KEY_ENV = "BLOCKCHAIN_OFFCHAIN_VAULT_KEY"
_KEY_FILE_ENV = "BLOCKCHAIN_OFFCHAIN_VAULT_KEY_FILE"


def project_root_from_records_dir(records_dir: Path) -> Path:
    """records_dir = <project>/outputs/records/<tag> -> project root."""
    return records_dir.resolve().parent.parent.parent


def is_vault_enabled() -> bool:
    raw = os.environ.get(_ENABLED_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _expected_key_file(project_root: Path) -> Path:
    override = os.environ.get(_KEY_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (project_root / ".offchain_vault_key").resolve()


def vault_secret_string(project_root: Path) -> str:
    """
    Canonical UTF-8 secret for Fernet (same derivation everywhere).
    Prefer on-disk key file if present; otherwise environment variable only.
    """
    key_file = _expected_key_file(project_root)
    if key_file.is_file():
        lines = key_file.read_text(encoding="utf-8").splitlines()
        s = (lines[0].strip() if lines else "")
        if s:
            return s
    env_s = os.environ.get(_KEY_ENV, "").strip()
    if env_s:
        return env_s
    raise RuntimeError(
        f"Vault secret missing: add first line to {_expected_key_file(project_root)} "
        f"or set {_KEY_ENV} (same string everyone uses for this dataset)."
    )


def records_root_from_records_dir(records_dir: Path) -> Path:
    """records_dir = .../outputs/records/<tag> -> .../outputs/records."""
    return records_dir.resolve().parent


def path_is_under_datasets_resolved(resolved: Path, records_root: Path) -> bool:
    """True if resolved path is inside outputs/records/_datasets/."""
    try:
        ds = (records_root / "_datasets").resolve()
        if not ds.is_dir():
            return False
        resolved.relative_to(ds)
        return True
    except ValueError:
        return False


def _fernet(project_root: Path):
    """Symmetric cipher derived from vault_secret_string (file or env only)."""
    from cryptography.fernet import Fernet

    secret = vault_secret_string(project_root).encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


def read_text_under_records(path: Path, records_dir: Path) -> str:
    """
    Read text from path; decrypt if file is vault-wrapped under _datasets/.
    Encrypted files require vault enabled + matching BLOCKCHAIN_OFFCHAIN_VAULT_KEY.
    """
    records_root = records_root_from_records_dir(records_dir)
    project_root = project_root_from_records_dir(records_dir)
    p = Path(path)
    raw = p.read_bytes()
    resolved = p.resolve()

    if not path_is_under_datasets_resolved(resolved, records_root):
        return raw.decode("utf-8")

    if raw.startswith(_FILE_MAGIC):
        if not is_vault_enabled():
            raise RuntimeError(
                f"File appears vault-encrypted ({path}). Set {_ENABLED_ENV}=1 and {_KEY_ENV} "
                "to match .offchain_vault_key (first line)."
            )
        require_vault_auth(project_root)
        f = _fernet(project_root)
        plain = f.decrypt(raw[len(_FILE_MAGIC) :])
        return plain.decode("utf-8")

    return raw.decode("utf-8")


def write_text_under_records(path: Path, text: str, records_dir: Path) -> None:
    """Write text; encrypt whole file when vault is on and path is under _datasets/."""
    records_root = records_root_from_records_dir(records_dir)
    project_root = project_root_from_records_dir(records_dir)
    p = Path(path)
    resolved = p.resolve()
    data = text.encode("utf-8")

    if is_vault_enabled() and path_is_under_datasets_resolved(resolved, records_root):
        require_vault_auth(project_root)
        f = _fernet(project_root)
        data = _FILE_MAGIC + f.encrypt(data)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def require_vault_auth(project_root: Path) -> None:
    """Raise RuntimeError if vault is enabled but key is missing or inconsistent."""
    if not is_vault_enabled():
        return
    presented_env = os.environ.get(_KEY_ENV, "").strip().encode("utf-8")
    key_file = _expected_key_file(project_root)
    from_file = b""
    if key_file.is_file():
        lines = key_file.read_text(encoding="utf-8").splitlines()
        from_file = (lines[0].strip() if lines else "").encode("utf-8")

    if presented_env and from_file and not secrets.compare_digest(presented_env, from_file):
        raise RuntimeError(
            "Off-chain vault: BLOCKCHAIN_OFFCHAIN_VAULT_KEY does not match the first line of the key file."
        )
    if not presented_env and not from_file:
        raise RuntimeError(
            f"Off-chain vault enabled ({_ENABLED_ENV}=1): set {_KEY_ENV} or create {_expected_key_file(project_root)} "
            "with one line — the shared secret for this dataset (curator + collaborators use the same string; "
            "never commit it to Git)."
        )


def _vault_audit_path(records_dir: Path) -> Path:
    base = records_dir.parent / "_vault"
    base.mkdir(parents=True, exist_ok=True)
    return base / "access_audit.jsonl"


def _is_under_datasets(path: Path, records_dir: Path) -> bool:
    try:
        rs = records_root_from_records_dir(records_dir)
        return path_is_under_datasets_resolved(path.resolve(), rs)
    except OSError:
        return False


def maybe_audit_datasets_path(
    action: str,
    path: Path | str,
    records_dir: Path,
    detail: str | None = None,
) -> None:
    """Log one line if vault is enabled and path lies under outputs/records/_datasets/."""
    if not is_vault_enabled():
        return
    p = Path(path)
    if not _is_under_datasets(p, records_dir):
        return
    line: dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "path": str(p.resolve()),
        "records_tag": records_dir.name,
        "pid": os.getpid(),
        "argv0": sys.argv[0] if sys.argv else "",
    }
    if detail:
        line["detail"] = detail
    ap = _vault_audit_path(records_dir)
    with ap.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
