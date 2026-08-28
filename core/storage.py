"""Atomic JSON persistence with bounded backups and audit records."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time as _time
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from .models import StateLedger, iso_now


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            pass
    return str(value)


class LedgerStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.ledger_dir = self.data_dir / "ledgers"
        self.backup_dir = self.data_dir / "backups"
        self.audit_file = self.data_dir / "audit.jsonl"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def user_token(user_key: str) -> str:
        return hashlib.sha256(user_key.encode("utf-8")).hexdigest()[:24]

    def ledger_path(self, user_key: str) -> Path:
        return self.ledger_dir / f"{self.user_token(user_key)}.json"

    def backup_path(self, user_key: str) -> Path:
        return self.backup_dir / f"{self.user_token(user_key)}.json.bak"

    def load(self, user_key: str) -> StateLedger:
        path = self.ledger_path(user_key)
        backup = self.backup_path(user_key)
        for candidate in (path, backup):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("ledger root must be an object")
                return StateLedger.from_dict(payload, user_key=user_key)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return StateLedger(user_key=user_key)

    def sessions(self) -> list[dict[str, Any]]:
        """Return persisted private sessions ordered by their latest activity."""
        sessions: list[dict[str, Any]] = []
        for path in self.ledger_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                user_key = str(payload.get("user_key", "")).strip()
                if not user_key:
                    continue
                sessions.append(
                    {
                        "session_id": user_key,
                        "updated_at": str(payload.get("updated_at", "")),
                        "message_watermark": int(payload.get("message_watermark", 0)),
                    }
                )
            except (
                OSError,
                AttributeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
        return sorted(
            sessions,
            key=lambda item: (item["updated_at"], item["message_watermark"]),
            reverse=True,
        )

    def user_keys(self) -> list[str]:
        """Enumerate persisted ledger owners without exposing hashed filenames."""
        return [item["session_id"] for item in self.sessions()]

    def save(self, ledger: StateLedger) -> None:
        path = self.ledger_path(ledger.user_key)
        backup = self.backup_path(ledger.user_key)
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(
            _json_safe(ledger.to_dict()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if path.exists():
            shutil.copy2(path, backup)
        temporary.write_text(payload, encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        # Windows: os.replace raises PermissionError (WinError 5) while any
        # other handle to the destination is open (e.g. a concurrent ledger
        # scan, antivirus, or backup tool). Such handles are short-lived, so
        # retry with backoff before giving up.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                _time.sleep(0.05 * 2**attempt)

    def append_audit(self, user_key: str, action: str, detail: dict[str, Any]) -> None:
        record = {
            "at": iso_now(),
            "user_token": self.user_token(user_key),
            "action": action,
            "detail": detail,
        }
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
