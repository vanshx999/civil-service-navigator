import json
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("[]")


def _read() -> list[dict]:
    _ensure()
    return json.loads(USERS_FILE.read_text())


def _write(users: list[dict]):
    _ensure()
    USERS_FILE.write_text(json.dumps(users, indent=2))


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_token() -> str:
    return secrets.token_hex(32)


def register_user(username: str, password: str, email: str = "") -> dict:
    users = _read()
    if any(u["username"] == username for u in users):
        return {"success": False, "error": "Username already exists"}
    token = create_token()
    user = {
        "username": username,
        "password": hash_password(password),
        "email": email,
        "token": token,
        "created_at": datetime.now().isoformat(),
        "is_admin": False,
    }
    users.append(user)
    _write(users)
    return {"success": True, "token": token, "username": username}


def login_user(username: str, password: str) -> dict:
    users = _read()
    pw = hash_password(password)
    for u in users:
        if u["username"] == username and u["password"] == pw:
            token = create_token()
            u["token"] = token
            _write(users)
            return {"success": True, "token": token, "username": username, "email": u.get("email", "")}
    return {"success": False, "error": "Invalid username or password"}


def verify_token(token: str) -> Optional[dict]:
    users = _read()
    for u in users:
        if u.get("token") == token:
            return u
    return None
