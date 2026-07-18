import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

from src.auth_utils import register_user, login_user, verify_token

logger = logging.getLogger(__name__)
router = APIRouter()

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


class RegisterBody(BaseModel):
    username: str
    password: str
    email: str = ""


class LoginBody(BaseModel):
    username: str
    password: str


@router.get("/login")
async def login_page():
    p = STATIC_DIR / "login.html"
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>Not found</h1>")


@router.get("/register")
async def register_page():
    p = STATIC_DIR / "register.html"
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>Not found</h1>")


@router.post("/api/auth/register")
async def api_register(body: RegisterBody):
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(body.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    result = register_user(body.username, body.password, body.email)
    if not result["success"]:
        raise HTTPException(409, result["error"])
    return result


@router.post("/api/auth/login")
async def api_login(body: LoginBody):
    result = login_user(body.username, body.password)
    if not result["success"]:
        raise HTTPException(401, result["error"])
    return result


@router.get("/api/auth/profile")
async def api_profile(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return {
        "username": user["username"],
        "email": user.get("email", ""),
        "created_at": user.get("created_at", ""),
        "is_admin": user.get("is_admin", False),
    }
