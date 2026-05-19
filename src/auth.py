"""
AI Career Copilot — Authentication
------------------------------------
Google OAuth login + SQLite user database.
Each user gets their own CV, jobs, and applications.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from jose import JWTError, jwt
from httpx import AsyncClient
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY           = os.getenv("SECRET_KEY", "fallback-secret")
ALGORITHM            = "HS256"
BASE_URL             = os.getenv("BASE_URL", "http://localhost:8000")

DB_FILE = "users.db"


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id   TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT,
            picture     TEXT,
            created_at  TEXT,
            cv_text     TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER,
            date_applied    TEXT,
            title           TEXT,
            company         TEXT,
            location        TEXT,
            recruiter_name  TEXT,
            recruiter_email TEXT,
            match_score     INTEGER,
            status          TEXT DEFAULT 'Applied',
            follow_up_date  TEXT,
            notes           TEXT,
            link            TEXT,
            job_type        TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_google_id(google_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE google_id = ?", (google_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "google_id": row[1], "email": row[2],
            "name": row[3], "picture": row[4], "created_at": row[5],
            "cv_text": row[6]
        }
    return None


def create_or_update_user(google_id, email, name, picture):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (google_id, email, name, picture, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(google_id) DO UPDATE SET
            name=excluded.name,
            picture=excluded.picture
    """, (google_id, email, name, picture, datetime.now().isoformat()))
    conn.commit()
    user_id = c.lastrowid or get_user_by_google_id(google_id)["id"]
    conn.close()
    return user_id


def update_cv(user_id: int, cv_text: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET cv_text = ? WHERE id = ?", (cv_text, user_id))
    conn.commit()
    conn.close()


def get_user_applications(user_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM applications WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    cols = ["id","user_id","date_applied","title","company","location",
            "recruiter_name","recruiter_email","match_score","status",
            "follow_up_date","notes","link","job_type"]
    return [dict(zip(cols, row)) for row in rows]


def add_application(user_id: int, job: dict, recruiter_name: str,
                    recruiter_email: str, notes: str = ""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Check duplicate
    c.execute("SELECT id FROM applications WHERE user_id=? AND title=? AND company=?",
              (user_id, job.get("title"), job.get("company")))
    if c.fetchone():
        conn.close()
        return "already_logged"

    today = datetime.now().strftime("%Y-%m-%d")
    fu    = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    c.execute("""
        INSERT INTO applications
        (user_id,date_applied,title,company,location,recruiter_name,
         recruiter_email,match_score,status,follow_up_date,notes,link,job_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (user_id, today, job.get("title",""), job.get("company",""),
          job.get("location",""), recruiter_name, recruiter_email,
          job.get("match_score",0), "Applied", fu, notes,
          job.get("link",""), job.get("job_type","")))
    conn.commit()
    conn.close()
    return fu


# ── JWT Tokens ────────────────────────────────────────────────────────────────

def create_token(user_id: int, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode(
        {"user_id": user_id, "email": email, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )


def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("token")
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (payload["user_id"],))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0], "google_id": row[1], "email": row[2],
        "name": row[3], "picture": row[4], "created_at": row[5],
        "cv_text": row[6]
    }


# ── OAuth Routes ──────────────────────────────────────────────────────────────

@router.get("/auth/login")
async def login():
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  f"{BASE_URL}/auth/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
    }
    from urllib.parse import urlencode
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url)


@router.get("/auth/callback")
async def callback(code: str, request: Request):
    async with AsyncClient() as client:
        # Exchange code for token
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  f"{BASE_URL}/auth/callback",
                "grant_type":    "authorization_code",
            }
        )
        tokens = token_res.json()

        # Get user info
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        userinfo = userinfo_res.json()

    user_id = create_or_update_user(
        google_id = userinfo["sub"],
        email     = userinfo["email"],
        name      = userinfo.get("name",""),
        picture   = userinfo.get("picture",""),
    )

    token    = create_token(user_id, userinfo["email"])
    response = RedirectResponse("/")
    response.set_cookie("token", token, max_age=30*24*3600, httponly=True)
    return response


@router.get("/auth/logout")
async def logout():
    response = RedirectResponse("/")
    response.delete_cookie("token")
    return response


@router.get("/auth/me")
async def me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"logged_in": False})
    return JSONResponse({
        "logged_in": True,
        "name":      user["name"],
        "email":     user["email"],
        "picture":   user["picture"],
        "has_cv":    bool(user.get("cv_text")),
    })
