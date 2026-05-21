"""
AI Career Copilot — Google OAuth Handler
------------------------------------------
Handles Google Sign-In for multi-user support.
"""

import os
import sqlite3
import secrets
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# OAuth setup
config = Config(environ={
    "GOOGLE_CLIENT_ID":     os.getenv("GOOGLE_CLIENT_ID", ""),
    "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
})

oauth = OAuth(config)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
)

DB_FILE     = "users.db"
SECRET_KEY  = os.getenv("SECRET_KEY", secrets.token_hex(32))
RENDER_URL  = os.getenv("BASE_URL", "https://ai-career-copilot-0b72.onrender.com")


def get_or_create_user(email, name, picture):
    """Get existing user or create new one."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT id, email, name, picture FROM users WHERE email=?", (email,))
    user = c.fetchone()
    if user:
        conn.close()
        return {"id": user[0], "email": user[1], "name": user[2], "picture": user[3]}
    # Create new user
    c.execute(
        "INSERT INTO users (email, name, picture, created_at) VALUES (?,?,?,?)",
        (email, name, picture, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    user_id = c.lastrowid
    conn.close()
    return {"id": user_id, "email": email, "name": name, "picture": picture}


def get_user_scores(user_id):
    """Get job scores for a specific user."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM job_scores
        WHERE user_id=?
        ORDER BY match_score DESC
    """, (user_id,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


def save_user_scores(user_id, scores):
    """Save scored jobs for a specific user."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    # Clear old scores for this user
    c.execute("DELETE FROM job_scores WHERE user_id=?", (user_id,))
    # Insert new scores
    for job in scores:
        c.execute("""
            INSERT INTO job_scores
            (user_id, title, company, location, job_type, priority,
             published, match_score, verdict, matched_skills,
             missing_skills, reason, link, scraped_at, scored_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id,
            job.get("title",""), job.get("company",""), job.get("location",""),
            job.get("job_type",""), job.get("priority",3),
            job.get("published",""), job.get("match_score",0),
            job.get("verdict",""), job.get("matched_skills",""),
            job.get("missing_skills",""), job.get("reason",""),
            job.get("link",""), job.get("scraped_at",""),
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
    conn.commit()
    conn.close()


@router.get("/auth/login")
async def login(request: Request):
    """Redirect to Google OAuth."""
    redirect_uri = f"{RENDER_URL}/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if not userinfo:
            return JSONResponse({"error": "No user info"}, status_code=400)

        user = get_or_create_user(
            email   = userinfo["email"],
            name    = userinfo.get("name", ""),
            picture = userinfo.get("picture", ""),
        )

        # Set session cookie
        response = RedirectResponse(url="/")
        response.set_cookie(
            key      = "session_token",
            value    = userinfo["email"],
            httponly = True,
            max_age  = 7 * 24 * 3600,  # 7 days
            samesite = "lax",
        )
        return response

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/auth/logout")
async def logout():
    """Clear session."""
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response


@router.get("/auth/me")
async def get_me(request: Request):
    """Get current logged-in user."""
    token = request.cookies.get("session_token")
    if not token:
        return JSONResponse({"logged_in": False})
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT id, email, name, picture, cv_text FROM users WHERE email=?", (token,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "logged_in": True,
            "id":        row[0],
            "email":     row[1],
            "name":      row[2],
            "picture":   row[3],
            "has_cv":    bool(row[4]),
        }
    return JSONResponse({"logged_in": False})
