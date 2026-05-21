"""
AI Career Copilot — Auth Module
"""
import sqlite3
import os
from fastapi import APIRouter, Request
from datetime import datetime

DB_FILE = "users.db"
router  = APIRouter()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            picture TEXT,
            cv_text TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            company TEXT,
            location TEXT,
            recruiter_name TEXT,
            recruiter_email TEXT,
            match_score INTEGER,
            status TEXT DEFAULT 'Applied',
            follow_up_date TEXT,
            notes TEXT,
            link TEXT,
            job_type TEXT,
            date_applied TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS job_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            company TEXT,
            location TEXT,
            job_type TEXT,
            priority INTEGER,
            published TEXT,
            match_score INTEGER,
            verdict TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            reason TEXT,
            link TEXT,
            scraped_at TEXT,
            scored_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_current_user(request: Request):
    try:
        token = request.cookies.get("session_token")
        if not token:
            return None
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("SELECT id, email, name, picture FROM users WHERE email=?", (token,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"id": row[0], "email": row[1],
                    "name": row[2], "picture": row[3]}
        return None
    except:
        return None

def get_user_applications(user_id: int):
    try:
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("SELECT * FROM applications WHERE user_id=? ORDER BY date_applied DESC", (user_id,))
        rows = c.fetchall()
        cols = [d[0] for d in c.description]
        conn.close()
        return [dict(zip(cols, row)) for row in rows]
    except:
        return []

def add_application(user_id: int, job: dict, recruiter_name: str,
                    recruiter_email: str, notes: str = ""):
    from datetime import timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    fu    = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("""
            INSERT INTO applications
            (user_id,title,company,location,recruiter_name,recruiter_email,
             match_score,status,follow_up_date,notes,link,job_type,date_applied)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id,
            job.get("title",""), job.get("company",""), job.get("location",""),
            recruiter_name, recruiter_email, job.get("match_score",0),
            "Applied", fu, notes, job.get("link",""), job.get("job_type",""), today
        ))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def update_cv(user_id: int, cv_text: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("UPDATE users SET cv_text=? WHERE id=?", (cv_text, user_id))
        conn.commit()
        conn.close()
        return True
    except:
        return False

@router.get("/auth/user")
def get_user(request: Request):
    user = get_current_user(request)
    return user or {}

@router.post("/auth/logout")
def logout():
    from fastapi.responses import JSONResponse
    resp = JSONResponse({"status": "logged out"})
    resp.delete_cookie("session_token")
    return resp
