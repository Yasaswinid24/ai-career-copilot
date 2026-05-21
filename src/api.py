"""
AI Career Copilot — FastAPI Backend
-------------------------------------
Powers the dashboard with real actions:
- ATS check per job
- Email generation
- Application tracking
All from the browser — no terminal needed.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(__file__))
from recruiter_finder import find_recruiter

load_dotenv()

# Import auth
from src.google_auth import (
    router as google_router, get_user_scores,
    save_user_scores, get_or_create_user
)
from src.auth import (init_db, get_current_user, get_user_applications,
                      add_application, update_cv, router as auth_router)

# Init database
init_db()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

app = FastAPI()
app.include_router(auth_router)
app.include_router(google_router)
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY","changeme"), https_only=False, same_site="lax")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESUME_FILE  = "src/resume.txt"
TRACKER_FILE = "applications.csv"
TRACKER_COLS = [
    "date_applied","title","company","location",
    "recruiter_name","recruiter_email","match_score",
    "status","follow_up_date","notes","link","job_type"
]


def load_resume():
    with open(RESUME_FILE) as f:
        return f.read()


def load_tracker():
    try:
        return pd.read_csv(TRACKER_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=TRACKER_COLS)
        df.to_csv(TRACKER_FILE, index=False)
        return df


def save_tracker(df):
    df.to_csv(TRACKER_FILE, index=False)


# ── Models ────────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    title:    str
    company:  str
    location: str = ""
    link:     str = ""
    job_type: str = ""
    match_score: int = 0
    matched_skills: str = ""
    missing_skills: str = ""
    description: str = ""


class ApplyRequest(BaseModel):
    job:             dict
    recruiter_name:  str
    recruiter_email: str
    notes:           str = ""


class StatusUpdate(BaseModel):
    title:   str
    company: str
    status:  str
    notes:   str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
def get_jobs(request: Request):
    try:
        # Check if user is logged in
        token = request.cookies.get("session_token")
        if token:
            import sqlite3
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE email=?", (token,))
            row = c.fetchone()
            conn.close()
            if row:
                user_id = row[0]
                scores = get_user_scores(user_id)
                if scores:
                    return scores

        # Fall back to shared matched_jobs.csv for non-logged-in users
        df = pd.read_csv("matched_jobs.csv")
        df = df[~df["verdict"].isin(["Error","API Error"])]
        df = df.sort_values(
            ["priority","match_score"],
            ascending=[True,False]
        )
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/tracker")
def get_tracker(request: Request):
    user = get_current_user(request)
    if user:
        return get_user_applications(user["id"])
    # Fallback to CSV for backward compatibility
    df = load_tracker()
    return df.fillna("").to_dict(orient="records")


@app.post("/api/ats")
def run_ats(req: JobRequest):
    resume = load_resume()
    prompt = f"""You are a strict ATS specialist. Analyse this CV against this job.
Use the FULL 0-100 range. Do NOT default to 85.

CV:
{resume}

Job: {req.title} at {req.company} ({req.job_type})

IGNORE pre-computed fields. Do a FRESH analysis from the job description below.

{req.description[:2000] if req.description else "Title: " + req.title + " at " + req.company}
{req.description[:1500] if req.description else "Not available"}

STRICT SCORING RUBRIC:
90-100: 90 percent+ of job keywords present. Perfect alignment.
75-89:  70-89 percent of keywords. Minor gaps only.
60-74:  50-69 percent of keywords. Several important gaps.
40-59:  30-49 percent of keywords. Major gaps.
0-39:   Less than 30 percent. Wrong profile entirely.

HARD PENALTIES:
- Job requires German C1/C2, CV shows A2: -25 points
- Job requires 3+ years experience: -20 points
- Job completely unrelated to candidate: -40 points
- Missing critical tech (AWS, C++, ROS): -10 per item

BONUSES:
- Published research matching job domain: +10
- Direct industry experience match: +10
- All core technical skills present: +5

Respond ONLY with valid JSON:
{{
  "ats_score": <integer 0-100>,
  "critical_missing": ["kw1","kw2"],
  "found_keywords": ["kw1","kw2"],
  "top_suggestion": "one specific actionable suggestion",
  "summary": "2 specific sentences explaining the score"
}}"""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        raw = r.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/recruiter")
def get_recruiter(req: JobRequest):
    try:
        contacts, domain = find_recruiter(req.company)
        return {"contacts": contacts, "domain": domain}
    except Exception as e:
        return {"error": str(e), "contacts": [], "domain": ""}


@app.post("/api/email")
def generate_email(req: dict):
    resume  = load_resume()
    job     = req.get("job", {})
    name    = req.get("recruiter_name", "Hiring Team")
    email   = req.get("recruiter_email", "")
    greeting = f"Dear {name.split()[0]}" if name and name not in ["Hiring Team","Recruiting Team","HR Team"] else "Dear Hiring Team"

    prompt = f"""Write a cold job application email.

CANDIDATE CV:
{resume}

JOB: {job.get('title')} at {job.get('company')} ({job.get('job_type')})
RECRUITER: {name} <{email}>

Rules:
- Start with: {greeting},
- Max 150 words
- Mention role and company by name
- Reference 1-2 specific projects with real numbers from CV
- Mention available 20hrs/week as Werkstudent in Cottbus
- End with clear call to action
- Sound human not robotic

Output email body only. No subject. No explanation."""

    subj_prompt = f"""Write email subject for job application.
Role: {job.get('title')} at {job.get('company')}
Max 10 words. Professional. Don't start with 'Application for'.
Output subject line only."""

    try:
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=400, temperature=0.7,
        )
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":subj_prompt}],
            max_tokens=30, temperature=0.7,
        )
        return {
            "subject": r2.choices[0].message.content.strip(),
            "body":    r1.choices[0].message.content.strip(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/apply")
def log_application(req: ApplyRequest):
    df    = load_tracker()
    job   = req.job
    today = datetime.now().strftime("%Y-%m-%d")
    fu    = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    # Check duplicate
    exists = df[
        (df["title"]   == job.get("title")) &
        (df["company"] == job.get("company"))
    ]
    if not exists.empty:
        return {"status": "already_logged",
                "message": "Already applied to this job"}

    new_row = {
        "date_applied":    today,
        "title":           job.get("title",""),
        "company":         job.get("company",""),
        "location":        job.get("location",""),
        "recruiter_name":  req.recruiter_name,
        "recruiter_email": req.recruiter_email,
        "match_score":     job.get("match_score",""),
        "status":          "Applied",
        "follow_up_date":  fu,
        "notes":           req.notes,
        "link":            job.get("link",""),
        "job_type":        job.get("job_type",""),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_tracker(df)
    return {"status": "logged", "follow_up_date": fu}


@app.post("/api/status")
def update_status(req: StatusUpdate):
    df = load_tracker()
    mask = (df["title"] == req.title) & (df["company"] == req.company)
    if not mask.any():
        return {"error": "Application not found"}
    df.loc[mask, "status"] = req.status
    if req.notes:
        df.loc[mask, "notes"] = req.notes
    save_tracker(df)
    return {"status": "updated"}


@app.get("/api/followups")
def get_followups():
    df    = load_tracker()
    today = datetime.now().date()
    try:
        due = df[
            (df["status"] == "Applied") &
            (pd.to_datetime(df["follow_up_date"]).dt.date <= today)
        ]
        return due.fillna("").to_dict(orient="records")
    except:
        return []




# ── Serve dashboard ───────────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/")
def serve_root():
    return FileResponse("dashboard.html")

@app.get("/dashboard.html")
def serve_dashboard():
    return FileResponse("dashboard.html")

@app.post("/api/cv")
async def upload_cv(req: dict, request: Request):
    user = get_current_user(request)
    if not user:
        return {"error": "Not logged in"}
    update_cv(user["id"], req.get("cv_text",""))
    return {"status": "saved"}


# ── Per-user CV upload and scoring ────────────────────────────────────────────

from fastapi import UploadFile, File
from src.google_auth import save_user_scores

@app.post("/api/cv/upload")
async def upload_cv(request: Request, cv_text: str = ""):
    """Save user's CV text and trigger per-user scoring."""
    token = request.cookies.get("session_token")
    if not token:
        return {"error": "Not logged in"}

    import sqlite3
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE users SET cv_text=? WHERE email=?", (cv_text, token))
    conn.commit()
    c.execute("SELECT id FROM users WHERE email=?", (token,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "User not found"}

    return {"status": "saved", "user_id": row[0],
            "message": "CV saved. Click Score My CV to get personalised results."}


@app.post("/api/cv/score")
async def score_cv_for_user(request: Request):
    """Score all current jobs against the user's uploaded CV using RAG."""
    import sys, os
    sys.path.append(os.path.dirname(__file__))

    token = request.cookies.get("session_token")
    if not token:
        return {"error": "Not logged in"}

    import sqlite3
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, cv_text FROM users WHERE email=?", (token,))
    row = c.fetchone()
    conn.close()

    if not row or not row[1]:
        return {"error": "No CV found. Please upload your CV first."}

    user_id = row[0]
    cv_text = row[1]

    try:
        from src.rag_matcher import run_rag_matching
        import tempfile, pandas as pd

        # Run RAG matching with user's CV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_output = f.name

        result_df = run_rag_matching(cv_text, "jobs.csv", tmp_output)

        if result_df is not None:
            scores = result_df.fillna("").to_dict(orient="records")
            save_user_scores(user_id, scores)
            os.unlink(tmp_output)
            return {
                "status": "scored",
                "total": len(scores),
                "apply_now": len([s for s in scores if s.get("verdict") == "Apply Now"]),
                "message": f"Scored {len(scores)} jobs against your CV"
            }
        return {"error": "Scoring failed"}
    except Exception as e:
        return {"error": str(e)}
