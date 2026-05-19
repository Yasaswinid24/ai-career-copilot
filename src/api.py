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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(__file__))
from recruiter_finder import find_recruiter

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

app = FastAPI()
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
def get_jobs():
    try:
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
def get_tracker():
    df = load_tracker()
    return df.fillna("").to_dict(orient="records")


@app.post("/api/ats")
def run_ats(req: JobRequest):
    resume = load_resume()
    prompt = f"""You are an ATS specialist. Analyse this CV for this job.

CV:
{resume}

Job: {req.title} at {req.company} ({req.job_type})
Already matched: {req.matched_skills}
Missing: {req.missing_skills}

Respond ONLY with JSON:
{{
  "ats_score": <0-100>,
  "critical_missing": ["kw1","kw2"],
  "found_keywords": ["kw1","kw2"],
  "top_suggestion": "one specific thing to add to CV",
  "summary": "2 sentence assessment"
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
