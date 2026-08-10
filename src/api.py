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
from src.cover_letter import generate_cover_letter, save_draft as save_cl_draft
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
    with open(RESUME_FILE, encoding="utf-8") as f:
        return f.read()


def load_tracker():
    try:
        return pd.read_csv(TRACKER_FILE, encoding="utf-8")
    except FileNotFoundError:
        df = pd.DataFrame(columns=TRACKER_COLS)
        df.to_csv(TRACKER_FILE, index=False, encoding="utf-8")
        return df


def save_tracker(df):
    df.to_csv(TRACKER_FILE, index=False, encoding="utf-8")


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
        # Check if user is logged in — serve personalised scores
        token = request.cookies.get("session_token")
        if token:
            import sqlite3
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE email=?", (token,))
            row = c.fetchone()
            conn.close()
            if row:
                scores = get_user_scores(row[0])
                if scores:
                    return scores

        # Fall back in order of usefulness:
        #   matched_jobs.csv   scored
        #   jobs_eligible.csv  filtered but unscored
        #   jobs.csv           everything, including off-domain and senior
        for fname in ["matched_jobs.csv", "jobs_eligible.csv", "jobs.csv"]:
            try:
                df = pd.read_csv(fname, encoding="utf-8")
            except FileNotFoundError:
                continue
            if "verdict" in df.columns:
                df = df[~df["verdict"].isin(["Error", "API Error"])]
                df = df.sort_values(["priority", "match_score"],
                                    ascending=[True, False])
            else:
                # Not yet scored — newest first.
                if "published" in df.columns:
                    df = df.sort_values(["priority", "published"],
                                        ascending=[True, False])
                df["match_score"] = 0
                df["verdict"]     = "Unscored"
                df["reason"]      = "Not yet scored — run full pipeline"
            return df.fillna("").to_dict(orient="records")

        return []
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
    prompt = f"""You are an ATS keyword analyst. Your job is evidence-based only.

STRICT RULES:
1. ONLY report what is EXPLICITLY written in the job description below
2. NEVER invent or assume requirements not stated in the text
3. Do NOT add German requirements unless the JD explicitly mentions German
4. Do NOT add experience years unless the JD explicitly states them
5. Every item in critical_missing MUST be a word/phrase found in the JD text
6. ATOMIC ITEMS ONLY: 1-4 words each. Never copy a whole requirement
   sentence. "Gute Kenntnisse in SQL, Power BI und Power Platform" is THREE
   items, and each must be checked against the CV separately.

CANDIDATE CV:
{resume}

JOB: {req.title} at {req.company}

JOB DESCRIPTION (your ONLY source of truth):
{req.description[:2000] if req.description else req.title + " at " + req.company}

INSTRUCTIONS:
- Read the job description above
- List every technical skill, tool, language explicitly mentioned
- Check each against the CV
- critical_missing = skills in JD but NOT in CV
- found_keywords = skills in BOTH JD and CV
- Score = percentage of JD requirements covered by CV

SCORING:
90-100: CV covers 90%+ of explicit JD requirements
75-89:  CV covers 70-89%
60-74:  CV covers 50-69%
40-59:  CV covers 30-49%
0-39:   CV covers less than 30%

Respond ONLY with valid JSON:
{{
  "ats_score": <integer 0-100>,
  "critical_missing": ["only items explicitly in JD but missing from CV"],
  "found_keywords": ["items in both JD and CV"],
  "top_suggestion": "one specific skill from the JD to add to CV",
  "summary": "2 sentences based only on evidence from the JD"
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
        result = json.loads(raw)

        # Same deterministic guard as rag_matcher: the model repeatedly
        # reports skills as missing that are present in the CV. Whether a
        # string occurs in a document is not a judgement call.
        import re as _re
        cv_norm = _re.sub(r"[^a-z0-9+#]", "", resume.lower())
        still_missing, recovered = [], []
        for item in result.get("critical_missing") or []:
            key = _re.sub(r"[^a-z0-9+#]", "", str(item).lower())
            if len(key) >= 3 and key in cv_norm:
                recovered.append(item)
            else:
                still_missing.append(item)
        if recovered:
            result["critical_missing"] = still_missing
            found = result.get("found_keywords") or []
            result["found_keywords"] = found + [r for r in recovered
                                                if r not in found]
        return result
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
- Reference 1-2 specific projects with real numbers from the CV above
- State availability and location as given in the CV — do not invent them
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
    except Exception:
        return []


# ── Cover Letter ──────────────────────────────────────────────────────────────

class CoverLetterRequest(BaseModel):
    job_role:       str
    company:        str
    tone:           str = "Cambridge 10/10 framework"
    language:       str = "English"
    output_format:  str = "Email body only"
    recruiter_name: str = ""
    key_points:     str = ""
    job_description:str = ""
    save_draft:     bool = False


@app.post("/api/cover_letter")
async def api_cover_letter(req: CoverLetterRequest, request: Request):
    """Generate a cover letter using the Cambridge 10/10 framework."""
    # Load user CV if logged in
    cv_text = ""
    token = request.cookies.get("session_token")
    if token:
        import sqlite3
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT cv_text FROM users WHERE email=?", (token,))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            cv_text = row[0]

    result = generate_cover_letter(
        job_role        = req.job_role,
        company         = req.company,
        tone            = req.tone,
        language        = req.language,
        output_format   = req.output_format,
        recruiter_name  = req.recruiter_name,
        key_points      = req.key_points,
        job_description = req.job_description,
        resume_text     = cv_text,
    )

    if result.get("error"):
        return {"error": result["error"]}

    # Save draft if requested
    if req.save_draft:
        save_cl_draft(
            req.job_role, req.company,
            result["subject"], result["body"],
            req.language, req.tone
        )

    return result


@app.get("/api/cover_letter/drafts")
def get_cover_letter_drafts():
    """Return list of saved cover letter drafts from outreach_drafts.txt"""
    drafts = []
    try:
        with open("outreach_drafts.txt", "r", encoding="utf-8") as f:
            content = f.read()
        blocks = content.strip().split("=" * 55)
        for block in blocks:
            block = block.strip()
            if not block or "COVER LETTER" not in block:
                continue
            lines = block.split("\n")
            draft = {}
            body_lines = []
            in_body = False
            for line in lines:
                if line.startswith("DATE:"):
                    draft["date"] = line.replace("DATE:", "").strip()
                elif line.startswith("JOB:"):
                    draft["job"] = line.replace("JOB:", "").strip()
                elif line.startswith("COMPANY:"):
                    draft["company"] = line.replace("COMPANY:", "").strip()
                elif line.startswith("TONE:"):
                    draft["tone"] = line.replace("TONE:", "").strip()
                elif line.startswith("LANGUAGE:"):
                    draft["language"] = line.replace("LANGUAGE:", "").strip()
                elif line.startswith("SUBJECT:"):
                    draft["subject"] = line.replace("SUBJECT:", "").strip()
                    in_body = True
                elif in_body and line:
                    body_lines.append(line)
            draft["body"] = "\n".join(body_lines).strip()
            if draft.get("job"):
                drafts.append(draft)
    except FileNotFoundError:
        pass
    return list(reversed(drafts))  # newest first


# ── Job Summarizer ────────────────────────────────────────────────────────────

@app.post("/api/summarise")
def summarise_job(req: JobRequest):
    """Summarise a job and give a quick verdict."""
    prompt = f"""Summarise this job for a candidate in 4 bullet points.
Be direct and specific. Tell them exactly what the role involves and whether it's worth applying.

JOB: {req.title} at {req.company} ({req.location})
TYPE: {req.job_type}
DESCRIPTION: {req.description[:1500] if req.description else 'Not available'}

Format your response as JSON:
{{
  "tldr": "One sentence: what this job actually is",
  "good_fit": ["reason 1", "reason 2"],
  "watch_out": ["potential issue 1"],
  "verdict": "Apply now / Worth a look / Skip",
  "time_to_apply": "estimated minutes to apply"
}}

Output valid JSON only."""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        raw = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ── Interview Prep ────────────────────────────────────────────────────────────

class InterviewRequest(BaseModel):
    prep_type: str   # technical | behavioural | paper | mock | german | takehome
    job_role:  str = ""
    company:   str = ""


@app.post("/api/interview_prep")
def interview_prep(req: InterviewRequest):
    """Generate interview prep content based on type."""
    resume = load_resume()

    prompts = {
        "technical": f"""Generate 10 technical ML interview questions for a Werkstudent/intern role.
Tailor them to this candidate's specific stack.

CANDIDATE: {resume}
ROLE: {req.job_role or 'ML Werkstudent'}

For each question provide:
- The question
- What the interviewer is testing
- A strong answer based on the candidate's actual projects

Format as JSON array:
[{{"question": "...", "tests": "...", "answer": "..."}}]
Output valid JSON only.""",

        "behavioural": f"""Generate 8 behavioural interview questions for a German ML Werkstudent role.
Use STAR method (Situation, Task, Action, Result).

CANDIDATE: {resume}
ROLE: {req.job_role or 'ML Werkstudent'} at {req.company or 'German tech company'}

Base answers on real experience from the CV.
Format as JSON array:
[{{"question": "...", "star_answer": {{"situation": "...", "task": "...", "action": "...", "result": "..."}}}}]
Output valid JSON only.""",

        "paper": """Generate 5 questions an interviewer might ask about this paper:
"Bilingual Speech Translation Between Tamil and Telugu" — AIST-2024, Springer
DOI: 10.1007/978-3-031-91331-0_8
Topics: S2UT model, HuBERT, HiFi-GAN, low-resource NLP, unit-to-unit translation

For each question provide a strong, confident answer.
Format as JSON array:
[{"question": "...", "answer": "..."}]
Output valid JSON only.""",

        "german": f"""Provide guidance for a candidate interviewing for a Werkstudent ML
role at a German company. Take the candidate's actual language levels from
the CV below — do not assume them.

CANDIDATE CV:
{resume}

Cover:
1. How to address the German language level question professionally
2. How to discuss Werkstudent availability
3. How to handle visa/residence permit questions
4. German salary expectations for an ML Werkstudent (range + how to discuss)
5. Key cultural differences in German interviews vs international norms
6. 3 sample German phrases that show effort at the candidate's stated level

Format as JSON:
{{"topics": [{{"title": "...", "advice": "...", "example": "..."}}]}}
Output valid JSON only.""",

        "takehome": """Describe the 5 most common ML take-home assignment types at German tech companies.
For each: what they test, typical timeframe, tips for success, and a template structure.

Format as JSON array:
[{"type": "...", "tests": "...", "timeframe": "...", "tips": ["..."], "template": "..."}]
Output valid JSON only.""",
    }

    prompt = prompts.get(req.prep_type)
    if not prompt:
        return {"error": f"Unknown prep type: {req.prep_type}"}

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5,
        )
        raw = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return {"type": req.prep_type, "content": json.loads(raw)}
    except Exception as e:
        return {"error": str(e)}


# ── Follow-up Email ───────────────────────────────────────────────────────────

class FollowUpRequest(BaseModel):
    title:          str
    company:        str
    date_applied:   str
    recruiter_name: str = ""
    recruiter_email:str = ""


@app.post("/api/followup_email")
def generate_followup_email(req: FollowUpRequest):
    """Generate a polite follow-up email for an application."""
    greeting = f"Dear {req.recruiter_name.split()[0]}" if req.recruiter_name else "Dear Hiring Team"
    resume = load_resume()
    prompt = f"""Write a short, polite follow-up email for a job application.

Role: {req.title}
Company: {req.company}
Applied on: {req.date_applied}

CANDIDATE CV (use the name and details from here — do not invent any):
{resume[:800]}

Rules:
- Start with: {greeting},
- Maximum 80 words
- Reference the original application date
- Ask if there is any update without being pushy
- End with one sentence showing continued interest
- Do NOT say "I look forward to hearing from you"

Output only the email body. No subject. No explanation."""

    subj_prompt = f"""Write a follow-up email subject line.
Original role: {req.title} at {req.company}
Max 8 words. Professional. Do not start with "Following up on".
Output only the subject line."""

    try:
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.7,
        )
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": subj_prompt}],
            max_tokens=25, temperature=0.7,
        )
        return {
            "subject": r2.choices[0].message.content.strip(),
            "body":    r1.choices[0].message.content.strip(),
        }
    except Exception as e:
        return {"error": str(e)}


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
    cv_text = req.get("cv_text", "")
    user = get_current_user(request)
    if user:
        # Logged in — save to user profile in DB
        update_cv(user["id"], cv_text)
        return {"status": "saved", "mode": "user_profile"}
    else:
        # Not logged in — save to shared resume.txt as fallback
        try:
            with open("src/resume.txt", "w", encoding="utf-8") as f:
                f.write(cv_text)
            return {"status": "saved", "mode": "shared_resume"}
        except Exception as e:
            return {"error": str(e)}


# ── Per-user CV upload and scoring ────────────────────────────────────────────

from fastapi import UploadFile, File
from src.google_auth import save_user_scores

@app.post("/api/cv/upload")
async def upload_cv_text(request: Request, cv_text: str = ""):  # renamed: was duplicate
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
    """Score all current jobs against the user's uploaded CV."""
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
        # rag_matcher v3 renamed run_rag_matching -> run_matching when the
        # FAISS retrieval was removed. Importing the old name raises
        # ImportError and the Score My CV button fails silently.
        from src.rag_matcher import run_matching
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_output = f.name

        # Score the filtered list, not every posting — jobs.csv now holds
        # off-domain and senior roles too, since the eligibility checker
        # annotates instead of deleting.
        jobs_file = ("jobs_eligible.csv" if os.path.exists("jobs_eligible.csv")
                     else "jobs.csv")
        result_df = run_matching(cv_text, jobs_file, tmp_output)

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


# ── Pipeline triggers ─────────────────────────────────────────────────────────

import subprocess, threading, time as _time

# In-memory pipeline state
_pipeline_state = {
    "scrape":  {"status": "idle", "step": "", "started_at": None, "finished_at": None, "error": None},
    "full":    {"status": "idle", "step": "", "started_at": None, "finished_at": None, "error": None},
}

SCRAPE_STEPS = [
    ("src/scraper.py",             "Scraping jobs from Arbeitsagentur…"),
    ("src/freshness_check.py",     "Removing stale jobs (>14 days)…"),
    ("src/eligibility_checker.py", "Filtering ineligible roles…"),
]

FULL_STEPS = [
    ("src/scraper.py",             "Scraping jobs from Arbeitsagentur…"),
    ("src/freshness_check.py",     "Removing stale jobs (>14 days)…"),
    ("src/eligibility_checker.py", "Filtering ineligible roles…"),
    ("src/rag_matcher.py",         "Scoring jobs against your CV…"),
]

def run_pipeline_bg(pipeline_key: str, steps: list):
    """Run pipeline steps sequentially, updating state at each step."""
    state = _pipeline_state[pipeline_key]
    state["status"]      = "running"
    state["started_at"]  = _time.strftime("%H:%M:%S")
    state["finished_at"] = None
    state["error"]       = None

    def _run():
        try:
            for script, label in steps:
                state["step"] = label
                result = subprocess.run(
                    # sys.executable, not "python3": that name does not exist
                    # on Windows, so local runs failed while Render worked.
                    [sys.executable, script],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)) + "/.."
                )
                if result.returncode != 0:
                    state["status"] = "error"
                    state["error"]  = f"{script} failed: {result.stderr[-300:] if result.stderr else 'unknown error'}"
                    state["step"]   = ""
                    return
            state["status"]      = "done"
            state["step"]        = "Complete!"
            state["finished_at"] = _time.strftime("%H:%M:%S")
        except Exception as e:
            state["status"] = "error"
            state["error"]  = str(e)
            state["step"]   = ""

    t = threading.Thread(target=_run, daemon=True)
    t.start()


@app.post("/api/pipeline/scrape")
def trigger_scrape():
    state = _pipeline_state["scrape"]
    if state["status"] == "running":
        return {"status": "already_running", "message": "Scrape already in progress.", "step": state["step"]}
    run_pipeline_bg("scrape", SCRAPE_STEPS)
    return {"status": "started", "message": "Scraping started!"}


@app.post("/api/pipeline/full")
def trigger_full_pipeline(request: Request):
    state = _pipeline_state["full"]
    if state["status"] == "running":
        return {"status": "already_running", "message": "Pipeline already running.", "step": state["step"]}
    run_pipeline_bg("full", FULL_STEPS)
    return {"status": "started", "message": "Full pipeline started!"}


@app.get("/api/pipeline/status")
def get_pipeline_status():
    """Poll this to get live status of both pipelines."""
    return {
        "scrape": _pipeline_state["scrape"],
        "full":   _pipeline_state["full"],
    }