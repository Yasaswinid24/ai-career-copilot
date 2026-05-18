"""
AI Career Copilot — ATS Resume Tailor Agent
---------------------------------------------
PRESERVATION RULES (hardcoded, never violated):
- All project names, metrics, and outcomes are FROZEN
- All experience facts are FROZEN  
- Publication details are FROZEN
- Only Profile Summary wording and Skills ORDER may change
- Only TRUE skills that are missing from CV text can be added
"""

import os
import json
import requests
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "matched_jobs.csv"
OUTPUT_DIR  = "tailored_cvs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FROZEN FACTS — these are never touched by the agent ──────────────────────
FROZEN_FACTS = """
FROZEN — NEVER MODIFY THESE:
- Fractal Analytics internship: BERT sentiment pipeline, 95%+ accuracy, 65% time reduction
- BTU Research Assistant: Music AI, PyTorch, Librosa, MFCCs, spectrograms
- Publication: AIST-2024 Springer, DOI 10.1007/978-3-031-91331-0_8
- Job Email Classification: XLM-RoBERTa, AUC 0.95, CNN vs transformer, 8 F1 points
- Speech Translation: HuBERT, HiFi-GAN, ASR+MT+TTS, basis of publication
- Drowsiness Detection: CNN, OpenCV, 99%+ accuracy, 2 second alert
- Sentiment Analysis: BERT/DistilBERT/CRF, F1 improved 6 points
- AI Career Copilot: LangGraph, Groq, RAG, FAISS, FastAPI, Docker
- Education: BTU Cottbus M.Sc. AI 2025-2027, Amrita B.Tech CSE 2020-2024 CGPA 7.2
- Languages: English C1, German A2, Telugu Native, Tamil Professional
"""


def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()


def fetch_job_description(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","nav","footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return text[:3000]
    except Exception:
        pass
    return None


def analyse_ats(resume, job, job_description=None):
    jd_section = f"\nFULL JOB DESCRIPTION:\n{job_description}" if job_description else ""

    prompt = f"""
You are an ATS specialist. Analyse this CV against this job posting.

CANDIDATE CV:
{resume}

JOB:
Title: {job.get('title')}
Company: {job.get('company')}
{jd_section}

Respond ONLY with valid JSON:
{{
  "ats_score": <integer 0-100>,
  "found_keywords": ["keyword1", "keyword2"],
  "missing_critical": ["keyword1", "keyword2"],
  "missing_nice_to_have": ["keyword1", "keyword2"],
  "suggestions": [
    {{
      "section": "Profile Summary or Skills only",
      "add": "exact short text to add",
      "reason": "why this keyword matters for this role",
      "is_truthful": true
    }}
  ],
  "summary": "2 sentence assessment"
}}

IMPORTANT: Only suggest adding keywords that are genuinely true based on the CV.
Never suggest fabricating experience. Only suggest rewording or reordering.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)


def generate_tailored_sections(resume, job, analysis):
    missing   = ", ".join(analysis.get("missing_critical", []))
    nice      = ", ".join(analysis.get("missing_nice_to_have", []))
    found     = ", ".join(analysis.get("found_keywords", []))

    prompt = f"""
You are an expert CV writer. Your job is to tailor ONLY the Profile Summary
and Skills sections of this CV for a specific job.

ORIGINAL CV:
{resume}

TARGET JOB:
Title: {job.get('title')}
Company: {job.get('company')}

KEYWORDS ALREADY IN CV (keep these):
{found}

MISSING KEYWORDS TO NATURALLY WEAVE IN (only if truthful):
Critical: {missing}
Nice to have: {nice}

{FROZEN_FACTS}

STRICT RULES — you will be penalised for breaking these:
1. NEVER change any project name, metric, accuracy score, or outcome
2. NEVER invent skills or experience that are not in the original CV
3. NEVER remove any existing content
4. ONLY reword the Profile Summary to echo job language naturally
5. ONLY reorder the Skills section to put most relevant skills first
6. You MAY add a missing keyword to Skills if it is clearly demonstrated
   in the projects (e.g. "data pipelines" if they built one)
7. Every added keyword must be something Yasaswini can honestly defend
   in an interview

Output format — two sections only:

PROFILE SUMMARY (tailored):
[max 4 sentences, naturally mentions the role type and 1-2 key skills]

SKILLS (reordered and lightly updated):
[same skills as original, reordered by relevance, max 2 new honest additions]

Then add:

WHAT CHANGED:
[bullet list of exactly what you changed and why — be specific]

WHAT YOU DID NOT CHANGE:
[confirm all projects, metrics, and facts are preserved]
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def save_report(job, analysis, tailored, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ATS TAILOR REPORT\n{'='*55}\n")
        f.write(f"Job:     {job.get('title')}\n")
        f.write(f"Company: {job.get('company')}\n")
        f.write(f"Link:    {job.get('link')}\n")
        f.write(f"{'='*55}\n\n")
        f.write(f"ATS SCORE: {analysis.get('ats_score')}/100\n\n")
        f.write(f"SUMMARY:\n{analysis.get('summary')}\n\n")
        f.write(f"KEYWORDS FOUND:\n")
        for k in analysis.get('found_keywords',[]): f.write(f"  + {k}\n")
        f.write(f"\nCRITICAL MISSING:\n")
        for k in analysis.get('missing_critical',[]): f.write(f"  ! {k}\n")
        f.write(f"\nNICE TO HAVE:\n")
        for k in analysis.get('missing_nice_to_have',[]): f.write(f"  ~ {k}\n")
        f.write(f"\nSUGGESTIONS:\n")
        for s in analysis.get('suggestions',[]):
            f.write(f"\n  [{s.get('section')}]\n")
            f.write(f"  Add:    {s.get('add')}\n")
            f.write(f"  Reason: {s.get('reason')}\n")
            f.write(f"  Truthful: {s.get('is_truthful')}\n")
        f.write(f"\n{'='*55}\n")
        f.write(f"TAILORED SECTIONS:\n{'='*55}\n\n")
        f.write(tailored)


def main():
    print("=" * 55)
    print("  AI Career Copilot — ATS Resume Tailor")
    print("  Your projects and facts are NEVER modified")
    print("=" * 55)

    resume = load_resume()

    try:
        df = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("matched_jobs.csv not found. Run resume_matcher.py first.")
        return

    top = df[df["verdict"] == "Apply Now"].sort_values(
        "match_score", ascending=False
    ).head(15).reset_index(drop=True)

    print("\nTop jobs to tailor your CV for:\n")
    for i, row in top.iterrows():
        print(f"  [{i+1:2d}] {row['match_score']:3d} | {row['title'][:45]:<45} | {row['company'][:25]}")

    print()
    choice = input("Enter job number (or q to quit): ").strip()
    if choice.lower() == 'q':
        return

    try:
        job = top.iloc[int(choice)-1].to_dict()
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print(f"\nSelected: {job['title']} at {job['company']}")
    print("Fetching job description...")
    jd = fetch_job_description(job.get('link',''))
    print(f"{'Got ' + str(len(jd)) + ' chars of job description' if jd else 'Using title only — no description fetched'}")

    print("\nRunning ATS analysis...")
    analysis = analyse_ats(resume, job, jd)

    # Print ATS results
    print(f"\n{'='*55}")
    print(f"ATS SCORE: {analysis.get('ats_score')}/100")
    print(f"{'='*55}")
    print(f"\n{analysis.get('summary')}")
    print(f"\nFOUND IN YOUR CV:")
    for k in analysis.get('found_keywords',[])[:8]:
        print(f"  + {k}")
    print(f"\nCRITICAL MISSING:")
    for k in analysis.get('missing_critical',[]):
        print(f"  ! {k}")
    print(f"\nNICE TO HAVE:")
    for k in analysis.get('missing_nice_to_have',[]):
        print(f"  ~ {k}")
    print(f"\nSUGGESTIONS:")
    for s in analysis.get('suggestions',[]):
        truthful = s.get('is_truthful', True)
        marker = "OK" if truthful else "SKIP - NOT TRUTHFUL"
        print(f"\n  [{s.get('section')}] [{marker}]")
        print(f"  Add:    {s.get('add')}")
        print(f"  Reason: {s.get('reason')}")

    print(f"\n{'='*55}")
    go = input("\nGenerate tailored Profile Summary + Skills? (y/n): ").strip().lower()

    if go != 'y':
        return

    print("\nGenerating tailored sections (preserving all your facts)...")
    tailored = generate_tailored_sections(resume, job, analysis)

    print(f"\n{'='*55}")
    print(tailored)
    print(f"{'='*55}")

    # Build filepath
    company_clean = job.get('company','').replace(' ','_')[:20]
    role_clean    = job.get('title','').replace(' ','_')[:20]
    filepath      = f"{OUTPUT_DIR}/{role_clean}_{company_clean}.txt"

    save_report(job, analysis, tailored, filepath)
    print(f"\nFull report saved to: {filepath}")
    print("\nNext steps:")
    print("  1. Open Overleaf")
    print("  2. Replace ONLY your Profile Summary and Skills with the tailored versions above")
    print("  3. Verify all your project metrics are still intact")
    print("  4. Compile and apply")


if __name__ == "__main__":
    main()
