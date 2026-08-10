"""
AI Career Copilot — Unified Apply Workflow v2

Changes from v1:
  * encoding="utf-8" when reading resume.txt. Windows defaults to cp1252
    and mangles or crashes on any umlaut.
  * Contact selection guards. v1 did contacts[int(x)-1] inside a try whose
    except handler was contacts[0] — fine in practice, since find_recruiter
    always falls back to three generic addresses, but entering "0" silently
    selected contacts[-1], the LAST contact, with no error. Now validated.
  * Error message pointed at resume_matcher.py; the pipeline uses
    rag_matcher.py.
  * Shows why a job scored well (matched/missing skills) before you commit
    to writing an email about it.
"""

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from recruiter_finder import find_recruiter

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

RESUME_FILE = "src/resume.txt"
JOBS_FILE = "matched_jobs.csv"
OUTPUT_FILE = "outreach_drafts.txt"


def load_resume():
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        return f.read()


def generate_email(resume, job, recruiter_name):
    named = recruiter_name and recruiter_name not in (
        "Recruiting Team", "HR Team", "Careers", "Unknown")
    greeting = f"Dear {recruiter_name.split()[0]}" if named else "Dear Hiring Team"

    prompt = f"""You are an expert career coach helping a student write
a cold email to apply for a job in Germany.

CANDIDATE PROFILE:
{resume}

JOB DETAILS:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Contract type: {job.get('job_type')}
Match score: {job.get('match_score')}/100
Skills that matched: {job.get('matched_skills')}
Gaps: {job.get('missing_skills')}

Write a cold email starting with: {greeting},

STRICT RULES:
- Maximum 150 words
- Mention the specific role and company by name
- Reference 1-2 specific projects or achievements with real numbers,
  drawn from the CV above — do not invent any
- State availability and location as given in the CV
- End with one clear call to action
- Sound human and specific, not a template
- Never start with "I am writing to express my interest"
- Do not mention the match score or that a tool scored this job

Output only the email body. No subject line. No explanation.
"""
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.7)
    return r.choices[0].message.content.strip()


def generate_subject(job):
    prompt = f"""Write a short email subject line for a job application.
Role: {job.get('title')}
Company: {job.get('company')}

Rules:
- Maximum 10 words
- Professional and specific
- Do not start with "Application for"
- Output only the subject line, nothing else
"""
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=30, temperature=0.7)
    return r.choices[0].message.content.strip()


def save_draft(job, recruiter, subject, email, status):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"DATE:      {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"STATUS:    {status}\n")
        f.write(f"JOB:       {job.get('title')}\n")
        f.write(f"COMPANY:   {job.get('company')}\n")
        f.write(f"RECRUITER: {recruiter.get('name')} — {recruiter.get('email')}\n")
        f.write(f"SCORE:     {job.get('match_score')}/100\n")
        f.write(f"LINK:      {job.get('link','N/A')}\n")
        f.write(f"{'='*55}\n")
        f.write(f"SUBJECT: {subject}\n\n")
        f.write(email + "\n")


def pick_contact(contacts, prompt="Pick a contact number (Enter for first): "):
    """Validated selection. Rejects 0 and negatives, which in v1 silently
    indexed from the end of the list."""
    if not contacts:
        return None
    for i, c in enumerate(contacts, 1):
        print(f"  [{i}] {c['name']:<25} {c['email']:<35} "
              f"{c.get('confidence', 0)}% confidence")
    raw = input(f"\n{prompt}").strip()
    if not raw:
        return contacts[0]
    if not raw.isdigit() or not (1 <= int(raw) <= len(contacts)):
        print(f"  Not a valid choice — using {contacts[0]['name']}")
        return contacts[0]
    return contacts[int(raw) - 1]


def main():
    print("=" * 55)
    print("  AI Career Copilot — Full Apply Workflow v2")
    print("=" * 55)

    resume = load_resume()

    try:
        df = pd.read_csv(JOBS_FILE, encoding="utf-8")
    except FileNotFoundError:
        print(f"{JOBS_FILE} not found. Run rag_matcher.py first.")
        return

    top = (df[df["verdict"] == "Apply Now"]
           .sort_values("match_score", ascending=False)
           .head(20).reset_index(drop=True))
    if top.empty:
        print("\nNo jobs scored 'Apply Now' yet.")
        return

    print("\nTop jobs:\n")
    for i, row in top.iterrows():
        print(f"  [{i+1:2d}] {row['match_score']:3d} | "
              f"{str(row['title'])[:45]:<45} | {str(row['company'])[:25]}")

    choice = input("\nEnter job number (or q to quit): ").strip()
    if choice.lower() == "q":
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(top)):
        print("Invalid choice.")
        return
    job = top.iloc[int(choice) - 1].to_dict()

    print(f"\nSelected: {job['title']} at {job['company']}")
    print(f"  matched: {job.get('matched_skills','')}")
    print(f"  gaps:    {job.get('missing_skills','')}")
    print(f"  why:     {job.get('reason','')}")

    print("\nStep 1 — Finding recruiter...")
    contacts, domain = find_recruiter(job.get("company", ""))
    if not contacts:
        print(f"  No contacts found for {domain}. Apply via the posting link:")
        print(f"  {job.get('link','')}")
        return

    print("\nContacts found:")
    recruiter = pick_contact(contacts)
    print(f"\nEmailing: {recruiter['name']} — {recruiter['email']}")

    print("\nStep 2 — Generating personalised email...")
    subject = generate_subject(job)
    email = generate_email(resume, job, recruiter["name"])

    while True:
        print(f"\n{'='*55}")
        print(f"TO:      {recruiter['name']} <{recruiter['email']}>")
        print(f"SUBJECT: {subject}")
        print(f"{'='*55}")
        print(email)
        print(f"{'='*55}")
        print("\n  [s] Save as ready to send")
        print("  [r] Regenerate email")
        print("  [e] Edit subject line")
        print("  [c] Change recruiter contact")
        print("  [q] Quit without saving")

        action = input("\nYour choice: ").strip().lower()

        if action == "s":
            save_draft(job, recruiter, subject, email, "READY TO SEND")
            print(f"\nSaved to {OUTPUT_FILE}")
            print("\nNext steps:")
            print(f"  1. Open: {job.get('link','the job link')}")
            print(f"  2. Send email to: {recruiter['email']}")
            print("  3. Log it with tracker.py")
            break
        elif action == "r":
            print("\nRegenerating...\n")
            subject = generate_subject(job)
            email = generate_email(resume, job, recruiter["name"])
        elif action == "e":
            new = input("New subject line: ").strip()
            if new:
                subject = new
        elif action == "c":
            recruiter = pick_contact(contacts, "Choose contact: ")
            print(f"Switched to: {recruiter['name']}")
        elif action == "q":
            save_draft(job, recruiter, subject, email, "DISCARDED")
            break
        else:
            print("Please enter s, r, e, c, or q.")


if __name__ == "__main__":
    main()