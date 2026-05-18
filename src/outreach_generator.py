"""
AI Career Copilot — Outreach Generator Agent
----------------------------------------------
Picks a job from matched_jobs.csv, writes a
personalised cold email, you review and approve,
then it gets saved to outreach_drafts.txt.
Nothing is ever sent automatically.
"""

import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

RESUME_FILE  = "src/resume.txt"
JOBS_FILE    = "matched_jobs.csv"
OUTPUT_FILE  = "outreach_drafts.txt"


def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()


def generate_subject(job):
    prompt = f"""Write a short email subject line for a job application.
Role: {job.get('title')}
Company: {job.get('company')}
Candidate: Yasaswini Dharmavarapu, M.Sc. AI student at BTU Cottbus

Rules:
- Maximum 10 words
- Professional and specific
- Do not start with "Application for"
- Output only the subject line, nothing else
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=30,
        temperature=0.7,
    )
    return r.choices[0].message.content.strip()


def generate_email(resume, job):
    prompt = f"""You are an expert career coach helping a student write
a cold email to apply for a job in Germany.

CANDIDATE PROFILE:
{resume}

JOB DETAILS:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Match Score: {job.get('match_score')}/100
Matched Skills: {job.get('matched_skills')}
Missing Skills: {job.get('missing_skills')}

Write a short professional cold email from Yasaswini to the hiring team.

STRICT RULES:
- Maximum 150 words — recruiters stop reading after this
- Mention the specific role and company by name
- Reference 1-2 specific projects or skills that match this role
- Mention she is available 20hrs/week as Werkstudent
- Mention she is based in Cottbus, Germany
- End with one clear call to action
- Sound like a real human, not a template
- Do not use phrases like "I am writing to express my interest"
- Do not list skills like a robot — weave them naturally
- Write in English unless the job title is fully in German

Output only the email body. No subject line. No explanation.
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return r.choices[0].message.content.strip()


def save_draft(job, subject, email, status="approved"):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"DATE:    {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"STATUS:  {status}\n")
        f.write(f"JOB:     {job.get('title')}\n")
        f.write(f"COMPANY: {job.get('company')}\n")
        f.write(f"LINK:    {job.get('link','N/A')}\n")
        f.write(f"SCORE:   {job.get('match_score')}/100\n")
        f.write(f"{'='*55}\n")
        f.write(f"SUBJECT: {subject}\n\n")
        f.write(email + "\n")


def main():
    print("=" * 55)
    print("  AI Career Copilot — Outreach Generator")
    print("  Nothing sends automatically. You approve first.")
    print("=" * 55)

    resume = load_resume()

    try:
        df = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("matched_jobs.csv not found. Run resume_matcher.py first.")
        return

    # Show top Apply Now jobs
    top = df[df["verdict"] == "Apply Now"].sort_values(
        "match_score", ascending=False
    ).head(20).reset_index(drop=True)

    print("\nTop jobs to write outreach for:\n")
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

    print(f"\nGenerating email for:")
    print(f"  Role:    {job['title']}")
    print(f"  Company: {job['company']}")
    print(f"  Score:   {job['match_score']}/100")
    print(f"\nPlease wait...\n")

    subject = generate_subject(job)
    email   = generate_email(resume, job)

    while True:
        print("=" * 55)
        print(f"SUBJECT: {subject}")
        print("=" * 55)
        print(email)
        print("=" * 55)

        print("\nOptions:")
        print("  [s] Save and mark as ready to send")
        print("  [r] Regenerate a fresh version")
        print("  [e] Edit subject line")
        print("  [q] Quit without saving")

        action = input("\nYour choice: ").strip().lower()

        if action == 's':
            save_draft(job, subject, email, status="READY TO SEND")
            print(f"\nDraft saved to {OUTPUT_FILE}")
            print(f"Next step: copy the email, open {job.get('link','the job link')}")
            print("and send it to the recruiter or use the apply form.")
            break

        elif action == 'r':
            print("\nRegenerating...\n")
            subject = generate_subject(job)
            email   = generate_email(resume, job)

        elif action == 'e':
            new_subject = input("Enter new subject line: ").strip()
            if new_subject:
                subject = new_subject
                print(f"Subject updated to: {subject}")

        elif action == 'q':
            save_draft(job, subject, email, status="DISCARDED")
            print("Exiting without saving as ready.")
            break

        else:
            print("Please enter s, r, e, or q.")


if __name__ == "__main__":
    main()
