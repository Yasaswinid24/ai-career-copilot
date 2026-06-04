"""
AI Career Copilot — Unified Apply Workflow
-------------------------------------------
Step 1: Pick a job
Step 2: Find recruiter contact
Step 3: Generate personalised email with their name
Step 4: You review and approve
Step 5: Save to outreach_drafts.txt
"""

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from recruiter_finder import find_recruiter

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "matched_jobs.csv"
OUTPUT_FILE = "outreach_drafts.txt"


def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()


def generate_email(resume, job, recruiter_name, recruiter_email):
    greeting = f"Dear {recruiter_name}" if recruiter_name and recruiter_name != "Recruiting Team" else "Dear Hiring Team"

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

RECRUITER: {recruiter_name} ({recruiter_email})

Write a cold email starting with: {greeting},

STRICT RULES:
- Maximum 150 words
- Address the recruiter by first name if their full name is known
- Mention the specific role and company by name
- Reference 1-2 specific projects or achievements with real numbers
- Mention available 20hrs/week as Werkstudent in Cottbus, Germany
- End with one clear call to action
- Sound human and specific — not a template
- Never start with "I am writing to express my interest"

Output only the email body. No subject line. No explanation.
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return r.choices[0].message.content.strip()


def generate_subject(job, recruiter_name):
    prompt = f"""Write a short email subject line for a job application.
Role: {job.get('title')}
Company: {job.get('company')}
Recruiter: {recruiter_name}

Rules:
- Maximum 10 words
- Professional and specific
- Do not start with "Application for"
- Output only the subject line, nothing else
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.7,
    )
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


def main():
    print("=" * 55)
    print("  AI Career Copilot — Full Apply Workflow")
    print("=" * 55)

    resume = load_resume()

    try:
        df = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("matched_jobs.csv not found. Run resume_matcher.py first.")
        return

    top = df[df["verdict"] == "Apply Now"].sort_values(
        "match_score", ascending=False
    ).head(20).reset_index(drop=True)

    print("\nTop jobs:\n")
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

    # Step 1 — Find recruiter
    print("\nStep 1 — Finding recruiter...")
    contacts, domain = find_recruiter(job.get('company',''))

    print(f"\nContacts found:")
    for i, c in enumerate(contacts):
        print(f"  [{i+1}] {c['name']:<25} {c['email']:<35} {c['confidence']}% confidence")

    print()
    rec_choice = input("Pick a contact number to email (or press Enter for first): ").strip()
    try:
        recruiter = contacts[int(rec_choice)-1] if rec_choice else contacts[0]
    except (ValueError, IndexError):
        recruiter = contacts[0]

    print(f"\nEmailing: {recruiter['name']} — {recruiter['email']}")

    # Step 2 — Generate email
    print("\nStep 2 — Generating personalised email...")
    subject = generate_subject(job, recruiter['name'])
    email   = generate_email(resume, job, recruiter['name'], recruiter['email'])

    # Step 3 — Review loop
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

        if action == 's':
            save_draft(job, recruiter, subject, email, "READY TO SEND")
            print(f"\nSaved to {OUTPUT_FILE}")
            print(f"\nNext steps:")
            print(f"  1. Open: {job.get('link','the job link')}")
            print(f"  2. Send email to: {recruiter['email']}")
            print(f"  3. Log your application in applications.csv (coming next)")
            break

        elif action == 'r':
            print("\nRegenerating...\n")
            subject = generate_subject(job, recruiter['name'])
            email   = generate_email(resume, job, recruiter['name'], recruiter['email'])

        elif action == 'e':
            new = input("New subject line: ").strip()
            if new:
                subject = new

        elif action == 'c':
            for i, c in enumerate(contacts):
                print(f"  [{i+1}] {c['name']} — {c['email']}")
            pick = input("Choose contact: ").strip()
            try:
                recruiter = contacts[int(pick)-1]
                print(f"Switched to: {recruiter['name']}")
            except (ValueError, IndexError):
                print("Invalid — keeping current contact")

        elif action == 'q':
            save_draft(job, recruiter, subject, email, "DISCARDED")
            break
        else:
            print("Please enter s, r, e, c, or q.")


if __name__ == "__main__":
    main()
