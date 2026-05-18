"""
AI Career Copilot — Outreach Generator Agent
----------------------------------------------
Reads matched_jobs.csv, lets you pick a job,
then generates a personalised cold email using Groq AI.
You review and approve before anything is sent.
"""

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "matched_jobs.csv"
OUTPUT_FILE = "outreach_drafts.txt"


def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()


def generate_email(resume, job):
    prompt = f"""
You are an expert career coach helping a student write a cold email to apply for a job in Germany.

CANDIDATE PROFILE:
{resume}

JOB DETAILS:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Match Score: {job.get('match_score')}/100
Matched Skills: {job.get('matched_skills')}

Write a short, professional cold email from Yasaswini to the hiring team at {job.get('company')}.

Rules:
- Maximum 150 words
- Mention the specific role and company by name
- Reference 1-2 specific skills or projects that match this role
- Mention she is available 20hrs/week as Werkstudent
- Mention she is based in Cottbus, Germany
- End with a clear call to action
- Sound human, not robotic or generic
- Write in English unless the job title is fully in German

Only output the email text. No subject line yet. No explanation.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_subject(job):
    prompt = f"""
Write a short email subject line for a job application.
Role: {job.get('title')}
Company: {job.get('company')}
Candidate: Yasaswini Dharmavarapu, M.Sc. AI student at BTU Cottbus

Rules:
- Maximum 10 words
- Professional and specific
- Do not use "Application for" — be more creative
- Output only the subject line, nothing else
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    print("=" * 55)
    print("  AI Career Copilot — Outreach Generator")
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
    ).head(15).reset_index(drop=True)

    print("\nTop jobs to apply to:\n")
    for i, row in top.iterrows():
        print(f"  [{i+1:2d}] Score {row['match_score']:3d} | {row['title'][:45]:<45} | {row['company'][:30]}")

    print()
    choice = input("Enter job number to generate outreach email (or q to quit): ").strip()

    if choice.lower() == 'q':
        return

    try:
        idx = int(choice) - 1
        job = top.iloc[idx].to_dict()
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print(f"\nGenerating email for: {job['title']} at {job['company']}...")
    print("Please wait...\n")

    subject = generate_subject(job)
    email   = generate_email(resume, job)

    print("=" * 55)
    print(f"SUBJECT: {subject}")
    print("=" * 55)
    print(email)
    print("=" * 55)

    # Ask for approval
    print("\nOptions:")
    print("  [s] Save this draft")
    print("  [r] Regenerate a new version")
    print("  [q] Quit without saving")

    while True:
        action = input("\nYour choice: ").strip().lower()

        if action == 's':
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"\n{'='*55}\n")
                f.write(f"JOB: {job['title']} at {job['company']}\n")
                f.write(f"LINK: {job.get('link','N/A')}\n")
                f.write(f"SUBJECT: {subject}\n")
                f.write(f"{'='*55}\n")
                f.write(email + "\n")
            print(f"\nDraft saved to {OUTPUT_FILE}")
            break

        elif action == 'r':
            print("\nRegenerating...\n")
            subject = generate_subject(job)
            email   = generate_email(resume, job)
            print("=" * 55)
            print(f"SUBJECT: {subject}")
            print("=" * 55)
            print(email)
            print("=" * 55)

        elif action == 'q':
            print("Exiting without saving.")
            break

        else:
            print("Please enter s, r, or q.")


if __name__ == "__main__":
    main()
