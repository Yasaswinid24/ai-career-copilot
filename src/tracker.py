"""
AI Career Copilot — Application Tracker Agent
-----------------------------------------------
Logs every application you make.
Tracks status and reminds you to follow up.
Saves to applications.csv
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

TRACKER_FILE = "applications.csv"
JOBS_FILE    = "matched_jobs.csv"

COLUMNS = [
    "date_applied", "title", "company", "location",
    "recruiter_name", "recruiter_email", "match_score",
    "status", "follow_up_date", "notes", "link"
]

STATUSES = [
    "Applied",
    "Follow-up sent",
    "Interview scheduled",
    "Interview done",
    "Offer received",
    "Rejected",
    "Withdrawn"
]


def load_tracker():
    try:
        return pd.read_csv(TRACKER_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(TRACKER_FILE, index=False)
        return df


def save_tracker(df):
    df.to_csv(TRACKER_FILE, index=False)


def log_application(job, recruiter_name="", recruiter_email="", notes=""):
    df = load_tracker()

    # Check if already logged
    existing = df[
        (df["title"] == job.get("title")) &
        (df["company"] == job.get("company"))
    ]
    if not existing.empty:
        print(f"\nAlready logged: {job.get('title')} at {job.get('company')}")
        print(f"Status: {existing.iloc[0]['status']}")
        return

    today      = datetime.now().strftime("%Y-%m-%d")
    follow_up  = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    new_row = {
        "date_applied":    today,
        "title":           job.get("title",""),
        "company":         job.get("company",""),
        "location":        job.get("location",""),
        "recruiter_name":  recruiter_name,
        "recruiter_email": recruiter_email,
        "match_score":     job.get("match_score",""),
        "status":          "Applied",
        "follow_up_date":  follow_up,
        "notes":           notes,
        "link":            job.get("link",""),
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_tracker(df)
    print(f"\nLogged: {job.get('title')} at {job.get('company')}")
    print(f"Follow-up reminder set for: {follow_up}")


def check_followups():
    df = load_tracker()
    if df.empty:
        print("No applications logged yet.")
        return

    today = datetime.now().date()
    due = df[
        (df["status"] == "Applied") &
        (pd.to_datetime(df["follow_up_date"]).dt.date <= today)
    ]

    if due.empty:
        print("\nNo follow-ups due today.")
    else:
        print(f"\n{len(due)} follow-up(s) due:\n")
        for _, row in due.iterrows():
            print(f"  ! {row['title']} at {row['company']}")
            print(f"    Applied: {row['date_applied']}")
            print(f"    Contact: {row['recruiter_name']} — {row['recruiter_email']}")
            print()


def generate_followup(row):
    prompt = f"""Write a short polite follow-up email for a job application.

Details:
- Role: {row['title']}
- Company: {row['company']}
- Applied on: {row['date_applied']}
- Recruiter: {row['recruiter_name']}
- Candidate: Yasaswini Dharmavarapu, M.Sc. AI student at BTU Cottbus

Rules:
- Maximum 80 words
- Polite and professional
- Reference the original application
- Ask if there is any update
- Do not be pushy
- Address recruiter by first name if known

Output only the email body.
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )
    return r.choices[0].message.content.strip()


def update_status():
    df = load_tracker()
    if df.empty:
        print("No applications logged yet.")
        return

    print("\nYour applications:\n")
    for i, row in df.iterrows():
        print(f"  [{i+1:2d}] {row['status']:<22} | {row['title'][:40]:<40} | {row['company'][:25]}")

    print()
    choice = input("Enter number to update status (or q to quit): ").strip()
    if choice.lower() == 'q':
        return

    try:
        idx = int(choice) - 1
        row = df.iloc[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print(f"\nCurrent status: {row['status']}")
    print("\nNew status options:")
    for i, s in enumerate(STATUSES):
        print(f"  [{i+1}] {s}")

    pick = input("\nEnter status number: ").strip()
    try:
        new_status = STATUSES[int(pick)-1]
        df.at[idx, "status"] = new_status

        # If interview scheduled — ask for date
        if new_status == "Interview scheduled":
            date = input("Interview date (YYYY-MM-DD): ").strip()
            df.at[idx, "notes"] = f"Interview: {date}"

        save_tracker(df)
        print(f"Updated to: {new_status}")
    except (ValueError, IndexError):
        print("Invalid choice.")


def show_dashboard():
    df = load_tracker()
    if df.empty:
        print("\nNo applications yet. Start applying!")
        return

    print(f"\n{'='*55}")
    print(f"  APPLICATION TRACKER — {len(df)} total")
    print(f"{'='*55}")

    for status in STATUSES:
        count = len(df[df["status"] == status])
        if count > 0:
            bar = "█" * count
            print(f"  {status:<22} {bar} {count}")

    print(f"\nRecent applications:")
    recent = df.sort_values("date_applied", ascending=False).head(5)
    for _, row in recent.iterrows():
        print(f"  {row['date_applied']} | {row['status']:<20} | {row['title'][:35]} @ {row['company'][:20]}")


def main():
    print("=" * 55)
    print("  AI Career Copilot — Application Tracker")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  [1] Log a new application")
        print("  [2] Check follow-ups due today")
        print("  [3] Generate follow-up email")
        print("  [4] Update application status")
        print("  [5] View dashboard")
        print("  [q] Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == '1':
            try:
                df = pd.read_csv(JOBS_FILE)
                top = df[df["verdict"] == "Apply Now"].sort_values(
                    "match_score", ascending=False
                ).head(20).reset_index(drop=True)
                print()
                for i, row in top.iterrows():
                    print(f"  [{i+1:2d}] {row['title'][:45]:<45} | {row['company'][:25]}")
                pick = input("\nJob number: ").strip()
                job  = top.iloc[int(pick)-1].to_dict()
                rname  = input("Recruiter name (Enter to skip): ").strip()
                remail = input("Recruiter email (Enter to skip): ").strip()
                notes  = input("Notes (Enter to skip): ").strip()
                log_application(job, rname, remail, notes)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            check_followups()

        elif choice == '3':
            df = load_tracker()
            due = df[df["status"] == "Applied"]
            if due.empty:
                print("No applications with Applied status.")
                continue
            print()
            for i, row in due.iterrows():
                print(f"  [{i+1}] {row['title']} at {row['company']}")
            pick = input("\nPick application number: ").strip()
            try:
                row = due.iloc[int(pick)-1]
                print("\nGenerating follow-up email...")
                followup = generate_followup(row)
                print(f"\n{'-'*45}")
                print(followup)
                print(f"{'-'*45}")
                save = input("\nSave this follow-up? (y/n): ").strip().lower()
                if save == 'y':
                    df.at[row.name, "status"] = "Follow-up sent"
                    save_tracker(df)
                    print("Status updated to: Follow-up sent")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            update_status()

        elif choice == '5':
            show_dashboard()

        elif choice == 'q':
            break
        else:
            print("Please enter 1-5 or q.")


if __name__ == "__main__":
    main()
