"""
AI Career Copilot — Daily Runner
"""
import os, sys, subprocess, socket, pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

def header(text):
    print(f"\n{'='*55}\n  {text}\n{'='*55}")

def check_keys():
    print("\nAPI Keys:")
    for name, key in [("Groq","GROQ_API_KEY"),("Hunter","HUNTER_API_KEY"),("Adzuna","ADZUNA_APP_ID")]:
        val = os.getenv(key)
        print(f"  {name:<8} {'OK' if val else 'MISSING'}")

def check_freshness():
    try:
        df = pd.read_csv("jobs.csv")
        latest = pd.to_datetime(df["scraped_at"]).max()
        age = datetime.now() - latest
        print(f"\n  Last scraped: {latest.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Data age:     {age.days}d {age.seconds//3600}h")
        if age > timedelta(days=2):
            print("  Status: STALE — rescrape recommended")
            return True
        print("  Status: FRESH")
        return False
    except:
        print("  No data yet — scrape first")
        return True

def check_followups():
    try:
        df = pd.read_csv("applications.csv")
        today = datetime.now().date()
        due = df[
            (df["status"] == "Applied") &
            (pd.to_datetime(df["follow_up_date"]).dt.date <= today)
        ]
        if not due.empty:
            print(f"\n  ACTION NEEDED — {len(due)} follow-up(s) due:")
            for _, r in due.iterrows():
                print(f"    ! {r['title']} at {r['company']}")
        else:
            print(f"\n  No follow-ups due today")
        print(f"  Total applications: {len(df)}")
    except:
        print("\n  No applications logged yet")

def run(script):
    subprocess.run([sys.executable, script],
                   cwd=os.path.dirname(os.path.abspath(__file__)))

def git_push(message):
    subprocess.run(["git", "add", "jobs.csv", "matched_jobs.csv", "thesis_roles.csv"])
    result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
    if "nothing to commit" in result.stdout:
        print("  Nothing new to push")
    else:
        subprocess.run(["git", "push"])
        print("  Live site updated!")

def start_dashboard():
    port = 8080
    while True:
        try:
            s = socket.socket()
            s.bind(("", port))
            s.close()
            break
        except OSError:
            port += 1
    print(f"\nStarting dashboard on port {port}")
    print("Open Ports tab → click globe icon")
    print("Press Ctrl+C to stop\n")
    subprocess.run([sys.executable, "-m", "http.server", str(port)])

def main():
    print("="*55)
    print(f"  AI Career Copilot — {datetime.now().strftime('%A, %d %B %Y')}")
    print("="*55)

    header("System Status")
    check_keys()

    header("Data Freshness")
    check_freshness()

    header("Follow-up Reminders")
    check_followups()

    header("What do you want to do?")
    print("""
  [1] Full pipeline  (scrape + filter + score + push + dashboard)
  [2] Scrape fresh jobs only
  [3] Check eligibility of scraped jobs
  [4] Score jobs against my CV
  [5] Apply to a job  (recruiter + email)
  [6] Tailor CV for a job  (ATS check)
  [7] Application tracker
  [8] Open dashboard
  [9] Rescore jobs only (resume updated, jobs unchanged)
  [q] Quit
""")

    while True:
        choice = input("Your choice: ").strip().lower()

        if choice == '1':
            run("src/scraper.py")
            run("src/freshness_check.py")
            run("src/eligibility_checker.py")
            run("src/rag_matcher.py")
            print("\nPushing to GitHub...")
            git_push("Auto-update fresh jobs")
            start_dashboard()

        elif choice == '2':
            run("src/scraper.py")
            run("src/freshness_check.py")
            run("src/eligibility_checker.py")
            print("\nPushing to GitHub...")
            git_push("Fresh job listings")

        elif choice == '3':
            run("src/eligibility_checker.py")

        elif choice == '4':
            run("src/rag_matcher.py")
            print("\nPushing to GitHub...")
            git_push("Updated job scores")

        elif choice == '5':
            run("src/apply.py")

        elif choice == '6':
            run("src/ats_tailor.py")

        elif choice == '7':
            run("src/tracker.py")

        elif choice == '8':
            start_dashboard()

        elif choice == '9':
            subprocess.run([sys.executable, "-c",
                "import os; os.remove('matched_jobs.csv') if os.path.exists('matched_jobs.csv') else None"])
            run("src/rag_matcher.py")
            print("\nPushing to GitHub...")
            git_push("Rescore jobs")

        elif choice == 'q':
            print("\nGood luck today!")
            break

        else:
            print("Enter 1-9 or q")

if __name__ == "__main__":
    main()
