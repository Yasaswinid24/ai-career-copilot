"""
AI Career Copilot — Freshness Checker
Only keeps jobs posted within 14 days.
Anything older is a waste of your time.
"""
import pandas as pd
from datetime import datetime, timedelta

JOBS_FILE = "jobs.csv"
MAX_AGE_DAYS = 14  # Only apply to jobs posted within 2 weeks

def classify_freshness(date_str):
    try:
        posted = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        age = (datetime.now().date() - posted).days
        if age <= 3:    return "Hot",    age  # apply today
        elif age <= 7:  return "Fresh",  age  # apply this week
        elif age <= 14: return "Recent", age  # still worth it
        else:           return "Stale",  age  # skip
    except:
        return "Unknown", 0

def main():
    print("=" * 55)
    print("  Freshness Checker — 14 day maximum")
    print("=" * 55)

    df = pd.read_csv(JOBS_FILE)
    print(f"\nJobs before filter: {len(df)}")

    df[["freshness","age_days"]] = df["published"].apply(
        lambda x: pd.Series(classify_freshness(x))
    )

    print(f"\nFreshness breakdown:")
    for label in ["Hot","Fresh","Recent","Stale","Unknown"]:
        count = len(df[df["freshness"]==label])
        bar   = "█" * (count//2)
        note  = ""
        if label == "Hot":    note = "← apply TODAY"
        if label == "Fresh":  note = "← apply this week"
        if label == "Recent": note = "← still worth it"
        if label == "Stale":  note = "← skip"
        print(f"  {label:<8} {bar} {count} {note}")

    # Keep only Hot, Fresh, Recent
    before   = len(df)
    df_keep  = df[df["freshness"].isin(["Hot","Fresh","Recent","Unknown"])]
    df_stale = df[df["freshness"] == "Stale"]
    removed  = before - len(df_keep)

    print(f"\nRemoved {removed} stale jobs (older than {MAX_AGE_DAYS} days)")
    print(f"Remaining: {len(df_keep)} fresh jobs")

    if not df_stale.empty:
        print(f"\nStale jobs removed (sample):")
        for _, row in df_stale.head(5).iterrows():
            print(f"  ✗ [{row['age_days']}d old] {row['title'][:50]} @ {row['company'][:25]}")

    print(f"\nHot jobs — apply TODAY:")
    hot = df_keep[df_keep["freshness"]=="Hot"].sort_values("age_days")
    for _, row in hot.head(10).iterrows():
        print(f"  🔥 [{row['age_days']}d] {row['title'][:50]} @ {row['company'][:25]}")

    df_keep.to_csv(JOBS_FILE, index=False)
    print(f"\nSaved {len(df_keep)} fresh jobs to {JOBS_FILE}")

if __name__ == "__main__":
    main()
