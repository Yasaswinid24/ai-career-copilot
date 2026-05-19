"""
AI Career Copilot — Job Discovery Agent
-----------------------------------------
Targets Werkstudent and Praktikum roles specifically.
Full-time roles are deprioritised.
"""

import requests
import pandas as pd
import urllib3
from datetime import datetime

urllib3.disable_warnings()

HEADERS = {
    "X-API-Key": "jobboerse-jobsuche",
    "User-Agent": "Jobsuche/2.9.2 (de.arbeitsagentur.jobboerse; build:1077; iOS 15.1.0) Alamofire/5.4.4",
}

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/app/jobs"

# ── Job type priorities ───────────────────────────────────────────────────────
JOB_TYPES = [
    {"angebotsart": 34, "label": "Werkstudent",  "priority": 1},
    {"angebotsart": 4,  "label": "Praktikum",    "priority": 2},
    {"angebotsart": 1,  "label": "Full-time",    "priority": 3},
]

# ── Search queries focused on AI/ML ──────────────────────────────────────────
SEARCH_QUERIES = [
    "Machine Learning",
    "Artificial Intelligence",
    "Deep Learning",
    "Natural Language Processing",
    "Data Science",
    "Computer Vision",
    "AI Engineer",
]

OUTPUT_FILE = "jobs.csv"


def search_jobs(keyword, angebotsart, label, page=1, size=25):
    params = {
        "was":          keyword,
        "wo":           "Deutschland",
        "page":         page,
        "size":         size,
        "angebotsart":  angebotsart,
    }
    try:
        r = requests.get(
            BASE_URL, headers=HEADERS,
            params=params, timeout=10, verify=False
        )
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  Error {r.status_code}")
            return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def parse_results(data, query, label, priority):
    jobs = []
    if not data or "stellenangebote" not in data:
        return jobs

    for job in data["stellenangebote"]:
        jobs.append({
            "title":        job.get("titel", "N/A"),
            "company":      job.get("arbeitgeber", "N/A"),
            "location":     job.get("arbeitsort", {}).get("ort", "N/A"),
            "region":       job.get("arbeitsort", {}).get("region", "N/A"),
            "job_type":     label,
            "priority":     priority,
            "published":    job.get("aktuelleVeroeffentlichungsdatum", "N/A"),
            "start_date":   job.get("eintrittsdatum", "N/A"),
            "ref_number":   job.get("refnr", "N/A"),
            "link":         f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{job.get('refnr','')}",
            "search_query": query,
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    return jobs


def main():
    print("=" * 55)
    print("  AI Career Copilot — Job Discovery Agent")
    print("  Targeting: Werkstudent > Praktikum > Full-time")
    print("=" * 55)

    all_jobs = []

    for job_type in JOB_TYPES:
        label       = job_type["label"]
        angebotsart = job_type["angebotsart"]
        priority    = job_type["priority"]

        print(f"\n── {label} roles ──────────────────────────")

        for query in SEARCH_QUERIES:
            print(f"  Searching: '{query}'...")
            data = search_jobs(query, angebotsart, label)
            if data:
                total = data.get("maxErgebnisse", 0)
                jobs  = parse_results(data, query, label, priority)
                print(f"  Found {len(jobs)} / {total} available")
                all_jobs.extend(jobs)

    # Remove duplicates by ref number
    seen   = set()
    unique = []
    for job in all_jobs:
        if job["ref_number"] not in seen:
            seen.add(job["ref_number"])
            unique.append(job)

    # Sort — Werkstudent first, then Praktikum, then Full-time
    df = pd.DataFrame(unique)
    df = df.sort_values(["priority", "title"])

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"Total unique jobs: {len(df)}")
    print(f"\nBreakdown:")
    for label in ["Werkstudent", "Praktikum", "Full-time"]:
        count = len(df[df["job_type"] == label])
        bar   = "█" * (count // 2)
        print(f"  {label:<12} {bar} {count}")

    print(f"\nTop Werkstudent roles:")
    ws = df[df["job_type"] == "Werkstudent"].head(5)
    print(ws[["title","company","location"]].to_string(index=False))

    print(f"\nTop Praktikum roles:")
    pk = df[df["job_type"] == "Praktikum"].head(5)
    print(pk[["title","company","location"]].to_string(index=False))


if __name__ == "__main__":
    main()
