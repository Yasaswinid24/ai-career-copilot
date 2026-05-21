"""
AI Career Copilot — Job Discovery Agent v2
-------------------------------------------
Now fetches full job descriptions from Arbeitsagentur.
Real descriptions = real ATS matching.
"""

import requests
import pandas as pd
import urllib3
import time
import random
from bs4 import BeautifulSoup
from datetime import datetime

urllib3.disable_warnings()

HEADERS_API = {
    "X-API-Key": "jobboerse-jobsuche",
    "User-Agent": "Jobsuche/2.9.2 (de.arbeitsagentur.jobboerse; build:1077; iOS 15.1.0) Alamofire/5.4.4",
}

HEADERS_WEB = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/app/jobs"

JOB_TYPES = [
    {"angebotsart": 34, "label": "Werkstudent",  "priority": 1},
    {"angebotsart": 4,  "label": "Praktikum",    "priority": 2},
    {"angebotsart": 1,  "label": "Full-time",    "priority": 3},
]

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
        "was": keyword, "wo": "Deutschland",
        "page": page, "size": size,
        "angebotsart": angebotsart,
    }
    try:
        r = requests.get(BASE_URL, headers=HEADERS_API,
                        params=params, timeout=10, verify=False)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Request failed: {e}")
    return None


def fetch_job_description(link):
    """Fetch full job description from Arbeitsagentur job page."""
    try:
        r = requests.get(link, headers=HEADERS_WEB, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            # Find job description section
            if "Stellenbeschreibung" in text:
                idx = text.find("Stellenbeschreibung")
                desc = text[idx:idx+3000]
            else:
                desc = text[:3000]
            return desc.strip()
    except Exception:
        pass
    return ""


def parse_results(data, query, label, priority):
    jobs = []
    if not data or "stellenangebote" not in data:
        return jobs
    for job in data["stellenangebote"]:
        jobs.append({
            "title":       job.get("titel", "N/A"),
            "company":     job.get("arbeitgeber", "N/A"),
            "location":    job.get("arbeitsort", {}).get("ort", "N/A"),
            "region":      job.get("arbeitsort", {}).get("region", "N/A"),
            "job_type":    label,
            "priority":    priority,
            "published":   job.get("aktuelleVeroeffentlichungsdatum", "N/A"),
            "start_date":  job.get("eintrittsdatum", "N/A"),
            "ref_number":  job.get("refnr", "N/A"),
            "link":        f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{job.get('refnr','')}",
            "search_query": query,
            "scraped_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "description": "",  # filled in next step
        })
    return jobs


def main():
    print("=" * 55)
    print("  AI Career Copilot — Job Discovery Agent v2")
    print("  Now fetching full job descriptions")
    print("=" * 55)

    all_jobs = []

    # Step 1 — Collect job listings
    for job_type in JOB_TYPES:
        label       = job_type["label"]
        angebotsart = job_type["angebotsart"]
        priority    = job_type["priority"]
        print(f"\n── {label} roles ──")

        for query in SEARCH_QUERIES:
            print(f"  Searching: '{query}'...")
            data = search_jobs(query, angebotsart, label)
            if data:
                jobs = parse_results(data, query, label, priority)
                print(f"  Found {len(jobs)} jobs")
                all_jobs.extend(jobs)

    # Remove duplicates
    seen   = set()
    unique = []
    for job in all_jobs:
        if job["ref_number"] not in seen:
            seen.add(job["ref_number"])
            unique.append(job)

    print(f"\nTotal unique jobs: {len(unique)}")

    # Step 2 — Fetch job descriptions
    print(f"\nFetching job descriptions (this takes a few minutes)...")
    print("Each job page fetched with polite delay to avoid blocking.\n")

    for i, job in enumerate(unique):
        print(f"  [{i+1}/{len(unique)}] Fetching: {job['title'][:50]}")
        desc = fetch_job_description(job["link"])
        job["description"] = desc
        if desc:
            print(f"    Got {len(desc)} chars")
        else:
            print(f"    No description fetched")
        # Polite delay
        time.sleep(random.uniform(1.5, 3.0))

    # Step 3 — Sort and save
    df = pd.DataFrame(unique)
    df = df.sort_values(["priority", "title"])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"Saved {len(df)} jobs with descriptions to {OUTPUT_FILE}")
    print(f"\nBreakdown:")
    for label in ["Werkstudent","Praktikum","Full-time"]:
        count = len(df[df["job_type"]==label])
        print(f"  {label:<12} {'█'*(count//2)} {count}")

    # Show how many got descriptions
    with_desc = len(df[df["description"].str.len() > 100])
    print(f"\nJobs with full description: {with_desc}/{len(df)}")


if __name__ == "__main__":
    main()
