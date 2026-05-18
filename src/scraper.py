"""
AI Career Copilot — Job Discovery Agent
Uses Bundesagentur für Arbeit (Germany's official job portal)
No registration or API key signup needed.
"""

import requests
import pandas as pd
from datetime import datetime

HEADERS = {
    "X-API-Key": "jobboerse-jobsuche",
    "User-Agent": "Jobsuche/2.9.2 (de.arbeitsagentur.jobboerse; build:1077; iOS 15.1.0) Alamofire/5.4.4",
}

SEARCH_QUERIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "AI Engineer",
]

OUTPUT_FILE = "jobs.csv"
BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/app/jobs"


def search_jobs(keyword, page=1, size=25):
    params = {
        "was": keyword,        # job title keyword
        "wo": "Deutschland",   # location
        "page": page,
        "size": size,
        "angebotsart": 1,      # 1=jobs, 4=internships, 34=working student
    }
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  Error {response.status_code} for '{keyword}'")
            return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def parse_results(data, query):
    jobs = []
    if not data or "stellenangebote" not in data:
        return jobs
    for job in data["stellenangebote"]:
        jobs.append({
            "title": job.get("titel", "N/A"),
            "company": job.get("arbeitgeber", "N/A"),
            "location": job.get("arbeitsort", {}).get("ort", "N/A"),
            "region": job.get("arbeitsort", {}).get("region", "N/A"),
            "job_type": job.get("angebotsart", "N/A"),
            "published": job.get("eintrittsdatum", "N/A"),
            "ref_number": job.get("refnr", "N/A"),
            "link": f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{job.get('refnr', '')}",
            "search_query": query,
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    return jobs


def main():
    print("=" * 55)
    print("  AI Career Copilot — Job Discovery Agent")
    print("  Source: Bundesagentur für Arbeit (Official)")
    print("=" * 55)

    import urllib3
    urllib3.disable_warnings()  # suppress SSL warnings cleanly

    all_jobs = []

    for query in SEARCH_QUERIES:
        print(f"\nSearching: '{query}'...")
        data = search_jobs(query)
        if data:
            total = data.get("maxErgebnisse", 0)
            print(f"  Total available: {total}")
            jobs = parse_results(data, query)
            print(f"  Fetched: {len(jobs)} jobs")
            all_jobs.extend(jobs)
        else:
            print(f"  No results returned.")

    # Remove duplicates by reference number
    seen = set()
    unique = []
    for job in all_jobs:
        if job["ref_number"] not in seen:
            seen.add(job["ref_number"])
            unique.append(job)

    print(f"\nTotal unique jobs found: {len(unique)}")

    if unique:
        df = pd.DataFrame(unique)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
        print("\nPreview:")
        print(df[["title", "company", "location"]].head(10).to_string())
    else:
        print("No jobs found.")

if __name__ == "__main__":
    main()
