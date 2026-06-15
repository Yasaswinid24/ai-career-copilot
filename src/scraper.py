"""
AI Career Copilot — Job Discovery Agent v3
-------------------------------------------
Two-stage approach:
  Stage 1: Fast scrape — title + company + location only (2-3 min)
  Stage 2: Fetch JD only for jobs that pass eligibility filter (~3-4 min)

Total: ~5-7 min with real JD content for accurate LLM scoring.
Only jobs posted within last 7 days are kept.
"""

import requests
import pandas as pd
import urllib3
import time
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date

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

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAX_AGE_DAYS        = 7    # Only jobs posted in last 7 days
MAX_JOBS_TOTAL      = 200  # Cap on scraping
MAX_DESC_FETCH      = 80   # Max JDs to fetch (after eligibility filter)
DESC_DELAY          = (1.0, 2.0)  # Polite delay between JD fetches (seconds)
# ─────────────────────────────────────────────────────────────────────────────

JOB_TYPES = [
    {"angebotsart": 34, "label": "Werkstudent", "priority": 1},
    {"angebotsart": 4,  "label": "Praktikum",   "priority": 2},
    {"angebotsart": 1,  "label": "Full-time",   "priority": 3},
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
CUTOFF_DATE = (datetime.now() - timedelta(days=MAX_AGE_DAYS)).date()

# ── Eligibility words (same logic as eligibility_checker.py) ─────────────────
INELIGIBLE_WORDS = [
    "senior", "lead", "principal", "director", "head of",
    "leiter", "leitung", "chief", "teamlead", "teamleiter",
    "postdoc", "postdoctoral", "phd candidate",
]
AUSBILDUNG_WORDS = [
    "duales studium", "duale hochschule", "dhbw", "dual study",
    "ausbildung", "auszubildende", "azubi", "berufsausbildung",
]
STUDENT_SIGNALS = [
    "werkstudent", "praktikum", "internship", "junior",
    "student", "berufseinsteiger", "entry level",
]


def is_fresh(date_str):
    try:
        posted = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        return posted >= CUTOFF_DATE
    except Exception:
        return True


def quick_eligible(title):
    """Fast title-based check — returns False if clearly ineligible."""
    t = title.lower()
    for w in AUSBILDUNG_WORDS:
        if w in t:
            return False
    for w in INELIGIBLE_WORDS:
        if w in t:
            # Allow if student signal also present
            if not any(s in t for s in STUDENT_SIGNALS):
                return False
    return True


def classify_freshness(date_str):
    try:
        posted = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        age    = (date.today() - posted).days
        if age == 0: return "Today",     age
        if age == 1: return "Yesterday", age
        if age <= 3: return "Hot",       age
        if age <= 7: return "Fresh",     age
        return "Recent", age
    except Exception:
        return "Unknown", 0


def search_jobs(keyword, angebotsart, page=1, size=25):
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
        print(f"    Request error: {e}")
    return None


def parse_results(data, query, label, priority):
    jobs = []
    if not data or "stellenangebote" not in data:
        return jobs
    for job in data["stellenangebote"]:
        published = job.get("aktuelleVeroeffentlichungsdatum", "")
        if not is_fresh(published):
            continue
        jobs.append({
            "title":        job.get("titel", "N/A"),
            "company":      job.get("arbeitgeber", "N/A"),
            "location":     job.get("arbeitsort", {}).get("ort", "N/A"),
            "region":       job.get("arbeitsort", {}).get("region", "N/A"),
            "job_type":     label,
            "priority":     priority,
            "published":    published,
            "ref_number":   job.get("refnr", "N/A"),
            "link":         f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{job.get('refnr','')}",
            "search_query": query,
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "description":  "",
            "freshness":    "",
            "age_days":     0,
        })
    return jobs


def fetch_description(link):
    """Fetch full job description from Arbeitsagentur job page."""
    try:
        r = requests.get(link, headers=HEADERS_WEB, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            # Try to find the job description section
            for marker in ["Stellenbeschreibung", "Ihre Aufgaben", "Job Description", "Aufgaben"]:
                if marker in text:
                    idx  = text.find(marker)
                    return text[idx:idx + 3000].strip()
            return text[:3000].strip()
    except Exception as e:
        print(f"    Fetch error: {e}")
    return ""


def main():
    print("=" * 55)
    print("  AI Career Copilot — Job Discovery v3")
    print(f"  Stage 1: Scrape titles (7-day filter)")
    print(f"  Stage 2: Fetch JDs for eligible jobs only")
    print(f"  Cutoff: {CUTOFF_DATE.strftime('%d %b %Y')}")
    print("=" * 55)

    # ── STAGE 1: Fast scrape ──────────────────────────────────────────────────
    print("\n── Stage 1: Scraping job listings ──")
    all_jobs = []

    for job_type in JOB_TYPES:
        label       = job_type["label"]
        angebotsart = job_type["angebotsart"]
        priority    = job_type["priority"]
        print(f"\n  {label}:")

        for query in SEARCH_QUERIES:
            if len(all_jobs) >= MAX_JOBS_TOTAL:
                print(f"  Hit cap of {MAX_JOBS_TOTAL} — stopping")
                break
            print(f"    '{query}'...", end=" ", flush=True)
            data = search_jobs(query, angebotsart)
            if data:
                jobs = parse_results(data, query, label, priority)
                print(f"{len(jobs)} fresh")
                all_jobs.extend(jobs)
            else:
                print("no results")
            time.sleep(random.uniform(0.3, 0.7))

    # Deduplicate by ref_number
    seen, unique = set(), []
    for job in all_jobs:
        if job["ref_number"] not in seen:
            seen.add(job["ref_number"])
            unique.append(job)

    print(f"\n  Total unique fresh jobs: {len(unique)}")

    # Classify freshness
    for job in unique:
        label, age       = classify_freshness(job["published"])
        job["freshness"] = label
        job["age_days"]  = age

    # ── STAGE 2: Fetch JDs for eligible jobs only ─────────────────────────────
    print(f"\n── Stage 2: Fetching job descriptions ──")

    # Quick eligibility pre-filter on title
    eligible   = [j for j in unique if quick_eligible(j["title"])]
    ineligible = [j for j in unique if not quick_eligible(j["title"])]

    print(f"  Eligible (by title):   {len(eligible)}")
    print(f"  Ineligible (filtered): {len(ineligible)} — skipping JD fetch")

    # Sort eligible: Werkstudent first, then freshest
    eligible.sort(key=lambda j: (j["priority"], j["age_days"]))

    # Fetch JDs — cap at MAX_DESC_FETCH
    to_fetch = eligible[:MAX_DESC_FETCH]
    print(f"  Fetching JDs for top {len(to_fetch)} eligible jobs...\n")

    for i, job in enumerate(to_fetch):
        print(f"  [{i+1:3d}/{len(to_fetch)}] {job['title'][:50]:<50} @ {job['company'][:20]}", end=" ", flush=True)
        desc = fetch_description(job["link"])
        job["description"] = desc
        if desc:
            print(f"✓ {len(desc)} chars")
        else:
            print("✗ no desc")
        time.sleep(random.uniform(*DESC_DELAY))

    # Merge back — ineligible jobs get no description
    all_final = to_fetch + eligible[MAX_DESC_FETCH:] + ineligible

    # Sort and save
    df = pd.DataFrame(all_final)
    if not df.empty:
        df = df.sort_values(["priority", "age_days"], ascending=[True, True])
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    with_desc = len(df[df["description"].str.len() > 100]) if not df.empty else 0
    print(f"\n{'='*55}")
    print(f"Saved {len(df)} jobs to {OUTPUT_FILE}")
    print(f"Jobs with full JD: {with_desc}/{len(df)}")

    print(f"\nBreakdown by type:")
    for lbl in ["Werkstudent", "Praktikum", "Full-time"]:
        c = len(df[df["job_type"] == lbl]) if not df.empty else 0
        print(f"  {lbl:<12} {'█'*(c//2)} {c}")

    print(f"\nFreshness:")
    if not df.empty:
        for lbl in ["Today", "Yesterday", "Hot", "Fresh"]:
            c = len(df[df["freshness"] == lbl])
            if c: print(f"  {lbl:<12} {'█'*(c//2)} {c}")

    print(f"\nStage 2 complete — run rag_matcher.py next for scoring.")


if __name__ == "__main__":
    main()