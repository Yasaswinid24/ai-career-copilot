"""
AI Career Copilot — Resume Matching Agent
"""
import os
import pandas as pd
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "jobs.csv"
OUTPUT_FILE = "matched_jobs.csv"
MODEL       = "llama-3.1-8b-instant"

def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()

def score_job(resume, job, retries=3):
    prompt = f"""You are a technical recruiter evaluating job fit.

CANDIDATE RESUME:
{resume}

JOB:
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Location: {job.get('location', 'N/A')}

Respond ONLY with valid JSON, no extra text:
{{
  "match_score": <integer 0-100>,
  "verdict": "<Apply Now | Worth Considering | Skip>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reason": "<2 sentence explanation>"
}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {err[:80]}")
                break

    return {
        "match_score": 0,
        "verdict": "Error",
        "matched_skills": [],
        "missing_skills": [],
        "reason": "Could not evaluate."
    }

def main():
    print("=" * 55)
    print("  AI Career Copilot — Resume Matching Agent")
    print("=" * 55)

    resume = load_resume()
    print(f"\nResume loaded: {len(resume)} characters")

    try:
        df = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("jobs.csv not found. Run scraper.py first.")
        return

    # Skip already-matched jobs if output exists
    already_done = set()
    try:
        existing = pd.read_csv(OUTPUT_FILE)
        already_done = set(zip(existing["title"], existing["company"]))
        print(f"Resuming — {len(already_done)} already matched, {len(df)-len(already_done)} remaining")
    except FileNotFoundError:
        existing = pd.DataFrame()

    print(f"Total jobs: {len(df)}\n")
    results = []

    for i, row in df.iterrows():
        job = row.to_dict()
        key = (job.get("title"), job.get("company"))

        if key in already_done:
            continue

        title_short   = str(job.get("title",""))[:45]
        company_short = str(job.get("company",""))[:25]
        print(f"[{i+1}/{len(df)}] {title_short} @ {company_short}")

        score = score_job(resume, job)

        results.append({
            "title":          job.get("title"),
            "company":        job.get("company"),
            "location":       job.get("location"),
            "match_score":    score.get("match_score", 0),
            "verdict":        score.get("verdict"),
            "matched_skills": ", ".join(score.get("matched_skills", [])),
            "missing_skills": ", ".join(score.get("missing_skills", [])),
            "reason":         score.get("reason"),
            "link":           job.get("link"),
            "scraped_at":     job.get("scraped_at"),
        })

        # Save every 10 jobs so progress is never lost
        if len(results) % 10 == 0:
            batch = pd.DataFrame(results)
            combined = pd.concat([existing, batch], ignore_index=True)
            combined.to_csv(OUTPUT_FILE, index=False)
            print(f"    Progress saved — {len(results)} matched so far")

        # Polite delay — avoids rate limits
        time.sleep(4)

    # Final save
    if results:
        batch = pd.DataFrame(results)
        combined = pd.concat([existing, batch], ignore_index=True)
        combined = combined.sort_values("match_score", ascending=False)
        combined.to_csv(OUTPUT_FILE, index=False)

    # Load final results for summary
    try:
        final = pd.read_csv(OUTPUT_FILE)
        final = final[final["verdict"] != "Error"]
        final = final.sort_values("match_score", ascending=False)

        print(f"\n{'='*55}")
        print(f"COMPLETE — {len(final)} jobs evaluated")
        print(f"\nTop 10 matches:\n")
        top = final.head(10)[["title","company","match_score","verdict"]]
        print(top.to_string(index=False))

        print(f"\nApply Now:         {len(final[final.verdict=='Apply Now'])}")
        print(f"Worth Considering: {len(final[final.verdict=='Worth Considering'])}")
        print(f"Skip:              {len(final[final.verdict=='Skip'])}")
    except Exception as e:
        print(f"Summary error: {e}")

if __name__ == "__main__":
    main()
