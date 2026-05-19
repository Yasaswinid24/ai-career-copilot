"""
AI Career Copilot — Smart Matcher
-----------------------------------
Only scores NEW jobs not already in matched_jobs.csv
Saves time — instead of 2 hours, takes 5-10 minutes daily.
"""

import os
import pandas as pd
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

RESUME_FILE  = "src/resume.txt"
JOBS_FILE    = "jobs.csv"
OUTPUT_FILE  = "matched_jobs.csv"


def load_resume():
    with open(RESUME_FILE, "r") as f:
        return f.read()


def score_job(resume, job, retries=3):
    prompt = f"""You are a technical recruiter evaluating job fit for a Masters student.

CANDIDATE:
{resume}

JOB:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Type: {job.get('job_type')}

Respond ONLY with valid JSON:
{{
  "match_score": <0-100>,
  "verdict": "<Apply Now | Worth Considering | Skip>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reason": "<2 sentence explanation>"
}}"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )
            raw = r.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            return json.loads(raw)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                break

    return {"match_score": 0, "verdict": "Error",
            "matched_skills": [], "missing_skills": [],
            "reason": "Could not evaluate."}


def main():
    print("=" * 55)
    print("  AI Career Copilot — Smart Matcher")
    print("  Only scores NEW jobs — saves time daily")
    print("=" * 55)

    resume = load_resume()

    # Load all jobs
    try:
        all_jobs = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("jobs.csv not found. Run scraper.py first.")
        return

    # Load already scored jobs
    try:
        scored_df  = pd.read_csv(OUTPUT_FILE)
        scored_keys = set(zip(
            scored_df["title"].str.strip(),
            scored_df["company"].str.strip()
        ))
        print(f"\nAlready scored: {len(scored_keys)} jobs")
    except FileNotFoundError:
        scored_df   = pd.DataFrame()
        scored_keys = set()

    # Find only new jobs
    new_jobs = all_jobs[~all_jobs.apply(
        lambda r: (str(r["title"]).strip(),
                   str(r["company"]).strip()) in scored_keys,
        axis=1
    )].reset_index(drop=True)

    print(f"Total jobs:     {len(all_jobs)}")
    print(f"New to score:   {len(new_jobs)}")

    if new_jobs.empty:
        print("\nAll jobs already scored — nothing to do!")
        print("Run scraper.py first to get fresh jobs.")
        return

    estimated = len(new_jobs) * 5.5 / 60
    print(f"Estimated time: {estimated:.1f} minutes\n")

    results = []

    for i, row in new_jobs.iterrows():
        job = row.to_dict()
        title   = str(job.get("title",""))[:45]
        company = str(job.get("company",""))[:25]
        jtype   = job.get("job_type","")

        print(f"[{i+1}/{len(new_jobs)}] [{jtype[:2]}] {title} @ {company}")

        score = score_job(resume, job)
        results.append({
            "title":          job.get("title"),
            "company":        job.get("company"),
            "location":       job.get("location"),
            "job_type":       job.get("job_type"),
            "priority":       job.get("priority"),
            "published":      job.get("published"),
            "match_score":    score.get("match_score", 0),
            "verdict":        score.get("verdict"),
            "matched_skills": ", ".join(score.get("matched_skills", [])),
            "missing_skills": ", ".join(score.get("missing_skills", [])),
            "reason":         score.get("reason"),
            "link":           job.get("link"),
            "scraped_at":     job.get("scraped_at"),
        })

        # Save every 10 jobs
        if len(results) % 10 == 0:
            batch    = pd.DataFrame(results)
            combined = pd.concat([scored_df, batch], ignore_index=True)
            combined.to_csv(OUTPUT_FILE, index=False)
            print(f"    Saved — {len(results)} new jobs scored so far")

        time.sleep(4)

    # Final save
    if results:
        batch    = pd.DataFrame(results)
        combined = pd.concat([scored_df, batch], ignore_index=True)
        combined = combined.sort_values(
            ["priority","match_score"],
            ascending=[True, False]
        )
        combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    try:
        final = pd.read_csv(OUTPUT_FILE)
        final = final[~final["verdict"].isin(["Error","API Error"])]
        ws    = final[final["job_type"] == "Werkstudent"]
        pk    = final[final["job_type"] == "Praktikum"]

        print(f"\n{'='*55}")
        print(f"COMPLETE")
        print(f"{'='*55}")
        print(f"\nTop Werkstudent matches:")
        print(ws.sort_values("match_score", ascending=False)
                .head(5)[["title","company","match_score","verdict"]]
                .to_string(index=False))
        print(f"\nTop Praktikum matches:")
        print(pk.sort_values("match_score", ascending=False)
                .head(5)[["title","company","match_score","verdict"]]
                .to_string(index=False))
        print(f"\nApply Now:         {len(final[final.verdict=='Apply Now'])}")
        print(f"Worth Considering: {len(final[final.verdict=='Worth Considering'])}")
    except Exception as e:
        print(f"Summary error: {e}")


if __name__ == "__main__":
    main()
