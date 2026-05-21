"""
AI Career Copilot — Resume Matching Agent v2
---------------------------------------------
Uses strict scoring rubric to properly differentiate jobs.
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

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "jobs.csv"
OUTPUT_FILE = "matched_jobs.csv"


def load_resume():
    with open(RESUME_FILE) as f:
        return f.read()


def score_job(resume, job, retries=3):
    prompt = f"""You are a strict technical recruiter scoring a candidate for a job.
Be HONEST and SPECIFIC. Do NOT give everyone 85. Differentiate clearly.

CANDIDATE PROFILE:
{resume}

JOB:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Type: {job.get('job_type','Unknown')}

STRICT SCORING RUBRIC — follow exactly:

90-100: Perfect match. All required skills present. Student/intern role. AI/ML domain. Published research is a bonus.
75-89:  Good match. Most skills present. Minor gaps. Role is appropriate level.
60-74:  Partial match. Some relevant skills but missing key requirements. Role level borderline.
40-59:  Weak match. Few relevant skills. Role may be too senior or wrong domain.
0-39:   Poor match. Wrong domain, too senior, or completely different field.

PENALISE HARD FOR:
- Job requires German C1/C2 but candidate is A2 (-20 points)
- Job is clearly sales, legal, finance, HR with no AI component (-40 points)
- Job requires 3+ years experience for a student (-30 points)
- Job is completely unrelated to AI/ML/data (-50 points)

REWARD FOR:
- Werkstudent or Praktikum role explicitly (+5 points)
- Role matches candidate's published research topic (+10 points)
- Company is top German tech firm (BMW, Bosch, Airbus, Zalando, SAP) (+5 points)
- Role matches specific projects in candidate's CV (+10 points)

Respond ONLY with valid JSON:
{{
  "match_score": <integer 0-100 — be strict, use full range>,
  "verdict": "<Apply Now if score>=75 | Worth Considering if 50-74 | Skip if <50>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reason": "<2 specific sentences explaining the score — mention actual skills or gaps>"
}}"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
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
                print(f"    Error: {err[:80]}")
                break

    return {"match_score":0,"verdict":"Error",
            "matched_skills":[],"missing_skills":[],
            "reason":"Could not evaluate."}


def main():
    print("=" * 55)
    print("  AI Career Copilot — Resume Matcher v2")
    print("  Strict scoring — full 0-100 range used")
    print("=" * 55)

    resume = load_resume()
    print(f"\nResume loaded: {len(resume)} chars")

    try:
        df = pd.read_csv(JOBS_FILE)
    except FileNotFoundError:
        print("jobs.csv not found.")
        return

    # Skip already matched
    already_done = set()
    try:
        existing = pd.read_csv(OUTPUT_FILE)
        existing = existing[existing["verdict"] != "Error"]
        already_done = set(zip(existing["title"], existing["company"]))
        print(f"Resuming — {len(already_done)} already matched")
    except FileNotFoundError:
        existing = pd.DataFrame()

    print(f"Total jobs: {len(df)}")
    print(f"To score: {len(df) - len(already_done)}\n")

    results = []

    for i, row in df.iterrows():
        job = row.to_dict()
        key = (job.get("title"), job.get("company"))

        if key in already_done:
            continue

        print(f"[{i+1}/{len(df)}] {str(job.get('title',''))[:45]} @ {str(job.get('company',''))[:25]}")

        score = score_job(resume, job)
        results.append({
            "title":          job.get("title"),
            "company":        job.get("company"),
            "location":       job.get("location"),
            "job_type":       job.get("job_type",""),
            "priority":       job.get("priority", 3),
            "freshness":      job.get("freshness",""),
            "published":      job.get("published",""),
            "match_score":    score.get("match_score", 0),
            "verdict":        score.get("verdict"),
            "matched_skills": ", ".join(score.get("matched_skills", [])),
            "missing_skills": ", ".join(score.get("missing_skills", [])),
            "reason":         score.get("reason"),
            "link":           job.get("link"),
            "scraped_at":     job.get("scraped_at"),
            "published":      job.get("published",""),
        })

        if len(results) % 10 == 0:
            batch = pd.DataFrame(results)
            combined = pd.concat([existing, batch], ignore_index=True)
            combined.to_csv(OUTPUT_FILE, index=False)
            print(f"    Saved {len(results)} so far")

        time.sleep(4)

    # Final save
    if results:
        batch    = pd.DataFrame(results)
        combined = pd.concat([existing, batch], ignore_index=True)
        combined = combined.sort_values(
            ["priority","match_score"], ascending=[True, False]
        )
        combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    try:
        final = pd.read_csv(OUTPUT_FILE)
        final = final[~final["verdict"].isin(["Error","API Error"])]

        print(f"\n{'='*55}")
        print(f"Score distribution:")
        print(final["match_score"].describe().to_string())

        print(f"\nScore breakdown:")
        bins = [(90,100,"Excellent"),(75,89,"Good"),
                (60,74,"Partial"),(40,59,"Weak"),(0,39,"Poor")]
        for low, high, label in bins:
            count = len(final[
                (final["match_score"]>=low) &
                (final["match_score"]<=high)
            ])
            print(f"  {label} ({low}-{high}): {'█'*(count//2)} {count}")

        print(f"\nTop 10 matches:")
        top = final.head(10)[["title","company","match_score","verdict","job_type"]]
        print(top.to_string(index=False))

        print(f"\nApply Now:         {len(final[final.verdict=='Apply Now'])}")
        print(f"Worth Considering: {len(final[final.verdict=='Worth Considering'])}")
        print(f"Skip:              {len(final[final.verdict=='Skip'])}")
    except Exception as e:
        print(f"Summary error: {e}")


if __name__ == "__main__":
    main()
