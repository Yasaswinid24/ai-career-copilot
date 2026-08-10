"""
AI Career Copilot — Job Matching Engine v3

WHY THE RAG WAS REMOVED
-----------------------
v2 embedded the CV into FAISS and retrieved top_k=3 chunks per job. The CV
indexed to 15 chunks, so the model saw 20% of it and reasonably concluded
the rest was absent. The result was an inverted missing_skills field: it
listed CV lines that retrieval had hidden, labelled as gaps.

Proof from a real run — every "missing" item was a verbatim line from the
CV's own SKILLS section:
    "FAISS vector search, structured prompting"                  <- one CV line
    "HuBERT, HiFi-GAN, OpenCV, ADAS-applicable computer vision"  <- another
    "CNN, LSTM, Transformer architectures"                       <- another
One job reported FAISS as missing while the next reported it as matched,
from the same CV. Another claimed the candidate lacked "Transformer
architectures" while listing BERT and XLM-RoBERTa as matched.

Retrieval exists to fit a corpus that exceeds the context window. A ~480
word CV does not. Sending it whole is both simpler and correct.

If you want RAG back for the CV line on your resume, put it on the JOB side
instead: descriptions now run 3-5k chars, so retrieving the requirements
section out of a long ad is a real retrieval problem. That is a better
answer in an interview than retrieval over 15 chunks.

OTHER CHANGES
-------------
  * missing_skills is now explicitly defined in the prompt as "requirements
    stated IN THE JOB AD that the CV does not evidence". v2 never defined
    it, which is what let the model invert it.
  * The model must quote the job ad for each gap, which suppresses
    inventions like "CNN and LSTM are crucial for 3D metal printing".
  * Drops sentence-transformers, faiss and torch — no model download, no
    30s startup.
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

JOBS_FILE = "jobs_eligible.csv"
OUTPUT_FILE = "matched_jobs.csv"
RESUME_FILE = "src/resume.txt"
REQUEST_DELAY = 2.0
JD_CHARS = 3000          # descriptions are full now; v2 truncated at 800


def known_facts(job):
    """Facts eligibility_checker already derived from the text. Stating them
    is cheaper and more consistent than asking an 8B model to re-derive."""
    facts = []
    reason = str(job.get("reason", "") or "")
    if job.get("eligibility"):
        facts.append(f"- Eligibility screen: {job['eligibility']} ({reason})")
    if job.get("job_type"):
        facts.append(f"- Contract type: {job['job_type']}")
    m = re.search(r"(\d+)\+? years", reason)
    if m:
        facts.append(f"- Employer requires {m.group(1)}+ years experience")
    lvl = re.search(r"\b(deutsch\w*|german)\b[^.]{0,60}?\b([ABC][12])\b",
                    str(job.get("description", "") or ""), re.IGNORECASE)
    if lvl:
        facts.append(f"- German level named in the ad: {lvl.group(2).upper()}")
    if job.get("homeoffice"):
        facts.append("- Home office is possible")
    return "\n".join(facts) or "- none extracted"


def _norm(t):
    return re.sub(r"[^a-z0-9+#]", "", t.lower())


def verify_gaps(result, cv_text):
    """Move any 'missing' skill that literally appears in the CV over to matched.

    The LLM judges fit; whether a string occurs in a document is not a
    judgement call and should not be delegated to one. Even the 70B reported
    SQL and Power BI as missing against a CV containing SQL seven times and
    Power BI three times. This guard makes that class of error impossible
    rather than merely less likely.
    """
    cv_norm = _norm(cv_text)
    missing, recovered = [], []
    for item in result.get("missing_skills") or []:
        item = str(item).strip()
        if not item:
            continue
        key = _norm(item)
        # Require 3+ chars so "C" or "R" don't match inside unrelated words.
        if len(key) >= 3 and key in cv_norm:
            recovered.append(item)
        else:
            missing.append(item)
    if recovered:
        result["missing_skills"] = missing
        matched = result.get("matched_skills") or []
        result["matched_skills"] = matched + [r for r in recovered
                                              if r not in matched]
        result["_recovered"] = recovered
    return result


def score_job(cv_text, job, retries=3):
    description = str(job.get("description", "") or "")

    prompt = f"""You are a strict technical recruiter scoring a candidate for a job.
Be HONEST and SPECIFIC. Use the FULL 0-100 range. Do NOT default to 85.

=== CANDIDATE CV (complete) ===
{cv_text}

=== ALREADY VERIFIED ABOUT THIS POSTING (treat as fact) ===
{known_facts(job)}

=== JOB POSTING ===
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Contract type: {job.get('job_type','Unknown')}

Description:
{description[:JD_CHARS] if description else 'Not available'}

=== HOW TO FILL EACH FIELD ===

matched_skills: skills that appear in BOTH the job posting and the CV.
  If the posting does not name a skill, it does not belong here, however
  impressive that skill is.

missing_skills: requirements STATED IN THE JOB POSTING ABOVE that the CV
  does not evidence.
  - Read requirements only from the job posting text. Never from the CV.
  - A CV skill the posting does not ask for is NOT missing. It is irrelevant
    to this job. Leave it out entirely.
  - ATOMIC ITEMS ONLY: 1-4 words each, one skill per item. Never copy a whole
    requirement sentence or bullet. If the ad says "Gute Kenntnisse in SQL,
    Power BI, Power Platform und GenAI-Technologien", that is FOUR separate
    items, and you must check EACH ONE against the CV separately. Copying the
    bullet whole is how a skill the candidate has gets reported as missing.
  - Before listing any item, search the CV's SKILLS section for it. If it is
    there in any form, it is matched, not missing.
  - Do not treat a specific technology as missing when the CV shows an
    example of it (BERT and XLM-RoBERTa ARE transformer architectures).
  - German and English name the same tools: SQL = SQL, Datenanalyse = data
    analysis, Kenntnisse in X = experience with X.
  - An empty list is a valid and common answer.

reason: two sentences. Name a concrete skill overlap and a concrete gap,
  each traceable to the posting text. Do not invent requirements.

SCORING RUBRIC:
90-100: Nearly every stated requirement met. Core domain of the CV.
75-89:  Most requirements met, gaps are nice-to-haves only.
60-74:  Real overlap but one or more stated requirements unmet.
40-59:  Few relevant skills, or wrong level, or adjacent domain.
0-39:   Wrong domain or a disqualifying requirement.

HARD CAPS — apply these BEFORE the rubric, they override it:
  - The ad requires a degree in a field the CV does not have (accounting,
    finance, law, medicine, mechanical engineering, business administration):
    MAXIMUM 35. Programming skill does not substitute for a required degree.
  - The role's core function is not software, data, ML or AI (accounting,
    audit, sales, HR, procurement, logistics, marketing): MAXIMUM 30, even
    if the ad mentions Excel, SQL or "analytics".
  - The ad requires German above the level stated in the CV: subtract 25.
  - A named tool or framework central to the role is absent from the CV:
    MAXIMUM 74. It cannot be "minor gaps" if a core requirement is unmet.

CALIBRATION: most postings are NOT a strong fit. If you find yourself giving
80+ to most jobs, you are being too generous. Reserve 90+ for a posting whose
requirements the CV meets almost line for line. A score of 82 for a job in an
unrelated profession is wrong no matter how transferable Python looks.

Weigh the verified facts above; do not re-derive them.

Respond ONLY with valid JSON:
{{
  "match_score": <integer 0-100>,
  "verdict": "<Apply Now | Worth Considering | Skip>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["requirement from the ad the CV lacks"],
  "reason": "<2 specific sentences>"
}}"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0.1)
            raw = r.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            score = int(result.get("match_score", 0))
            result["match_score"] = score
            result["verdict"] = ("Apply Now" if score >= 75
                                 else "Worth Considering" if score >= 50
                                 else "Skip")
            return verify_gaps(result, cv_text)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {err[:80]}")
                break

    return {"match_score": 0, "verdict": "Error", "matched_skills": [],
            "missing_skills": [], "reason": "Could not evaluate."}


def load_already_scored(output_file, rescore):
    if rescore or not os.path.exists(output_file):
        return set(), []
    try:
        prev = pd.read_csv(output_file, encoding="utf-8")
    except Exception:
        return set(), []
    if "ref_number" not in prev.columns:
        print("  existing matched_jobs.csv predates ref_number — "
              "rescoring everything once")
        return set(), []
    keep = prev[prev["verdict"] != "Error"]
    return set(keep["ref_number"].dropna().astype(str)), keep.to_dict("records")


def run_matching(cv_text, jobs_file=JOBS_FILE, output_file=OUTPUT_FILE,
                 rescore=False, limit=None, eligible_only=False):
    words = len(cv_text.split())
    print(f"CV loaded: {words} words, {len(cv_text)} chars — sent in full")
    if words < 150:
        print("  WARNING: that is very short for a CV. Check src/resume.txt "
              "is the current version — every score depends on it.")

    try:
        df = pd.read_csv(jobs_file, encoding="utf-8")
    except FileNotFoundError:
        print(f"{jobs_file} not found. Run eligibility_checker.py first.")
        return

    if eligible_only and "eligibility" in df.columns:
        before = len(df)
        df = df[df["eligibility"] == "eligible"]
        print(f"--eligible-only: {before} -> {len(df)} jobs")

    done, results = load_already_scored(output_file, rescore)
    todo = (df[~df["ref_number"].astype(str).isin(done)]
            if "ref_number" in df.columns else df)
    if limit:
        todo = todo.head(limit)

    print(f"\n{len(df)} jobs, {len(done)} already scored, {len(todo)} to score")
    if todo.empty:
        print("Nothing new. Use --rescore after updating your CV.")
        return pd.DataFrame(results)
    print(f"Estimated time: {len(todo) * REQUEST_DELAY / 60:.0f} min\n")
    recovered_total = 0

    for n, (_, row) in enumerate(todo.iterrows(), 1):
        job = row.to_dict()
        print(f"[{n}/{len(todo)}] {str(job.get('title',''))[:45]} "
              f"@ {str(job.get('company',''))[:25]}")
        s = score_job(cv_text, job)
        if s.get("_recovered"):
            print(f"    corrected false gaps: {', '.join(s['_recovered'])}")
            recovered_total += len(s["_recovered"])
        results.append({
            "ref_number": job.get("ref_number"),
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "job_type": job.get("job_type", ""),
            "priority": job.get("priority", 3),
            "eligibility": job.get("eligibility", ""),
            "published": job.get("published", ""),
            "match_score": s.get("match_score", 0),
            "verdict": s.get("verdict"),
            "matched_skills": ", ".join(s.get("matched_skills", [])),
            "missing_skills": ", ".join(s.get("missing_skills", [])),
            "reason": s.get("reason"),
            "link": job.get("link"),
            "scraped_at": job.get("scraped_at"),
        })
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False,
                                         encoding="utf-8")
            print(f"    checkpoint — {len(results)} saved")
        time.sleep(REQUEST_DELAY)

    out = pd.DataFrame(results).sort_values(["priority", "match_score"],
                                            ascending=[True, False])
    out.to_csv(output_file, index=False, encoding="utf-8")

    valid = out[out["verdict"] != "Error"]
    print("\nScore distribution:")
    for low, high, label in [(90, 100, "Excellent"), (75, 89, "Good"),
                             (60, 74, "Partial"), (40, 59, "Weak"),
                             (0, 39, "Poor")]:
        c = len(valid[(valid.match_score >= low) & (valid.match_score <= high)])
        print(f"  {label:<10} ({low:>2}-{high:>3}): {'#' * (c // 2)} {c}")
    for v in ("Apply Now", "Worth Considering", "Skip"):
        print(f"{v:<18} {len(valid[valid.verdict == v])}")

    # Sanity check on the field that was broken in v2.
    empty = (out["missing_skills"].fillna("") == "").sum()
    print(f"\nPostings with no gaps listed: {empty}/{len(out)}")
    if recovered_total:
        print(f"  False gaps corrected in code: {recovered_total} "
              f"(skills the model called missing that are in the CV)")
    # Long items mean whole requirement bullets were copied instead of being
    # split into skills — that is what makes a CV skill look missing.
    longest = max((len(i.strip()) for v in out["missing_skills"].fillna("")
                   for i in str(v).split(",")), default=0)
    if longest > 45:
        print(f"  WARNING: longest gap item is {longest} chars — the model is "
              f"copying requirement sentences, not atomic skills")
    if len(valid) >= 5:
        spread = valid.match_score.max() - valid.match_score.min()
        print(f"  Score spread: {spread} points "
              f"({'too narrow — model is anchoring' if spread < 25 else 'ok'})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true",
                    help="rescore everything (use after updating your CV)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", default=JOBS_FILE)
    ap.add_argument("--eligible-only", action="store_true",
                    help="skip borderline full-time roles")
    args = ap.parse_args()

    with open(RESUME_FILE, encoding="utf-8") as f:
        cv = f.read()
    run_matching(cv, jobs_file=args.jobs, rescore=args.rescore,
                 limit=args.limit, eligible_only=args.eligible_only)