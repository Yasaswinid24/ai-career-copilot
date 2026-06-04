"""
AI Career Copilot — RAG Matcher v3
------------------------------------
- Upgraded to llama-3.3-70b-versatile
- Stronger scoring prompt
- Better JSON error handling
"""

import os
import json
import time
import pandas as pd
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

RESUME_FILE = "src/resume.txt"
JOBS_FILE   = "jobs.csv"
OUTPUT_FILE = "matched_jobs.csv"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def embed_texts(texts):
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings.astype("float32")


def build_cv_index(cv_text):
    chunk_size = 300
    words  = cv_text.split()
    chunks = [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def retrieve_relevant_cv_sections(cv_index, cv_chunks, job_text, top_k=3):
    job_embedding = embed_texts([job_text])
    scores, indices = cv_index.search(job_embedding, top_k)
    relevant = [cv_chunks[i] for i in indices[0] if i < len(cv_chunks)]
    return "\n".join(relevant)


def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        lines = raw.split("\n")
        for i in range(len(lines)-1, 0, -1):
            attempt = "\n".join(lines[:i]) + "\n}"
            try:
                json.loads(attempt)
                return attempt
            except:
                continue
    return raw


def score_job_rag(cv_index, cv_chunks, full_cv, job, retries=3):
    description = job.get("description", "")
    if description and len(description) > 100:
        job_text = description[:2000]
    else:
        job_text = f"""
    Title: {job.get("title","")}
    Company: {job.get("company","")}
    Location: {job.get("location","")}
    Type: {job.get("job_type","")}
    """

    relevant_cv = retrieve_relevant_cv_sections(
        cv_index, cv_chunks, job_text, top_k=3
    )

    prompt = f"""You are a strict technical recruiter scoring a candidate for a job.
Use the FULL 0-100 range. Be HONEST. Do NOT give everyone 75-85.

MOST RELEVANT CV SECTIONS (retrieved via semantic search):
{relevant_cv}

CANDIDATE SUMMARY:
- M.Sc. AI student at BTU Cottbus (semester 2, started April 2025)
- Published at AIST-2024 Springer: speech translation, NLP
- Software Engineer at Availity: GPT-4, LangChain, clinical AI (production)
- ML Intern at Fractal Analytics: BERT, 95% accuracy, 65% automation
- Projects: LangGraph multi-agent, RAG/FAISS, FastAPI, Docker
- Available 20hrs/week Werkstudent OR 6-month full-time Praktikum
- German: A2 (beginner) | English: C1 (fluent)

JOB TO EVALUATE:
Title: {job.get("title")}
Company: {job.get("company")}
Location: {job.get("location")}
Type: {job.get("job_type","Unknown")}
Description: {job.get("description","")[:1000] if job.get("description") else "Not available"}

SCORING RUBRIC:
90-100: Perfect. All skills match. Student/intern role. LLM/agentic/NLP domain.
75-89:  Good. Most skills match. Minor gaps. Appropriate level.
60-74:  Partial. Some skills match but missing key requirements.
40-59:  Weak. Few relevant skills. Wrong domain or level.
0-39:   Poor. Completely wrong domain or too senior.

HARD PENALTIES:
- German C1/C2 required, candidate is A2:          -35 points
- Role is sales, retail, marketing, legal, HR:     -60 points
- Role requires PhD or 3+ years experience:        -40 points
- Role is Ausbildung/dual study:                   -70 points
- Role is pure C++, embedded, hardware, robotics:  -35 points
- Role is financial risk (VaR, scorecard, COBOL):  -40 points
- Role is senior/lead/manager/architect:           -40 points
- Role is completely unrelated to AI/ML/data:      -60 points

BONUSES:
- Werkstudent or Praktikum explicitly:             +5
- Matches LangChain/LangGraph/RAG/agentic:         +15
- Matches NLP/speech/LLM research:                +10
- English-only or international team:              +5
- Published research matches role topic:           +10

IMPORTANT: Most jobs should score 40-70. Reserve 75+ for genuine matches.
Airbus Space Robotics, TIMOCOM, Lidl, BSH should score below 50.
CHECK24 Data Science, Bayer Agentic AI should score above 75.

Respond ONLY with valid JSON:
{{
  "match_score": <integer 0-100>,
  "verdict": "<Apply Now if >=75 | Worth Considering if 50-74 | Skip if <50>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reason": "<2 specific sentences mentioning actual skills or gaps>"
}}"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            raw = r.choices[0].message.content.strip()
            raw = clean_json(raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    JSON error (attempt {attempt+1}): {str(e)[:60]}")
            if attempt < retries - 1:
                time.sleep(3)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 20 * (attempt + 1)
                print(f"    Rate limited -- waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {err[:80]}")
                break

    return {
        "match_score": 0, "verdict": "Error",
        "matched_skills": [], "missing_skills": [],
        "reason": "Could not evaluate."
    }


def run_rag_matching(cv_text, jobs_file=JOBS_FILE, output_file=OUTPUT_FILE):
    print("Building CV vector index...")
    cv_index, cv_chunks = build_cv_index(cv_text)
    print(f"CV indexed: {len(cv_chunks)} chunks, {cv_index.ntotal} vectors")

    try:
        df = pd.read_csv(jobs_file)
    except FileNotFoundError:
        print(f"{jobs_file} not found.")
        return

    already_done = set()
    try:
        existing = pd.read_csv(output_file)
        existing = existing[~existing["verdict"].isin(["Error", "API Error"])]
        already_done = set(zip(existing["title"], existing["company"]))
        print(f"Resuming -- {len(already_done)} already matched")
    except FileNotFoundError:
        existing = pd.DataFrame()

    to_score = df[~df.apply(
        lambda r: (r.get("title"), r.get("company")) in already_done, axis=1
    )]

    print(f"Total jobs: {len(df)} | To score: {len(to_score)}\n")

    results = []

    for i, (_, row) in enumerate(to_score.iterrows()):
        job = row.to_dict()
        print(f"[{i+1}/{len(to_score)}] {str(job.get('title',''))[:45]} @ {str(job.get('company',''))[:25]}")

        score = score_job_rag(cv_index, cv_chunks, cv_text, job)
        results.append({
            "title":          job.get("title"),
            "company":        job.get("company"),
            "location":       job.get("location"),
            "job_type":       job.get("job_type", ""),
            "priority":       job.get("priority", 3),
            "llm_score":      job.get("llm_score", 0),
            "published":      job.get("published", ""),
            "match_score":    score.get("match_score", 0),
            "verdict":        score.get("verdict"),
            "matched_skills": ", ".join(score.get("matched_skills", [])),
            "missing_skills": ", ".join(score.get("missing_skills", [])),
            "reason":         score.get("reason"),
            "link":           job.get("link"),
            "scraped_at":     job.get("scraped_at"),
        })

        if len(results) % 10 == 0:
            batch    = pd.DataFrame(results)
            combined = pd.concat([existing, batch], ignore_index=True)
            combined.to_csv(output_file, index=False)
            print(f"    Saved {len(results)} results")

        time.sleep(5)

    if results:
        batch    = pd.DataFrame(results)
        combined = pd.concat([existing, batch], ignore_index=True)
        combined = combined.sort_values(
            ["match_score", "llm_score"], ascending=[False, False]
        )
        combined.to_csv(output_file, index=False)

    try:
        final = pd.read_csv(output_file)
        valid = final[~final["verdict"].isin(["Error", "API Error"])]

        print(f"\n{'='*60}")
        bins = [(90,100,"Excellent"),(75,89,"Good"),
                (60,74,"Partial"),(40,59,"Weak"),(0,39,"Poor")]
        for low, high, label in bins:
            count = len(valid[(valid["match_score"]>=low)&(valid["match_score"]<=high)])
            print(f"  {label:<12} ({low}-{high}): {'#'*(max(count//2,0))} {count}")

        print(f"\nApply Now:         {len(valid[valid.verdict=='Apply Now'])}")
        print(f"Worth Considering: {len(valid[valid.verdict=='Worth Considering'])}")
        print(f"Skip:              {len(valid[valid.verdict=='Skip'])}")

        print(f"\nTop 10 matches:")
        top = valid.head(10)[["title","company","match_score","verdict"]]
        print(top.to_string(index=False))

    except Exception as e:
        print(f"Summary error: {e}")


if __name__ == "__main__":
    with open(RESUME_FILE) as f:
        cv = f.read()
    run_rag_matching(cv)
