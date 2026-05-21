"""
AI Career Copilot — RAG Matching Engine
-----------------------------------------
Real RAG implementation:
1. Embed CV sections using sentence-transformers
2. Embed job descriptions into FAISS vector index
3. Retrieve semantically relevant chunks
4. Send only relevant context to Groq — not raw full text
5. Score per user based on their uploaded CV
"""

import os
import json
import numpy as np
import faiss
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

# Load embedding model once
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, free, local
print("Model loaded.")


def chunk_text(text, chunk_size=80):
    """Split text into overlapping chunks for better retrieval."""
    words  = text.split()
    chunks = []
    step = chunk_size // 3  # 50% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(texts):
    """Embed a list of texts into vectors."""
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def build_cv_index(cv_text):
    """
    Build a FAISS index from the user's CV.
    Returns: index, chunks list
    """
    chunks     = chunk_text(cv_text)
    embeddings = embed_texts(chunks)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatIP(dim)  # Inner product = cosine sim on normalized vecs
    index.add(embeddings)
    return index, chunks


def retrieve_relevant_cv_sections(cv_index, cv_chunks, job_text, top_k=3):
    """
    Given a job description, retrieve the most relevant
    sections of the CV using semantic search.
    """
    job_embedding = embed_texts([job_text])
    scores, indices = cv_index.search(job_embedding, top_k)
    relevant = [cv_chunks[i] for i in indices[0] if i < len(cv_chunks)]
    return "\n".join(relevant)


def score_job_rag(cv_index, cv_chunks, full_cv, job, retries=3):
    """
    Score a job using RAG:
    1. Retrieve relevant CV sections for this job
    2. Send only relevant context to Groq
    3. Apply strict scoring rubric
    """
    # Use real job description if available, else fall back to title
    description = job.get('description', '')
    if description and len(description) > 100:
        job_text = description[:2000]  # Use real JD for RAG retrieval
    else:
        job_text = f"""
    Title: {job.get('title','')}
    Company: {job.get('company','')}
    Location: {job.get('location','')}
    Type: {job.get('job_type','')}
    """

    # RAG: retrieve relevant CV sections
    relevant_cv = retrieve_relevant_cv_sections(
        cv_index, cv_chunks, job_text, top_k=3
    )

    prompt = f"""You are a strict technical recruiter scoring a candidate for a job.
Be HONEST and SPECIFIC. Use the FULL 0-100 range. Do NOT default to 85.

MOST RELEVANT CV SECTIONS FOR THIS JOB (retrieved via semantic search):
{relevant_cv}

FULL CANDIDATE PROFILE SUMMARY:
- M.Sc. AI student at BTU Cottbus (semester 1)
- Published at AIST-2024 (Springer) — speech translation, NLP
- ML Intern at Fractal Analytics — BERT, 95% accuracy, 65% automation
- Projects: agentic AI, RAG, NLP, computer vision, speech AI
- Available 20hrs/week as Werkstudent, based in Cottbus Germany
- German level: A2 (beginner)
- English level: C1

JOB TO EVALUATE:
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Type: {job.get('job_type','Unknown')}
Description: {job.get('description','')[:800] if job.get('description') else 'Not available'}

STRICT SCORING RUBRIC:
90-100: Perfect. All skills match. Student/intern role. AI/ML domain. Publication relevant.
75-89:  Good. Most skills match. Minor gaps. Appropriate level.
60-74:  Partial. Some skills match. Missing key requirements.
40-59:  Weak. Few relevant skills. Wrong level or domain.
0-39:   Poor. Wrong domain, too senior, completely unrelated.

HARD PENALTIES (apply these):
- Job requires German C1/C2, candidate is A2: -25 points
- Job is sales, retail, legal, HR with no AI: -50 points
- Job requires 3+ years experience: -30 points
- Job is Ausbildung or Duales Studium: -60 points
- Job is completely unrelated to tech/AI: -60 points

BONUSES (apply these):
- Werkstudent or Praktikum role: +5
- Matches published research topic (NLP, speech, translation): +10
- Top German tech company (BMW, Bosch, Airbus, Zalando, SAP, Mercedes): +5
- Role matches specific project in CV (agentic AI, RAG, computer vision): +10

Respond ONLY with valid JSON:
{{
  "match_score": <integer 0-100>,
  "verdict": "<Apply Now if >=75 | Worth Considering if 50-74 | Skip if <50>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reason": "<2 specific sentences — mention actual skills or gaps, be direct>"
}}"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
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
                print(f"    Error: {err[:80]}")
                break

    return {"match_score":0,"verdict":"Error",
            "matched_skills":[],"missing_skills":[],
            "reason":"Could not evaluate."}


def run_rag_matching(cv_text, jobs_file="jobs.csv", output_file="matched_jobs.csv"):
    """
    Full RAG matching pipeline for a given CV.
    Can be called per-user with their uploaded CV.
    """
    print("Building CV vector index...")
    cv_index, cv_chunks = build_cv_index(cv_text)
    print(f"CV indexed: {len(cv_chunks)} chunks, {cv_index.ntotal} vectors")

    try:
        df = pd.read_csv(jobs_file)
    except FileNotFoundError:
        print(f"{jobs_file} not found.")
        return

    print(f"Scoring {len(df)} jobs using RAG...\n")
    results = []

    for i, row in df.iterrows():
        job = row.to_dict()
        print(f"[{i+1}/{len(df)}] {str(job.get('title',''))[:45]} @ {str(job.get('company',''))[:25]}")

        score = score_job_rag(cv_index, cv_chunks, cv_text, job)
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
        })

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"    Saved {len(results)} results")

        time.sleep(8)

    # Final save + sort
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(
        ["priority","match_score"], ascending=[True, False]
    )
    df_out.to_csv(output_file, index=False)

    # Score distribution
    valid = df_out[~df_out["verdict"].isin(["Error","API Error"])]
    print(f"\nScore distribution:")
    bins = [(90,100,"Excellent"),(75,89,"Good"),
            (60,74,"Partial"),(40,59,"Weak"),(0,39,"Poor")]
    for low, high, label in bins:
        count = len(valid[
            (valid["match_score"]>=low) & (valid["match_score"]<=high)
        ])
        print(f"  {label} ({low}-{high}): {'█'*(max(count//2,0))} {count}")

    print(f"\nApply Now:         {len(valid[valid.verdict=='Apply Now'])}")
    print(f"Worth Considering: {len(valid[valid.verdict=='Worth Considering'])}")
    print(f"Skip:              {len(valid[valid.verdict=='Skip'])}")

    return df_out


if __name__ == "__main__":
    # Test with local resume.txt
    with open("src/resume.txt") as f:
        cv = f.read()
    run_rag_matching(cv)
