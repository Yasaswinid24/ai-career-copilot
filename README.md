
# AI Career Copilot

A job-search assistant for the German market. It scrapes fresh postings, filters out the ones you are not eligible for, scores the rest against your CV with an LLM, and then helps with the application: ATS keyword check, cover letter, recruiter contact, email draft, and follow-up tracking. Everything is available from a web dashboard with Google sign-in.

**Stack:** Python · FastAPI · Groq (Llama 3.1 8B Instant) · pandas · SQLite · Google OAuth · Hunter.io · Render

## What it does

| Stage | What happens | File |
|---|---|---|
| **Discover** | Pulls postings (Werkstudent, internship, thesis and entry-level searches) from the Bundesagentur für Arbeit job search, in two stages: a fast listing scrape, then the full job description only for postings that pass the filter. | `src/scraper.py` |
| **Freshness** | Labels each posting Hot (up to 3 days), Fresh (up to 7), Recent (up to 14) or Stale, so you apply to new postings first. | `src/freshness_check.py` |
| **Eligibility** | Marks each posting eligible, borderline or ineligible from title and description: seniority, years of experience required, German level named in the ad, contract type. Never deletes rows. | `src/eligibility_checker.py` |
| **Match** | Scores every eligible posting 0 to 100 against your CV and lists matched and missing skills, each missing skill backed by a quote from the job ad. | `src/rag_matcher.py` |
| **ATS check and tailoring** | Compares CV and job ad keywords. Only the profile summary wording and skills order may change, and only with skills you really have. Projects, metrics and experience are frozen. | `src/ats_tailor.py` |
| **Cover letter** | English, German or bilingual. Email body, full letter or .docx. | `src/cover_letter.py` |
| **Recruiter finder** | Looks up recruiter contacts through the Hunter.io free API, with generic fallbacks. | `src/recruiter_finder.py` |
| **Apply and track** | Drafts the email, logs the application and sets a follow-up date. The dashboard lists follow-ups that are due. | `src/apply.py`, `src/tracker.py` |
| **Dashboard and API** | FastAPI backend and a single-page dashboard with search, filters, per-job actions, interview preparation and follow-up emails. Multi-user through Google sign-in. | `src/api.py`, `dashboard.html` |

## How it works

```
Bundesagentur job search
        |
  [scraper]  listings first, full description only for jobs that pass the filter
        |
  [freshness_check]  Hot / Fresh / Recent / Stale
        |
  [eligibility_checker]  eligible / borderline / ineligible + reason (annotates, never deletes)
        |
  [matcher]  CV + job ad + known facts  ->  score, matched skills, missing skills (verified)
        |
  jobs_eligible.csv / matched_jobs.csv
        |
  Dashboard (FastAPI)  ->  ATS check  ->  cover letter  ->  recruiter  ->  email  ->  tracker
```

### Design decisions
- **The matcher sends the whole CV, not retrieved chunks.** An earlier version indexed the CV in a FAISS vector store and retrieved 3 chunks per job. The CV only split into 15 chunks, so the model saw about 20% of it and reported skills as "missing" that were in the CV. Retrieval exists for text that does not fit into the context window, and a CV does. The version here sends it whole, and retrieval would make more sense on the job side, where descriptions run 3 to 5 thousand characters.
- **Missing skills are verified.** The prompt defines a gap as a requirement stated in the job ad that the CV does not show, the model must quote the ad for each gap, and `verify_gaps` drops any "missing" skill that appears verbatim in the CV.
- **The eligibility check annotates and never deletes.** An early version overwrote the job list with the filtered result, which destroyed data and made it impossible to tune the filter. It now keeps every posting with an eligibility column and a reason.
- **Whole-word matching in the filter.** Substring matching had rejected good jobs ("Ausbildungsmöglichkeit" is a perk, not an apprenticeship, and "Lead Generation" is not a leadership role).
- **Frozen facts in the generators.** The CV tailor and the cover letter generator get a fixed list of true facts and may not invent projects, metrics or skills.
- **Known facts instead of re-deriving.** Years of experience, contract type and German level are extracted from the text once, then given to the model as facts.

## Run it locally

You need Python 3.11 and a free [Groq API key](https://console.groq.com). Hunter.io and Google OAuth are optional.

```bash
git clone https://github.com/Yasaswinid24/ai-career-copilot.git
cd ai-career-copilot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key
HUNTER_API_KEY=optional_for_recruiter_lookup
GOOGLE_CLIENT_ID=optional_for_google_sign_in
GOOGLE_CLIENT_SECRET=optional_for_google_sign_in
SECRET_KEY=any_long_random_string
BASE_URL=http://localhost:8000
```

Put your own CV as plain text in `src/resume.txt`, then choose how to start:

```bash
python run.py                       # daily menu: scrape, filter, score, apply, tailor, track
uvicorn src.api:app --reload        # dashboard and API at http://localhost:8000
```

The `run.py` menu shows data freshness and due follow-ups, then offers: full pipeline, scrape only, eligibility check, score jobs, apply, tailor CV, tracker, dashboard and rescoring.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/jobs` | Scored jobs |
| `POST /api/ats` | ATS keyword check for one job |
| `POST /api/cover_letter` | Generate a cover letter |
| `POST /api/email`, `POST /api/followup_email` | Application and follow-up emails |
| `POST /api/recruiter` | Recruiter contact lookup |
| `POST /api/apply`, `POST /api/status` | Log an application, update its status |
| `GET /api/tracker`, `GET /api/followups` | Applications and due follow-ups |
| `POST /api/summarise`, `POST /api/interview_prep` | Job summary and interview preparation |
| `POST /api/cv`, `POST /api/cv/upload`, `POST /api/cv/score` | Save or upload a CV, score jobs against it |
| `POST /api/pipeline/scrape`, `POST /api/pipeline/full`, `GET /api/pipeline/status` | Run the pipeline from the browser |
| `GET /auth/login`, `/auth/callback`, `/auth/me`, `/auth/logout` | Google sign-in |

## Deployment
The FastAPI app is deployed on Render. Set the environment variables above in the service settings. `runtime.txt` pins the Python version, and `.devcontainer/` provides a ready-to-use GitHub Codespaces setup.

## Limitations
- Built for the German job market. The scraper depends on the Bundesagentur search service, which is not an official public API and can change.
- Scoring uses a small model (Llama 3.1 8B Instant) for speed and the free tier. Scores are a ranking aid, not a judgement.
- SQLite stores users and applications. On hosts with an ephemeral disk this data is lost on redeploy, and a hosted database would be the next step.
- Job lists, application logs and a sample CV are committed as sample data. Do not commit your own personal data.
