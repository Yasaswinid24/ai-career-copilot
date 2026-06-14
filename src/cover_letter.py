"""
AI Career Copilot — Cover Letter Generator
-------------------------------------------
Cambridge 10/10 framework:
  - Hook paragraph: specific connection between employer focus and candidate experience
  - Skill-led paragraphs: skill name as first words, then evidence with numbers
  - Closing: one clear call to action

Supports: English, German, Bilingual
Formats:  email body, full cover letter, .docx
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

RESUME_FILE = "src/resume.txt"

FROZEN_FACTS = """
CANDIDATE FACTS — never alter these:
- Name: Yasaswini Dharmavarapu
- Degree: M.Sc. Artificial Intelligence, BTU Cottbus (2025–2027)
- Location: Cottbus, Germany. Available 20 hrs/week as Werkstudent.
- Publication: AIST-2024 Springer, bilingual Tamil–Telugu speech translation, DOI: 10.1007/978-3-031-91331-0_8
- Fractal Analytics ML Intern: BERT fine-tuned, 95%+ accuracy, 65% preprocessing automation
- BTU Research Assistant: music synthesis AI, PyTorch + Librosa, 50% data prep reduction
- AI Career Copilot: LangGraph, Groq API, RAG, FAISS, FastAPI, Docker — live in production
- Job Email Classification: XLM-RoBERTa, AUC 0.95, 8 F1-point improvement
- Driver Drowsiness Detection: CNN, 99%+ accuracy, 2-second alert
- Languages: English C1, German A2 (B1 ongoing), Telugu native, Tamil professional
"""

CAMBRIDGE_FRAMEWORK = """
Cambridge 10/10 Cover Letter Framework:
1. HOOK paragraph (2-3 sentences):
   - Open with the single most relevant thing about the EMPLOYER (their mission, product, challenge)
   - Immediately connect it to ONE specific achievement from the candidate's CV with a real number
   - Do NOT start with "I am writing to apply..."

2. SKILL paragraphs (2-3 paragraphs):
   - Start each paragraph with the skill name as the first 1-3 words
   - Follow with specific evidence: project name, metric, tool used
   - Each paragraph = one skill, one story, one number

3. CLOSING (2-3 sentences):
   - Restate availability (20 hrs/week Werkstudent)
   - One clear call to action (request a call/interview)
   - No "I look forward to hearing from you" clichés
"""


def load_resume():
    try:
        with open(RESUME_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return FROZEN_FACTS


def generate_cover_letter(
    job_role: str,
    company: str,
    tone: str = "Cambridge 10/10 framework",
    language: str = "English",
    output_format: str = "Email body only",
    recruiter_name: str = "",
    key_points: str = "",
    job_description: str = "",
    resume_text: str = ""
) -> dict:
    """
    Generate a cover letter for a given job.
    Returns dict with: subject, body, language, tone, word_count
    """
    resume = resume_text or load_resume()
    greeting = f"Dear {recruiter_name.split()[0]}" if recruiter_name else "Dear Hiring Team"
    if language == "German":
        greeting = f"Sehr geehrte/r {recruiter_name}" if recruiter_name else "Sehr geehrte Damen und Herren"

    jd_section = f"\nJOB DESCRIPTION:\n{job_description[:2000]}" if job_description else ""
    points_section = f"\nKEY POINTS TO EMPHASISE:\n{key_points}" if key_points else ""

    framework_note = CAMBRIDGE_FRAMEWORK if "Cambridge" in tone else f"Tone: {tone}"

    if language == "English":
        lang_instruction = "Write entirely in English."
    elif language == "German":
        lang_instruction = "Write entirely in German (formal Sie-form). Use natural, fluent German."
    else:
        lang_instruction = "Write the cover letter in English, then add a German translation below separated by a line '--- Deutsche Version ---'."

    format_instruction = {
        "Email body only": "Output only the email body (no subject line, no address header). Max 200 words.",
        "Full cover letter (with header)": "Include a professional header with date, candidate name/contact, then the full letter. Max 300 words.",
        "Cover letter as .docx": "Output only the email body. Max 200 words. (The .docx will be generated separately.)",
    }.get(output_format, "Output only the email body. Max 200 words.")

    prompt = f"""You are an expert career coach specialising in German tech job applications.

CANDIDATE CV:
{resume}

{FROZEN_FACTS}

TARGET ROLE: {job_role} at {company}
{jd_section}
{points_section}

FRAMEWORK:
{framework_note}

LANGUAGE: {lang_instruction}

FORMAT: {format_instruction}

STRICT RULES:
- Start with: {greeting},
- Reference the company by name in the opening hook
- Include at least 2 specific metrics from the CV (numbers, percentages, accuracies)
- Mention availability as Werkstudent 20hrs/week in Cottbus
- Never use "I am writing to apply", "I look forward to hearing from you", or "passionate about"
- Sound human and specific — not a generic template
- Every claim must be true and verifiable from the CV

Output ONLY the cover letter body. No explanation. No preamble."""

    subject_prompt = f"""Write a professional email subject line for a job application.
Role: {job_role}
Company: {company}
Recruiter: {recruiter_name or 'Hiring Team'}

Rules:
- Maximum 10 words
- Do NOT start with "Application for"
- Be specific and memorable
- Output ONLY the subject line, nothing else."""

    try:
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": subject_prompt}],
            max_tokens=30,
            temperature=0.7,
        )
        body    = r1.choices[0].message.content.strip()
        subject = r2.choices[0].message.content.strip()
        return {
            "subject":    subject,
            "body":       body,
            "language":   language,
            "tone":       tone,
            "word_count": len(body.split()),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "job":        f"{job_role} at {company}",
            "error":      None,
        }
    except Exception as e:
        return {"error": str(e), "body": "", "subject": ""}


def save_draft(job_role, company, subject, body, language, tone):
    """Save a cover letter draft to outreach_drafts.txt"""
    output_file = "outreach_drafts.txt"
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"DATE:     {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"TYPE:     COVER LETTER\n")
        f.write(f"JOB:      {job_role}\n")
        f.write(f"COMPANY:  {company}\n")
        f.write(f"TONE:     {tone}\n")
        f.write(f"LANGUAGE: {language}\n")
        f.write(f"{'='*55}\n")
        f.write(f"SUBJECT: {subject}\n\n")
        f.write(body + "\n")
    return output_file


def main():
    """CLI interface for cover letter generation"""
    print("=" * 55)
    print("  AI Career Copilot — Cover Letter Generator")
    print("  Cambridge 10/10 framework by default")
    print("=" * 55)

    job_role = input("\nJob role: ").strip()
    company  = input("Company:  ").strip()

    print("\nTone options:")
    tones = [
        "Cambridge 10/10 framework",
        "Professional and concise",
        "Enthusiastic and warm",
        "Technical and detailed",
    ]
    for i, t in enumerate(tones):
        print(f"  [{i+1}] {t}")
    t_pick = input("Tone [1]: ").strip()
    tone = tones[int(t_pick)-1] if t_pick.isdigit() and 1 <= int(t_pick) <= len(tones) else tones[0]

    print("\nLanguage options:")
    langs = ["English", "German", "Bilingual (English + German)"]
    for i, l in enumerate(langs):
        print(f"  [{i+1}] {l}")
    l_pick = input("Language [1]: ").strip()
    language = langs[int(l_pick)-1] if l_pick.isdigit() and 1 <= int(l_pick) <= len(langs) else langs[0]

    recruiter = input("\nRecruiter name (Enter to skip): ").strip()
    key_points = input("Key points to emphasise (Enter to skip): ").strip()

    print("\nPaste job description (Enter twice to skip):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    jd = "\n".join(lines)

    print(f"\nGenerating {tone} cover letter in {language}...")
    result = generate_cover_letter(
        job_role=job_role,
        company=company,
        tone=tone,
        language=language,
        recruiter_name=recruiter,
        key_points=key_points,
        job_description=jd,
    )

    if result.get("error"):
        print(f"\nError: {result['error']}")
        return

    print(f"\n{'='*55}")
    print(f"SUBJECT: {result['subject']}")
    print(f"{'='*55}")
    print(result["body"])
    print(f"{'='*55}")
    print(f"Words: {result['word_count']} | Language: {language} | Tone: {tone}")

    save = input("\nSave this draft? (y/n): ").strip().lower()
    if save == 'y':
        path = save_draft(job_role, company, result["subject"], result["body"], language, tone)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()