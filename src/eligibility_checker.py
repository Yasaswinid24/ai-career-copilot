"""
AI Career Copilot — Eligibility Checker v3
-------------------------------------------
Annotates every posting with an eligibility verdict. Never deletes anything.

What was wrong with v2:

  1. DESTRUCTIVE. `df_apply.to_csv(JOBS_FILE)` wrote the *filtered* set back
     over jobs.csv, so ineligible rows were gone forever. You could never
     re-tune the filter against the full data or audit what it threw away.
     It also explains why jobs.csv held 89 rows when the same queries return
     600+: most were deleted before you ever saw them.

  2. SUBSTRING MATCHING. `if word in title` with no word boundary:
       "duale"      matched "duale Karrierewege"      -> killed a Werkstudent role
       "ausbildung" matched "Ausbildungsmöglichkeit"  -> that's a PERK, not an apprenticeship
       "lead"       matched "Lead Generation Analytics" -> that's a data job
     Each one silently deleted a job worth seeing.

  3. The docstring claimed it removed roles by experience requirement, but
     classify() only ever read `title`. The check did not exist. Descriptions
     are full and reliable since scraper v3, so now it does.

  4. The senior check had a "but it says Werkstudent" rescue; the Ausbildung
     check did not, so "Werkstudent ... duales Studium" was killed outright.

v3 writes:
    jobs.csv          every posting + eligibility/reason columns  (analytics reads this)
    jobs_eligible.csv eligible + borderline only                  (your daily job hunt)
    thesis_roles.csv  thesis roles                                (unchanged)
"""

import re

import pandas as pd

JOBS_FILE = "jobs.csv"
ELIGIBLE_FILE = "jobs_eligible.csv"
THESIS_FILE = "thesis_roles.csv"

# --------------------------------------------------------------------
# Patterns, not substrings. \b marks a word boundary, so "ausbildung"
# no longer fires inside "Ausbildungsmöglichkeit".
# German compounds need explicit alternatives rather than a bare stem.
# --------------------------------------------------------------------
TOO_SENIOR = re.compile(
    r"\b(senior|sr\.|principal|staff|director|chief|head\s+of|vp\b"
    r"|team\s?lead|tech\s?lead"
    r"|lead\s+(engineer|developer|scientist|analyst|architect|data|ml|software|consultant)"
    r"|abteilungsleiter\w*|teamleiter\w*|gruppenleiter\w*|projektleiter\w*"
    r"|bereichsleiter\w*|referatsleiter\w*|leitung\s+(der|des|von)"
    r"|postdoc\w*|phd\s+(candidate|student)|doktorand\w*)\b",
    re.IGNORECASE)

# Deliberately NOT matching a bare "lead" — "Lead Generation Analytics" is a
# data role, not a leadership one. Only "lead <role-noun>" and the German
# compound titles count, which is why the noun list above is explicit.

AUSBILDUNG = re.compile(
    r"\b(duales\s+studium|duale[sr]?\s+(hochschule|studiengang)|dhbw"
    r"|dual\s+study|ausbildung(splatz|sjahr|sbeginn)?\b(?!\s*(smöglichkeit|angebot))"
    r"|auszubildende[rn]?|azubi|berufsausbildung|vocational\s+training"
    r"|fachinformatiker\w*\s+ausbildung)\b",
    re.IGNORECASE)

# "Ausbildungsmöglichkeit"/"Weiterbildung" are benefits, not apprenticeships.
AUSBILDUNG_PERK = re.compile(
    r"ausbildungsm[öo]glichkeit|weiterbildung|fortbildung|ausbildungsangebot",
    re.IGNORECASE)

THESIS = re.compile(
    r"\b(abschlussarbeit|master\s?arbeit|bachelor\s?arbeit|masterand\w*"
    r"|bachelorand\w*|(master|bachelor|diploma)\s+thesis|\bthesis\b"
    r"|dissertation|studienarbeit|forschungsarbeit)\b",
    re.IGNORECASE)

# --------------------------------------------------------------------
# Domain gate. The Bundesagentur search matches on description text, so
# "Data Science" returns any ad that mentions Excel or a dashboard. That
# put Legal Intern, Supply Chain Logistics and Social Media & Content
# Creation into the scoring queue.
#
# Asking the LLM to reject them did not work: told that a non-technical
# role caps at 30, the 70B instead applied the "core tool missing" cap of
# 74 and parked five unrelated jobs there. Whether a title is a data role
# is a keyword question, so it is answered here in code.
# --------------------------------------------------------------------
IN_DOMAIN = re.compile(
    r"\b(data|daten|analytics|analyse|analyst\w*|analytik"
    r"|machine\s?learning|deep\s?learning|\bml\b|\bai\b|\bki\b"
    r"|artificial\s+intelligence|k[üu]nstliche\s+intelligenz"
    r"|nlp|llm|genai|computer\s?vision|bildverarbeitung"
    r"|software|entwickl\w*|developer|engineer\w*|informatik\w*"
    r"|programmier\w*|\bit\b|digitalisierung|digital"
    r"|business\s+intelligence|\bbi\b|power\s?bi|tableau|dashboard"
    r"|reporting|datenbank|database|\bsql\b|etl|data\s?warehouse"
    r"|cloud|devops|automation|automatisierung|robotik|robotics"
    r"|forschung|research|wissenschaftlich\w*|scientist"
    r"|python|cyber\s?security|informationssicherheit)\b",
    re.IGNORECASE)

# Titles that contain a domain word but are not the job. "Data Protection
# Officer" is legal work; "Vertrieb Data Solutions" is sales.
OFF_DOMAIN = re.compile(
    r"\b(vertrieb|sales|einkauf|procurement|beschaffung"
    r"|supply\s?chain|logistik|logistics|lager"
    r"|immobilien|real\s?estate|facility"
    r"|social\s?media|content\s+creation|marketing|kommunikation"
    r"|communications|redaktion|personal\w*|\bhr\b|recruiting"
    r"|legal|recht\w*|jurist\w*|datenschutzbeauftragt\w*"
    r"|controlling|controller|buchhaltung|accounting|steuer\w*"
    r"|finance|finanz\w*|audit|revision"
    r"|lean\s+management|operational\s+excellence"
    r"|qualit[äa]tssicherung|arbeitssicherheit"
    r"|bid\s+(&|and)?\s*proposal|tender|ausschreibung)\b",
    re.IGNORECASE)

STUDENT_SIGNAL = re.compile(
    r"\b(werk\s?student\w*|working\s+student|praktikum|praktikant\w*"
    r"|internship|intern\b|junior|studentische\s+hilfskraft|hiwi"
    r"|berufseinsteiger\w*|entry[\s-]level|einsteiger\w*|trainee|graduate)\b",
    re.IGNORECASE)

# --------------------------------------------------------------------
# Domain relevance, from the TITLE.
#
# The Bundesagentur search matches on description text, so any ad that
# happens to mention Excel or "Datenanalyse" surfaces under a "Data
# Science" query. That put Legal Intern, Supply Chain Logistics and Social
# Media & Content Creation into the scoring queue.
#
# This is deliberately a title check done in code, not another instruction
# to the LLM. Asking the model to cap non-technical roles produced Social
# Media at 74 and Legal Intern at 45 — and worse, the literal number in the
# prompt ("MAXIMUM 74") became an anchor the model clustered on.
# --------------------------------------------------------------------
IN_DOMAIN = re.compile(
    r"\b(data|daten|analytic|analyse|analyst|scien|ki\b|ai\b|artificial"
    r"|machine learning|deep learning|\bml\b|\bnlp\b|llm|genai"
    r"|software|entwickl|develop|engineer|ingenieur|informatik|\bit\b"
    r"|bi\b|business intelligence|power bi|sql|datenbank|database"
    r"|digitalisierung|digital|automation|automatisierung|cloud"
    r"|cyber|security|computer vision|robotic|programmier)",
    re.IGNORECASE)

# Titles that clearly belong to another profession, even when they mention
# a tool. "Werkstudent Controlling mit Power BI" is a finance role.
OUT_OF_DOMAIN = re.compile(
    r"\b(legal|recht\w*|jurist\w*|vertrieb|sales|einkauf|procurement"
    r"|supply chain|logistik|logistics|immobilien|real estate"
    r"|social media|content creation|marketing|kommunikation"
    r"|communications|personal\b|\bhr\b|recruiting|buchhaltung"
    r"|controlling|accounting|steuer\w*|audit|lean management"
    r"|qualitätssicherung|quality assurance|instandhaltung"
    r"|produktion\b|fertigung|montage|lager\b|verkauf)\b",
    re.IGNORECASE)


def in_domain(title):
    """True when the title reads as a data/AI/software role."""
    if OUT_OF_DOMAIN.search(title) and not IN_DOMAIN.search(title):
        return False
    if OUT_OF_DOMAIN.search(title) and IN_DOMAIN.search(title):
        # Both present: the out-of-domain noun usually names the department,
        # e.g. "Werkstudent Data Analytics im Vertrieb" is still a data role.
        # Whichever appears first in the title wins.
        return (IN_DOMAIN.search(title).start()
                < OUT_OF_DOMAIN.search(title).start())
    return bool(IN_DOMAIN.search(title))


# Experience gates, read from the description. The check the old docstring
# promised but never implemented.
# Matches "5 Jahre Berufserfahrung", "at least 3 years experience",
# "3+ years of relevant experience", "mindestens 2 Jahre". The words between
# the number and "experience" vary a lot, so allow a short gap rather than
# trying to enumerate them.
EXPERIENCE = re.compile(
    r"(\d+)\s*\+?\s*(?:bis\s*\d+\s*)?(?:jahre?n?|years?)"
    r"(?:[\s\w.,()-]{0,40}?)?"
    r"(?:erfahrung|berufserfahrung|experience)",
    re.IGNORECASE)

MAX_YEARS = 2          # a semester-3 M.Sc. student can credibly claim ~2


def required_years(description):
    """Highest stated experience requirement, or None."""
    if not isinstance(description, str) or not description:
        return None
    years = [int(m.group(1)) for m in EXPERIENCE.finditer(description)
             if m.group(1) and m.group(1).isdigit() and int(m.group(1)) <= 40]
    return max(years) if years else None


def classify(row):
    title = str(row.get("title", "") or "")
    desc = str(row.get("description", "") or "")
    job_type = str(row.get("job_type", "") or "")

    student = bool(STUDENT_SIGNAL.search(title)) or job_type in (
        "Werkstudent", "Praktikum", "Trainee")

    # Thesis first — these are a distinct track, not a rejection.
    if THESIS.search(title) or job_type == "Thesis":
        return "thesis", "Abschlussarbeit — good for later"

    # Ausbildung. The student-signal rescue now applies here too, and a
    # mention of training as a *benefit* no longer counts.
    if AUSBILDUNG.search(title) and not AUSBILDUNG_PERK.search(title):
        if student:
            return "eligible", "Ausbildung wording but explicit student role"
        return "ineligible", "Ausbildung/apprenticeship"
    if job_type == "Ausbildung" and not student:
        return "ineligible", "Ausbildung/apprenticeship"

    # Domain gate, before seniority — an unrelated profession is a harder
    # blocker than a senior title.
    off = OFF_DOMAIN.search(title)
    if off and not IN_DOMAIN.search(title):
        return "off-domain", f"Not a data/software role: '{off.group(0)}'"
    if not IN_DOMAIN.search(title):
        return "off-domain", "No data/AI/software term in title"

    # Seniority.
    m = TOO_SENIOR.search(title)
    if m:
        if student:
            return "eligible", f"Senior wording ('{m.group(0)}') but student role"
        return "ineligible", f"Too senior: '{m.group(0)}'"

    # Experience gate, from the full description.
    yrs = required_years(desc)
    if yrs is not None and yrs > MAX_YEARS and not student:
        return "ineligible", f"Requires {yrs}+ years experience"

    if not in_domain(title):
        return "off-domain", "Not a data/AI/software role"

    if student:
        return "eligible", f"Student role ({job_type or 'title signal'})"

    if yrs is not None and yrs <= MAX_YEARS:
        return "eligible", f"Only {yrs} years experience required"

    return "borderline", "Full-time, no student signal"


def bar(n, scale=3):
    return "#" * min(n // scale, 40)


def main():
    print("=" * 55)
    print("  AI Career Copilot — Eligibility Checker v3")
    print("  Annotates. Does not delete.")
    print("=" * 55)

    df = pd.read_csv(JOBS_FILE)
    print(f"\nTotal jobs: {len(df)}")
    if df.empty:
        print("jobs.csv is empty — nothing to classify.")
        return

    df[["eligibility", "reason"]] = df.apply(
        lambda r: pd.Series(classify(r)), axis=1)

    counts = df["eligibility"].value_counts()
    print("\nBreakdown:")
    for key, label in [("eligible", "Eligible   (apply now)"),
                       ("borderline", "Borderline (check first)"),
                       ("thesis", "Thesis     (save for later)"),
                       ("off-domain", "Off-domain (wrong profession)"),
                       ("ineligible", "Ineligible (kept, flagged)")]:
        n = int(counts.get(key, 0))
        print(f"  {label:<28} {bar(n)} {n}")

    off = df[df.eligibility == "off-domain"]
    if not off.empty:
        print(f"\nOff-domain sample (excluded from scoring):")
        for t in off["title"].head(6):
            print(f"     {str(t)[:60]}")

    print("\nWhy postings were flagged ineligible:")
    for reason, n in (df[df.eligibility == "ineligible"]["reason"]
                      .value_counts().head(8).items()):
        print(f"  {n:>4}  {reason}")

    off = df[df.eligibility == "off-domain"]
    if not off.empty:
        print(f"\nOff-domain sample (excluded from scoring, kept in jobs.csv):")
        for t in off["title"].head(6):
            print(f"  - {str(t)[:60]}")

    # ---- write ------------------------------------------------------
    # jobs.csv keeps EVERY row. The analytics pipeline reads this one and
    # needs the whole market, not the slice that suits one student.
    df.to_csv(JOBS_FILE, index=False, encoding="utf-8")

    df_elig = df[df.eligibility.isin(["eligible", "borderline"])]
    df_elig.to_csv(ELIGIBLE_FILE, index=False, encoding="utf-8")
    df[df.eligibility == "thesis"].to_csv(THESIS_FILE, index=False,
                                          encoding="utf-8")

    print(f"\n  {JOBS_FILE:<20} {len(df):>4} rows (all postings, annotated)")
    print(f"  {ELIGIBLE_FILE:<20} {len(df_elig):>4} rows (apply now)")
    print(f"  {THESIS_FILE:<20} {len(df[df.eligibility=='thesis']):>4} rows")

    for label in ("Werkstudent", "Praktikum"):
        sub = df_elig[df_elig.get("job_type", "") == label].head(5)
        if not sub.empty:
            print(f"\nTop {label} roles:")
            for _, r in sub.iterrows():
                print(f"  * {str(r['title'])[:50]} @ {str(r['company'])[:25]}")


if __name__ == "__main__":
    main()