"""
AI Career Copilot — Eligibility Checker v3
-----------------------------------------
Combines original seniority/level filtering with:
- German C1/C2 hard reject
- Expanded domain mismatch hard reject
- LLM signal detection + score boost
- Deduplication
"""

import re
import pandas as pd

JOBS_FILE = "jobs.csv"

INELIGIBLE_TITLE_WORDS = [
    "senior", "lead", "principal", "director", "head of",
    "leiter", "leitung", "chief", "teamlead", "teamleiter",
    "postdoc", "postdoctoral", "phd candidate",
]

AUSBILDUNG_WORDS = [
    "duales studium", "duale hochschule", "dhbw", "dual study",
    "ausbildung", "auszubildende", "azubi",
    "berufsausbildung", "vocational",
]

THESIS_WORDS = [
    "abschlussarbeit", "masterarbeit", "bachelorarbeit",
    "thesis", "master thesis", "dissertation",
]

STUDENT_SIGNALS = [
    "werkstudent", "praktikum", "internship", "junior",
    "student", "berufseinsteiger", "entry level", "einstieg",
]

GERMAN_C1_PATTERNS = [
    r"verhandlungssicher",
    r"c1[\s\-]?niveau",
    r"c2[\s\-]?niveau",
    r"sehr gute deutsch",
    r"sichere deutsch",
    r"sicheres deutsch",
    r"zwingend.*deutsch",
    r"muttersprachlich.*deutsch",
    r"fließende.*deutsch",
    r"mind\.?\s*c1",
    r"mindestens c1",
    r"c1 oder muttersprache",
    r"deutschkenntnisse.*voraussetzung",
    r"voraussetzung.*deutschkenntnisse",
    r"sehr gute kenntnisse.*deutsch",
]

DOMAIN_REJECT_PATTERNS = [
    r"\bcobol\b", r"\bpl/i\b", r"\bmainframe\b",
    r"value.at.risk", r"\bscorecard\b", r"kreditrisiko",
    r"risikomodell", r"ratingmodell",
    r"\branorex\b",
    r"lackier", r"schweißen",
    r"kundenberater.*verkauf", r"verkaufsberater",
    r"brand marketing manager", r"marketing manager",
    r"filialleiter", r"einzelhandel",
    r"datenschutz.*compliance", r"rechtsanwalt",
    r"steuerberatung", r"buchführung",
    r"doctoral researcher", r"doktorand",
    r"\bphd\b.*position", r"promotion.*stelle",
    r"pflegefachkraft", r"krankenpflege",
    r"strategic executive assistant",
]

LLM_KEYWORDS = [
    "langchain", "langgraph", "llm", "large language model",
    "rag", "retrieval.augmented", "agentic", "multi.agent",
    "gpt", "claude", "openai", "anthropic",
    "prompt engineering", "faiss", "vector database",
    "hugging face", "transformers",
    "sprachmodell", "generative ki", "generative ai",
]


def count_llm_signals(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in LLM_KEYWORDS if re.search(kw, t))


def check_german_required(text: str):
    t = text.lower()
    for pattern in GERMAN_C1_PATTERNS:
        if re.search(pattern, t):
            return True, pattern
    return False, ""


def check_domain_mismatch(text: str, llm_score: int):
    t = text.lower()
    for pattern in DOMAIN_REJECT_PATTERNS:
        if re.search(pattern, t):
            if llm_score >= 2:
                return False, ""
            return True, pattern
    return False, ""


def classify(row):
    title    = str(row.get("title", "")).lower()
    job_type = str(row.get("job_type", ""))
    desc     = str(row.get("description", "")).lower()
    text     = title + " " + desc

    llm_score = count_llm_signals(text)

    german_required, german_pattern = check_german_required(text)
    if german_required:
        return "ineligible", f"German C1/C2 required ({german_pattern})"

    domain_mismatch, domain_pattern = check_domain_mismatch(text, llm_score)
    if domain_mismatch:
        return "ineligible", f"Domain mismatch ({domain_pattern})"

    for word in AUSBILDUNG_WORDS:
        if word in title:
            return "ineligible", "Ausbildung/apprenticeship role"

    for word in INELIGIBLE_TITLE_WORDS:
        if word in title:
            for signal in STUDENT_SIGNALS:
                if signal in title:
                    return "eligible", "Senior title but has student signal"
            return "ineligible", f"Too senior: '{word}'"

    for word in THESIS_WORDS:
        if word in title:
            return "thesis", "Abschlussarbeit — good for later"

    if job_type in ["Werkstudent", "Praktikum"]:
        return "eligible", "Werkstudent/Praktikum"

    for signal in STUDENT_SIGNALS:
        if signal in title:
            return "eligible", f"Student signal: {signal}"

    return "borderline", "Full-time, no student signal"


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "ref_number" in df.columns:
        df = df.drop_duplicates(subset=["ref_number"], keep="first")
    df = df.drop_duplicates(subset=["title", "company"], keep="first")
    after = len(df)
    if before > after:
        print(f"  Removed {before - after} duplicates")
    return df


def main():
    print("=" * 60)
    print("  AI Career Copilot — Eligibility Checker v3")
    print("  M.Sc. student + German + Domain + Dedup")
    print("=" * 60)

    df = pd.read_csv(JOBS_FILE)
    print(f"\nTotal jobs loaded: {len(df)}")

    df = remove_duplicates(df)
    print(f"After dedup: {len(df)}")

    df["llm_score"] = df.apply(
        lambda row: count_llm_signals(
            str(row.get("title", "")) + " " +
            str(row.get("description", ""))
        ), axis=1
    )

    df[["eligibility", "reason"]] = df.apply(
        lambda row: pd.Series(classify(row)), axis=1
    )

    eligible   = df[df["eligibility"] == "eligible"]
    borderline = df[df["eligibility"] == "borderline"]
    thesis     = df[df["eligibility"] == "thesis"]
    ineligible = df[df["eligibility"] == "ineligible"]

    print(f"\nBreakdown:")
    print(f"  Eligible   (apply now)      {'#'*(len(eligible)//3)} {len(eligible)}")
    print(f"  Borderline (check first)    {'#'*(len(borderline)//3)} {len(borderline)}")
    print(f"  Thesis     (save for later) {'#'*(len(thesis)//3)} {len(thesis)}")
    print(f"  Ineligible (removed)        {'#'*(len(ineligible)//3)} {len(ineligible)}")

    german_filtered = ineligible[ineligible["reason"].str.contains("German", na=False)]
    domain_filtered = ineligible[ineligible["reason"].str.contains("Domain", na=False)]
    other_filtered  = ineligible[~ineligible["reason"].str.contains("German|Domain", na=False)]

    print(f"\nIneligible breakdown:")
    print(f"  German C1/C2:    {len(german_filtered)}")
    print(f"  Domain mismatch: {len(domain_filtered)}")
    print(f"  Level/other:     {len(other_filtered)}")

    print(f"\nRemoved jobs:")
    for _, row in ineligible.iterrows():
        print(f"  x {row['title'][:55]} -- {row['reason']}")

    if not thesis.empty:
        print(f"\nThesis roles (save for later):")
        for _, row in thesis.iterrows():
            print(f"  {row['title'][:50]} -- {row['company'][:25]}")

    df_apply = df[df["eligibility"].isin(["eligible", "borderline"])].copy()
    df_apply = df_apply.sort_values(
        ["priority", "llm_score"], ascending=[True, False]
    )
    df_thesis = df[df["eligibility"] == "thesis"]

    df_apply.to_csv(JOBS_FILE, index=False)
    df_thesis.to_csv("thesis_roles.csv", index=False)
    ineligible.to_csv("filtered_out.csv", index=False)

    print(f"\nSaved {len(df_apply)} jobs to jobs.csv")
    print(f"Saved {len(df_thesis)} thesis roles to thesis_roles.csv")
    print(f"Saved {len(ineligible)} filtered to filtered_out.csv")

    if not df_apply.empty:
        print(f"\nTop LLM/agentic signal jobs:")
        llm_top = df_apply[df_apply["llm_score"] >= 2].head(10)
        for _, r in llm_top.iterrows():
            print(f"  [{r['llm_score']} signals] {r['title'][:45]} @ {r['company'][:25]}")

        print(f"\nTop Werkstudent:")
        for _, r in df_apply[df_apply["job_type"]=="Werkstudent"].head(5).iterrows():
            print(f"  {r['title'][:50]} @ {r['company'][:25]}")

        print(f"\nTop Praktikum:")
        for _, r in df_apply[df_apply["job_type"]=="Praktikum"].head(5).iterrows():
            print(f"  {r['title'][:50]} @ {r['company'][:25]}")


if __name__ == "__main__":
    main()
