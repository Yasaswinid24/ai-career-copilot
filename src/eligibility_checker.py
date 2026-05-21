"""
AI Career Copilot — Eligibility Checker
-----------------------------------------
Filters for Masters student, semester 1.
Removes: Senior, PhD, Ausbildung, experience requirements
Flags: Abschlussarbeit (good later, not now)
"""

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
    "berufsausbildung", "vocational", "duales studium", "duale", "dhbw", "dual study",
]

THESIS_WORDS = [
    "abschlussarbeit", "masterarbeit", "bachelorarbeit",
    "thesis", "master thesis", "dissertation",
]

STUDENT_SIGNALS = [
    "werkstudent", "praktikum", "internship", "junior",
    "student", "berufseinsteiger", "entry level", "einstieg",
]


def classify(row):
    title    = str(row.get("title", "")).lower()
    job_type = str(row.get("job_type", ""))

    # Ausbildung — wrong level for Masters student
    for word in AUSBILDUNG_WORDS:
        if word in title:
            return "ineligible", f"Ausbildung/apprenticeship role"

    # Senior/Lead/PhD — too senior
    for word in INELIGIBLE_TITLE_WORDS:
        if word in title:
            for signal in STUDENT_SIGNALS:
                if signal in title:
                    return "eligible", "Senior title but has student signal"
            return "ineligible", f"Too senior: '{word}'"

    # Thesis roles — good later, flag for now
    for word in THESIS_WORDS:
        if word in title:
            return "thesis", "Abschlussarbeit — good for later"

    # Werkstudent and Praktikum — perfect
    if job_type in ["Werkstudent", "Praktikum"]:
        return "eligible", "Werkstudent/Praktikum"

    # Full-time with student signals — ok
    for signal in STUDENT_SIGNALS:
        if signal in title:
            return "eligible", f"Student signal: {signal}"

    return "borderline", "Full-time, no student signal"


def main():
    print("=" * 55)
    print("  AI Career Copilot — Eligibility Checker v2")
    print("  Filtering for M.Sc. student, semester 1")
    print("=" * 55)

    df = pd.read_csv(JOBS_FILE)
    print(f"\nTotal jobs: {len(df)}")

    df[["eligibility","reason"]] = df.apply(
        lambda row: pd.Series(classify(row)), axis=1
    )

    eligible   = df[df["eligibility"] == "eligible"]
    borderline = df[df["eligibility"] == "borderline"]
    thesis     = df[df["eligibility"] == "thesis"]
    ineligible = df[df["eligibility"] == "ineligible"]

    print(f"\nBreakdown:")
    print(f"  Eligible   (apply now)    {'█'*(len(eligible)//3)} {len(eligible)}")
    print(f"  Borderline (check first)  {'█'*(len(borderline)//3)} {len(borderline)}")
    print(f"  Thesis     (save for later) {'█'*(len(thesis)//3)} {len(thesis)}")
    print(f"  Ineligible (removed)      {'█'*(len(ineligible)//3)} {len(ineligible)}")

    print(f"\nRemoved jobs:")
    for _, row in ineligible.head(10).iterrows():
        print(f"  ✗ {row['title'][:55]} — {row['reason']}")

    print(f"\nThesis roles (save for later):")
    for _, row in thesis.head(5).iterrows():
        print(f"  📌 {row['title'][:55]} — {row['company'][:25]}")

    # Save eligible + borderline to jobs.csv
    # Save thesis separately so you don't lose them
    df_apply = df[df["eligibility"].isin(["eligible","borderline"])]
    df_thesis = df[df["eligibility"] == "thesis"]

    df_apply.to_csv(JOBS_FILE, index=False)
    df_thesis.to_csv("thesis_roles.csv", index=False)

    print(f"\nSaved {len(df_apply)} jobs to jobs.csv (apply now)")
    print(f"Saved {len(df_thesis)} thesis roles to thesis_roles.csv (for later)")

    print(f"\nTop Werkstudent roles:")
    ws = df_apply[df_apply["job_type"]=="Werkstudent"].head(5)
    for _, r in ws.iterrows():
        print(f"  ★ {r['title'][:50]} @ {r['company'][:25]}")

    print(f"\nTop Praktikum roles:")
    pk = df_apply[df_apply["job_type"]=="Praktikum"].head(5)
    for _, r in pk.iterrows():
        print(f"  ★ {r['title'][:50]} @ {r['company'][:25]}")


if __name__ == "__main__":
    main()
