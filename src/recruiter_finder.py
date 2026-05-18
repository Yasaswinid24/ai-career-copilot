"""
AI Career Copilot — Recruiter Finder
--------------------------------------
Finds recruiter contact details using Hunter.io free API.
Sign up free at hunter.io — 25 searches/month on free tier.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")

KNOWN_DOMAINS = {
    "brunel":     "brunel.com",
    "siemens":    "siemens.com",
    "deloitte":   "deloitte.com",
    "zalando":    "zalando.de",
    "bosch":      "bosch.com",
    "bmw":        "bmw.com",
    "google":     "google.com",
    "sap":        "sap.com",
    "telekom":    "telekom.com",
    "bechtle":    "bechtle.com",
    "optum":      "optum.com",
    "ferchau":    "ferchau.com",
    "hays":       "hays.de",
    "dachser":    "dachser.com",
    "delivery hero": "deliveryhero.com",
    "check24":    "check24.de",
}


def find_domain(company_name):
    name_lower = company_name.lower()
    for key, domain in KNOWN_DOMAINS.items():
        if key in name_lower:
            return domain
    # Guess from name — strip legal suffixes
    clean = name_lower
    for suffix in ["gmbh & co. kg","gmbh & co kg","& co. kg","& co kg",
                   "gmbh","se & co","se","ag","kg","ltd","nl","inc"]:
        clean = clean.replace(suffix, "")
    clean = clean.strip().split()[0] if clean.strip() else "company"
    clean = ''.join(c for c in clean if c.isalnum())
    return f"{clean}.de"


def search_hunter(domain):
    if not HUNTER_API_KEY:
        return None, "HUNTER_API_KEY not set in .env"

    try:
        r = requests.get(
            "https://api.hunter.io/v2/domain-search",
            params={"domain": domain, "api_key": HUNTER_API_KEY,
                    "limit": 5, "type": "personal"},
            timeout=8
        )
        data = r.json()

        if data.get("errors"):
            return None, str(data["errors"])

        emails = data.get("data", {}).get("emails", [])
        if not emails:
            return None, f"No emails found for {domain}"

        hr_keywords = ["hr","recruit","talent","hiring","people","career","bewerbung"]
        hr_emails = [
            e for e in emails
            if any(kw in (e.get("value","") + " " + (e.get("position") or "")).lower()
                   for kw in hr_keywords)
        ]

        results = hr_emails if hr_emails else emails[:3]
        contacts = []
        for e in results:
            first = e.get("first_name") or ""
            last  = e.get("last_name") or ""
            contacts.append({
                "name":       f"{first} {last}".strip() or "Unknown",
                "email":      e.get("value",""),
                "position":   e.get("position") or "Unknown",
                "confidence": e.get("confidence", 0),
            })
        return contacts, None

    except Exception as ex:
        return None, str(ex)


def fallback_contacts(domain):
    return [
        {"name": "Recruiting Team", "email": f"recruiting@{domain}",
         "position": "HR / Recruiting", "confidence": 30},
        {"name": "HR Team",         "email": f"hr@{domain}",
         "position": "HR",           "confidence": 25},
        {"name": "Careers",         "email": f"careers@{domain}",
         "position": "HR",           "confidence": 20},
    ]


def find_recruiter(company_name):
    domain   = find_domain(company_name)
    print(f"  Domain: {domain}")

    if HUNTER_API_KEY:
        contacts, error = search_hunter(domain)
        if contacts:
            print(f"  Hunter.io found {len(contacts)} contact(s)")
            return contacts, domain
        else:
            print(f"  Hunter.io: {error} — using fallback")
    else:
        print("  Hunter.io key not set — using fallback generic contacts")

    return fallback_contacts(domain), domain


if __name__ == "__main__":
    company  = input("Enter company name: ").strip()
    contacts, domain = find_recruiter(company)
    print(f"\nResults for {domain}:")
    for c in contacts:
        print(f"\n  Name:       {c['name']}")
        print(f"  Email:      {c['email']}")
        print(f"  Position:   {c['position']}")
        print(f"  Confidence: {c['confidence']}%")
