"""Find and download a >= 100-page open-access World Bank PDF for the Advanced Challenge."""
import json
import re
from pathlib import Path
import requests

from src.config import PROJECT_ROOT

ADV_DIR = PROJECT_ROOT / "starter" / "data" / "advanced"


def find_pdf_urls() -> list[str]:
    candidates = [
        "https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099355507062229876",
        "https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099104507112233955/p17724104cb4aa06b09bd70f8021175628",
        "https://openknowledge.worldbank.org/handle/10986/37618",
    ]
    found: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            print(url, "->", r.status_code, r.url)
            for m in re.findall(r'https://documents1\.worldbank\.org/[^\s"\'<>]+\.pdf', r.text):
                if m not in found:
                    found.append(m)
            for m in re.findall(r'/server/api/core/bitstreams/[a-f0-9-]+/content', r.text):
                full = "https://openknowledge.worldbank.org" + m
                if full not in found:
                    found.append(full)
        except Exception as exc:
            print("ERR:", url, exc)
    # Also try DSpace 7 REST search for handle 10986/37618
    try:
        r = requests.get(
            "https://openknowledge.worldbank.org/server/api/pid/find?id=10986/37618",
            headers=headers,
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            uuid = data.get("uuid")
            print("OKR item uuid:", uuid)
            if uuid:
                b = requests.get(
                    f"https://openknowledge.worldbank.org/server/api/core/items/{uuid}/bundles",
                    headers=headers,
                    timeout=20,
                ).json()
                for bundle in b.get("_embedded", {}).get("bundles", []):
                    if bundle.get("name") == "ORIGINAL":
                        bs_url = bundle["_links"]["bitstreams"]["href"]
                        bs = requests.get(bs_url, headers=headers, timeout=20).json()
                        for bit in bs.get("_embedded", {}).get("bitstreams", []):
                            if bit.get("name", "").lower().endswith(".pdf"):
                                found.append(bit["_links"]["content"]["href"])
                                print("Found OKR PDF:", bit.get("name"), bit.get("sizeBytes"), bit["_links"]["content"]["href"])
    except Exception as exc:
        print("OKR REST ERR:", exc)
    return found


if __name__ == "__main__":
    urls = find_pdf_urls()
    print("CANDIDATE PDFS:", json.dumps(urls, indent=2))
