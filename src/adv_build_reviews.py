"""Write manually audited pass/fail labels for Advanced Challenge test & dev traces."""
import json
from collections import defaultdict
from pathlib import Path

from src.config import PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "starter" / "data" / "advanced" / "adv_cases.json"
OUT_ROOT = PROJECT_ROOT / "results" / "advanced"


def summarize_reviews(cases: list[dict], traces: list[dict], reviews: dict[str, dict]) -> dict:
    case_by_id = {c["id"]: c for c in cases}
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
    per_case = []
    errors = 0
    for tr in traces:
        qid = tr["query_id"]
        c = case_by_id[qid]
        grp = c["group"]
        if tr.get("status") == "error":
            errors += 1
        rev = reviews[qid]
        passed = bool(rev["passed"])
        by_group[grp]["total"] += 1
        by_group[grp]["pass"] += int(passed)
        per_case.append(
            {
                "query_id": qid,
                "group": grp,
                "answerable": c["answerable"],
                "status": tr["status"],
                "passed": passed,
                "citations": tr.get("citations", []),
                "reason": rev["reason"],
            }
        )
    total = len(per_case)
    passed_total = sum(1 for x in per_case if x["passed"])
    group_summary = {
        g: {
            "pass": d["pass"],
            "total": d["total"],
            "ratio": f"{d['pass']}/{d['total']}",
            "pct": round(100.0 * d["pass"] / d["total"], 1) if d["total"] else 0.0,
        }
        for g, d in by_group.items()
    }
    return {
        "total": total,
        "passed": passed_total,
        "pass_ratio": f"{passed_total}/{total}",
        "pass_pct": round(100.0 * passed_total / total, 2) if total else 0.0,
        "errors": errors,
        "by_group": group_summary,
        "cases": per_case,
    }


def main() -> None:
    all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    test_reviews = {
        "at01": {"passed": True, "reason": "Exact 4-column values for Thailand & Vietnam matching Table 1 (PDF p.18); cited p018:v01."},
        "at02": {"passed": True, "reason": "Exact 2022-2040 NPV values for NZP & NZP with reforms and $133.8B difference matching Table 3 (PDF p.46)."},
        "at03": {"passed": True, "reason": "Correctly identified 'NZP with energy efficiency gains' (+0.2%) and 'NZP without reforms' (-73.6%) in Table 4 (PDF p.48)."},
        "at04": {"passed": True, "reason": "Correctly identified Wind (offshore) +$39B and Solar +429% in Table 5 (PDF p.52)."},
        "at05": {"passed": True, "reason": "Exact 267.18 TWh, 84.78%, 15.22%, Our World in Data matching Annex 1 (PDF p.85)."},
        "at06": {"passed": True, "reason": "Exact Baseline Real GDP 4.24% and RCP 4.5 2050 damages 9.36% (Agr 2.31%, Heat 2.92%, Flood 4.00%) matching Annex 3 (PDF p.89)."},
        "at07": {"passed": True, "reason": "All 5 cities (Hanoi, Haiphong, Da Nang, HCMC, Can Tho) and legend 'Relative Flood Risk' verified on Figure 11 (PDF p.38)."},
        "at08": {"passed": True, "reason": "Quang Ninh and 5+ labels (Na Duong, Thai Nguyen, Hongai, Campha, Da River/Song Da) verified on Figure 14 (PDF p.54)."},
        "at09": {"passed": True, "reason": "Private $184B (~3.4%), Public $130B (~2.4%), External $54B (~1%) verified on Figure 21 (PDF p.71)."},
        "at10": {"passed": True, "reason": "Listed all 7 High Urgency + High Impact actions matching Annex 4 Resilient Matrix (PDF p.94)."},
        "at11": {"passed": False, "reason": "FAIL: Cited p094:v00 (Resilient Pathway: 7 vs 4 actions) instead of p095 (Decarbonizing Pathway: 8 vs 5 actions)."},
        "at12": {"passed": True, "reason": "Exact $12,695 HIC threshold and crossing at T+54 verified on Figure 2 (PDF p.15)."},
        "at13": {"passed": True, "reason": "Exact 'Business/production interruption' 54% (11%, 35%, 38%, 16%) verified on Figure 6 (PDF p.23)."},
        "at14": {"passed": True, "reason": "Exact 9.8% vs 12.3% in 2050 and divergence ~2030 verified on Figure 8 (PDF p.28)."},
        "at15": {"passed": True, "reason": "Exact 2040 TWh (BAU 922/Coal 550, CPS 788/Coal 276, ADS 812/Coal 26) and emissions (CPS 246, ADS 63 MtCO2e) verified on Figure 13 (PDF p.51)."},
        "at16": {"passed": True, "reason": "Exact Electricity (+23.6%) and Transport (-6.3%) verified on Figure 20 (PDF p.66)."},
        "at17": {"passed": True, "reason": "Exact 364.43 MtCO2e (2018), 927.9 BAU, 844 NDC, +479.57 MtCO2e verified on Annex 1 (PDF p.85)."},
        "at18": {"passed": True, "reason": "Exact Agr $837M, Env Disaster $447M (Fig 3 p.17) and 12-14.5% GDP / $400-523B by 2050 (p.28)."},
        "at19": {"passed": True, "reason": "Exact 3 Box 6 highlights (p.50) and Coal -$19B (-72%), Offshore wind +$39B (+402%) from Table 5 (p.52)."},
        "at20": {"passed": True, "reason": "Exact $15.6B (p.59), 27.53M VND/ha, and 10.97 MtCO2e/yr (p.60 Footnote 75)."},
        "at21": {"passed": True, "reason": "Exact 2030 (5.6%, 6%, 6.2%), 2050 (7.6%, 7.9%, 10.6%) losses and ~$400M PFES verified on PDF p.31."},
        "at22": {"passed": True, "reason": "Exact 5 Hanoi PM2.5 sources (~35%, ~25%, 20%, 10%, ~7%) and coal/ammonia drivers verified on Box 5 (PDF p.47)."},
        "at23": {"passed": False, "reason": "FAIL: Misstated 'younger' as 'poorer' and inverted 'Kinh 37% less likely' to 'ethnic minority 37% more likely' (PDF p.64)."},
        "at24": {"passed": True, "reason": "Exact 72/85 lacking dedicated unit and 74/85 lacking green credit appraisal procedure (PDF p.72)."},
        "at25": {"passed": True, "reason": "Searched 4 times and abstained cleanly on JETP $15.5B breakdown (absent in July 2022 CCDR)."},
        "at26": {"passed": True, "reason": "Searched and abstained cleanly on VinFast charging stations (absent)."},
        "at27": {"passed": True, "reason": "Searched and abstained cleanly on HOSE 2025 carbon credit price (absent)."},
        "at28": {"passed": True, "reason": "Searched 5 times and abstained cleanly on Thi Vai LNG capacity (absent)."},
        "at29": {"passed": True, "reason": "Searched 2 times and abstained cleanly on Ben Tre 2023 mangrove budget (absent)."},
        "at30": {"passed": True, "reason": "Searched 2 times and abstained cleanly on lithium mining/recycling capacity (absent)."},
    }

    test_cases = [c for c in all_cases if c["split"] == "test"]
    test_dir = OUT_ROOT / "multimodal_vision_test"
    test_traces = json.loads((test_dir / "traces.json").read_text(encoding="utf-8"))
    test_summary = summarize_reviews(test_cases, test_traces, test_reviews)
    (test_dir / "reviews.json").write_text(
        json.dumps(test_reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (test_dir / "eval_summary.json").write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in test_summary.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
