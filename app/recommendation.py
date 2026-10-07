"""
Rule-based recommendation + summary from LSTM analysis output.
"""


def _top_category(by_cat: dict):
    """Return category with most entries, or None."""
    if not by_cat:
        return None
    return max(by_cat, key=lambda k: len(by_cat[k]))


def _dedupe_sentences(sentences: list, max_items: int = 3) -> list:
    """Remove near-duplicates, keep top N."""
    seen = set()
    result = []
    for s in sentences:
        key = s.lower().strip()[:60]
        if key not in seen:
            seen.add(key)
            result.append(s)
        if len(result) >= max_items:
            break
    return result


def build_summary(analysis: dict, store_name: str) -> str:
    """One-liner strategic summary."""
    pos = analysis["positif_pct"]
    neg = analysis["negatif_pct"]

    pros_cats = analysis.get("pros_by_category", {})
    cons_cats = analysis.get("cons_by_category", {})

    top_pro = _top_category(pros_cats)
    top_con = _top_category(cons_cats)

    parts = []
    if pos >= 75:
        parts.append(f"{store_name} has an excellent reputation")
    elif pos >= 55:
        parts.append(f"{store_name} has a solid reputation")
    elif neg >= 40:
        parts.append(f"{store_name} receives many complaints")
    else:
        parts.append(f"{store_name} has a mixed reputation")

    if top_pro:
        parts.append(f"strongest in {top_pro.lower()}")
    if top_con and len(cons_cats[top_con]) >= 2:
        parts.append(f"needs improvement in {top_con.lower()}")

    return ", ".join(parts) + "."


def pick_winner(analysis_a: dict, analysis_b: dict, name_a: str, name_b: str) -> dict:
    """Pick winner using composite score (positif_pct - negatif_pct)."""
    def score(a):
        return a["positif_pct"] - a["negatif_pct"]

    score_a = score(analysis_a)
    score_b = score(analysis_b)

    if score_a > score_b + 3:
        winner = name_a
        reason = (
            f"{name_a} leads with {analysis_a['positif_pct']}% positive sentiment "
            f"(vs {analysis_b['positif_pct']}%) and a lower complaint rate "
            f"({analysis_a['negatif_pct']}% vs {analysis_b['negatif_pct']}%)."
        )
    elif score_b > score_a + 3:
        winner = name_b
        reason = (
            f"{name_b} leads with {analysis_b['positif_pct']}% positive sentiment "
            f"(vs {analysis_a['positif_pct']}%) and a lower complaint rate "
            f"({analysis_b['negatif_pct']}% vs {analysis_a['negatif_pct']}%)."
        )
    else:
        winner = "Tie"
        reason = (
            f"Both gyms perform similarly. "
            f"{name_a}: {analysis_a['positif_pct']}% positive, "
            f"{name_b}: {analysis_b['positif_pct']}% positive."
        )

    return {"pemenang": winner, "alasan": reason, "top_pros": {
        name_a: _top_category(analysis_a.get("pros_by_category", {})),
        name_b: _top_category(analysis_b.get("pros_by_category", {})),
    }}