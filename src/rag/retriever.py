"""
retriever.py — Smart retrieval with automatic topic and bank detection.
Classifies queries into deposit/credit/branch and optionally filters by bank.
"""

from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from src.rag.store import query_vector_store
from src.config import TOP_K_RESULTS

# ─── Topic Keywords ────────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "deposit": [
        # Հայերեն
        "ավանդ", "դեպոզիտ", "տոկոս", "տոկոսադրույք", "արժույթ","currency", "դրամ", "դոլար", "եվրո", "ռուբli",
        "ժամկետ", "գումար", "նվազագույն", "առավելագույն", "համալրել",
        "խնայողություն", "եկամուտ", "երաշխիք", "կուտակել", "ֆիքսված",
        # English
        "deposit", "interest", "currency", "term", "savings",
        "minimum", "maximum", "replenishment", "guarantee", "fixed",
        # Currencies
        "AMD", "USD", "EUR", "RUB", "RUR", "դրամ",
    ],
    "credit": [
        # Հայերեն
        "վարկ", "վարկային", "կրեդիտ", "մարում", "տոկոս",
        "ապահովված", "գրավ", "երաշխավոր", "օվերդրաֆթ",
        "վարկային գիծ", "ժամկետ", "փոխառություն", "հիփոթեք",
        "ավտոմեքենա", "ոսկի", "ուսում", "սպառողական",
        # English
        "credit", "loan", "repayment", "collateral", "overdraft",
        "mortgage", "car", "gold", "student", "consumer",
        "interest", "secured", "unsecured",
    ],
    "branch": [
         # Հայերեն
        "մասնաճյուղ", "հասցե", "հեռախոս", "աշխատանքային",
        "ժամ", "բաց", "քաղաք", "մարզ",
        # English
        "branch", "address", "phone", "working hours",
        "location", "office", "open",
        # Cities — Հայաստանի քաղաքներ
        "Երևան", "Գյումրի", "Վանաձոր", "Աբովյան", "Հրազդան",
        "Իջևան", "Մարտունի", "Սիսիան", "Գոռիս", "Կապան",
        "Yerevan", "Gyumri", "Vanadzor", "Abovyan", "Hrazdan",
        "Ijevan", "Martuni", "Sisian", "Goris", "Kapan",
        # Regions
        "Կոտայք", "Լոռի", "Շիրակ", "Սյունիք", "Տավուշ",
        "Արարատ", "Արմավիր", "Գեղարքունիք",
    ],
}

BANK_CANONICAL = {
    "araratbank": ["արարատ", "ararat"],
    "ameriabank": ["ամերիա", "ameria"],
    "acba_bank": ["acba", "ակբա"],
}

# ─── Detection ─────────────────────────────────────────────────────

def detect_topic(query: str) -> Optional[str]:
    query_lower = query.lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                scores[topic] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def detect_bank(query: str) -> Optional[str]:
    query_lower = query.lower()
    best_bank = None
    best_score = 0.4

    words = query_lower.split()
    for bank_id, keywords in BANK_CANONICAL.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                return bank_id
            for word in words:
                score = SequenceMatcher(None, kw.lower(), word).ratio()
                if score > best_score:
                    best_score = score
                    best_bank = bank_id

    return best_bank


def is_allowed_topic(query: str) -> Tuple[bool, Optional[str]]:
    topic = detect_topic(query)
    return (topic is not None, topic)

# ─── Retrieval ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    topic = detect_topic(query)
    bank = detect_bank(query)
    results = query_vector_store(
        query=query,
        top_k=top_k,
        filter_bank=bank,
        filter_type=topic,
    )
    return results, topic, bank


def format_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        meta = r["metadata"]
        bank_name = meta.get("bank", "")
        source = meta.get("source_url", "")
        parts.append(f"[{bank_name}]\n{r['text']}\n(Source: {source})")
    return "\n\n---\n\n".join(parts)