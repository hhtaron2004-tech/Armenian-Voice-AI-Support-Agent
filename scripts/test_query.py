"""
test_query.py — Test the RAG pipeline with text queries (no voice needed).
Use this to verify the knowledge base and guardrails work correctly.

Usage:
    python scripts/test_query.py
    python scripts/test_query.py --query "Արարատ բանկի ավանդի տոկոսն ինչ է?"
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import run_rag_pipeline

TEST_QUERIES = [
    # On-topic: deposits
    "Արարատ բանկի ավանդի տոկոսն ինչ է?",
    "Ի՞նչ արժույթներով կարող եմ ավանդ բացել",
    # On-topic: credits
    "Ի՞նչ վարկեր կան հասանելի",
    # On-topic: branches
    "Արարատ բանկի Երևանյան մասնաճյուղերը ո՞րն են",
    # Off-topic: should be refused
    "Ո՞վ է Հայաստանի վարչապետը",
    "Եղանակն ի՞նչ է այսօր",
    "Ինձ կօգնե՞ս Python կոդ գրել",
]


def run_tests(queries):
    print("=" * 60)
    print("Armenian Bank Voice AI — RAG Pipeline Test")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Հարց: {query}")
        print("-" * 40)
        answer, is_on_topic = run_rag_pipeline(query)
        status = "✅ ON-TOPIC" if is_on_topic else "🚫 OFF-TOPIC"
        print(f"{status}")
        print(f"Պատասխան: {answer}")

    print("\n" + "=" * 60)
    print("Test complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query to test")
    args = parser.parse_args()

    if args.query:
        run_tests([args.query])
    else:
        run_tests(TEST_QUERIES)


if __name__ == "__main__":
    main()