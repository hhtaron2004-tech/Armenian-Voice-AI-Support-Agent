"""
llm.py — GPT-4o-mini wrapper + RAG pipeline for Armenian bank customer support.
"""

from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from src.config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, TOP_K_RESULTS
from src.rag.retriever import retrieve, format_context, is_allowed_topic

client = OpenAI(api_key=OPENAI_API_KEY)

OFF_TOPIC_RESPONSE = (
    "Ես կարող եմ պատասխանել միայն հետևյալ թեմաների վերաբերյալ՝ "
    "ավանդներ, վարկեր և մասնաճյուղեր։ "
    "Խնդրում եմ տվեք հարց այդ թեմաներից մեկի վերաբերյալ։"
)

# ─── LLM ───────────────────────────────────────────────────────────

def ask_llm(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    if not context:
        return OFF_TOPIC_RESPONSE

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-6:])

    user_message = (
        f"Provided data:\n{context}\n\n"
        f"Customer question: {query}"
    )

    print("\n" + "=" * 50)
    print("[LLM] Sending to GPT:")
    print(user_message)
    print("=" * 50 + "\n")

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()

# ─── RAG Pipeline ──────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[str, bool]:
    allowed, topic = is_allowed_topic(query)

    if not allowed:
        return OFF_TOPIC_RESPONSE, False

    results, detected_topic, detected_bank = retrieve(
        query=query,
        top_k=TOP_K_RESULTS,
    )

    context = format_context(results)

    answer = ask_llm(
        query=query,
        context=context,
        conversation_history=conversation_history,
    )

    return answer, True