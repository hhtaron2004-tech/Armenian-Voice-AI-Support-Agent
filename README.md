# 🇦🇲 Armenian Bank Voice AI Support Agent

> End-to-end Voice AI customer support agent for Armenian banks, built with open-source LiveKit framework. The agent understands and speaks **Armenian**, answering questions strictly about **deposits**, **credits**, and **branch locations**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack & Model Choices](#tech-stack--model-choices)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Data Sources](#data-sources)
- [Setup Instructions](#setup-instructions)
- [Running the Agent](#running-the-agent)
- [Testing](#testing)
- [Guardrails & Safety](#guardrails--safety)
- [Scalability](#scalability)
- [Frontend](#frontend)
- [Known Limitations](#known-limitations)

---

## 🎯 Project Overview

This project implements a production-ready **Voice AI customer support agent** for three Armenian banks:

| Bank                     | Color | Coverage |
|--------------------------|-------|----------|
| AraratBank (Արարատ Բանկ) | 🔴 Red | Deposits, Credits, Branches |
| ACBA Bank (ԱԿԲԱ Բանկ)    | 🟢 Green | Deposits, Credits, Branches |
| Ameriabank (Ամերիա Բանկ) | 🔵 Blue | Deposits, Credits, Branches |

The agent:
- Understands spoken **Armenian** via OpenAI STT (`gpt-4o-transcribe`)
- Retrieves answers from a **RAG (Retrieval-Augmented Generation)** knowledge base built from official bank websites
- Generates responses using **GPT-4o-mini**
- Speaks back in **Armenian** using OpenAI TTS
- Strictly refuses to answer questions outside of its three allowed topics

---

## 🏗️ Architecture

```
        User speaks Armenian
                │
                ▼
        ┌─────────────────┐
        │   LiveKit Room  │  ← Open-source WebRTC server
        │   (WebRTC)      │
        └────────┬────────┘
                 │ Audio frames
                 ▼
        ┌─────────────────┐
        │  Silero VAD     │  ← Voice Activity Detection
        │                 │     Detects when user is speaking
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  OpenAI STT     │  ← Armenian speech → text
        │ gpt-4o-transcr. │     language="hy", with Armenian
        │                 │     bank vocabulary prompt
        └────────┬────────┘
                 │ Armenian text
                 ▼
┌──────────────────────────────────────┐
│            RAG Pipeline              │
│                                      │
│  1. Topic Detection (keyword-based)  │
│     deposit / credit / branch        │
│                                      │
│  2. Bank Detection (fuzzy matching)  │
│     AraratBank / ACBA / Ameriabank   │
│                                      │
│  3. ChromaDB Vector Search           │
│     multilingual embeddings          │
│     cosine similarity                │
│                                      │
│  4. Context Formatting               │
│     top-8 relevant chunks            │
└────────────────┬─────────────────────┘
                 │ Retrieved context + query
                 ▼
        ┌─────────────────┐
        │  GPT-4o-mini    │  ← LLM: generates Armenian answer
        │                 │     grounded strictly in retrieved
        │                 │     data, temperature=0.2
        └────────┬────────┘
                 │ Armenian text answer
                 ▼
        ┌─────────────────┐
        │  OpenAI TTS     │  ← Armenian text → speech
        │  (tts-1, nova)  │     voice="nova", speed=0.85
        └────────┬────────┘
                 │ Audio
                 ▼
        User hears Armenian response
```

---

## 🤖 Tech Stack & Model Choices

### Voice Framework: LiveKit (Open Source)
LiveKit provides a full open-source WebRTC stack with a Python agent SDK (`livekit-agents`). The `AgentSession` class handles VAD, audio buffering, and turn-taking out of the box. Self-hosted via Docker with no cloud dependency.

### STT: OpenAI `gpt-4o-transcribe`
The `gpt-4o-transcribe` model provides highly accurate Armenian speech recognition. The `language="hy"` parameter forces Armenian transcription, and a custom `prompt` parameter primes the model with Armenian bank vocabulary to reduce transcription errors on proper nouns such as bank names.

### LLM: GPT-4o-mini
GPT-4o-mini offers the best balance of cost, speed, and Armenian language understanding. `temperature=0.2` ensures factual, consistent responses. It reliably follows strict system prompt guardrails, refusing off-topic questions and always responding in Armenian.

### TTS: OpenAI TTS (`tts-1`, voice: `nova`)
OpenAI TTS supports Armenian text natively, delivers low latency, and reuses the same API key as the LLM. The `nova` voice performs best for non-English accents. `speed=0.85` ensures clear, natural-paced Armenian speech.

### Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
This sentence-transformers model supports 50+ languages including Armenian. It is fast, lightweight (117MB), runs on CPU, and produces high-quality semantic embeddings for Armenian text — essential for matching Armenian user queries to Armenian bank data chunks.

### Vector Store: ChromaDB
ChromaDB is lightweight, embedded (no separate server needed), persists to disk, supports cosine similarity, and has a simple Python API. Metadata filtering allows simultaneous filtering by bank and topic type.

### VAD: Silero VAD
Silero VAD accurately detects voice activity and handles Armenian speech well. It is the recommended VAD for LiveKit agents and is included in the `livekit-plugins-silero` package.

---

## 📁 Project Structure

```
Armenian-Voice-AI-Support-Agent/
│
├── data/                          # Bank knowledge base (JSON)
│   ├── ameriabank/
│   │   ├── deposits.json
│   │   ├── credits.json
│   │   └── branches.json
│   ├── acba_bank/
│   │   ├── deposits.json
│   │   ├── credits.json
│   │   └── branches.json
│   └── araratbank/
│       ├── deposits.json
│       ├── credits.json
│       └── branches.json
│
├── embeddings/
│   └── chroma_db/                 # Persisted ChromaDB vector store
│
├── src/
│   ├── config.py                  # All settings and constants
│   ├── rag/
│   │   ├── store.py               # Data loading, chunking, embedding, ChromaDB
│   │   └── retriever.py           # Smart retrieval + topic/bank detection
│   ├── llm.py                     # GPT-4o-mini wrapper + RAG pipeline
│   └── main.py                    # LiveKit agent entry point
│
├── scripts/
│   ├── build_embeddings.py        # One-time vector DB builder
│   └── test_query.py              # Text-only RAG pipeline tester
│
├── frontend/
│   └── index.html                 # Web client for voice interaction
│
├── .env                           # API keys (not committed)
├── requirements.txt
├── start.sh                       # One-command startup script
└── README.md
```

---

## 📄 File Descriptions

### `src/config.py`
Central configuration file. Contains all settings: API keys (loaded from `.env`), model names, file paths, supported banks list, ChromaDB collection name, embedding model name, and the LLM system prompt. **Adding a new bank requires only editing `SUPPORTED_BANKS` here.**

### `src/rag/store.py`
Combined data pipeline module. Handles:
- **JSON Loading** — reads all bank JSON files from `data/`
- **Chunking** — converts deposits (flat chunks), credits (per-topic chunks), and branches (flat chunks) into a uniform chunk format
- **Embedding** — generates multilingual embeddings using sentence-transformers
- **ChromaDB** — builds and queries the vector store with optional bank/type filters

### `src/rag/retriever.py`
Smart retrieval module with automatic topic and bank detection:
- **`detect_topic()`** — keyword-based classification into deposit/credit/branch
- **`detect_bank()`** — fuzzy matching (SequenceMatcher) to identify which bank is mentioned, handling Armenian speech recognition errors
- **`is_allowed_topic()`** — guardrail: blocks off-topic queries before vector search
- **`retrieve()`** — main function: detects topic and bank, queries ChromaDB with filters
- **`format_context()`** — formats retrieved chunks into a single context string for GPT

### `src/llm.py`
LLM wrapper and RAG pipeline orchestrator:
- **`ask_llm()`** — sends query and retrieved context to GPT-4o-mini, maintains conversation history (last 3 turns), enforces Armenian-only responses
- **`run_rag_pipeline()`** — the main pipeline: topic check → retrieve → format context → ask GPT → return answer

### `src/main.py`
LiveKit agent entry point:
- **`ArmenianBankAgent`** class — extends LiveKit's `Agent`; greets user on entry in Armenian, processes each voice turn through the RAG pipeline
- **`prewarm()`** — preloads the embedding model and VAD before each job starts, reducing response latency
- **`entrypoint()`** — connects to LiveKit room, configures `AgentSession` with VAD + STT + LLM + TTS, starts the agent

### `scripts/build_embeddings.py`
One-time setup script. Reads all JSON files from `data/`, generates embeddings, and saves them to ChromaDB. Must be run before starting the agent, and re-run whenever JSON data is updated.

### `scripts/test_query.py`
Development testing tool. Tests the full RAG pipeline with text queries (no voice required). Includes predefined test cases for on-topic and off-topic queries. Use the `--query` flag for a single query test.

### `frontend/index.html`
Web client for voice interaction. Features bank branding, token input, connect/disconnect controls, microphone toggle, real-time status indicator, chat history display, and audio output for agent responses.

### `start.sh`
One-command startup script. Launches the LiveKit Docker server, HTTP server for the frontend, the Python agent, generates a JWT token, and dispatches the agent to the room after the user connects.

---

## 🏦 Data Sources

All knowledge base data was scraped from official Armenian bank websites:

| Bank | Website | Topics |
|------|---------|--------|
| AraratBank | https://www.araratbank.am | Deposits, Credits, Branches |
| ACBA Bank | https://www.acba.am | Deposits, Credits, Branches |
| Ameriabank | https://ameriabank.am | Deposits, Credits, Branches |

### Data Format

Each bank has three JSON files:

**deposits.json** — flat records with `content`, `metadata` (bank, type, product, topic, source_url), and `structured` data fields.

**credits.json** — records with a `topics` dict containing keys such as overview, amount, term, interest, fees, documents, collateral, and eligibility.

**branches.json** — flat records with `content` (address, phone, working hours) and `metadata` (bank, city, region).

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Armenian-Voice-AI-Support-Agent.git
cd Armenian-Voice-AI-Support-Agent
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

### 4. Build the vector store (one-time)

```bash
PYTHONPATH=. python scripts/build_embeddings.py
```

Expected output:
```
Building Armenian Bank Knowledge Base
✅ Vector store built successfully!
   Total chunks: 324
```

---

## 🚀 Running the Agent

### Option 1: One-command startup (recommended)

```bash
bash start.sh
```

This will:
1. Start LiveKit server (Docker)
2. Start HTTP server for the frontend
3. Start the Python agent
4. Generate a JWT token
5. Wait for you to connect in the browser, then dispatch the agent to the room

### Option 2: Manual startup

**Terminal 1 — LiveKit server:**
```bash
sudo docker run --rm \
  -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server:latest --dev --bind 0.0.0.0
```

**Terminal 2 — Agent:**
```bash
PYTHONPATH=. python src/main.py dev
```

**Terminal 3 — Frontend:**
```bash
python -m http.server 8080 --directory frontend
```

**Terminal 4 — Generate token:**
```bash
python -c "
import jwt, time
payload = {
    'iss': 'devkey', 'sub': 'user1',
    'iat': int(time.time()), 'exp': int(time.time()) + 3600,
    'video': {'room': 'test', 'roomJoin': True, 'canPublish': True, 'canSubscribe': True}
}
print(jwt.encode(payload, 'secret', algorithm='HS256'))
"
```

**Terminal 5 — Dispatch agent:**
```bash
python -c "
import asyncio
from livekit.api import LiveKitAPI
from livekit.api.agent_dispatch_service import CreateAgentDispatchRequest

async def main():
    api = LiveKitAPI('http://localhost:7880', 'devkey', 'secret')
    await api.agent_dispatch.create_dispatch(
        CreateAgentDispatchRequest(room='test', agent_name=''))
    await api.aclose()

asyncio.run(main())
"
```

**Browser:** Open `http://localhost:8080/index.html`, paste the token, click Connect, then speak in Armenian.

---

## 🧪 Testing

### Text-based RAG pipeline test (no voice required)

```bash
# Run all predefined test cases
PYTHONPATH=. python scripts/test_query.py

# Test a single query
PYTHONPATH=. python scripts/test_query.py --query "Արарат բanki avandi tokosn inch e?"
```

### Example test results

```
[1] Query: Araratbank-i avandi tokosn inch e?
✅ ON-TOPIC
Answer: AraratBank-i avandi tokosadruytky kazmum e minchev 8.5% AMD, 4.25% USD, 2.4% EUR

[7] Query: Ov e Hayastani varchapety?
🚫 OFF-TOPIC
Answer: I can only answer questions about deposits, credits, and branches...
```

### Predefined test queries

**On-topic (should be answered):**
- "Արարատ բանկի ավանդի տոկոսն ինչ է?" — AraratBank deposit interest rate
- "Ի՞նչ վարկեր կան հասանելի" — Available credit products
- "Արարատ բանկի Երևանյան մասնաճյուղերը ո՞րն են" — Komitas branch phone number

**Off-topic (should be refused):**
- "Ո՞վ է Հայաստանի վարչապետը"
- "Եղանակն ի՞նչ է այսօր"
- "Ինձ կօգնե՞ս Python կոդ գրել"

---

## 🔒 Guardrails & Safety

The agent enforces strict topic restrictions at **two independent levels**:

### Level 1: Keyword Detection (pre-LLM)
`retriever.py` classifies the query using extensive Armenian and English keyword lists for each topic. Off-topic queries are rejected **before any LLM or vector search call** — zero cost, zero latency.

**Deposit keywords:** ավanд, депозit, токос, арժуйт, AMD, USD, EUR, драм, долар, евро, хнайогутюн...

**Credit keywords:** варк, варкаин, кредit, маруm, грав, ерашавор, овердрафт, хипотек, авто, оски...

**Branch keywords:** масначяух, хасде, херакхос, ашхатанкаин жам, Ереvan, Гюмри, Ваnadзор...

### Level 2: LLM System Prompt
Even if a query passes keyword detection, the system prompt instructs GPT-4o-mini to:
- Answer ONLY from provided knowledge base data
- Politely refuse in Armenian if context is insufficient
- Never invent information
- Always respond in Armenian only, never mixing languages

---

## 📈 Scalability

Adding a new bank requires only **4 steps**:

1. Create `data/new_bank_name/` with `deposits.json`, `credits.json`, `branches.json` following the existing format
2. Add `"new_bank_name"` to `SUPPORTED_BANKS` in `src/config.py`
3. Add bank name keywords to `BANK_CANONICAL` in `src/rag/retriever.py`
4. Run `PYTHONPATH=. python scripts/build_embeddings.py`

No other code changes required.

---

## 🎨 Frontend Design

The web client (`frontend/index.html`) features:
- **Dark banking aesthetic** — professional dark theme with subtle radial gradient backgrounds
- **Bank color coding** — AraratBank (`#c0392b` red), ACBA (`#27ae60` green), Ameriabank (`#2980b9` blue)
- **Playfair Display** typography for headers — elegant, banking-appropriate serif font
- **Real-time status indicator** — animated dot for connected / speaking / thinking states
- **Chat history** — conversation displayed as chat bubbles with user and agent avatars
- **Microphone toggle** — clear visual feedback when the microphone is active
- **Audio output** — agent speech played back via LiveKit track subscription

---

## 💰 Estimated API Costs

| Component | Model | Estimated Cost |
|-----------|-------|----------------|
| STT | gpt-4o-transcribe | ~$0.006/min |
| LLM | gpt-4o-mini | ~$0.15/1M tokens |
| TTS | tts-1 | ~$0.015/1K chars |
| **Total (30 test queries)** | | **< $1.00** |

---

## ⚠️ Known Limitations

- **STT accuracy** — The STT model occasionally misrecognizes Armenian proper nouns (bank names). Mitigated by fuzzy matching in `retriever.py` and vocabulary priming via the STT prompt parameter.
- **TTS language mixing** — OpenAI TTS may occasionally pronounce English words within Armenian text in English. Mitigated by system prompt instructions to avoid English terms.
- **Memory usage** — The multilingual embedding model (~800MB RAM) is loaded per worker process. Pre-warming via `prewarm()` reduces latency but increases baseline memory.
- **Number pronunciation** — TTS may pronounce numbers and percentages inconsistently. The system prompt instructs Armenian-style number pronunciation.

---

## 🛠️ Requirements

```
openai>=2.0.0
python-dotenv>=1.0.0
livekit>=1.0.0
livekit-agents>=1.5.0
livekit-plugins-openai>=1.5.0
livekit-plugins-silero>=1.5.0
chromadb>=0.5.0
sentence-transformers>=2.7.0
torch>=2.1.0
numpy>=1.26.0
tqdm>=4.66.0
PyJWT>=2.8.0
```

---

## 📝 License

This project was developed as a technical assessment. All bank data is sourced from publicly available official bank websites for educational purposes.
