# 🧠 Context-Aware AI Interview System with Conversational Memory

**UptoSkill Internship — Task 2**  
**Built with LangChain + Groq + FAISS + SentenceTransformers**  
**Full-Stack Web Application (FastAPI + HTML/CSS/JS)**

---

## 📌 Table of Contents

1. [What is This Project?](#-what-is-this-project)
2. [How Does It Work?](#-how-does-it-work)
3. [Tech Stack — Why Each Choice?](#-tech-stack--why-each-choice)
4. [Project Structure](#-project-structure)
5. [Setup Guide — Step by Step](#-setup-guide--step-by-step)
6. [How We Developed It](#-how-we-developed-it)
7. [Code Walkthrough](#-code-walkthrough)
8. [API Reference](#-api-reference)
9. [Frontend Features](#-frontend-features)
10. [Design Decisions & Optimizations](#-design-decisions--optimizations)

---

## 🎯 What is This Project?

A **real-time AI-powered interview system** that conducts intelligent technical interviews with:

| Feature | Description |
|---|---|
| **Conversational Memory (RAG)** | Remembers ALL previous answers using hybrid short-term + long-term memory |
| **Intelligent Follow-ups** | Each question is based on what you've said — not random |
| **Real-time Evaluation** | Scores answers on relevance, clarity, and technical depth (0–10) |
| **Repetition Detection** | Cosine similarity (threshold > 0.85) flags repeated answers |
| **Contradiction Detection** | LLM-as-Judge checks if new answers contradict previous ones |
| **Chat-style Web UI** | Modern dark-themed interface — feels like a real chat |

### Project Evolution

The project originally started as a single Colab notebook (`UPTOSKILL_task2_nootbook.ipynb`) and has now been fully converted into a **modular full-stack web application** located in the `backend/` and `frontend/` folders.

---

## 🔄 How Does It Work?

```
User clicks "Start Interview"
        │
        ▼
┌──────────────────────────┐
│  POST /start             │ → Creates a new session
│  Returns first question  │ → "Tell me about yourself..."
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  User types answer       │
│  POST /answer            │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Backend Processing Pipeline:                │
│  1. Retrieve hybrid context (short + long)   │
│  2. Detect repetition (cosine similarity)    │
│  3. Check contradiction (LLM-as-Judge)       │
│  4. Evaluate answer (relevance/clarity/depth)│
│  5. Store in memory (JSON + FAISS)           │
│  6. Generate next follow-up question         │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Response to UI:         │
│  - Score (0-10)          │
│  - Feedback text         │
│  - Repetition alert      │
│  - Contradiction alert   │
│  - Next question         │
└──────────────────────────┘
           │
    (Repeats for 10 questions)
           │
           ▼
    🎉 Interview Complete!
```

### Hybrid Memory System

| Memory Type | Implementation | Purpose |
|---|---|---|
| **Short-term** | Last 5 Q&A pairs from in-memory list | Immediate conversation context |
| **Long-term** | FAISS vector store (similarity search) | Retrieves semantically relevant past answers |

Both are combined into a single `full_context` string passed to every LLM call — this is what makes it **context-aware**.

---

## 🛠 Tech Stack — Why Each Choice?

### Backend

| Technology | Role | Why? |
|---|---|---|
| **Python 3.10+** | Language | Industry standard for AI/ML |
| **FastAPI** | Web framework | Async, auto-docs, fast for AI APIs |
| **Uvicorn** | ASGI server | Production-grade server for FastAPI |
| **LangChain** | AI orchestration | Chains prompts + LLMs + parsers cleanly |
| **Groq API** | LLM provider | Ultra-fast inference (Llama-3.1-8B) — free tier |
| **FAISS** | Vector database | Fast similarity search, runs locally, no external DB |
| **SentenceTransformers** | Embeddings | `all-MiniLM-L6-v2` runs locally — no API calls |
| **python-dotenv** | Config | Loads `.env` for secure API keys |

### Frontend

| Technology | Role | Why? |
|---|---|---|
| **HTML5** | Structure | Semantic, accessible |
| **Vanilla CSS** | Styling | Premium dark theme, glassmorphism, no framework overhead |
| **Vanilla JavaScript** | Logic | Lightweight, no build step |
| **Inter (Google Fonts)** | Typography | Modern, professional look |

### Why Groq + Llama 3.1?

- **Speed**: ~500 tokens/sec — near-instant responses
- **Cost**: Free tier with generous rate limits
- **No GPU needed**: All inference happens via Groq's API

### Why FAISS (not Pinecone/ChromaDB)?

- **Zero setup**: No external database or API keys
- **In-memory**: Perfect for per-session interview data
- **Fast**: Optimized C++ under the hood

---

## 📁 Project Structure

```
Task_2/
│
├── .env                          # Environment variables (GROQ_API_KEY)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── UPTOSKILL_task2_nootbook.ipynb # Original Colab notebook
│
├── backend/                      # FastAPI backend (Python)
│   ├── __init__.py               # Package marker
│   ├── main.py                   # FastAPI app — routes + session management
│   ├── llm_chain.py              # LLM setup, prompt templates, chains
│   ├── memory.py                 # ConversationStorage + InterviewMemory (FAISS)
│   └── analyzer.py               # SemanticAnalyzer (repetition + contradiction)
│
└── frontend/                     # Static web frontend
    ├── index.html                # Main HTML page
    ├── style.css                 # Premium dark theme CSS (566 lines)
    └── script.js                 # Frontend logic (API calls, UI rendering)
```

### Module Dependencies

```
main.py
  ├── memory.py (ConversationStorage, InterviewMemory)
  ├── analyzer.py (SemanticAnalyzer)
  └── llm_chain.py (followup_chain, evaluation_chain)

memory.py → llm_chain.py (embed_model)
analyzer.py → llm_chain.py (semantic_model, llm)
llm_chain.py ← Foundation (no internal imports)
```

---

## 🚀 Setup Guide — Step by Step

### Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- **Groq API Key** — free at [console.groq.com/keys](https://console.groq.com/keys)

### Step 1: Clone the Project

```bash
git clone <your-repo-url>
cd Task_2
```

### Step 2: Create a Virtual Environment

```bash
# Using venv:
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# OR using Conda:
conda create -n interview-ai python=3.10 -y
conda activate interview-ai
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
```
fastapi, uvicorn[standard], langchain, langchain-groq, langchain-community,
langchain-huggingface, faiss-cpu, sentence-transformers, python-dotenv
```

### Step 4: Configure API Key

Edit `.env` in the project root:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

> ⚠️ **Never commit your API key to Git!** Add `.env` to `.gitignore`.

### Step 5: Run the Application

> ⚠️ **IMPORTANT**: Make sure your terminal is in the root project folder (`Task_2`). Do **not** `cd` into the `backend` folder, and do **not** run `python main.py`.

Run the FastAPI application using Uvicorn:

```bash
uvicorn backend.main:app --reload --port 8000
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 6: Open in Browser

Go to **http://localhost:8000** → Click **"Start Interview"** → Answer 10 questions!

### Step 7 (Optional): API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🧑‍💻 How We Developed It

### Phase 1: Research & Prototype (Google Colab)

Built a working single-file prototype (`UPTOSKILL_task2_nootbook.ipynb`) using research from:

- **Medium article**: Metadata enrichment, contextual retrieval, FAISS indexing, prompt clarity
- **Redis blog**: Long-term memory (#7), semantic caching for repetition, LLM-as-Judge (#9)
- **Mentor guidance**: `InterviewConversationMemory` + `SemanticAnalyzer` class design
- **GitHub projects** (TalentRAG / HiringHelp-Chatbot): Interview scoring pipeline

### Phase 2: Full-Stack Web App Conversion

| Original (Single File) | Extracted To | Key Change |
|---|---|---|
| Model initialization | `backend/llm_chain.py` | Uses `python-dotenv` instead of hardcoded keys |
| `ConversationStorage` + `InterviewMemory` | `backend/memory.py` | In-memory only (no file persistence) |
| `SemanticAnalyzer` | `backend/analyzer.py` | Returns `dict` instead of JSON string |
| `run_interview()` loop | `backend/main.py` | Replaced with FastAPI endpoints |

Added:
- **Session management** — UUID-based, supports multiple concurrent users
- **Frontend** — Chat-style UI with dark theme, glassmorphism, animations
- **Error handling** — Groq rate limits, LLM failures, graceful degradation

### Phase 3: Polish

- End-to-end testing (10-question flow)
- Rate limit handling (429 responses)
- Fallback responses on LLM failure
- Responsive design for mobile

---

## 📖 Code Walkthrough

### `backend/llm_chain.py` — Foundation Module

Initializes all AI models (loaded **once** at module level):

```python
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=512)
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
```

Defines two LCEL chains:

| Chain | Input | Output | Purpose |
|---|---|---|---|
| `followup_chain` | context + last_answer | String | Generate next question |
| `evaluation_chain` | question + answer + context | JSON | Score the answer |

### `backend/memory.py` — Hybrid Memory

| Method | What It Does |
|---|---|
| `add_interaction()` | Stores in both in-memory list AND FAISS |
| `get_short_term_context()` | Returns last 5 Q&A pairs |
| `retrieve_long_term()` | FAISS similarity search (threshold ≥ 0.75) |
| `get_full_context()` | Combines short + long-term |

### `backend/analyzer.py` — Semantic Analysis

- **`detect_repetition()`** — Cosine similarity ≥ 0.85 = repetition flagged
- **`check_contradiction()`** — LLM-as-Judge returns `{contradiction, explanation, severity}`

### `backend/main.py` — FastAPI Application

`POST /answer` pipeline: Retrieve context → Detect repetition → Check contradiction → Evaluate → Store → Generate follow-up

---

## 📡 API Reference

### `POST /start`

Initializes a new interview session.

**Response:**
```json
{
  "session_id": "uuid-string",
  "question": "Tell me about yourself and your experience with Python and data analysis."
}
```

### `POST /answer`

Submits an answer and returns evaluation + next question.

**Request:**
```json
{ "session_id": "uuid", "answer": "Your answer text" }
```

**Response:**
```json
{
  "next_question": "Follow-up question...",
  "score": 7.5,
  "feedback": "Good depth on pandas, could elaborate on visualization.",
  "repetition": false,
  "repetition_similarity": 0.312,
  "contradiction": { "contradiction": "false", "explanation": "...", "severity": "low" },
  "questions_asked": 3,
  "interview_complete": false
}
```

| Error Code | Cause |
|---|---|
| `404` | Invalid session_id |
| `400` | Interview complete or empty answer |
| `429` | Groq rate limit |
| `500` | Internal error |

---

## 🎨 Frontend Features

| Component | Description |
|---|---|
| **Welcome Screen** | Feature list + "Start Interview" button |
| **Chat Messages** | AI (left, dark card) + User (right, gradient bubble) |
| **Evaluation Card** | Color-coded score: 🟢 ≥7 · 🟡 4–6 · 🔴 <4 |
| **Alert Badges** | 🔁 Repetition · ⚠️ Contradiction · ✅ No issues |
| **Loading Dots** | Animated typing indicator while AI processes |
| **Input Bar** | Auto-resize textarea + send button + counters |

**Design:** Dark theme (`#0a0e17`) · Glassmorphism · Gradient accents · Slide-in animations · Responsive (640px breakpoint)

---

## ⚙️ Design Decisions & Optimizations

| Technique | Source | Implementation |
|---|---|---|
| Hybrid Memory | Redis blog #7 | Short-term (last 5) + Long-term (FAISS) |
| Semantic Caching | Redis blog | Cosine similarity for repetition |
| LLM-as-Judge | Redis blog #9 | Contradiction detection |
| Metadata Enrichment | Medium article | Timestamp, topic, score on FAISS docs |
| Context Window Optimization | Medium article | Threshold filtering on retrieval |
| Module-level Loading | Best practice | Models loaded once, not per-request |
| Graceful Degradation | Production pattern | Fallback responses on LLM failure |

### Key Trade-offs

| Decision | Reasoning |
|---|---|
| In-memory sessions | Interviews are ephemeral — no need for DB persistence |
| FAISS over ChromaDB | No external setup needed; sufficient for 10 Q&A per session |
| Vanilla JS over React | Single page — framework overhead not justified |
| `temperature=0.3` | Consistency matters more than creativity in evaluations |
| Cosine threshold 0.85 | Conservative — avoids false positive repetition flags |

---

## 🏁 Summary

This project demonstrates: **AI system design** (multi-component pipeline with memory), **RAG** (FAISS vector search), **LangChain** (LCEL chains, prompt templates), **full-stack development** (FastAPI + static frontend), and **production patterns** (error handling, rate limits, graceful degradation).

---

**Built with ❤️ for the UptoSkill Internship**  
**Stack:** Python · FastAPI · LangChain · Groq · FAISS · SentenceTransformers · HTML · CSS · JavaScript
