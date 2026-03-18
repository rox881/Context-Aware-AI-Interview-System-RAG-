"""
LLM Chain Module
Initializes Groq LLM, embeddings, and prompt chains.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from sentence_transformers import SentenceTransformer

# Load .env file if present
load_dotenv()

# ── Models (loaded once at module level) ──────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Set it via a .env file or your shell before starting the server."
    )

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Prompt Templates ─────────────────────────────────────────────────────────

followup_prompt = ChatPromptTemplate.from_template("""
You are an expert technical interviewer. Generate ONE intelligent follow-up question based on the candidate's last answer and full context.
Keep it natural, probing deeper into skills/experience.

FULL CONTEXT:
{full_context}

LAST ANSWER: {last_answer}

Generate ONLY the question (no explanation):
""")

evaluation_prompt = ChatPromptTemplate.from_template("""
Evaluate the candidate's answer on a scale of 0-10 for each criterion.
Return JSON only.

Criteria:
- relevance: how directly it answers the question
- clarity: communication quality
- technical_depth: depth of knowledge shown

Question: {question}
Answer: {answer}
Full Context: {context}

Output JSON:
{{
  "relevance": X,
  "clarity": X,
  "technical_depth": X,
  "overall_score": X,
  "feedback": "short constructive feedback"
}}
""")

# ── Chains ────────────────────────────────────────────────────────────────────

followup_chain = followup_prompt | llm | StrOutputParser()
evaluation_chain = evaluation_prompt | llm | JsonOutputParser()
