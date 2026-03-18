"""
Semantic Analyzer Module
Repetition detection (cosine similarity) + LLM-based contradiction detection.
"""

import json
from typing import List, Tuple

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from backend.llm_chain import semantic_model, llm


class SemanticAnalyzer:
    """Handles repetition detection and contradiction checking."""

    def __init__(self):
        self.model = semantic_model

    def cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.model.encode(text1, normalize_embeddings=True)
        emb2 = self.model.encode(text2, normalize_embeddings=True)
        return float(np.dot(emb1, emb2))

    def detect_repetition(
        self, new_answer: str, history_answers: List[str], threshold: float = 0.85
    ) -> Tuple[bool, float]:
        """Semantic caching + repetition detection."""
        max_sim = 0.0
        for past in history_answers:
            sim = self.cosine_similarity(new_answer, past)
            if sim >= threshold:
                return True, sim
            max_sim = max(max_sim, sim)
        return False, max_sim

    def check_contradiction(self, new_answer: str, context: str) -> dict:
        """LLM-as-Judge contradiction detection. Returns parsed dict."""
        prompt = ChatPromptTemplate.from_template("""
You are a strict HR interviewer. Check if the candidate's NEW ANSWER contradicts any previous statements.

CONTEXT (previous answers):
{context}

NEW ANSWER: {new_answer}

Respond in JSON only:
{{
  "contradiction": "true or false",
  "explanation": "brief reason",
  "severity": "low/medium/high"
}}
""")
        chain = prompt | llm | JsonOutputParser()
        try:
            result = chain.invoke({"context": context, "new_answer": new_answer})
            return result
        except Exception:
            return {
                "contradiction": "false",
                "explanation": "Unable to parse LLM response",
                "severity": "low",
            }
