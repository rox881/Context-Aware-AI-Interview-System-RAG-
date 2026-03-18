"""
Memory Module
In-memory conversation storage + hybrid short-term / long-term (FAISS RAG) memory.
Each session gets its own storage and memory instances.
"""

from datetime import datetime
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.llm_chain import embed_model


class ConversationStorage:
    """In-memory conversation history for a single session."""

    def __init__(self):
        self.history: List[Dict] = []

    def add_interaction(self, interaction: Dict):
        self.history.append(interaction)


class InterviewMemory:
    """Hybrid memory: short-term buffer + FAISS long-term RAG retrieval."""

    def __init__(self, storage: ConversationStorage):
        self.storage = storage
        self.embeddings = embed_model
        self.vectorstore: FAISS = self._build_vectorstore()
        self.short_term_limit = 5  # last N exchanges (context window optimization)

    def _build_vectorstore(self) -> FAISS:
        if not self.storage.history:
            dummy = Document(
                page_content="Initial empty context",
                metadata={"timestamp": datetime.now().isoformat()},
            )
            return FAISS.from_documents([dummy], self.embeddings)

        docs = []
        for entry in self.storage.history:
            content = f"Q: {entry['question']}\nA: {entry['answer']}"
            doc = Document(
                page_content=content,
                metadata={
                    "timestamp": entry["timestamp"],
                    "topic": entry.get("topic", "general"),
                    "score": entry.get("eval_score", 0),
                },
            )
            docs.append(doc)
        return FAISS.from_documents(docs, self.embeddings)

    def add_interaction(
        self,
        question: str,
        answer: str,
        topic: str = "general",
        eval_score: float = 0.0,
    ):
        interaction = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "eval_score": eval_score,
        }
        self.storage.add_interaction(interaction)

        # Add to FAISS (long-term memory)
        doc = Document(
            page_content=f"Q: {question}\nA: {answer}",
            metadata={
                "timestamp": interaction["timestamp"],
                "topic": topic,
                "score": eval_score,
            },
        )
        self.vectorstore.add_documents([doc])

    def get_short_term_context(self) -> str:
        """Short-term memory: last N turns."""
        recent = self.storage.history[-self.short_term_limit :]
        return "\n\n".join(
            [f"Q: {r['question']}\nA: {r['answer']}" for r in recent]
        )

    def retrieve_long_term(
        self, query: str, k: int = 4, threshold: float = 0.75
    ) -> str:
        """Long-term RAG retrieval with relevance threshold."""
        docs_with_score = self.vectorstore.similarity_search_with_score(query, k=k)
        relevant = []
        for doc, score in docs_with_score:
            sim = 1 / (1 + score)
            if sim >= threshold:
                relevant.append(doc.page_content)
        return "\n\n".join(relevant) if relevant else ""

    def get_full_context(self, current_question: str) -> str:
        """Hybrid memory: short + long-term (optimized context window)."""
        short = self.get_short_term_context()
        long = self.retrieve_long_term(current_question)
        return (
            f"=== RECENT CONVERSATION (Short-term) ===\n{short}\n\n"
            f"=== RELEVANT PAST ANSWERS (Long-term RAG) ===\n{long}"
        )
