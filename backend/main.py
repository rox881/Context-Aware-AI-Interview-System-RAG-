"""
FastAPI Main Application
Serves the AI Interview Web App — API endpoints + static frontend.
"""

import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.memory import ConversationStorage, InterviewMemory
from backend.analyzer import SemanticAnalyzer
from backend.llm_chain import followup_chain, evaluation_chain

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Interview System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session State ─────────────────────────────────────────────────────────────

STARTER_QUESTION = (
    "Tell me about yourself and your experience with Python and data analysis."
)


class SessionState:
    """Holds all state for a single interview session."""

    def __init__(self):
        self.storage = ConversationStorage()
        self.memory = InterviewMemory(self.storage)
        self.analyzer = SemanticAnalyzer()
        self.current_question: str = STARTER_QUESTION
        self.history_answers: list[str] = []
        self.questions_asked: int = 0
        self.max_questions: int = 10


sessions: Dict[str, SessionState] = {}

# ── Request / Response Models ─────────────────────────────────────────────────


class StartResponse(BaseModel):
    session_id: str
    question: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class AnswerResponse(BaseModel):
    next_question: str
    score: float
    feedback: str
    repetition: bool
    repetition_similarity: float
    contradiction: dict
    questions_asked: int
    interview_complete: bool


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.post("/start", response_model=StartResponse)
async def start_interview():
    """Initialize a new interview session and return the first question."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionState()
    return StartResponse(session_id=session_id, question=STARTER_QUESTION)


@app.post("/answer", response_model=AnswerResponse)
async def submit_answer(req: AnswerRequest):
    """Process a candidate answer and return evaluation + next question."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new interview.")

    if session.questions_asked >= session.max_questions:
        raise HTTPException(status_code=400, detail="Interview is complete. Start a new session.")

    answer = req.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")

    try:
        # 1. Retrieve hybrid context
        full_context = session.memory.get_full_context(session.current_question)

        # 2. Repetition detection
        is_repeated, sim_score = session.analyzer.detect_repetition(
            answer, session.history_answers
        )

        # 3. Contradiction check
        contradiction = session.analyzer.check_contradiction(answer, full_context)

        # 4. Evaluate answer
        try:
            eval_result = evaluation_chain.invoke({
                "question": session.current_question,
                "answer": answer,
                "context": full_context,
            })
        except Exception:
            eval_result = {
                "relevance": 5,
                "clarity": 5,
                "technical_depth": 5,
                "overall_score": 5,
                "feedback": "Evaluation temporarily unavailable.",
            }

        overall_score = float(eval_result.get("overall_score", 5))
        feedback = eval_result.get("feedback", "")

        # 5. Store in memory
        session.memory.add_interaction(
            question=session.current_question,
            answer=answer,
            topic="technical",
            eval_score=overall_score,
        )
        session.history_answers.append(answer)
        session.questions_asked += 1

        # 6. Generate next follow-up question
        interview_complete = session.questions_asked >= session.max_questions
        if not interview_complete:
            try:
                next_q = followup_chain.invoke({
                    "full_context": full_context,
                    "last_answer": answer,
                })
                session.current_question = next_q.strip()
            except Exception:
                session.current_question = (
                    "Can you elaborate further on your previous answer?"
                )
        else:
            session.current_question = ""

        return AnswerResponse(
            next_question=session.current_question,
            score=overall_score,
            feedback=feedback,
            repetition=is_repeated,
            repetition_similarity=round(sim_score, 3),
            contradiction=contradiction,
            questions_asked=session.questions_asked,
            interview_complete=interview_complete,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Groq API rate limit reached. Please wait a moment and try again.",
            )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ── Serve Frontend ────────────────────────────────────────────────────────────


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/style.css")
async def serve_css():
    return FileResponse("frontend/style.css", media_type="text/css")


@app.get("/script.js")
async def serve_js():
    return FileResponse("frontend/script.js", media_type="application/javascript")
