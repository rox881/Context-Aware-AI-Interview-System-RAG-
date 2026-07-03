/* ═══════════════════════════════════════════════════════════════════════════
   AI Interview System — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── DOM References ───────────────────────────────────────────────────

    const welcomeScreen     = document.getElementById("welcome-screen");
    const messagesContainer = document.getElementById("messages-container");
    const loadingIndicator  = document.getElementById("loading-indicator");
    const inputBar          = document.getElementById("input-bar");
    const answerInput       = document.getElementById("answer-input");
    const btnSend           = document.getElementById("btn-send");
    const btnNewSession     = document.getElementById("btn-new-session");
    const btnStartWelcome   = document.getElementById("btn-start-welcome");
    const statusText        = document.getElementById("status-text");
    const questionCounter   = document.getElementById("question-counter");
    const charCounter       = document.getElementById("char-counter");
    const chatArea          = document.getElementById("chat-area");

    // ── Configuration ────────────────────────────────────────────────────

    // Backend API base URL — ensures requests always reach FastAPI (port 8000)
    // even when the HTML is opened via Live Server (port 5500) or file://.
    const API_BASE = "http://localhost:8000";

    // ── State ────────────────────────────────────────────────────────────

    let sessionId       = null;
    let questionsAsked  = 0;
    const maxQuestions  = 10;
    let isProcessing    = false;

    // ── Utilities ────────────────────────────────────────────────────────

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatArea.scrollTop = chatArea.scrollHeight;
        });
    }

    function setLoading(show) {
        loadingIndicator.classList.toggle("hidden", !show);
        isProcessing = show;
        btnSend.disabled = show || answerInput.value.trim() === "";
        if (show) scrollToBottom();
    }

    function updateCounter() {
        questionCounter.textContent = `Question ${questionsAsked} / ${maxQuestions}`;
    }

    function getScoreClass(score) {
        if (score >= 7) return "high";
        if (score >= 4) return "medium";
        return "low";
    }

    // ── Message Rendering ────────────────────────────────────────────────

    function addAiMessage(text) {
        const div = document.createElement("div");
        div.className = "message ai";
        div.innerHTML = `
            <span class="msg-label">🤖 Interviewer</span>
            <div class="msg-bubble">${escapeHtml(text)}</div>
        `;
        messagesContainer.appendChild(div);
        scrollToBottom();
    }

    function addUserMessage(text) {
        const div = document.createElement("div");
        div.className = "message user";
        div.innerHTML = `
            <span class="msg-label">👤 You</span>
            <div class="msg-bubble">${escapeHtml(text)}</div>
        `;
        messagesContainer.appendChild(div);
        scrollToBottom();
    }

    function addEvalCard(data) {
        const scoreClass = getScoreClass(data.score);
        const div = document.createElement("div");
        div.className = "eval-card";

        let alertsHtml = "";

        // Repetition alert
        if (data.repetition) {
            alertsHtml += `
                <div class="alert-badge repetition">
                    🔁 Repetition Detected (similarity: ${(data.repetition_similarity * 100).toFixed(0)}%)
                </div>
            `;
        }

        // Contradiction alert
        const ct = data.contradiction || {};
        const isContradiction = String(ct.contradiction).toLowerCase() === "true";
        if (isContradiction) {
            alertsHtml += `
                <div class="alert-badge contradiction">
                    ⚠️ Contradiction (${ct.severity || "unknown"}): ${escapeHtml(ct.explanation || "")}
                </div>
            `;
        }

        if (!data.repetition && !isContradiction) {
            alertsHtml += `
                <div class="alert-badge no-issue">✅ No issues detected</div>
            `;
        }

        div.innerHTML = `
            <div class="eval-header">
                <span class="label">📊 Evaluation</span>
                <span class="score-badge ${scoreClass}">${data.score.toFixed(1)} / 10</span>
            </div>
            <p class="eval-feedback">${escapeHtml(data.feedback)}</p>
            ${alertsHtml}
        `;
        messagesContainer.appendChild(div);
        scrollToBottom();
    }

    function addInterviewComplete() {
        const div = document.createElement("div");
        div.className = "interview-complete";
        div.innerHTML = `
            <h3>🎉 Interview Complete!</h3>
            <p>You've answered all ${maxQuestions} questions. Click "New Interview" to start again.</p>
        `;
        messagesContainer.appendChild(div);
        scrollToBottom();
    }

    function escapeHtml(text) {
        const el = document.createElement("span");
        el.textContent = text;
        return el.innerHTML;
    }

    // ── API Calls ────────────────────────────────────────────────────────

    async function startInterview() {
        // Reset UI
        messagesContainer.innerHTML = "";
        welcomeScreen.style.display = "none";
        inputBar.classList.remove("hidden");
        questionsAsked = 0;
        updateCounter();
        statusText.textContent = "Interview Active";

        setLoading(true);

        try {
            const res = await fetch(API_BASE + "/start", { method: "POST" });
            if (!res.ok) throw new Error(`Server error: ${res.status}`);

            const data = await res.json();
            sessionId = data.session_id;

            setLoading(false);
            addAiMessage(data.question);
            answerInput.focus();
        } catch (err) {
            setLoading(false);
            addAiMessage(`❌ Failed to start interview: ${err.message}`);
        }
    }

    async function submitAnswer() {
        const answer = answerInput.value.trim();
        if (!answer || isProcessing || !sessionId) return;

        addUserMessage(answer);
        answerInput.value = "";
        charCounter.textContent = "0 / 2000";
        autoResizeTextarea();
        btnSend.disabled = true;

        setLoading(true);

        try {
            const res = await fetch(API_BASE + "/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, answer }),
            });

            if (res.status === 429) {
                setLoading(false);
                addAiMessage("⏳ Rate limit reached. Please wait a moment and try again.");
                return;
            }

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${res.status}`);
            }

            const data = await res.json();
            questionsAsked = data.questions_asked;
            updateCounter();

            setLoading(false);

            // Show evaluation
            addEvalCard(data);

            if (data.interview_complete) {
                addInterviewComplete();
                inputBar.classList.add("hidden");
                statusText.textContent = "Completed";
            } else {
                // Show next question
                addAiMessage(data.next_question);
                answerInput.focus();
            }
        } catch (err) {
            setLoading(false);
            addAiMessage(`❌ Error: ${err.message}`);
        }
    }

    // ── Textarea Auto-resize ─────────────────────────────────────────────

    function autoResizeTextarea() {
        answerInput.style.height = "auto";
        answerInput.style.height = Math.min(answerInput.scrollHeight, 120) + "px";
    }

    // ── Event Listeners ──────────────────────────────────────────────────

    btnStartWelcome.addEventListener("click", startInterview);
    btnNewSession.addEventListener("click", startInterview);

    btnSend.addEventListener("click", submitAnswer);

    answerInput.addEventListener("input", () => {
        const len = answerInput.value.length;
        charCounter.textContent = `${len} / 2000`;
        btnSend.disabled = answerInput.value.trim() === "" || isProcessing;
        autoResizeTextarea();
    });

    answerInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitAnswer();
        }
    });
})();
