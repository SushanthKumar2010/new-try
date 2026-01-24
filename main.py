import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# ======================
# CONFIG
# ======================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

MODEL_NAME = "gemini-2.5-flash-lite"

ALLOWED_BOARDS = {"ICSE", "CBSE"}

# ======================
# APP SETUP
# ======================

app = FastAPI(
    title="Class 9 & 10 AI Tutor",
    description="FastAPI backend for ICSE and CBSE Class 9 & 10 tutoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# GEMINI CLIENT
# ======================

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================
# ROUTES
# ======================

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- INTRO (FIX) ----------
@app.post("/api/intro")
def get_intro(payload: dict):
    board = (payload.get("board") or "").strip().upper()
    class_level = (payload.get("class_level") or "10").strip()
    subject = (payload.get("subject") or "General").strip()

    if board in ALLOWED_BOARDS:
        board_text = f"{board} Class"
    else:
        board_text = "Class"

    intro = (
        f"Hello! I'm here to help you with your {board_text} "
        f"{class_level} {subject} questions.\n\n"
        "Let's start with your first question. "
        "Please type your problem."
    )

    return {"intro": intro}

# ---------- SIMPLE CHAT ----------
@app.post("/api/chat")
def simple_chat(data: dict):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return {"response": (response.text or "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- MAIN ASK ----------
@app.post("/api/ask")
def ask_question(payload: dict):
    board = (payload.get("board") or "ICSE").strip().upper()
    class_level = (payload.get("class_level") or "10").strip()
    subject = (payload.get("subject") or "General").strip()
    chapter = (payload.get("chapter") or "General").strip()
    question = (payload.get("question") or "").strip()

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    prompt = f"""
You are an expert {board} Class {class_level} teacher.

Board: {board}
Subject: {subject}
Chapter: {chapter}

A student from Class {class_level} has asked the following question:

\"\"\"{question}\"\"\"

Your task is to answer strictly according to the {board} syllabus and exam pattern.

REQUIREMENTS:
- Explain the concept clearly and correctly.
- Use only {board} Class {class_level} level methods.
- Show all important steps and working where required (Maths, Physics, Chemistry).
- Keep the explanation concise but conceptually strong.
- Mention a common mistake ONLY if it is relevant.
- Focus on how answers are expected in board exams.

STRICT ANSWERING RULES (VERY IMPORTANT):

1. Use PLAIN TEXT ONLY.
   - NO Markdown
   - NO HTML
   - NO LaTeX
   - NO emojis
   - NO special formatting commands

2. Allowed mathematical symbols ONLY:
   - Degrees: 30°
   - Fractions: 1/2
   - Equals sign: =
   - Plus or minus: + −
   - Square root: √

3. Do NOT use:
   - LaTeX-style syntax (\\sin, \\frac, ^, _)
   - Markdown symbols (**, ##, -, etc.)

4. Write mathematics in NORMAL SCHOOL STYLE.
   Example: sin 30° = 1/2

5. Keep the answer:
   - SHORT
   - CLEAR
   - CONCEPTUALLY DEEP
   - STRICTLY exam-oriented

6. Follow this STRUCTURE exactly:
   - Core idea
   - Explanation in 2 to 4 lines
   - ONE simple value or example if useful
   - Final answer or result

7. IMPORTANT HIGHLIGHTING RULES:
   - Highlight ONLY the most important formulas, definitions, or final answers
   - Use ONLY SINGLE ASTERISKS like *this*
   - NEVER use double asterisks **
   - NEVER over-highlight
   - The MAIN FINAL RESULT must be inside single asterisks

8. Do NOT mention:
   - AI
   - Instructions
   - Formatting rules
   - Any external syllabus or board

9. Language must be:
   - Simple
   - Calm
   - Clear
   - Suitable for Class {class_level} students
"""


    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "I could not generate an answer."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    return {
        "answer": answer,
        "meta": {
            "board": board,
            "class_level": class_level,
            "subject": subject,
            "chapter": chapter,
        },
    }

# ======================
# LOCAL DEV ONLY
# ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)

