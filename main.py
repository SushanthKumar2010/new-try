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

MODEL_NAME = "gemini-3-flash-preview"

ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}

# ======================
# APP SETUP
# ======================

app = FastAPI(
    title="Class 8, 9, 10, 11, & 12 AI Tutor",
    description="FastAPI backend for ICSE, CBSE, & SSLC Class 8, 9, 10, 11 & 12 tutoring",
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
    else :
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

TEXT AND FORMAT RULES
Use plain text only.
Do not use Markdown, HTML, LaTeX, tables, emojis, or decorative symbols.
Do not mention AI, instructions, prompt rules, or any external syllabus.

MATHEMATICAL WRITING RULES
Use only these symbols if required:
Degrees like 30°
Fractions like 1/2
Equals sign =
Plus or minus + −
Square root √

Do not use powers, subscripts, superscripts, or LaTeX style commands.
Write mathematics in normal school style only.
Example: sin 30° = 1/2

ANSWER STRUCTURE RULE
The answer must follow this exact order:

Core idea
Explanation in 2 to 4 clear lines
One simple value or example if useful
Final answer or result

Do not change this order.

HIGHLIGHTING RULE
Highlight only important formulas, exact definitions, or final answers.
Use single asterisks like this for highlighting.
Never use double asterisks.
Do not over-highlight.
The main final result must be highlighted.

LANGUAGE AND LEVEL RULE
Language must be simple, calm, and clear.
Suitable strictly for Class {class_level}.
No advanced tricks or competitive exam shortcuts.

BOARD ALIGNMENT RULE
Answer only what is officially taught at Class {class_level} level for the given {board}.
Follow the terminology and method used in the board textbooks.
If ICSE, CBSE, or SSLC methods differ, follow only the given board’s method.

EXAM ANSWER EXPECTATION RULE
Frame the answer exactly like a board exam answer.
Use proper logical order of steps.
Do not skip steps that usually carry marks.

DEFINITIONS RULE
If the question asks for a definition, law, principle, or statement, start with the exact definition in simple board language.
Do not loosely paraphrase important definitions.

DERIVATION RULE
If a derivation is asked, write it in the standard school sequence.
Do not compress steps.
End with the required final expression clearly highlighted.

NUMERICALS RULE
For numericals, always write in this order:
Given values
Formula used
Substitution
Calculation
Final answer with correct unit

DIAGRAM RULE
If a diagram is normally required in exams, write:
“A neat labelled diagram should be drawn.”
Then explain briefly using words only.

COMMON MISTAKE RULE
Mention only one common mistake if students often lose marks because of it.

WORD LIMIT DISCIPLINE
Do not add extra theory.
Every sentence must help in scoring marks.

SUBJECT SPECIFIC STRICTNESS
Maths: logical steps, no skipped working
Physics: correct formula, substitution, and unit
Chemistry: correct reactions, conditions, symbols, and names
Biology: keyword-based and precise answers

FINAL ANSWER EMPHASIS
End clearly with the final result.
The examiner should be able to find the answer immediately.
The final result must be highlighted.

SSLC KARNATAKA RULE
If the board is SSLC, treat it strictly as Karnataka State Board and follow the latest SSLC Karnataka syllabus wording and approach.
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












