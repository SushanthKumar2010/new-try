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

# ======================
# APP SETUP
# ======================

app = FastAPI(
    title="ICSE AI Tutor",
    description="FastAPI backend for ICSE Class 10 tutor using Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# GEMINI CLIENT (NEW SDK)
# ======================

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================
# ROUTES
# ======================

@app.get("/")
def root():
    return {"status": "Backend running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

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
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
def ask_icse_question(payload: dict):
    class_level = (payload.get("class_level") or "10").strip()
    subject = (payload.get("subject") or "General").strip()
    chapter = (payload.get("chapter") or "General").strip()
    question = (payload.get("question") or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    prompt = f"""
You are an expert ICSE Class {class_level} tutor.

Subject: {subject}
Chapter: {chapter}

Student Question:
\"\"\"{question}\"\"\"  

Requirements:
- Give a clear, step-by-step solution.
- Use ICSE Class 9 & 10 (depends on the user input) level language and methods.
- Show all important working (for Maths/Physics/Chem).
- Mention common mistakes if relevant.
- Keep the answer structured and exam-focused.

STRICT ANSWERING RULES (VERY IMPORTANT):

1. Use PLAIN TEXT only.
   - NO Markdown
   - NO HTML
   - NO LaTeX
   - NO $, backslashes, or formatting commands

2. Allowed math symbols ONLY:
   - Degrees: 30°
   - Fractions: 1/2
   - Equals sign: =
   - Plus / minus: + −
   - Square root: √

3. Do NOT use:
   - LaTeX-style syntax (\\sin, \\frac, ^, _)
   - Markdown symbols (**, ##, -, etc.)
   - Emojis

4. Write mathematics in NORMAL SCHOOL STYLE.
   Example: sin 30° = 1/2

5. Keep the answer:
   - SHORT
   - CONCEPTUALLY DEEP
   - ICSE Class 10 exam-oriented

6. Structure the answer clearly:
   - Core idea first
   - Explanation in 2 to 4 lines
   - ONE simple example or value if useful
   - Final result or key point

7. Highlight IMPORTANT content using ASTERISKS:
   - Put ONLY the most important words or sentences between SINGLE asterisks like *this*
   - Highlight ONLY key ideas, formulas, definitions, or exam points
   - Do NOT over-highlight

8. Asterisk rules (VERY STRICT):
   - Use ONLY single asterisks *
   - NEVER use double asterisks **
   - NEVER use asterisks for bullet points or decoration
   - Asterisks are mandatory ONLY for highlighting important sentences and words in the answer and not any other thing, no matter what, just focus on pure answers and not other stuff

9. Do NOT mention:
   - AI
   - Formatting rules
   - Instructions

10. Keep language:
   - Simple
   - Clear
   - Calm
   - Exam-focused


Preferred Structure:
- Core idea
- Explanation (2 to 4 lines)
- Final result or value
- Key exam point
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        answer = (response.text or "I could not generate an answer.").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    return {
        "answer": answer,
        "meta": {
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














