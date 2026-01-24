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
1. Use plain text only (NO LaTeX, NO $ symbols, NO backslashes)
2. Use ONLY necessary math symbols:
   - Degrees: 30°
   - Fractions: 1/2
   - Equals sign: =
   - Plus / minus: + −
3. Do NOT use LaTeX-style commands like \\sin, \\frac, ^, _
4. Write math in normal school style (example: sin 30° = 1/2)
5. Keep the answer SHORT but CONCEPTUALLY DEEP
7. Prefer short bullet points or small paragraphs
8. Explain the core idea first
9. Add ONE simple example or value if useful
10. Be ICSE and exam-oriented
11. Do NOT use emojis
12. Do NOT mention AI or formatting rules
14. use √ instead of sqrt
15. highlight/bold the letters/sentences which are important, instead of using an asterisk
16.Highlight ONLY the most important words or sentences
17. Put important text ONLY between single asterisks like *this*
18. Do NOT use double asterisks **
19. Do NOT use Markdown
20. Do NOT use LaTeX, $, backslashes, or HTML
21. Use asterisks ONLY for highlighting (never for anything else)
22. Keep answers short, clear, and exam-oriented

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












