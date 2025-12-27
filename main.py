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

MODEL_NAME = "gemini-2.0-flash-lite"

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
- Use ICSE Class 10 level language and methods.
- Show all important working (for Maths/Physics/Chem).
- Mention common mistakes if relevant.
- Keep the answer structured and exam-focused.
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




