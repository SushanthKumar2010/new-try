import os
import asyncio
import json
import base64
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

app = FastAPI(title="AI Tutor Backend", version="4.1-safe")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}
SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "text/plain",
}
MAX_OUTPUT_TOKENS = 1200
MAX_QUESTION_LENGTH = 1500

client = genai.Client(api_key=GEMINI_API_KEY)

def is_greeting_or_casual(text: str) -> bool:
    """Detect if message is just a greeting or casual chat"""
    if not text:
        return False
    text_lower = text.lower().strip()
    greetings = {
        "hello", "hi", "hey", "hola", "namaste", "good morning", 
        "good afternoon", "good evening", "what's up", "whats up",
        "sup", "yo", "wassup", "how are you", "how r u", "hru"
    }
    if text_lower in greetings:
        return True
    if any(text_lower.startswith(g) for g in greetings) and len(text.split()) <= 4:
        return True
    return False

@app.get("/")
def root():
    return {"status": "running", "version": "4.1-safe"}

@app.post("/api/ask")
async def ask_question(payload: dict):
    board        = (payload.get("board")        or "ICSE").strip().upper()
    class_level  = (payload.get("class_level")  or "10").strip()
    subject      = (payload.get("subject")      or "General").strip()
    chapter      = (payload.get("chapter")      or "General").strip()
    question     = (payload.get("question")     or "").strip()
    model_choice = (payload.get("model")        or "t1").lower()
    files        =  payload.get("files")        or []

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    if not question and not files:
        raise HTTPException(status_code=400, detail="Question or file required")

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long. Please keep it under {MAX_QUESTION_LENGTH} characters."
        )

    if not question and files:
        question = "Please analyse this and answer any questions based on it."

    # SAFE MODEL SELECTION - using only stable model names
    # Changed from "gemini-2.5-flash-lite-preview-06-17" to "gemini-2.0-flash-exp"
    if model_choice == "t2":
        model_name = "gemini-2.0-flash-exp"  # More stable than 2.5
    else:
        model_name = "gemini-2.0-flash-exp"  # Same for both to ensure stability

    is_casual = is_greeting_or_casual(question) and not files
    
    if is_casual:
        prompt_text = f"""You are a friendly AI tutor for {board} Class {class_level} students.

Student said: "{question}"

Respond naturally and warmly as a helpful tutor would. Keep it brief (1-2 sentences). 
Invite them to ask any {subject} or other subject questions they need help with.
Be conversational and encouraging. Don't list topics or give structured explanations unless they ask a real question."""

    else:
        prompt_text = f"""You are a friendly, expert {board} Class {class_level} {subject} teacher.

Context: Board={board} | Class={class_level} | Subject={subject} | Chapter={chapter}

Student question: \"\"\"{question}\"\"\"

ANSWER RULES:
- Plain text only. No Markdown, LaTeX, HTML, or emojis.
- Math in school style: sin 30° = 1/2, not LaTeX.
- Wrap with *single asterisks* (never **) around: formulas, defined terms, laws, final answers, given values, units.
- Be friendly and conversational. Reference the board/class/subject naturally.
- Answer strictly at {board} Class {class_level} level.
- Frame answer how a board examiner expects it.

STRUCTURE:
1. Core idea (1-2 lines)
2. Explanation (2-4 lines, step-by-step for Maths/Physics/Chemistry)
3. Example if useful
4. *Final answer clearly stated*

SUBJECT-SPECIFIC:
- Maths: show every step.
- Physics: Given → Formula → Substitution → Answer with unit.
- Chemistry: correct reactions, conditions, symbols.
- Biology: keyword-based explanations.

Do NOT mention AI or instructions in your answer. Do NOT add motivation or off-topic facts."""

    if files:
        prompt_text += "\n\nFILE RULES: Read and analyse the attached file before answering. Solve all visible questions step by step."

    contents = []
    uploaded_file_names = []

    for f in files:
        mime = f.get("mimeType", "")
        b64  = f.get("base64",   "")
        name = f.get("name",     "file")

        if not b64 or not mime:
            continue

        if mime not in SUPPORTED_MIME_TYPES:
            continue

        try:
            raw_bytes = base64.b64decode(b64)
        except:
            continue

        if mime == "application/pdf":
            try:
                file_obj = io.BytesIO(raw_bytes)
                file_obj.name = name

                uploaded = client.files.upload(
                    file=file_obj,
                    config=types.UploadFileConfig(
                        mime_type="application/pdf",
                        display_name=name,
                    )
                )

                max_wait = 20
                waited = 0
                while uploaded.state.name == "PROCESSING" and waited < max_wait:
                    await asyncio.sleep(1)
                    waited += 1
                    uploaded = client.files.get(name=uploaded.name)

                if uploaded.state.name == "ACTIVE":
                    contents.append(types.Part.from_uri(
                        uri=uploaded.uri,
                        mime_type="application/pdf"
                    ))
                    uploaded_file_names.append(uploaded.name)
            except:
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except:
                    pass

        elif mime == "text/plain":
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                if len(text_content) > 8000:
                    text_content = text_content[:8000]
                prompt_text += f"\n\n--- {name} ---\n{text_content}\n---"
            except:
                pass

        else:
            try:
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
            except:
                pass

    contents.append(types.Part.from_text(text=prompt_text))

    async def stream():
        try:
            if is_casual:
                max_tokens = 150
                temp = 0.7
            else:
                max_tokens = MAX_OUTPUT_TOKENS
                temp = 0.4

            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temp,
                ),
            )

            loop = asyncio.get_event_loop()

            def get_next():
                return next(response_stream, None)

            while True:
                chunk = await loop.run_in_executor(None, get_next)
                if chunk is None:
                    break
                text = chunk.text or ""
                if text:
                    yield f"data: {json.dumps(text)}\n\n"

            yield "event: end\ndata: done\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

        finally:
            for file_name in uploaded_file_names:
                try:
                    client.files.delete(name=file_name)
                except:
                    pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
