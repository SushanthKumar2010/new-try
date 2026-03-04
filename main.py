import os
import asyncio
import json
import base64
import io
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

# =====================================================
# ENV
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Teengro AI Backend", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIG
# =====================================================
ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}

SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "text/plain",
}

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# MODELS
# =====================================================
MODEL_MAP = {
    "t1": "gemini-2.5-flash-lite",
    "t2": "gemini-2.5-flash",   # upgraded from preview
}

# =====================================================
# HEALTH
# =====================================================
@app.get("/")
def root():
    return {"status": "running", "version": "4.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api_key_present": bool(GEMINI_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =====================================================
# FILE UPLOAD
# =====================================================
async def upload_file_to_gemini(raw_bytes: bytes, mime_type: str, name: str):
    file_obj = io.BytesIO(raw_bytes)
    file_obj.name = name
    uploaded = client.files.upload(
        file=file_obj,
        config=types.UploadFileConfig(mime_type=mime_type, display_name=name),
    )
    for _ in range(30):
        if uploaded.state.name == "ACTIVE":
            return uploaded.uri, uploaded.name
        if uploaded.state.name == "FAILED":
            raise Exception(f"File processing failed: {name}")
        await asyncio.sleep(1)
        uploaded = client.files.get(name=uploaded.name)
    raise Exception(f"File processing timeout: {name}")

# =====================================================
# PROMPT BUILDER  (slim — ~60% fewer tokens)
# =====================================================
def build_prompt(board: str, class_level: str, subject: str, chapter: str, question: str) -> str:
    return f"""You are an expert {board} Class {class_level} {subject} teacher helping a student understand their syllabus.

Board: {board} | Class: {class_level} | Subject: {subject} | Chapter: {chapter}
Student question: \"{question}\"

ANSWER RULES (follow exactly):
1. Start DIRECTLY with the answer — no greetings, no "Hello", no preamble, no "Sure!".
2. Plain text only. No Markdown, no HTML, no LaTeX, no bullet symbols, no emojis.
3. Maths: school style only. Examples: sin 30° = 1/2  |  √2  |  x² + 3x = 0  |  3/4
4. Highlight key terms: wrap important formulas, definitions, and the final answer in *single asterisks*.
   - You may highlight multiple things, e.g. *formula*, *definition*, *final answer*.
5. Multi-step problems: number each step (1. 2. 3. ...).
6. Start definitions with the exact board-level definition first, then explain.
7. Keep answers concise but conceptually complete — match {board} exam answer style.
8. If a file/image is attached: analyse it carefully and solve all visible questions step by step.
9. Never repeat the question back. Never add a conclusion like "Hope this helps!"."""

# =====================================================
# ASK ROUTE
# =====================================================
@app.post("/api/ask")
async def ask_question(payload: dict):
    board        = (payload.get("board")       or "ICSE").strip().upper()
    class_level  = (payload.get("class_level") or "10").strip()
    subject      = (payload.get("subject")     or "General").strip()
    chapter      = (payload.get("chapter")     or "General").strip()
    question     = (payload.get("question")    or "").strip()
    model_key    = (payload.get("model")       or "t1").lower()
    files        =  payload.get("files")       or []

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")
    if not question and not files:
        raise HTTPException(status_code=400, detail="Question or file required")
    if not question and files:
        question = "Please analyse this and answer any questions visible."

    model_name = MODEL_MAP.get(model_key, MODEL_MAP["t1"])
    prompt_text = build_prompt(board, class_level, subject, chapter, question)

    # ── Build multimodal contents ──
    contents = []
    uploaded_file_names = []
    file_errors = []

    for f in files:
        mime = f.get("mimeType", "")
        b64  = f.get("base64",   "")
        name = f.get("name",     "file")
        if not b64 or not mime:
            continue
        if mime not in SUPPORTED_MIME_TYPES:
            file_errors.append(f"Unsupported: {name} ({mime})")
            continue
        try:
            raw_bytes = base64.b64decode(b64)
        except Exception as e:
            file_errors.append(f"Decode error: {name}")
            continue

        if mime == "application/pdf":
            try:
                uri, fname = await upload_file_to_gemini(raw_bytes, "application/pdf", name)
                contents.append(types.Part.from_uri(uri=uri, mime_type="application/pdf"))
                uploaded_file_names.append(fname)
            except Exception as e:
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except Exception:
                    file_errors.append(f"PDF failed: {name}")
        elif mime == "text/plain":
            try:
                prompt_text += f"\n\n[File: {name}]\n{raw_bytes.decode('utf-8', errors='replace')}"
            except Exception:
                file_errors.append(f"Text read failed: {name}")
        else:
            try:
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
            except Exception:
                file_errors.append(f"Image failed: {name}")

    if file_errors:
        prompt_text += "\n\n[Note — some files failed: " + "; ".join(file_errors) + "]"

    contents.append(types.Part.from_text(text=prompt_text))

    # ── Stream generator ──
    async def stream():
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
            )
            loop = asyncio.get_event_loop()
            get_next = lambda: next(response_stream, None)

            chunk_count = 0
            while True:
                try:
                    chunk = await loop.run_in_executor(None, get_next)
                    if chunk is None:
                        break
                    text = chunk.text or ""
                    if text:
                        chunk_count += 1
                        yield f"data: {json.dumps(text)}\n\n"
                except StopIteration:
                    break
                except Exception as e:
                    if chunk_count == 0:
                        raise
                    break

            yield "event: end\ndata: done\n\n"

        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower():
                yield "event: error\ndata: API quota exceeded — try again later.\n\n"
            elif "api key" in msg.lower():
                yield "event: error\ndata: API configuration error.\n\n"
            else:
                yield f"event: error\ndata: {msg}\n\n"
        finally:
            for fname in uploaded_file_names:
                try:
                    client.files.delete(name=fname)
                except Exception:
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

# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
