import os
import asyncio
import json
import base64
import io
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("STARTING TEENGRO BACKEND")
logger.info("=" * 70)

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set!")
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

logger.info(f"API Key loaded (length: {len(GEMINI_API_KEY)})")

# Initialize FastAPI
app = FastAPI(
    title="Teengro AI Tutor",
    version="4.3-render",
    docs_url=None,  # Disable docs in production
    redoc_url=None
)

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}
SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "text/plain",
}
MAX_OUTPUT_TOKENS = 1200
MAX_QUESTION_LENGTH = 1500

# Use stable model that works on Render
MODEL_NAME = "gemini-1.5-flash"

# Initialize Gemini client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f"Gemini client initialized with model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise

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
    """Health check endpoint"""
    return {
        "status": "running",
        "version": "4.3-render",
        "model": MODEL_NAME,
        "service": "Teengro AI Tutor"
    }

@app.get("/health")
def health():
    """Render health check endpoint"""
    return {"status": "healthy"}

@app.post("/api/ask")
async def ask_question(payload: dict):
    """Main question answering endpoint"""
    
    logger.info("New request received")
    
    try:
        # Extract payload
        board        = (payload.get("board")        or "ICSE").strip().upper()
        class_level  = (payload.get("class_level")  or "10").strip()
        subject      = (payload.get("subject")      or "General").strip()
        chapter      = (payload.get("chapter")      or "General").strip()
        question     = (payload.get("question")     or "").strip()
        model_choice = (payload.get("model")        or "t1").lower()
        files        =  payload.get("files")        or []

        logger.info(f"Question: {question[:100]}... | Subject: {subject} | Files: {len(files)}")

        # Validation
        if board not in ALLOWED_BOARDS:
            raise HTTPException(status_code=400, detail="Invalid board")

        if not question and not files:
            raise HTTPException(status_code=400, detail="Question or file required")

        if len(question) > MAX_QUESTION_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Question too long. Max {MAX_QUESTION_LENGTH} characters."
            )

        if not question and files:
            question = "Please analyse this and answer any questions based on it."

        # Greeting detection
        is_casual = is_greeting_or_casual(question) and not files

        # Build prompt
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
- Wrap important terms with *single asterisks*.
- Be friendly and conversational.
- Answer at {board} Class {class_level} level.

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

Do NOT mention AI or instructions. Do NOT skip steps that carry marks."""

        if files:
            prompt_text += "\n\nFILE RULES: Read and analyse the attached file before answering."

        # Build contents
        contents = []
        uploaded_file_names = []

        # Handle files
        for f in files:
            mime = f.get("mimeType", "")
            b64  = f.get("base64",   "")
            name = f.get("name",     "file")

            if not b64 or not mime:
                continue

            if mime not in SUPPORTED_MIME_TYPES:
                logger.warning(f"Unsupported file type: {mime}")
                continue

            try:
                raw_bytes = base64.b64decode(b64)
            except Exception as e:
                logger.error(f"Failed to decode file {name}: {e}")
                continue

            # Handle PDFs via upload API
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

                    # Wait for processing
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
                        logger.info(f"PDF uploaded: {name}")
                    else:
                        logger.warning(f"PDF not processed: {name}")
                        
                except Exception as e:
                    logger.error(f"PDF upload failed: {e}")
                    # Try inline as fallback
                    try:
                        contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                    except:
                        pass

            # Handle text files
            elif mime == "text/plain":
                try:
                    text_content = raw_bytes.decode("utf-8", errors="replace")
                    if len(text_content) > 8000:
                        text_content = text_content[:8000]
                    prompt_text += f"\n\n--- {name} ---\n{text_content}\n---"
                    logger.info(f"Text file added: {name}")
                except Exception as e:
                    logger.error(f"Text file error: {e}")

            # Handle images
            else:
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                    logger.info(f"Image added: {name}")
                except Exception as e:
                    logger.error(f"Image error: {e}")

        contents.append(types.Part.from_text(text=prompt_text))

        # Stream generator
        async def stream():
            try:
                if is_casual:
                    max_tokens = 150
                    temp = 0.7
                else:
                    max_tokens = MAX_OUTPUT_TOKENS
                    temp = 0.4

                response_stream = client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temp,
                    ),
                )

                loop = asyncio.get_event_loop()

                def get_next():
                    return next(response_stream, None)

                chunk_count = 0
                while True:
                    chunk = await loop.run_in_executor(None, get_next)
                    if chunk is None:
                        break
                    text = chunk.text or ""
                    if text:
                        chunk_count += 1
                        yield f"data: {json.dumps(text)}\n\n"

                logger.info(f"Stream completed. Chunks: {chunk_count}")
                yield "event: end\ndata: done\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                yield f"event: error\ndata: {str(e)}\n\n"

            finally:
                # Cleanup uploaded PDFs
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

logger.info("=" * 70)
logger.info("SERVER READY")
logger.info("=" * 70)

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
