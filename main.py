import base64  # Add this import at the top
from google.genai import types # Add this import at the top

# ... (keep your existing imports and setup) ...

@app.post("/api/ask")
def ask_question(payload: dict):
    board = (payload.get("board") or "ICSE").strip().upper()
    class_level = (payload.get("class_level") or "10").strip()
    subject = (payload.get("subject") or "General").strip()
    chapter = (payload.get("chapter") or "General").strip()
    question = (payload.get("question") or "").strip()
    
    # NEW: Get attachments from the frontend
    attachments = payload.get("attachments") or []

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    # Construct the base prompt (your existing prompt logic)
    system_instructions = f"""
    You are an expert {board} Class {class_level} teacher.
    Board: {board} | Subject: {subject} | Chapter: {chapter}
    
    A student has provided a question. If they have uploaded a photo or PDF, 
    read the content of that file carefully to answer.
    
    (Include all your existing STRICT RULES 1-25 here...)
    """

    # Prepare the "contents" list for Gemini
    # We start with the text prompt
    content_parts = [system_instructions]
    
    if question:
        content_parts.append(f"Student's Question text: {question}")

    # NEW: Process attachments (Images/PDFs)
    for auth in attachments:
        try:
            # The data comes as "data:image/png;base64,iVBOR..."
            # We need to strip the prefix to get the raw base64 string
            header, base64_data = auth["data"].split(",")
            mime_type = auth.get("mime_type") or auth.get("type") # match your JS key
            
            content_parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(base64_data),
                    mime_type=mime_type
                )
            )
        except Exception as e:
            print(f"Error processing attachment: {e}")
            continue

    try:
        # Call Gemini with the list of parts (Text + Images)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content_parts,
        )
        
        answer = (response.text or "").strip()
        if not answer:
            answer = "I could see the files, but I couldn't extract enough information to answer. Please try a clearer photo."
            
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
