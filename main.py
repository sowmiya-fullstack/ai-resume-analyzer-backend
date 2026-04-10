from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os
import fitz

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        return file_bytes.decode("utf-8", errors="ignore")

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files allowed")
    
    content = await file.read()
    
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 5MB")
    
    text = extract_text(content, file.filename)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a professional resume reviewer. Give structured feedback with these sections: 1) Overall Impression 2) Strengths 3) Areas to Improve 4) ATS Score (out of 100) 5) Top 3 Recommendations"},
            {"role": "user", "content": f"Review this resume:\n\n{text}"}
        ]
    )
    
    return {"feedback": response.choices[0].message.content}