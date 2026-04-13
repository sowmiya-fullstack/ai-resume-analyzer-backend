from fastapi import FastAPI, UploadFile, File, Form
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
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form("")
):
    content = await file.read()
    text = extract_text(content, file.filename)

    if job_description.strip():
        prompt = f"""Review this resume and compare it with the job description.
        
Resume:
{text}

Job Description:
{job_description}

Give feedback with these sections:
1. Overall Impression
2. Strengths
3. Areas to Improve
4. JD Match Score (out of 100) with explanation
5. Top 3 Recommendations to improve match"""
    else:
        prompt = f"""Review this resume and give structured feedback:
1. Overall Impression
2. Strengths  
3. Areas to Improve
4. ATS Score (out of 100)
5. Top 3 Recommendations

Resume:
{text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a professional resume reviewer and career coach."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"feedback": response.choices[0].message.content}