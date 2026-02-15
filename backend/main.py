"""
Meeting Transcript Analyzer - FastAPI Backend
Step 1: Connectivity & Foundation
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Meeting Transcript Analyzer")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AZURE OPENAI CLIENT ---
_azure_client = None


def get_azure_client():
    """Lazy-init Azure OpenAI client from env vars."""
    global _azure_client
    if _azure_client is None:
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise HTTPException(
                status_code=500,
                detail="AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set in .env",
            )
        _azure_client = AzureOpenAI(
            api_key=api_key,
            api_version="2025-04-14",
            azure_endpoint=endpoint.rstrip("/"),
        )
    return _azure_client


@app.get("/health")
async def health_check():
    """Health check endpoint for connectivity verification."""
    return {"status": "ok", "message": "Backend is running"}


@app.post("/ask-basic")
async def ask_basic():
    """
    Basic LLM ping: sends a hardcoded prompt to Azure GPT and returns the response.
    Used to verify Azure OpenAI connectivity.
    """
    try:
        client = get_azure_client()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-mini")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
                {"role": "user", "content": "Say hello and confirm you are connected. Use one short sentence."},
            ],
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
