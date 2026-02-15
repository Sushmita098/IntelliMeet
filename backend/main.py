"""
Meeting Transcript Analyzer - FastAPI Backend
Step 1: Connectivity & Foundation
Step 2: Ingestion & Vector Memory
"""
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pymongo import MongoClient
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
_embedding_client = None

# --- MONGODB ---
_mongo_client = None
_collection = None

CHUNK_SIZE = 800
DB_NAME = "meeting_analyzer_db"
COLLECTION_NAME = "transcripts"


def get_mongo_collection():
    """Lazy-init MongoDB collection."""
    global _mongo_client, _collection
    if _collection is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise HTTPException(
                status_code=500,
                detail="MONGO_URI must be set in .env for upload/ask",
            )
        _mongo_client = MongoClient(mongo_uri)
        _collection = _mongo_client[DB_NAME][COLLECTION_NAME]
    return _collection


def get_embedding_client():
    """Lazy-init Azure OpenAI client for embeddings (uses classic API version)."""
    global _embedding_client
    if _embedding_client is None:
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise HTTPException(
                status_code=500,
                detail="AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set in .env",
            )
        # Embeddings use classic API; Responses API (2025.x) may not support embeddings
        api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-08-01-preview")
        _embedding_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint.rstrip("/"),
        )
    return _embedding_client


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
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        _azure_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint.rstrip("/"),
        )
    return _azure_client


@app.get("/")
async def root():
    """Root route - redirects to docs."""
    return {"message": "Meeting Transcript Analyzer API", "docs": "/docs"}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into ~chunk_size character chunks."""
    if not text or not text.strip():
        return []
    text = text.strip()
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_embedding(text: str) -> list[float]:
    """Generate 1536-dimension embedding for text using ADA model."""
    client = get_embedding_client()
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
    response = client.embeddings.create(
        input=[text],
        model=deployment,
    )
    return response.data[0].embedding


@app.post("/upload")
async def upload_transcript(file: UploadFile = File(...)):
    """
    Upload a .txt transcript file. Chunks text, generates embeddings, and stores in MongoDB.
    """
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted")

    try:
        content = (await file.read()).decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File must be UTF-8 text: {e}")

    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="File is empty or contains no text")

    collection = get_mongo_collection()
    documents = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        documents.append({
            "text": chunk,
            "embedding": embedding,
            "filename": file.filename,
        })

    if documents:
        collection.insert_many(documents)

    return {"message": f"Successfully processed {len(documents)} chunks", "chunks": len(documents)}


@app.get("/health")
async def health_check():
    """Health check endpoint for connectivity verification."""
    return {"status": "ok", "message": "Backend is running"}


def _call_azure_llm(client, deployment: str) -> str:
    """
    Call Azure OpenAI. Uses Responses API for 2025.x, Chat Completions for legacy.
    """
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    prompt = "Say hello and confirm you are connected. Use one short sentence."

    # Use Responses API for 2025.x (Azure OpenAI newer API)
    if api_version.startswith("2025"):
        response = client.responses.create(
            model=deployment,
            input=prompt,
        )
        return response.output_text

    # Chat Completions API (legacy)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


@app.post("/ask-basic")
async def ask_basic():
    """
    Basic LLM ping: sends a hardcoded prompt to Azure GPT and returns the response.
    Used to verify Azure OpenAI connectivity.
    Supports both Responses API (2025.x) and Chat Completions (legacy).
    """
    try:
        client = get_azure_client()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-mini")
        answer = _call_azure_llm(client, deployment)
        return {"answer": answer}
    except Exception as e:
        err_msg = str(e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=err_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
