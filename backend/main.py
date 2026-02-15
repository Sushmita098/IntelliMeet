"""
Meeting Transcript Analyzer - FastAPI Backend
Step 1: Connectivity & Foundation
Step 2: Ingestion & Vector Memory
Step 3: Retrieval Augmented Generation (RAG)
"""
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
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


class AskRequest(BaseModel):
    """Request body for /ask (RAG)."""
    query: str


VECTOR_INDEX_NAME = "vector_index"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vector_search_fallback(collection, query_embedding: list[float], limit: int = 5) -> list[dict]:
    """
    Fallback when $vectorSearch is unavailable (e.g. local MongoDB without vector index).
    Fetches docs with embeddings and computes cosine similarity in Python.
    """
    docs = list(collection.find({"embedding": {"$exists": True}}, {"text": 1, "embedding": 1}))
    if not docs:
        return []
    scored = [(doc, _cosine_similarity(query_embedding, doc["embedding"])) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"text": doc["text"]} for doc, _ in scored[:limit]]


def _call_azure_rag(context: str, query: str) -> str:
    """
    Call Azure OpenAI with RAG context. Supports Responses API and Chat Completions.
    """
    client = get_azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    prompt = f"""Use the following context from a meeting transcript to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}"""

    if api_version.startswith("2025"):
        response = client.responses.create(
            model=deployment,
            input=prompt,
        )
        return response.output_text

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Answer based on the context. If unknown, say so."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    RAG endpoint: vectorizes query, searches MongoDB for relevant chunks, generates answer.
    Requires Vector Search Index on transcripts collection (see README).
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        # 1. Vectorize query
        query_embedding = get_embedding(query)

        # 2. Vector similarity search (try $vectorSearch first, fallback to Python similarity)
        collection = get_mongo_collection()
        results = None
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": VECTOR_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 5,
                    }
                },
                {"$project": {"text": 1, "_id": 0}},
            ]
            results = list(collection.aggregate(pipeline))
        except Exception:
            # Fallback for local MongoDB or when Vector Search Index is not configured
            results = _vector_search_fallback(collection, query_embedding, limit=5)

        if not results:
            return {
                "answer": "No relevant content found. Upload a transcript first and ensure the Vector Search Index exists."
            }

        context_text = "\n\n".join(r["text"] for r in results if r.get("text"))

        # 3. Generate answer with context
        answer = _call_azure_rag(context_text, query)
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
