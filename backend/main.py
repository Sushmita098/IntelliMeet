"""
Meeting Transcript Analyzer - FastAPI Backend
Step 1: Connectivity & Foundation
Step 2: Ingestion & Vector Memory
Step 3: Retrieval Augmented Generation (RAG)
Step 5: LangChain RAG-as-Tool & File-Scoped Chats
Step 6: OAuth & Docker
"""
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
from jose import JWTError, jwt
import bcrypt

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


# --- GLOBAL EXCEPTION HANDLER ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Ensure all exceptions return JSON responses."""
    import traceback
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
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
USERS_COLLECTION_NAME = "users"

# --- OAuth / JWT Configuration ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-min-32-chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

security = HTTPBearer()


def get_mongo_client():
    """Get MongoDB client."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise HTTPException(status_code=500, detail="MONGO_URI must be set in .env")
    return MongoClient(mongo_uri)


def get_mongo_collection():
    """Lazy-init MongoDB collection."""
    global _mongo_client, _collection
    if _collection is None:
        _mongo_client = get_mongo_client()
        _collection = _mongo_client[DB_NAME][COLLECTION_NAME]
    return _collection


def get_users_collection():
    """Get users collection."""
    client = get_mongo_client()
    return client[DB_NAME][USERS_COLLECTION_NAME]


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


# --- OAuth / Authentication Functions ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Hash a password. Bcrypt has a 72-byte limit."""
    # Ensure password is a string
    if not isinstance(password, str):
        password = str(password)
    # Bcrypt has a 72-byte limit; truncate if necessary
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate to 72 bytes (bcrypt's hard limit)
        password_bytes = password_bytes[:72]
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token and return user info."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"user_id": user_id, "email": payload.get("email")}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=72)
    name: str = Field(..., min_length=1)


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


@app.post("/auth/register")
async def register(user_data: UserRegister):
    """Register a new user."""
    try:
        users_collection = get_users_collection()
        if users_collection.find_one({"email": user_data.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed_password = get_password_hash(user_data.password)
        user = {
            "email": user_data.email,
            "hashed_password": hashed_password,
            "name": user_data.name,
            "created_at": datetime.utcnow().isoformat(),
        }
        users_collection.insert_one(user)
        return {"message": "User registered successfully", "email": user_data.email}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login")
async def login(user_data: UserLogin):
    """Login and get access token."""
    try:
        users_collection = get_users_collection()
        user = users_collection.find_one({"email": user_data.email})
        if not user or not verify_password(user_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        access_token = create_access_token(data={"sub": str(user["_id"]), "email": user["email"]})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"email": user["email"], "name": user.get("name", "")},
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.get("/auth/me")
async def get_current_user(user_info: dict = Depends(verify_token)):
    """Get current authenticated user info."""
    return user_info


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
async def upload_transcript(
    file: UploadFile = File(...), user_info: dict = Depends(verify_token)
):
    """
    Upload a .txt transcript file. Chunks text, generates embeddings, and stores in MongoDB.
    Requires authentication.
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
    user_id = user_info["user_id"]
    documents = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        documents.append({
            "text": chunk,
            "embedding": embedding,
            "filename": file.filename,
            "user_id": user_id,
        })

    if documents:
        collection.insert_many(documents)

    return {
        "message": f"Successfully processed {len(documents)} chunks",
        "chunks": len(documents),
        "filename": file.filename,
    }


@app.get("/files")
async def list_files(user_info: dict = Depends(verify_token)):
    """List unique filenames (files) that have been uploaded and indexed by the current user."""
    try:
        collection = get_mongo_collection()
        user_id = user_info["user_id"]
        files = collection.distinct("filename", {"user_id": user_id})
        return {"files": [f for f in files if f]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def _vector_search_fallback(
    collection,
    query_embedding: list[float],
    limit: int = 5,
    filename: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Fallback when $vectorSearch is unavailable. Fetches docs, optionally filtered by filename and user_id.
    """
    filt = {"embedding": {"$exists": True}}
    if filename:
        filt["filename"] = filename
    if user_id:
        filt["user_id"] = user_id
    docs = list(collection.find(filt, {"text": 1, "embedding": 1}))
    if not docs:
        return []
    scored = [(doc, _cosine_similarity(query_embedding, doc["embedding"])) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"text": doc["text"]} for doc, _ in scored[:limit]]


def search_transcript_scoped(
    query: str, filename: str | None = None, limit: int = 5, user_id: str | None = None
) -> str:
    """
    File-scoped RAG search: vectorize query, search chunks (optionally by filename and user_id), return context.
    Used by LangChain RAG tool and /chat endpoint.
    """
    if not filename:
        raise ValueError("filename is required for file-scoped search")
    query_embedding = get_embedding(query)
    collection = get_mongo_collection()
    results = None
    try:
        filter_dict = {"filename": filename}
        if user_id:
            filter_dict["user_id"] = user_id
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": limit,
                    "filter": filter_dict,
                }
            },
            {"$project": {"text": 1, "_id": 0}},
        ]
        results = list(collection.aggregate(pipeline))
    except Exception:
        results = _vector_search_fallback(
            collection, query_embedding, limit=limit, filename=filename, user_id=user_id
        )
    if not results:
        return "No relevant content found for this query in the transcript."
    return "\n\n".join(r["text"] for r in results if r.get("text"))


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
async def ask_question(req: AskRequest, user_info: dict = Depends(verify_token)):
    """
    RAG endpoint: vectorizes query, searches MongoDB for relevant chunks, generates answer.
    Requires authentication. Searches only the current user's transcripts.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    user_id = user_info["user_id"]

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
                        "filter": {"user_id": user_id},
                    }
                },
                {"$project": {"text": 1, "_id": 0}},
            ]
            results = list(collection.aggregate(pipeline))
        except Exception:
            # Fallback for local MongoDB or when Vector Search Index is not configured
            results = _vector_search_fallback(collection, query_embedding, limit=5, user_id=user_id)

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


# --- Step 5: LangChain RAG-as-Tool & File-Scoped Chats ---

def _create_rag_tool(filename: str, user_id: str | None = None):
    """Create a LangChain tool for RAG search scoped to a specific file and user."""
    from langchain_core.tools import tool

    @tool
    def search_transcript(query: str) -> str:
        """ONLY use this tool when the user's question is specifically about the meeting transcript content (e.g., what was discussed, decisions made, action items, participants' statements, topics covered in the meeting). 
        
        DO NOT use this tool for:
        - General knowledge questions (e.g., "what is the capital of India")
        - Questions about topics not related to the meeting
        - Questions you can answer from your own knowledge
        
        If the question is not about the meeting transcript, answer directly without using this tool.
        If you use this tool and find no relevant content, inform the user that the information is not in the transcript."""
        return search_transcript_scoped(query, filename=filename, limit=5, user_id=user_id)

    return search_transcript


def _get_langchain_llm():
    """Create LangChain AzureChatOpenAI (uses Chat Completions API)."""
    from langchain_openai import AzureChatOpenAI

    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-mini")
    # LangChain uses Chat Completions; use classic API if 2025 Responses API causes issues
    api_version = os.getenv("AZURE_OPENAI_AGENT_API_VERSION", "2024-08-01-preview")
    # Note: Some Azure OpenAI models only support temperature=1 (default)
    # Remove temperature parameter to use default, or set to 1 if needed
    return AzureChatOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,
        # temperature removed - using model default (1)
    )


_chat_memories: dict[str, list] = {}  # session_id -> list of (HumanMessage, AIMessage)


class ChatRequest(BaseModel):
    """Request body for /chat (file-scoped LangChain agent)."""
    file_id: str = Field(..., description="Filename of the transcript to chat about")
    message: str = Field(..., description="User message")
    session_id: str | None = Field(None, description="Session ID for multi-turn; omit for new chat")


@app.post("/chat")
async def chat(req: ChatRequest, user_info: dict = Depends(verify_token)):
    """
    File-scoped chat with LangChain agent. RAG is exposed as a tool; agent can call it multiple times.
    Requires authentication.
    """
    file_id = (req.file_id or "").strip()
    message = (req.message or "").strip()
    user_id = user_info["user_id"]
    session_id = req.session_id or str(uuid.uuid4())
    session_key = f"{user_id}:{session_id}"

    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # Verify file exists and belongs to user
    collection = get_mongo_collection()
    if collection.count_documents({"filename": file_id, "user_id": user_id}) == 0:
        raise HTTPException(
            status_code=404, detail=f"No transcript found for file: {file_id}"
        )

    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

        llm = _get_langchain_llm()
        tools = [_create_rag_tool(file_id, user_id=user_id)]
        llm_with_tools = llm.bind_tools(tools)

        # System message to guide the agent
        system_message = SystemMessage(content="""You are a helpful assistant that can answer questions about a meeting transcript.

IMPORTANT GUIDELINES:
1. For general knowledge questions (e.g., "what is the capital of India", "who is the president"), answer directly using your own knowledge. DO NOT use the search_transcript tool.

2. ONLY use the search_transcript tool when the user asks about:
   - Content from the meeting transcript (what was discussed, decisions made, action items)
   - Specific statements or topics mentioned in the meeting
   - Information that might be in the uploaded transcript

3. If you use the tool and find no relevant information, tell the user: "I don't have access to that information in the meeting transcript" or "That information is not available in the uploaded transcript."

4. If the question is not about the meeting transcript, answer directly without using any tools.

5. When you do use the tool and find relevant information, cite the transcript snippets in your answer.""")

        # Get or create chat history (scoped by user_id + session_id)
        history = _chat_memories.get(session_key, [])
        # Ensure system message is always first, then add history and new message
        if history and isinstance(history[0], SystemMessage):
            messages = list(history) + [HumanMessage(content=message)]
        else:
            # New conversation or history without system message
            messages = [system_message, HumanMessage(content=message)]

        # Tool loop: invoke LLM; if it calls tools, run them and continue until final answer
        max_turns = 10
        for _ in range(max_turns):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                answer = response.content if isinstance(response.content, str) else str(response.content or "")
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args") or {}
                tool_id = tc.get("id", "")
                tool_func = next((t for t in tools if t.name == tool_name), None)
                if tool_func:
                    result = tool_func.invoke(tool_args)
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id)
                    )
                else:
                    messages.append(
                        ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id)
                    )
        else:
            answer = messages[-1].content if hasattr(messages[-1], "content") else "No response generated."

        # Collect citations: Only include citations if the tool was actually called AND returned relevant content
        citations = []
        tool_was_called = any(isinstance(m, ToolMessage) for m in messages)
        
        if tool_was_called:
            for m in messages:
                if isinstance(m, ToolMessage) and getattr(m, "content", None):
                    # Tool returns chunks joined by "\n\n"; split into individual citations
                    raw = m.content if isinstance(m.content, str) else str(m.content)
                    # Skip if tool returned "no relevant content" message
                    if "No relevant content found" in raw or "not available" in raw.lower():
                        continue
                    for part in raw.split("\n\n"):
                        part = part.strip()
                        if part and len(part) > 20:  # Only include substantial citations
                            citations.append(part)
        
        # Deduplicate while preserving order, limit size for UI
        seen = set()
        unique_citations = []
        for c in citations:
            if c not in seen and len(unique_citations) < 10:
                seen.add(c)
                unique_citations.append(c[:500] + ("..." if len(c) > 500 else ""))

        # Persist history for multi-turn (keep last 20 messages, scoped by user_id + session_id)
        # Include system message if it exists, then keep last 19 user/assistant messages
        persisted_messages = []
        if messages and isinstance(messages[0], SystemMessage):
            persisted_messages.append(messages[0])
            remaining = messages[1:]
        else:
            remaining = messages
        # Keep last 19 messages (plus system message = 20 total)
        persisted_messages.extend(remaining[-19:] if len(remaining) > 19 else remaining)
        _chat_memories[session_key] = persisted_messages

        return {
            "answer": answer or "No response generated.",
            "session_id": session_id,
            "citations": unique_citations,
        }

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
