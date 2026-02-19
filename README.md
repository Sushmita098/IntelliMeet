# IntelliMeet

Meeting Transcript Analyzer using RAG (Retrieval-Augmented Generation).

## Project Structure

- **backend/** - FastAPI server (Python)
- **frontend/** - React app
- **documents/** - Project plan, technical documentation, and architecture design prompt

## Demo

**[Watch the IntelliMeet demo video](documents/Intellimeet_recording.mp4)** *(click to play)*

## Architecture

### System Overview & Data Flow

![IntelliMeet Architecture and RAG Flow](documents/IntelliMeet_FlowDiagram.png)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info + docs link |
| GET | `/health` | Connectivity check |
| POST | `/upload` | Upload .txt transcript |
| GET | `/files` | List indexed filenames |
| POST | `/ask-basic` | Simple LLM ping (Azure connectivity test) |
| POST | `/ask` | RAG query (global transcript search) - **Requires Auth** |
| POST | `/chat` | File-scoped chat with LangChain agent + RAG tool - **Requires Auth** |
| POST | `/auth/register` | User registration |
| POST | `/auth/login` | User login (returns JWT token) |
| GET | `/auth/me` | Get current user info - **Requires Auth** |
| POST | `/upload` | Upload .txt transcript - **Requires Auth** |
| GET | `/files` | List indexed filenames - **Requires Auth** |

### Components

| Component | Technology | Role |
|-----------|------------|------|
| Frontend | React 19 (CRA) | UI for upload, file selection, chat, RAG search |
| Backend | FastAPI | REST API, chunking, embeddings, RAG, chat |
| Embeddings | Azure OpenAI (text-embedding-ada-002) | 1536-d vectors for chunks and queries |
| LLM | Azure OpenAI (GPT) | Answer generation from RAG context and chat |
| Vector store | MongoDB Atlas | `transcripts` collection with vector index `vector_index` |
| Session store | In-memory dict | `_chat_memories`; ephemeral, not shared across workers |

### MongoDB Schema

**Collection:** `transcripts`

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk text (~800 chars) |
| `embedding` | float[1536] | Vector from ADA model |
| `filename` | string | Source file name |
| `user_id` | string | Owner user ID (for data isolation) |

**Collection:** `users`

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email (unique) |
| `hashed_password` | string | Bcrypt hashed password |
| `name` | string | User's display name |
| `created_at` | string | ISO timestamp |

**Index:** `vector_index` - Vector search index on `embedding`, cosine similarity, 1536 dimensions.

### Architecture Characteristics

- **Single process:** Backend runs as one uvicorn process (no horizontal scaling).
- **Synchronous upload:** Large files may cause timeouts; embedding calls block the request.
- **Stateless API (except chat):** Only `/chat` keeps state (`_chat_memories`).
- **File-scoped RAG:** Chunks filtered by `filename`; chat and RAG tool operate per-file.
- **Authentication:** JWT-based OAuth system with user registration/login (Step 6).
- **Data isolation:** All transcript chunks and chat sessions are scoped by `user_id`.
- **Dockerized:** Production-ready Docker setup with multi-stage builds (Step 6).

## Quick Start (Step 1)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment (already done if following setup)
uv venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt

# Copy .env.example to .env and fill in your values
copy .env.example .env

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm start
```

### 3. Manual: Create `.env`

Copy `backend/.env.example` to `backend/.env` and set:

- `AZURE_OPENAI_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Base URL (e.g. `https://your-resource.openai.azure.com`)
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Chat model deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` - Embedding model deployment (e.g. `text-embedding-ada-002`)
- `AZURE_OPENAI_API_VERSION` - Use `2025-04-01-preview` for Responses API
- `AZURE_OPENAI_AGENT_API_VERSION` - Use `2024-08-01-preview` for LangChain agent (Chat Completions API)
- `MONGO_URI` - MongoDB Atlas connection string (required for Step 2+)
- `JWT_SECRET_KEY` - Secret key for JWT token signing (min 32 characters, required for Step 6+)

### 4. MongoDB Vector Search Index (required for upload / RAG)

In MongoDB Atlas, create a **Vector Search Index** on the `transcripts` collection:

1. Go to your cluster -> Browse Collections -> `meeting_analyzer_db` -> `transcripts`
2. Search Indexes -> Create Search Index -> JSON Editor
3. Use this definition (index name: `vector_index`):

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

## Docker Implementation (Step 6)

### Architecture

The application is containerized using Docker with a multi-service architecture:

- **MongoDB Service:** MongoDB 7.0 with persistent volume storage
- **Backend Service:** FastAPI with Python 3.11-slim (multi-stage build)
- **Frontend Service:** React app built and served via Nginx Alpine

### Key Features

- **Multi-stage builds** for optimized image sizes
- **Health checks** for all services
- **Volume persistence** for MongoDB data
- **Hot reload** for backend development (volume mount)
- **Environment variable** management via `.env` file
- **Network isolation** via Docker bridge network

### Docker Files

- `backend/Dockerfile` - Multi-stage Python build
- `frontend/Dockerfile` - Multi-stage Node.js build with Nginx
- `docker-compose.yml` - Orchestration configuration
- `frontend/nginx.conf` - Production Nginx configuration
- `.dockerignore` files - Optimized build context

See [DOCKER.md](DOCKER.md) for detailed setup and deployment instructions.

## Security Implementation (Step 6)

### Authentication System

**Technology:** JWT (JSON Web Tokens) + Bcrypt password hashing

**Components:**
- **User Registration:** `POST /auth/register` - Creates new user accounts
- **User Login:** `POST /auth/login` - Returns JWT access token
- **Token Verification:** Bearer token authentication via `HTTPBearer` dependency
- **Password Hashing:** Direct bcrypt implementation (handles 72-byte limit)

### Security Features

1. **JWT Tokens:**
   - 7-day expiration
   - Signed with `JWT_SECRET_KEY` (configurable)
   - Contains user ID and email in payload

2. **Password Security:**
   - Bcrypt hashing with salt generation
   - Automatic truncation for passwords exceeding 72 bytes
   - Minimum 6 character requirement

3. **Data Isolation:**
   - All transcript chunks include `user_id` field
   - All queries filtered by authenticated user's `user_id`
   - Chat sessions scoped by `user_id` + `session_id`
   - Users can only access their own data

4. **Protected Endpoints:**
   - `/upload` - Requires authentication
   - `/files` - Requires authentication
   - `/ask` - Requires authentication
   - `/chat` - Requires authentication
   - `/auth/me` - Requires authentication

5. **Frontend Security:**
   - JWT token stored in `localStorage`
   - Authorization header added to all authenticated requests
   - Automatic logout on 401 responses
   - Login/register UI when not authenticated

### Security Considerations

**Implemented:**
- ✅ Password hashing (bcrypt)
- ✅ JWT token-based authentication
- ✅ User data isolation
- ✅ Protected API endpoints
- ✅ Secure password storage in MongoDB


## Q&A

### What would be required to productionize your solution, make it scalable and deploy it on a hyper-scaler such as AWS?

Upload the file and store it in S3, which will trigger a Lambda to do the chunking and embedding, finally storing it in MongoDB (serverless approach). Containerize and build Docker images, push to ECR, and deploy to ECS.

### RAG/LLM approach & decisions: Choices considered and final choice for LLM / embedding model / vector database / orchestration framework, prompt & context management, guardrails, quality, observability, hallucination mitigation, and re-ranking

#### 1. Choices considered and final choice

| Component | Choice | Notes |
|-----------|--------|-------|
| LLM | Azure OpenAI (GPT) | Enterprise-ready; personal preference for OpenAI stack |
| Embedding | `text-embedding-ada-002` | 1536-d vectors; cosine similarity in MongoDB |
| Vector DB | MongoDB Atlas | `$vectorSearch`; single data store for metadata + vectors |
| Orchestration | LangChain | Agent pattern; RAG exposed as a tool the agent calls on demand |
| **Authentication** | **JWT + Bcrypt** | **JWT tokens for stateless auth; bcrypt for password hashing (72-byte limit handled)** |
| **Containerization** | **Docker + Docker Compose** | **Multi-stage builds for optimized images; nginx for frontend; MongoDB included** |
| **Password Hashing** | **Bcrypt (direct)** | **Chose direct bcrypt over passlib to avoid initialization issues; handles 72-byte limit** |
| **Frontend Server** | **Nginx (Alpine)** | **Lightweight, production-ready static file serving with gzip compression** |
| **Backend Server** | **Uvicorn (Python 3.11)** | **ASGI server for FastAPI; multi-stage build reduces image size** |

#### 2. Prompt & context management

- Fixed RAG prompt instructs the model to answer only from context; say "I don't know" if not found.
- Top 5 chunks (~800 chars each) per query; file-scoped retrieval filters by `filename`.
- Agent tool description guides when to call RAG and supports multi-query.
- **System message** enforces strict guidelines: only use RAG tool for meeting-related questions, answer general knowledge directly.
- **Citation display** shows source chunks to users for transparency and verification.

#### 3. Guardrails & Hallucination Mitigation

**Hallucination Control:**
- **Prompt-based guardrails:** Explicit instructions in RAG prompt: "If the answer is not in the context, say so"
- **Context-only answers:** System message enforces answering only from provided context
- **Citation transparency:** Source chunks displayed to users for fact verification
- **Tool usage control:** System message prevents RAG tool calls for general knowledge questions
- **No-context handling:** Returns "No relevant content found" when search yields no results
- **Agent self-regulation:** LangChain agent instructed to admit when information is unavailable

**Re-ranking Strategy:**
- **Single-stage retrieval:** MongoDB `$vectorSearch` with cosine similarity
- **Candidate expansion:** `numCandidates: 50` retrieves broader candidate set
- **Top-K selection:** Returns top 5 most similar chunks (`limit: 5`)
- **No cross-encoder re-ranking:** Currently no secondary relevance scoring model
- **No similarity threshold:** All retrieved chunks included regardless of similarity score



**Potential Enhancements:**
1. **Add re-ranking layer:** Use cross-encoder model to re-rank top-K candidates after vector search
2. **Similarity thresholding:** Filter chunks below minimum cosine similarity (e.g., < 0.7)
3. **Answer verification:** Compare generated answer against retrieved chunks for consistency
4. **Confidence scoring:** Return relevance scores with citations for transparency
5. **Hybrid search:** Combine vector search with BM25/keyword matching for better recall

**Other Guardrails:**
- File-scoped RAG limits search to the selected transcript only.
- **User data isolation:** All queries filtered by `user_id`; users can only access their own transcripts.
- **JWT authentication:** Protected endpoints require valid JWT tokens.
- **Password security:** Bcrypt hashing with automatic truncation for passwords exceeding 72 bytes.


### Key technical decisions you made and why:**

I've chosen Python FastAPI since it's one of the best frameworks to create APIs easily, and React because Cursor could help me build the React frontend easily?it comes with libraries to show markdown formats in UI. Can also help integrate authentication easily in future versions. Using OpenAI for LLM and RAG is just a personal preference but wasn't necessary.

### Engineering standards you've followed (and maybe some that you skipped):**

*Considered:*
- Stateless API which doesn't save chat history (except in-memory for active sessions).
- Backend runs as a single uvicorn process.
- **Docker multi-stage builds** for optimized production images.
- **Health checks** for all services (MongoDB, backend, frontend).
- **Environment variable management** via `.env` files.
- **Data persistence** with Docker volumes for MongoDB.
- **Security headers** in nginx configuration (X-Frame-Options, X-Content-Type-Options, X-XSS-Protection).
- **Password validation** (minimum 6 characters, max 72 bytes for bcrypt compatibility).

*Skipped:*
- How to process large files?it might cause timeout errors.
- Didn't add voice-to-transcript functionality yet.
- **No password complexity requirements** (only length validation).
- **No rate limiting** on authentication endpoints.
- **No email verification** for user registration.
- **No password reset functionality**.
- **In-memory session storage** (not shared across multiple backend instances).

### How you used AI tools in your development process:**

Upon deciding the project I want to make, the first step was to decide the tech stack I want to use. The initial step was to prepare a detailed technical documentation which would outline the tech stack I want to use and how I want to use those. I used Gemini to lay out the technical documentation for the project (IntelliMeet).

Secondly, I wanted to create a Project Plan document?the idea here was to leverage Gemini to give me the detailed steps which would be needed for project development.

I used Cursor as my development tool. Although I have used very specific Cursor rules to make it follow my development plan and scope strictly?my certain guidelines and preferred development and build strategy, folder structure, etc.


### What you'd do differently with more time:**

- I would try to keep the history for a given file if the user selects clearing history and starting a fresh chat.
- We can limit the file size as well (less than 10MB).
- **Authentication mechanism implemented** (JWT-based OAuth).
- I'd like to implement streaming to visualize what is happening on the server side in real time?e.g. processing/embedding of file, tool calling during generating response, etc.
- Some more changes to the UI to make it look more engaging.
- **Persistent session storage** (Redis or database) instead of in-memory for multi-instance deployments.
- **Email verification** for user registration.
- **Password reset** functionality.
- **Rate limiting** on authentication endpoints to prevent brute force attacks.
- **OAuth 2.0 providers** (Google, Microsoft) integration instead of custom auth.

### Edge cases:

If you ask the bot "What is the capital of India?", it replies with "Delhi." But when you ask "What is the current timestamp?", it says "I don't have access to your device's real-time clock." Basically, the limitation of the backend orchestrator is limited by the tools it has access to. Also, since the LLM comes after the RAG operation and the context of the LLM is only the filtered/fetched chunks, there might be chances of missing context.

## Running the App

### Option 1: Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MongoDB: `localhost:27017` (username: `admin`, password: `changeme`)

**Note:** Create a `.env` file in the project root with your Azure OpenAI credentials (see Quick Start section).

### Option 2: Local Development

1. Start the backend: `cd backend && uvicorn main:app --reload --port 8000`
2. Start the frontend: `cd frontend && npm start`
3. Open http://localhost:3000 - register/login, then upload a .txt transcript and start chatting.

See [DOCKER.md](DOCKER.md) for detailed Docker setup instructions.
