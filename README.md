# IntelliMeet

Meeting Transcript Analyzer using RAG (Retrieval-Augmented Generation).

## Project Structure

- **backend/** ? FastAPI server (Python)
- **frontend/** ? React app
- **documents/** ? Project plan and technical documentation

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

- `AZURE_OPENAI_KEY` ? Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` ? Base URL (e.g. `https://your-resource.openai.azure.com`)
- `AZURE_OPENAI_DEPLOYMENT_NAME` ? Chat model deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` ? Embedding model deployment (e.g. `text-embedding-ada-002`)
- `AZURE_OPENAI_API_VERSION` ? Use `2025-04-01-preview` for Responses API
- `MONGO_URI` ? MongoDB Atlas connection string (required for Step 2+)

### 4. MongoDB Vector Search Index (required for upload / RAG)

In MongoDB Atlas, create a **Vector Search Index** on the `transcripts` collection:

1. Go to your cluster ? Browse Collections ? `meeting_analyzer_db` ? `transcripts`
2. Search Indexes ? Create Search Index ? JSON Editor
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

## Running the App

1. Start the backend: `cd backend && uvicorn main:app --reload --port 8000`
2. Start the frontend: `cd frontend && npm start`
3. Open http://localhost:3000 ? upload a .txt transcript, then use Ask Basic or Search (Step 3).
