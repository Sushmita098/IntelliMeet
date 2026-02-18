## **Project Plan Meeting Transcript Analyzer**

## **🟢 Step 1: Connectivity & Foundation** [COMPLETED]

**Goal:** Establish the communication bridge between React, FastAPI, and Azure OpenAI.

* **Task 1.1: Backend Environment Setup [COMPLETED]**  
  * Initialize a Python virtual environment.  
  * Install dependencies: `fastapi`, `uvicorn`, `openai`, `python-dotenv`, `python-multipart`.  
  * Create a `.env.example` file for Azure credentials and Mongo URI (copy to `.env` and fill values).  
* **Task 1.2: Frontend Initialization [COMPLETED]**  
  * Create a React app using `npx create-react-app`.  
  * Setup a basic layout with a "System Status" header.  
* **Task 1.3: Health Check & CORS [COMPLETED]**  
  * **Backend:** Create a `GET /health` endpoint and configure `CORSMiddleware` to allow the frontend origin.  
  * **Frontend:** Use `useEffect` to fetch from `/health` and display a "Connected" status.  
* **Task 1.4: Basic LLM Ping [COMPLETED]**  
  * **Backend:** Create a `POST /ask-basic` endpoint that sends a hardcoded prompt to **Azure GPT-5.2 mini**.  
  * **Frontend:** Create a text box and button to trigger this endpoint and display the AI's response.

---

## **🟡 Step 2: Ingestion & Vector Memory** [COMPLETED]

**Goal:** Enable file processing and storage in MongoDB.

* **Task 2.1: File Handling Logic [COMPLETED]**  
  * **Backend:** Implement a `POST /upload` endpoint to receive `.txt` files.  
  * **Frontend:** Create a file input field and an "Upload" button using `FormData`.  
* **Task 2.2: Text Chunking & ADA Embeddings [COMPLETED]**  
  * **Backend:** Write a utility to split text into ~800 character chunks.  
  * **Backend:** Integrate the **ADA model** to generate a 1536-dimension vector for each chunk.  
* **Task 2.3: MongoDB Persistence [COMPLETED]**  
  * **Backend:** Save the text chunks and their corresponding vectors into the MongoDB collection.  
  * **Database:** Manually create the **Vector Search Index** in the MongoDB Atlas dashboard (see README).  
* **Task 2.4: Integration Test [PENDING]**  
  * Verify that uploading a file in the UI results in multiple documents appearing in your MongoDB collection with valid `embedding` arrays.

---

## **🟠 Step 3: Retrieval Augmented Generation (RAG)** [COMPLETED]

**Goal:** Query the stored meeting data to get context-aware answers.

* **Task 3.1: Query Vectorization [COMPLETED]**  
  * **Backend:** Update the `POST /ask` endpoint to convert the user's incoming question into a vector using the ADA model.  
* **Task 3.2: Vector Similarity Search [COMPLETED]**  
  * **Backend:** Use the `$vectorSearch` aggregation pipeline in MongoDB to retrieve the top 3-5 most relevant chunks based on the query vector.  
* **Task 3.3: Context-Injected Generation [COMPLETED]**  
  * **Backend:** Construct a prompt for **GPT-5.2 mini** that includes the retrieved chunks as "Context" and the user's question as the "Query."  
* **Task 3.4: RAG UI Interface [COMPLETED]**  
  * **Frontend:** "Search Transcript (RAG)" interface with text input, Search button, and loading spinner.

---

## **🔴 Step 4: Specialized Analysis & Polish**

**Goal:** Optimize the bot for specific meeting outputs like action items and follow-ups.

* **Task 4.1: Specialized System Prompts \[NOT STARTED\]**  
  * **Backend:** Create logic to handle specific flags (e.g., "Summarize", "Action Items") with custom system instructions for the LLM.  
* **Task 4.2: Quick Action UI \[NOT STARTED\]**  
  * **Frontend:** Add "one-click" buttons for common tasks: "Extract Action Items" and "Identify Decisions."  
* **Task 4.3: Markdown Rendering \[NOT STARTED\]**  
  * **Frontend:** Install and integrate `react-markdown` so that the AI's bullet points and tables are rendered professionally.  
* **Task 4.4: Cleanup & Reset \[NOT STARTED\]**  
  * **Backend:** Create a `DELETE /clear` endpoint to purge the MongoDB collection.  
  * **Frontend:** Add a "Reset Session" button to start a new analysis.

---

## **🟣 Step 5: LangChain RAG-as-Tool & File-Scoped Chats** [COMPLETED]

**Goal:** Integrate LangChain with RAG exposed as a tool, enabling multi-turn chats scoped to each file and multiple RAG searches per conversation.

* **Task 5.1: Data Model for File-Scoped Chats [COMPLETED]**  
  * **Backend:** Chunks use `filename` in MongoDB. `GET /files` lists unique filenames.  
  * **Backend:** Session tracked via `session_id`; history kept in memory.  
  * **Frontend:** File selector (dropdown) for active chat context.  
* **Task 5.2: LangChain Integration [COMPLETED]**  
  * **Backend:** Added `langchain`, `langchain-openai`, `langchain-core`, `langgraph` to `requirements.txt`.  
  * **Backend:** LangChain `AzureChatOpenAI` with `create_react_agent` from LangGraph.  
  * **Backend:** RAG search scoped by `filename` in `search_transcript_scoped()`.  
* **Task 5.3: RAG as Tool [COMPLETED]**  
  * **Backend:** RAG exposed as LangChain tool `search_transcript(query)` scoped to file.  
  * **Backend:** Agent can call the tool multiple times per turn.  
* **Task 5.4: File-Scoped Chat API & UI [COMPLETED]**  
  * **Backend:** `POST /chat` accepts `file_id`, `message`, optional `session_id`; returns agent response.  
  * **Frontend:** File selector + chat panel scoped to selected file.  
  * **Frontend:** Chat history display and multi-turn conversation.

---

## **🔵 Step 6: OAuth Authentication & Docker Production Setup** [COMPLETED]

**Goal:** Implement basic OAuth authentication system and create production-ready Docker configuration.

* **Task 6.1: OAuth Backend Implementation [COMPLETED]**  
  * **Backend:** Add `python-jose[cryptography]` and `passlib[bcrypt]` dependencies.  
  * **Backend:** Implement user registration (`POST /auth/register`) and login (`POST /auth/login`) endpoints.  
  * **Backend:** JWT token generation and verification using `python-jose`.  
  * **Backend:** Password hashing with bcrypt via `passlib`.  
  * **Backend:** Store users in MongoDB `users` collection.  
  * **Backend:** Add `user_id` field to transcript chunks for data isolation.  
* **Task 6.2: Protected Routes & Authorization [COMPLETED]**  
  * **Backend:** Create `verify_token` dependency for JWT verification.  
  * **Backend:** Protect `/upload`, `/chat`, `/files` endpoints with authentication.  
  * **Backend:** Filter transcript queries by `user_id` to ensure data isolation.  
  * **Backend:** Scope chat sessions by `user_id` + `session_id`.  
* **Task 6.3: Frontend Authentication UI [COMPLETED]**  
  * **Frontend:** Add login and registration forms with email/password.  
  * **Frontend:** Store JWT token in `localStorage`.  
  * **Frontend:** Add `Authorization: Bearer <token>` header to authenticated API calls.  
  * **Frontend:** Show login/register UI when not authenticated; show chatbot when authenticated.  
  * **Frontend:** Add logout functionality and user info display in header.  
* **Task 6.4: Docker Production Setup [COMPLETED]**  
  * **Backend:** Create multi-stage `Dockerfile` for FastAPI backend.  
  * **Frontend:** Create multi-stage `Dockerfile` with nginx for React frontend.  
  * **Root:** Create `docker-compose.yml` with MongoDB, backend, and frontend services.  
  * **Root:** Add `.dockerignore` files for backend and frontend.  
  * **Frontend:** Add `nginx.conf` for production-ready static file serving.  
  * **Root:** Configure health checks, volumes, and networking in docker-compose.  

