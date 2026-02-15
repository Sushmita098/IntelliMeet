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

## **🟡 Step 2: Ingestion & Vector Memory**

**Goal:** Enable file processing and storage in MongoDB.

* **Task 2.1: File Handling Logic \[NOT STARTED\]**  
  * **Backend:** Implement a `POST /upload` endpoint to receive `.txt` files.  
  * **Frontend:** Create a file input field and an "Upload" button using `FormData`.  
* **Task 2.2: Text Chunking & ADA Embeddings \[NOT STARTED\]**  
  * **Backend:** Write a utility to split text into \~800 character chunks.  
  * **Backend:** Integrate the **ADA model** to generate a 1536-dimension vector for each chunk.  
* **Task 2.3: MongoDB Persistence \[NOT STARTED\]**  
  * **Backend:** Save the text chunks and their corresponding vectors into the MongoDB collection.  
  * **Database:** Manually create the **Vector Search Index** in the MongoDB Atlas dashboard.  
* **Task 2.4: Integration Test \[NOT STARTED\]**  
  * Verify that uploading a file in the UI results in multiple documents appearing in your MongoDB collection with valid `embedding` arrays.

---

## **🟠 Step 3: Retrieval Augmented Generation (RAG)**

**Goal:** Query the stored meeting data to get context-aware answers.

* **Task 3.1: Query Vectorization \[NOT STARTED\]**  
  * **Backend:** Update the `POST /ask` endpoint to convert the user's incoming question into a vector using the ADA model.  
* **Task 3.2: Vector Similarity Search \[NOT STARTED\]**  
  * **Backend:** Use the `$vectorSearch` aggregation pipeline in MongoDB to retrieve the top 3-5 most relevant chunks based on the query vector.  
* **Task 3.3: Context-Injected Generation \[NOT STARTED\]**  
  * **Backend:** Construct a prompt for **GPT-5.2 mini** that includes the retrieved chunks as "Context" and the user's question as the "Query."  
* **Task 3.4: RAG UI Interface \[NOT STARTED\]**  
  * **Frontend:** Implement a "Search" interface. Add a loading spinner that activates while the backend is searching and generating.  
  * **Testing:** Ask "What were the main decisions?" and ensure the answer comes from the uploaded text.

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

