# **Technical Documentation: Meeting Transcript Analyzer**

This document provides a comprehensive technical guide for building a meeting transcript analyzer using Retrieval-Augmented Generation (RAG).

## **1\. System Architecture**

The application follows a RAG pipeline to turn text transcripts into actionable insights.

* **Ingestion:** Upload `.txt` transcripts.  
* **Processing:** Split text into chunks and generate embeddings using the Azure OpenAI **ADA model** ($text-embedding-ada-002$).  
* **Storage:** Chunks and vector embeddings are stored in **MongoDB Atlas**.  
* **Retrieval & Generation:** User questions trigger a vector search; relevant chunks are sent to **Azure OpenAI GPT-5.2 mini** to generate a response.

## **2\. Backend: Python FastAPI**

The backend orchestrates the file processing, embedding generation, and LLM communication.

### **Dependencies (`requirements.txt`)**

```
fastapi
uvicorn
pymongo
openai
python-dotenv
python-multipart
```

### **Core Implementation (`main.py`)**

```
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLIENT INITIALIZATION ---
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2025-04-14", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["meeting_analyzer_db"]
collection = db["transcripts"]

@app.post("/upload")
async def upload_transcript(file: UploadFile = File(...)):
    try:
        content = (await file.read()).decode("utf-8")
        chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
        
        documents = []
        for chunk in chunks:
            embed_response = client.embeddings.create(
                input=[chunk], 
                model="text-embedding-ada-002"
            )
            embedding = embed_response.data[0].embedding
            documents.append({
                "text": chunk,
                "embedding": embedding,
                "filename": file.filename
            })
        
        if documents:
            collection.insert_many(documents)
        return {"message": f"Successfully processed {len(chunks)} chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: str):
    try:
        query_embed = client.embeddings.create(
            input=[query], 
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": query_embed,
                    "numCandidates": 50,
                    "limit": 5
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        context_text = "\n".join([r['text'] for r in results])
        
        response = client.chat.completions.create(
            model="gpt-5.2-mini",
            messages=[
                {"role": "system", "content": "Use the following context to answer the question. If unknown, say so."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## **3\. Frontend: React**

A simple interface to upload files and interact with the AI.

### **Implementation (`App.js`)**

```
import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleUpload = async () => {
    if (!file) return alert("Select a file!");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    await fetch("http://localhost:8000/upload", { method: "POST", body: formData });
    setLoading(false);
    alert("Transcript Indexed!");
  };

  const handleAsk = async () => {
    setLoading(true);
    const res = await fetch(`http://localhost:8000/ask?query=${encodeURIComponent(question)}`, { 
      method: "POST" 
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '600px', margin: '50px auto', fontFamily: 'Arial' }}>
      <h2>🎙️ Meeting Analyzer</h2>
      <div style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
        <h4>1. Upload Transcript</h4>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} disabled={loading}>{loading ? "Processing..." : "Upload"}</button>
      </div>
      <div style={{ marginTop: '30px' }}>
        <h4>2. Ask a Question</h4>
        <input style={{ width: '80%', padding: '10px' }} value={question} onChange={(e) => setQuestion(e.target.value)} />
        <button onClick={handleAsk} disabled={loading}>Ask</button>
      </div>
      {answer && <div style={{ marginTop: '20px', background: '#f9f9f9', padding: '15px' }}>{answer}</div>}
    </div>
  );
}

export default App;
```

## **4\. Database: MongoDB Atlas Configuration**

Create a **Search Index** on your collection (`transcripts`) using the JSON editor:

```
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

## **5\. Environment Variables (`.env`)**

```
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=[https://your-resource.openai.azure.com/](https://your-resource.openai.azure.com/)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

