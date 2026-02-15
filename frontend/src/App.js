import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [aiResponse, setAiResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        setBackendStatus(res.ok && data.status === 'ok' ? 'connected' : 'disconnected');
      } catch {
        setBackendStatus('disconnected');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file first.');
      return;
    }
    if (!file.name.toLowerCase().endsWith('.txt')) {
      setUploadStatus('Only .txt files are accepted.');
      return;
    }
    setLoading(true);
    setUploadStatus('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Success: ${data.message}`);
        setFile(null);
      } else {
        setUploadStatus(`Error: ${data.detail || res.statusText}`);
      }
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`);
    }
    setLoading(false);
  };

  const handleAskBasic = async () => {
    setLoading(true);
    setAiResponse('');
    try {
      const res = await fetch(`${API_BASE}/ask-basic`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setAiResponse(data.answer);
      } else {
        setAiResponse(`Error: ${data.detail || res.statusText}`);
      }
    } catch (err) {
      setAiResponse(`Error: ${err.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Meeting Transcript Analyzer</h1>
        <h2>System Status</h2>
        <p
          className={
            backendStatus === 'connected'
              ? 'status-connected'
              : backendStatus === 'checking'
              ? 'status-checking'
              : 'status-disconnected'
          }
        >
          {backendStatus === 'connected'
            ? 'Connected'
            : backendStatus === 'checking'
            ? 'Checking...'
            : 'Disconnected'}
        </p>
      </header>
      <main className="App-main">
        <section className="upload-section llm-section">
          <h3>1. Upload Transcript</h3>
          <p>Upload a .txt meeting transcript to chunk and index.</p>
          <div className="upload-controls">
            <input
              type="file"
              accept=".txt"
              onChange={(e) => {
                setFile(e.target.files?.[0] || null);
                setUploadStatus('');
              }}
            />
            <button onClick={handleUpload} disabled={loading}>
              {loading ? 'Processing...' : 'Upload'}
            </button>
          </div>
          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.startsWith('Error') ? 'error' : ''}`}>
              {uploadStatus}
            </div>
          )}
        </section>
        <section className="llm-section">
          <h3>2. Basic LLM Ping</h3>
          <p>Send a test prompt to Azure GPT to verify connectivity.</p>
          <button onClick={handleAskBasic} disabled={loading}>
            {loading ? 'Calling AI...' : 'Ask Basic'}
          </button>
          {aiResponse && (
            <div className="ai-response">{aiResponse}</div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
