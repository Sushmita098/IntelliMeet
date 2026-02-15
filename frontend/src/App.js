import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [aiResponse, setAiResponse] = useState('');
  const [loading, setLoading] = useState(false);

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
        <section className="llm-section">
          <h3>Basic LLM Ping</h3>
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
