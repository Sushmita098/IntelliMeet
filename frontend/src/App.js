import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
const CITATION_PREVIEW_CHARS = 100;

function citationPreview(text, charLimit = CITATION_PREVIEW_CHARS) {
  const trimmed = text.trim();
  const isTruncated = trimmed.length > charLimit;
  const preview = trimmed.slice(0, charLimit);
  return { preview: isTruncated ? preview + '...' : preview, full: trimmed, isTruncated };
}

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [expandedCitations, setExpandedCitations] = useState(() => new Set());
  
  // Auth state
  const [token, setToken] = useState(() => localStorage.getItem('auth_token') || null);
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem('auth_user');
    return stored ? JSON.parse(stored) : null;
  });
  const [showLogin, setShowLogin] = useState(true);
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authName, setAuthName] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const toggleCitation = (key) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const getAuthHeaders = () => {
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const handleLogin = async () => {
    if (!authEmail || !authPassword) {
      setAuthError('Email and password are required');
      return;
    }
    setAuthLoading(true);
    setAuthError('');
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      let data;
      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await res.json();
      } else {
        const text = await res.text();
        setAuthError(`Server error: ${res.status} ${res.statusText}`);
        return;
      }
      if (res.ok) {
        setToken(data.access_token);
        setUser(data.user);
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('auth_user', JSON.stringify(data.user));
        setAuthEmail('');
        setAuthPassword('');
      } else {
        setAuthError(data.detail || 'Login failed');
      }
    } catch (err) {
      setAuthError(`Error: ${err.message}`);
    }
    setAuthLoading(false);
  };

  const handleRegister = async () => {
    if (!authEmail || !authPassword || !authName) {
      setAuthError('Name, email, and password are required');
      return;
    }
    if (authPassword.length < 6) {
      setAuthError('Password must be at least 6 characters');
      return;
    }
    setAuthLoading(true);
    setAuthError('');
    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: authEmail, password: authPassword, name: authName }),
      });
      let data;
      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await res.json();
      } else {
        const text = await res.text();
        setAuthError(`Server error: ${res.status} ${res.statusText}. Please check backend logs.`);
        return;
      }
      if (res.ok) {
        setAuthError('');
        setShowLogin(true);
        setAuthName('');
        alert('Registration successful! Please login.');
      } else {
        setAuthError(data.detail || 'Registration failed');
      }
    } catch (err) {
      setAuthError(`Error: ${err.message}`);
    }
    setAuthLoading(false);
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    setFiles([]);
    setSelectedFile('');
    setChatMessages([]);
    setSessionId(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
  };

  const fetchFiles = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/files`, { headers: getAuthHeaders() });
      if (res.status === 401) {
        handleLogout();
        return;
      }
      const data = await res.json();
      if (res.ok && Array.isArray(data.files)) {
        setFiles(data.files);
        if (data.files.length > 0 && !selectedFile) setSelectedFile(data.files[0]);
      }
    } catch (err) {
      console.error('Failed to fetch files:', err);
    }
  };

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

  useEffect(() => {
    fetchFiles();
  }, [uploadStatus]);

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
        headers: getAuthHeaders(),
        body: formData,
      });
      if (res.status === 401) {
        handleLogout();
        setUploadStatus('Session expired. Please login again.');
        return;
      }
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Uploaded: ${data.filename}`);
        setFile(null);
        fetchFiles();
        if (data.filename) setSelectedFile(data.filename);
      } else {
        setUploadStatus(`Error: ${data.detail || res.statusText}`);
      }
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`);
    }
    setLoading(false);
  };

  const handleChatSend = async () => {
    if (!chatInput.trim() || !selectedFile) return;
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          file_id: selectedFile,
          message: userMsg,
          session_id: sessionId,
        }),
      });
      if (res.status === 401) {
        handleLogout();
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Session expired. Please login again.' },
        ]);
        return;
      }
      const data = await res.json();
      if (res.ok) {
        setSessionId(data.session_id);
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.answer, citations: data.citations || [] },
        ]);
      } else {
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${data.detail || res.statusText}` },
        ]);
      }
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
      ]);
    }
    setChatLoading(false);
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.value);
    setChatMessages([]);
    setSessionId(null);
  };

  // Show login/register UI if not authenticated
  if (!token || !user) {
    return (
      <div className="App">
        <header className="chatbot-header">
          <h1>Meeting Transcript Analyzer</h1>
          <span
            className={`status-dot ${backendStatus === 'connected' ? 'connected' : backendStatus === 'checking' ? 'checking' : 'disconnected'}`}
            title={backendStatus === 'connected' ? 'Connected' : backendStatus === 'checking' ? 'Checking...' : 'Disconnected'}
          />
        </header>
        <div className="auth-container">
          <div className="auth-box">
            <h2>{showLogin ? 'Login' : 'Register'}</h2>
            {authError && <div className="auth-error">{authError}</div>}
            {!showLogin && (
              <div className="auth-field">
                <label>Name:</label>
                <input
                  type="text"
                  value={authName}
                  onChange={(e) => setAuthName(e.target.value)}
                  placeholder="Your name"
                  disabled={authLoading}
                />
              </div>
            )}
            <div className="auth-field">
              <label>Email:</label>
              <input
                type="email"
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                placeholder="your@email.com"
                disabled={authLoading}
              />
            </div>
            <div className="auth-field">
              <label>Password:</label>
              <input
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="••••••"
                disabled={authLoading}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    showLogin ? handleLogin() : handleRegister();
                  }
                }}
              />
            </div>
            <button
              className="auth-submit"
              onClick={showLogin ? handleLogin : handleRegister}
              disabled={authLoading}
            >
              {authLoading ? 'Please wait...' : showLogin ? 'Login' : 'Register'}
            </button>
            <div className="auth-switch">
              {showLogin ? (
                <>
                  Don't have an account?{' '}
                  <button type="button" onClick={() => { setShowLogin(false); setAuthError(''); }}>
                    Register
                  </button>
                </>
              ) : (
                <>
                  Already have an account?{' '}
                  <button type="button" onClick={() => { setShowLogin(true); setAuthError(''); }}>
                    Login
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="chatbot-header">
        <h1>Meeting Transcript Analyzer</h1>
        <div className="header-right">
          <span className="user-info">Welcome, {user.name || user.email}</span>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
          <span
            className={`status-dot ${backendStatus === 'connected' ? 'connected' : backendStatus === 'checking' ? 'checking' : 'disconnected'}`}
            title={backendStatus === 'connected' ? 'Connected' : backendStatus === 'checking' ? 'Checking...' : 'Disconnected'}
          />
        </div>
      </header>

      <div className="chatbot-toolbar">
        <div className="toolbar-upload">
          <input
            type="file"
            accept=".txt"
            id="upload-input"
            onChange={(e) => {
              setFile(e.target.files?.[0] || null);
              setUploadStatus('');
            }}
          />
          <label htmlFor="upload-input" className="upload-label">
            Choose file
          </label>
          <button onClick={handleUpload} disabled={loading}>
            {loading ? 'Uploading...' : 'Upload'}
          </button>
          {uploadStatus && (
            <span className={`toolbar-status ${uploadStatus.startsWith('Error') ? 'error' : ''}`}>
              {uploadStatus}
            </span>
          )}
        </div>
        <div className="toolbar-select">
          <label htmlFor="file-select">Transcript:</label>
          <select
            id="file-select"
            value={selectedFile}
            onChange={handleFileChange}
            disabled={chatLoading}
          >
            <option value="">Select a transcript...</option>
            {files.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>
      </div>

      <main className="chatbot-main">
        <div className="chat-messages">
          {chatMessages.length === 0 && (
            <div className="chat-welcome">
              {selectedFile ? (
                <>Ask anything about <strong>{selectedFile}</strong>. The AI will search the transcript to answer.</>
              ) : (
                <>Upload a .txt transcript and select it above to start asking questions.</>
              )}
            </div>
          )}
          {chatMessages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              <div className="bubble-content">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
              {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                <div className="bubble-citations">
                  <span className="citations-label">From transcript:</span>
                  {msg.citations.map((cite, j) => {
                    const key = `${i}-${j}`;
                    const { preview, full, isTruncated } = citationPreview(cite);
                    const isExpanded = expandedCitations.has(key);
                    return (
                      <blockquote key={j} className="citation-quote">
                        <span className="citation-text">
                          {isTruncated && !isExpanded ? preview : full}
                        </span>
                        {isTruncated && (
                          <button
                            type="button"
                            className="citation-toggle"
                            onClick={() => toggleCitation(key)}
                            aria-expanded={isExpanded}
                          >
                            {isExpanded ? 'Show less ▲' : 'Show more ▼'}
                          </button>
                        )}
                      </blockquote>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
          {chatLoading && (
            <div className="chat-bubble assistant typing">
              <span className="typing-dots"><span></span><span></span><span></span></span>
            </div>
          )}
        </div>
        <div className="chat-input-area">
          <input
            type="text"
            placeholder={selectedFile ? 'Ask about this transcript...' : 'Select a transcript first'}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSend()}
            disabled={chatLoading || !selectedFile}
          />
          <button
            onClick={handleChatSend}
            disabled={chatLoading || !chatInput.trim() || !selectedFile}
          >
            Send
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
