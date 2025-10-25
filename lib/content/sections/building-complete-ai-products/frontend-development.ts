export const frontendDevelopment = {
  title: 'Frontend Development',
  id: 'frontend-development',
  content: `
# Frontend Development for AI Applications

## Introduction

Building the frontend for an AI application is fundamentally different from traditional web apps. You're dealing with streaming responses, long-running operations, real-time updates, complex state management, and unique UX patterns like "AI is thinking..."

This section covers building production-ready frontends for AI applications using React/Next.js with patterns from real products like ChatGPT, Claude, Cursor, and Perplexity.

### Key Challenges

**Streaming Responses**: LLMs return tokens one at a time, not all at once
**Long Operations**: Image/video generation takes 30s-2min, not milliseconds  
**Optimistic Updates**: Show immediate feedback while AI processes
**Error Recovery**: Handle failures gracefully mid-generation
**Cost Display**: Show users what they're spending in real-time

---

## Streaming Text Responses

### Server-Sent Events (SSE) Client

\`\`\`typescript
/**
 * SSE Hook for Streaming LLM Responses
 */

import { useState, useCallback } from 'react';

interface StreamingResponse {
  content: string;
  isStreaming: boolean;
  error: string | null;
}

export function useStreamingLLM() {
  const [response, setResponse] = useState<StreamingResponse>({
    content: ',
    isStreaming: false,
    error: null
  });

  const streamMessage = useCallback (async (prompt: string) => {
    setResponse({ content: ', isStreaming: true, error: null });

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!response.ok) throw new Error('Stream failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode (value);
        const lines = chunk.split('\\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse (data);
              setResponse (prev => ({
                ...prev,
                content: prev.content + parsed.token
              }));
            } catch (e) {
              console.error('Parse error:', e);
            }
          }
        }
      }

      setResponse (prev => ({ ...prev, isStreaming: false }));

    } catch (error: any) {
      setResponse (prev => ({
        ...prev,
        isStreaming: false,
        error: error.message
      }));
    }
  }, []);

  const cancel = useCallback(() => {
    // Implementation for canceling stream
    setResponse (prev => ({ ...prev, isStreaming: false }));
  }, []);

  return { ...response, streamMessage, cancel };
}

// Usage in Component
export function ChatInterface() {
  const { content, isStreaming, error, streamMessage, cancel } = useStreamingLLM();
  const [input, setInput] = useState(');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    await streamMessage (input);
    setInput(');
  };

  return (
    <div className="chat-interface">
      <div className="messages">
        <div className="message assistant">
          {content}
          {isStreaming && <span className="cursor">▊</span>}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={e => setInput (e.target.value)}
          disabled={isStreaming}
          placeholder="Ask anything..."
        />
        {isStreaming ? (
          <button type="button" onClick={cancel}>Stop</button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>

      {error && <div className="error">{error}</div>}
    </div>
  );
}
\`\`\`

---

## Long-Running Operations UI

### Progress Tracking for Image/Video Generation

\`\`\`typescript
/**
 * Job Status Polling Hook
 */

import { useState, useEffect } from 'react';

interface Job {
  id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: any;
  error?: string;
  estimatedTime?: number;
}

export function useJobStatus (jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const pollInterval = setInterval (async () => {
      try {
        const res = await fetch(\`/api/jobs/\${jobId}\`);
        const data = await res.json();
        
        setJob (data);

        // Stop polling when done
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval (pollInterval);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval (pollInterval);
  }, [jobId]);

  return job;
}

// Progress Bar Component
export function JobProgress({ job }: { job: Job }) {
  if (!job) return null;

  const getStatusMessage = () => {
    switch (job.status) {
      case 'queued':
        return \`Queued... Position: \${job.progress}\`;
      case 'processing':
        return \`Processing... \${Math.round (job.progress)}% complete\`;
      case 'completed':
        return 'Done!';
      case 'failed':
        return \`Failed: \${job.error}\`;
    }
  };

  const getTimeRemaining = () => {
    if (job.status !== 'processing' || !job.estimatedTime) return ';
    const remaining = Math.max(0, job.estimatedTime - (job.progress / 100) * job.estimatedTime);
    return \`~\${Math.round (remaining)}s remaining\`;
  };

  return (
    <div className="job-progress">
      <div className="status-bar">
        <div 
          className="progress-fill"
          style={{ 
            width: \`\${job.progress}%\`,
            backgroundColor: job.status === 'failed' ? '#ef4444' : '#3b82f6'
          }}
        />
      </div>
      
      <div className="status-text">
        <span>{getStatusMessage()}</span>
        <span className="time-remaining">{getTimeRemaining()}</span>
      </div>

      {job.status === 'processing' && (
        <div className="loading-spinner">
          <div className="spinner" />
        </div>
      )}
    </div>
  );
}

// Image Generation Component
export function ImageGenerator() {
  const [prompt, setPrompt] = useState(');
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJobStatus (jobId);
  const [results, setResults] = useState<string[]>([]);

  const generate = async () => {
    const res = await fetch('/api/images/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    const data = await res.json();
    setJobId (data.job_id);
  };

  useEffect(() => {
    if (job?.status === 'completed' && job.result?.url) {
      setResults (prev => [job.result.url, ...prev]);
      setJobId (null);
    }
  }, [job]);

  return (
    <div className="image-generator">
      <div className="input-section">
        <textarea
          value={prompt}
          onChange={e => setPrompt (e.target.value)}
          placeholder="Describe the image..."
          rows={3}
        />
        <button 
          onClick={generate}
          disabled={!prompt || jobId !== null}
        >
          Generate Image
        </button>
      </div>

      {job && <JobProgress job={job} />}

      <div className="results-grid">
        {results.map((url, i) => (
          <img key={i} src={url} alt="Generated" />
        ))}
      </div>
    </div>
  );
}
\`\`\`

---

## Optimistic Updates

### Immediate Feedback Pattern

\`\`\`typescript
/**
 * Optimistic UI Updates
 */

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isOptimistic?: boolean;
  error?: string;
}

export function useOptimisticChat() {
  const [messages, setMessages] = useState<Message[]>([]);

  const sendMessage = async (content: string) => {
    // Add user message immediately
    const userMessage: Message = {
      id: \`temp-\${Date.now()}\`,
      role: 'user',
      content
    };

    // Add optimistic assistant placeholder
    const assistantMessage: Message = {
      id: \`temp-assistant-\${Date.now()}\`,
      role: 'assistant',
      content: ',
      isOptimistic: true
    };

    setMessages (prev => [...prev, userMessage, assistantMessage]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: content })
      });

      const data = await response.json();

      // Replace optimistic message with real one
      setMessages (prev => 
        prev.map (msg => 
          msg.id === assistantMessage.id
            ? { ...msg, content: data.response, isOptimistic: false }
            : msg
        )
      );

    } catch (error: any) {
      // Mark optimistic message as failed
      setMessages (prev =>
        prev.map (msg =>
          msg.id === assistantMessage.id
            ? { ...msg, error: error.message, isOptimistic: false }
            : msg
        )
      );
    }
  };

  const retry = async (messageId: string) => {
    // Find message and retry
    const message = messages.find (m => m.id === messageId);
    if (!message) return;

    // Find previous user message
    const idx = messages.findIndex (m => m.id === messageId);
    const userMessage = messages[idx - 1];

    if (userMessage?.role === 'user') {
      // Remove failed message
      setMessages (prev => prev.filter (m => m.id !== messageId));
      // Retry
      await sendMessage (userMessage.content);
    }
  };

  return { messages, sendMessage, retry };
}

// Message Component
export function MessageBubble({ message, onRetry }: { 
  message: Message;
  onRetry: (id: string) => void;
}) {
  return (
    <div className={\`message \${message.role}\`}>
      <div className="content">
        {message.isOptimistic ? (
          <div className="thinking">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        ) : (
          message.content
        )}
      </div>

      {message.error && (
        <div className="error">
          <span>{message.error}</span>
          <button onClick={() => onRetry (message.id)}>
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
\`\`\`

---

## File Upload with Progress

### Drag-and-Drop File Upload

\`\`\`typescript
/**
 * File Upload with Progress
 */

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  result?: any;
  error?: string;
}

export function useFileUpload() {
  const [uploads, setUploads] = useState<Map<string, UploadProgress>>(new Map());

  const uploadFile = useCallback (async (file: File) => {
    const fileId = \`\${file.name}-\${Date.now()}\`;

    // Initialize progress
    setUploads (prev => new Map (prev).set (fileId, {
      file,
      progress: 0,
      status: 'uploading'
    }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100;
          setUploads (prev => {
            const newMap = new Map (prev);
            const current = newMap.get (fileId);
            if (current) {
              newMap.set (fileId, { ...current, progress });
            }
            return newMap;
          });
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const response = JSON.parse (xhr.responseText);
          
          setUploads (prev => {
            const newMap = new Map (prev);
            newMap.set (fileId, {
              file,
              progress: 100,
              status: 'processing',
              result: response
            });
            return newMap;
          });

          // Poll for processing completion
          pollProcessing (fileId, response.job_id);
        } else {
          throw new Error('Upload failed');
        }
      });

      xhr.addEventListener('error', () => {
        setUploads (prev => {
          const newMap = new Map (prev);
          newMap.set (fileId, {
            file,
            progress: 0,
            status: 'error',
            error: 'Upload failed'
          });
          return newMap;
        });
      });

      xhr.open('POST', '/api/upload');
      xhr.send (formData);

    } catch (error: any) {
      setUploads (prev => {
        const newMap = new Map (prev);
        newMap.set (fileId, {
          file,
          progress: 0,
          status: 'error',
          error: error.message
        });
        return newMap;
      });
    }
  }, []);

  const pollProcessing = async (fileId: string, jobId: string) => {
    const interval = setInterval (async () => {
      const res = await fetch(\`/api/jobs/\${jobId}\`);
      const data = await res.json();

      if (data.status === 'completed') {
        clearInterval (interval);
        setUploads (prev => {
          const newMap = new Map (prev);
          const current = newMap.get (fileId);
          if (current) {
            newMap.set (fileId, {
              ...current,
              status: 'completed',
              result: data.result
            });
          }
          return newMap;
        });
      } else if (data.status === 'failed') {
        clearInterval (interval);
        setUploads (prev => {
          const newMap = new Map (prev);
          const current = newMap.get (fileId);
          if (current) {
            newMap.set (fileId, {
              ...current,
              status: 'error',
              error: data.error
            });
          }
          return newMap;
        });
      }
    }, 2000);
  };

  return { uploads: Array.from (uploads.values()), uploadFile };
}

// Upload Component
export function FileUploader() {
  const { uploads, uploadFile } = useFileUpload();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach (uploadFile);
  }, [uploadFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  return (
    <div className="file-uploader">
      <div 
        {...getRootProps()} 
        className={\`dropzone \${isDragActive ? 'active' : '}\`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          <svg className="upload-icon" />
          <p>
            {isDragActive
              ? 'Drop files here...'
              : 'Drag files here or click to upload'}
          </p>
          <p className="hint">PDF, DOCX, TXT up to 10MB</p>
        </div>
      </div>

      <div className="uploads-list">
        {uploads.map((upload, i) => (
          <div key={i} className="upload-item">
            <div className="file-info">
              <span className="file-name">{upload.file.name}</span>
              <span className="file-size">
                {(upload.file.size / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>

            <div className="progress-section">
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: \`\${upload.progress}%\` }}
                />
              </div>
              
              <span className="status">
                {upload.status === 'uploading' && \`Uploading... \${Math.round (upload.progress)}%\`}
                {upload.status === 'processing' && 'Processing...'}
                {upload.status === 'completed' && '✓ Completed'}
                {upload.status === 'error' && \`✗ \${upload.error}\`}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
\`\`\`

---

## Real-Time Collaboration UI

### Live Cursors and Presence

\`\`\`typescript
/**
 * Real-Time Presence Component
 */

import { useEffect, useState } from 'react';

interface User {
  id: string;
  name: string;
  color: string;
  cursor?: { x: number; y: number };
  isTyping?: boolean;
}

export function usePresence (roomId: string) {
  const [users, setUsers] = useState<Map<string, User>>(new Map());
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const websocket = new WebSocket(\`wss://api.example.com/presence?room=\${roomId}\`);

    websocket.onopen = () => {
      // Send join message
      websocket.send(JSON.stringify({
        type: 'join',
        user: {
          id: 'user-123',
          name: 'Current User',
          color: '#3b82f6'
        }
      }));
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse (event.data);

      switch (data.type) {
        case 'user_joined':
          setUsers (prev => new Map (prev).set (data.user.id, data.user));
          break;
        
        case 'user_left':
          setUsers (prev => {
            const newMap = new Map (prev);
            newMap.delete (data.userId);
            return newMap;
          });
          break;
        
        case 'cursor_move':
          setUsers (prev => {
            const newMap = new Map (prev);
            const user = newMap.get (data.userId);
            if (user) {
              newMap.set (data.userId, { ...user, cursor: data.cursor });
            }
            return newMap;
          });
          break;
        
        case 'typing':
          setUsers (prev => {
            const newMap = new Map (prev);
            const user = newMap.get (data.userId);
            if (user) {
              newMap.set (data.userId, { ...user, isTyping: data.isTyping });
            }
            return newMap;
          });
          break;
      }
    };

    setWs (websocket);

    return () => websocket.close();
  }, [roomId]);

  const updateCursor = (x: number, y: number) => {
    ws?.send(JSON.stringify({
      type: 'cursor_move',
      cursor: { x, y }
    }));
  };

  const setTyping = (isTyping: boolean) => {
    ws?.send(JSON.stringify({
      type: 'typing',
      isTyping
    }));
  };

  return { 
    users: Array.from (users.values()),
    updateCursor,
    setTyping
  };
}

// Cursor Component
export function RemoteCursors({ users }: { users: User[] }) {
  return (
    <>
      {users.map (user => 
        user.cursor && (
          <div
            key={user.id}
            className="remote-cursor"
            style={{
              position: 'absolute',
              left: user.cursor.x,
              top: user.cursor.y,
              pointerEvents: 'none',
              transition: 'all 0.1s ease-out'
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 20 20"
              fill={user.color}
            >
              <path d="M0 0 L0 16 L4 12 L7 20 L9 19 L6 11 L12 11 Z" />
            </svg>
            <div
              className="cursor-label"
              style={{
                backgroundColor: user.color,
                color: 'white',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '12px',
                marginLeft: '20px'
              }}
            >
              {user.name}
            </div>
          </div>
        )
      )}
    </>
  );
}
\`\`\`

---

## State Management

### Zustand for AI App State

\`\`\`typescript
/**
 * Global State Management with Zustand
 */

import create from 'zustand';
import { persist } from 'zustand/middleware';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Generation {
  id: string;
  type: 'image' | 'video' | 'audio';
  prompt: string;
  status: string;
  result?: string;
}

interface AppState {
  // Chat
  conversations: Map<string, ChatMessage[]>;
  currentConversationId: string | null;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  createConversation: () => string;
  
  // Generations
  generations: Generation[];
  addGeneration: (generation: Generation) => void;
  updateGeneration: (id: string, updates: Partial<Generation>) => void;
  
  // User
  credits: number;
  deductCredits: (amount: number) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Chat state
      conversations: new Map(),
      currentConversationId: null,

      createConversation: () => {
        const id = \`conv-\${Date.now()}\`;
        set (state => ({
          conversations: new Map (state.conversations).set (id, []),
          currentConversationId: id
        }));
        return id;
      },

      addMessage: (conversationId, message) => {
        set (state => {
          const newConversations = new Map (state.conversations);
          const messages = newConversations.get (conversationId) || [];
          newConversations.set (conversationId, [...messages, message]);
          return { conversations: newConversations };
        });
      },

      // Generations state
      generations: [],

      addGeneration: (generation) => {
        set (state => ({
          generations: [generation, ...state.generations]
        }));
      },

      updateGeneration: (id, updates) => {
        set (state => ({
          generations: state.generations.map (gen =>
            gen.id === id ? { ...gen, ...updates } : gen
          )
        }));
      },

      // User state
      credits: 100,

      deductCredits: (amount) => {
        set (state => ({
          credits: Math.max(0, state.credits - amount)
        }));
      }
    }),
    {
      name: 'app-storage',
      partialize: (state) => ({
        conversations: Array.from (state.conversations.entries()),
        credits: state.credits
      })
    }
  )
);
\`\`\`

---

## Conclusion

Frontend development for AI applications requires:

1. **Streaming UI**: Handle token-by-token responses
2. **Long Operations**: Progress bars, polling, time estimates
3. **Optimistic Updates**: Immediate feedback
4. **File Upload**: Progress tracking, error handling
5. **Real-Time**: WebSocket, presence, cursors
6. **State Management**: Complex async state
7. **Error Recovery**: Retry, rollback, clear feedback

**Key Libraries**:
- **React/Next.js**: Framework
- **Zustand**: State management
- **react-dropzone**: File upload
- **WebSocket**: Real-time
- **Tailwind CSS**: Styling

These patterns create responsive, professional AI application frontends.
`,
};
