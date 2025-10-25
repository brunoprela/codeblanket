export const realTimeCollaboration = {
  title: 'Real-Time Collaboration & Multiplayer',
  id: 'real-time-collaboration',
  content: `
# Real-Time Collaboration & Multiplayer

## Introduction

Building real-time collaborative AI applications represents one of the most challenging and rewarding aspects of modern software development. Think Google Docs, but with AI assistance—multiple users editing simultaneously while AI provides suggestions, all in real-time.

This section covers how to build multiplayer AI experiences where:
- Multiple users collaborate in real-time
- AI assists all users simultaneously
- Changes sync instantly across all clients
- Conflicts are resolved automatically
- The experience feels seamless and instant

### Why Real-Time Collaboration Matters

**User Expectations**: Modern users expect Google Docs-level collaboration. Solo editing tools feel outdated.

**Team Productivity**: Real-time collaboration eliminates bottlenecks. No more "waiting for John to finish editing."

**Network Effects**: Collaborative features drive viral growth. Users invite team members, expanding your user base.

**Market Differentiation**: Adding multiplayer to your AI product creates competitive moats. It\'s technically challenging, so few competitors do it well.

### The Challenge

Real-time collaboration is **hard** because:

1. **Network Latency**: Changes take 50-300ms to propagate. Users expect instant feedback.
2. **Concurrent Edits**: User A and User B edit the same text simultaneously. Who wins?
3. **State Synchronization**: How do you keep all clients in sync without constant full-state transfers?
4. **Presence Awareness**: Users need to see where others are working (cursors, selections).
5. **AI Integration**: How does AI assist multiple users without conflicts?
6. **Scale**: Supporting 100+ simultaneous editors is exponentially harder than 2-3.

---

## Collaboration Architecture Fundamentals

### Synchronization Strategies

**Three approaches to real-time sync:**

1. **Pessimistic Locking** (Traditional)
   - User locks resource, makes changes, unlocks
   - Simple but terrible UX (blocking, slow)
   - Used by old systems (SVN, shared drives)

2. **Last-Write-Wins** (Naive)
   - Latest change overwrites previous
   - Simple but loses data
   - Unacceptable for production

3. **Conflict-Free Replicated Data Types (CRDTs)** (Modern)
   - Math-based convergence
   - All clients eventually reach same state
   - Used by Google Docs, Figma, Linear

### System Architecture

\`\`\`
┌──────────────────────────────────────────────────────────────┐
│              Real-Time Collaboration System                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  WebSocket  ┌─────────────────┐            │
│  │  Client A   │◄───────────►│  Collaboration  │            │
│  │  (Browser)  │             │     Server      │            │
│  └─────────────┘             │   (Node.js)     │            │
│                              └────────┬────────┘            │
│  ┌─────────────┐                     │                      │
│  │  Client B   │◄────────────────────┤                      │
│  │  (Browser)  │                     │                      │
│  └─────────────┘                     │                      │
│                                      │                      │
│  ┌─────────────┐                     │                      │
│  │  Client C   │◄────────────────────┤                      │
│  │  (Browser)  │                     │                      │
│  └─────────────┘                     │                      │
│                                      │                      │
│                              ┌───────▼────────┐             │
│                              │  Redis PubSub  │             │
│                              │  (Broadcast)   │             │
│                              └───────┬────────┘             │
│                                      │                      │
│                              ┌───────▼────────┐             │
│                              │   PostgreSQL   │             │
│                              │  (Persistence) │             │
│                              └────────────────┘             │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  AI Service (Separate Process)                      │    │
│  │  • Monitors document changes                        │    │
│  │  • Generates suggestions                            │    │
│  │  • Broadcasts AI edits as CRDT operations          │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
\`\`\`

**Key Components:**

1. **WebSocket Server**: Maintains persistent connections with all clients
2. **CRDT Engine**: Resolves concurrent edits without conflicts
3. **Redis PubSub**: Broadcasts changes across server instances (horizontal scaling)
4. **PostgreSQL**: Persists document state and operation history
5. **AI Service**: Generates suggestions that integrate with CRDT operations

---

## Conflict-Free Replicated Data Types (CRDTs)

### Understanding CRDTs

**The Problem**: Two users edit the same document:

\`\`\`
Initial: "Hello World"

User A: Insert "Beautiful " at position 6 → "Hello Beautiful World"
User B: Delete "World" (chars 6-11) → "Hello "

What's the final result?
\`\`\`

**Position-based editing fails** because positions shift. CRDT solves this with **identifiers**.

### CRDT Basics

Instead of positions, each character gets a unique, immutable identifier:

\`\`\`
"Hello World"
 ↓
[H₁, e₂, l₃, l₄, o₅, _₆, W₇, o₈, r₉, l₁₀, d₁₁]
\`\`\`

Operations reference IDs, not positions:
- **Insert**: "Insert 'Beautiful' after ₅"
- **Delete**: "Delete ₇₈₉₁₀₁₁"

IDs never change, so operations commute (order doesn't matter).

### Yjs - Production CRDT Library

**Yjs** is the most popular JavaScript CRDT library. Used by:
- Figma
- Linear
- Notion (partially)
- Many startups

\`\`\`typescript
/**
 * Yjs Integration for Real-Time Collaboration
 */

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { MonacoBinding } from 'y-monaco';
import * as monaco from 'monaco-editor';

// Create shared document
const ydoc = new Y.Doc();

// Create shared text type
const ytext = ydoc.getText('monaco');

// Connect to WebSocket server
const provider = new WebsocketProvider(
  'wss://collaboration.example.com',
  'document-123',
  ydoc
);

// Bind to Monaco editor
const editor = monaco.editor.create (document.getElementById('editor'), {
  value: ',
  language: 'javascript'
});

const binding = new MonacoBinding(
  ytext,
  editor.getModel(),
  new Set([editor]),
  provider.awareness  // For cursor/selection sharing
);

// Listen to document changes
ydoc.on('update', (update: Uint8Array) => {
  console.log('Document updated:', update);
  
  // Send to AI for analysis
  analyzeChangesWithAI(ytext.toString());
});

// Listen to remote changes
provider.on('sync', (isSynced: boolean) => {
  if (isSynced) {
    console.log('Document synchronized with server');
  }
});
\`\`\`

### CRDT Operations

\`\`\`typescript
/**
 * Understanding CRDT Operations
 */

// Insert operation
ytext.insert(5, 'Beautiful ');
// Generates: { type: 'insert', position: ID₅, content: 'Beautiful ' }

// Delete operation
ytext.delete(6, 5);  // Delete 5 characters starting at position 6
// Generates: { type: 'delete', start: ID₆, length: 5 }

// Operations are captured as binary updates
ydoc.on('update', (update: Uint8Array, origin: any) => {
  // Update is a compact binary representation
  // Can be sent over WebSocket efficiently
  websocket.send (update);
});

// Apply remote updates
websocket.on('message', (update: Uint8Array) => {
  Y.applyUpdate (ydoc, update);
  // Document automatically converges to correct state
});
\`\`\`

---

## WebSocket Architecture

### Server Implementation

\`\`\`typescript
/**
 * Collaboration WebSocket Server
 * Uses ws library + Yjs for CRDT
 */

import WebSocket from 'ws';
import * as Y from 'yjs';
import { setupWSConnection, setPersistence } from 'y-websocket/bin/utils';
import * as awarenessProtocol from 'y-protocols/awareness';

// Create WebSocket server
const wss = new WebSocket.Server({ port: 4000 });

// Store active documents in memory
const documents = new Map<string, Y.Doc>();

// Document persistence (PostgreSQL)
setPersistence({
  provider: 'postgres',
  bindState: async (docName: string, ydoc: Y.Doc) => {
    // Load document from database
    const persisted = await loadDocumentFromDB(docName);
    if (persisted) {
      Y.applyUpdate (ydoc, persisted);
    }
  },
  writeState: async (docName: string, ydoc: Y.Doc) => {
    // Save document to database
    const state = Y.encodeStateAsUpdate (ydoc);
    await saveDocumentToDB(docName, state);
  }
});

wss.on('connection', (ws: WebSocket, req: any) => {
  const docName = new URL(req.url, 'http://localhost').searchParams.get('doc');
  
  if (!docName) {
    ws.close();
    return;
  }

  console.log(\`Client connected to document: \${docName}\`);

  // Setup Yjs WebSocket connection
  setupWSConnection (ws, req, { docName });

  // Get or create document
  let ydoc = documents.get (docName);
  if (!ydoc) {
    ydoc = new Y.Doc();
    documents.set (docName, ydoc);
  }

  // Track active users for presence
  const awareness = awarenessProtocol.Awareness;
  
  ws.on('close', () => {
    console.log(\`Client disconnected from: \${docName}\`);
    
    // Clean up if no more clients
    if (wss.clients.size === 0) {
      documents.delete (docName);
    }
  });
});

console.log('Collaboration server running on ws://localhost:4000');
\`\`\`

### Client Integration

\`\`\`typescript
/**
 * React Component with Yjs Collaboration
 */

import React, { useEffect, useRef, useState } from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import CodeMirror from '@uiw/react-codemirror';
import { yCollab } from 'y-codemirror.next';

interface CollaborativeEditorProps {
  documentId: string;
  userId: string;
  userName: string;
}

export const CollaborativeEditor: React.FC<CollaborativeEditorProps> = ({
  documentId,
  userId,
  userName
}) => {
  const [doc] = useState(() => new Y.Doc());
  const [provider, setProvider] = useState<WebsocketProvider | null>(null);
  const [synced, setSynced] = useState (false);
  const [activeUsers, setActiveUsers] = useState<Map<number, any>>(new Map());

  useEffect(() => {
    // Connect to collaboration server
    const wsProvider = new WebsocketProvider(
      'wss://collab.example.com',
      documentId,
      doc
    );

    // Set user info for presence
    wsProvider.awareness.setLocalStateField('user', {
      id: userId,
      name: userName,
      color: getRandomColor()
    });

    // Track sync status
    wsProvider.on('sync', (isSynced: boolean) => {
      setSynced (isSynced);
    });

    // Track active users
    wsProvider.awareness.on('change', () => {
      setActiveUsers (new Map (wsProvider.awareness.getStates()));
    });

    setProvider (wsProvider);

    return () => {
      wsProvider.destroy();
    };
  }, [documentId, userId, userName]);

  const ytext = doc.getText('codemirror');

  return (
    <div className="collaborative-editor">
      <div className="editor-header">
        <div className="sync-status">
          {synced ? '✓ Synced' : '⟳ Syncing...'}
        </div>
        <div className="active-users">
          {Array.from (activeUsers.values())
            .filter (state => state.user)
            .map (state => (
              <div key={state.user.id} className="user-badge">
                <span 
                  className="user-indicator"
                  style={{ backgroundColor: state.user.color }}
                />
                {state.user.name}
              </div>
            ))}
        </div>
      </div>

      <CodeMirror
        value={ytext.toString()}
        extensions={[
          yCollab (ytext, provider?.awareness)
        ]}
        onChange={() => {
          // Changes automatically synced via Yjs
        }}
      />
    </div>
  );
};

function getRandomColor(): string {
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
  ];
  return colors[Math.floor(Math.random() * colors.length)];
}
\`\`\`

---

## Presence & Awareness

### Cursor Tracking

Show where other users are editing:

\`\`\`typescript
/**
 * Cursor and Selection Tracking
 */

import { Awareness } from 'y-protocols/awareness';

class CursorManager {
  private awareness: Awareness;
  private cursors: Map<number, HTMLElement> = new Map();

  constructor (awareness: Awareness, editorElement: HTMLElement) {
    this.awareness = awareness;

    // Listen to awareness changes
    awareness.on('change', (changes: any) => {
      this.updateCursors (changes, editorElement);
    });
  }

  updateCursors (changes: any, editorElement: HTMLElement) {
    // Added users
    changes.added.forEach((clientId: number) => {
      const state = this.awareness.getStates().get (clientId);
      if (state?.cursor) {
        this.createCursor (clientId, state, editorElement);
      }
    });

    // Updated users
    changes.updated.forEach((clientId: number) => {
      const state = this.awareness.getStates().get (clientId);
      if (state?.cursor) {
        this.updateCursor (clientId, state);
      }
    });

    // Removed users
    changes.removed.forEach((clientId: number) => {
      this.removeCursor (clientId);
    });
  }

  createCursor (clientId: number, state: any, editorElement: HTMLElement) {
    const cursor = document.createElement('div');
    cursor.className = 'remote-cursor';
    cursor.style.backgroundColor = state.user.color;
    cursor.style.position = 'absolute';
    
    // Position cursor
    const { line, ch } = state.cursor;
    const coords = this.getCoordinates (line, ch, editorElement);
    cursor.style.left = \`\${coords.x}px\`;
    cursor.style.top = \`\${coords.y}px\`;

    // Add user label
    const label = document.createElement('div');
    label.className = 'cursor-label';
    label.textContent = state.user.name;
    label.style.backgroundColor = state.user.color;
    cursor.appendChild (label);

    editorElement.appendChild (cursor);
    this.cursors.set (clientId, cursor);
  }

  updateCursor (clientId: number, state: any) {
    const cursor = this.cursors.get (clientId);
    if (!cursor) return;

    const { line, ch } = state.cursor;
    const coords = this.getCoordinates (line, ch, cursor.parentElement!);
    cursor.style.left = \`\${coords.x}px\`;
    cursor.style.top = \`\${coords.y}px\`;
  }

  removeCursor (clientId: number) {
    const cursor = this.cursors.get (clientId);
    if (cursor) {
      cursor.remove();
      this.cursors.delete (clientId);
    }
  }

  getCoordinates (line: number, ch: number, element: HTMLElement) {
    // Calculate pixel coordinates from line/character position
    // Implementation depends on editor library
    return { x: ch * 8, y: line * 20 };  // Simplified
  }

  // Update local cursor position
  updateLocalCursor (line: number, ch: number) {
    this.awareness.setLocalStateField('cursor', { line, ch });
  }
}

// Usage
const cursorManager = new CursorManager (provider.awareness, editorElement);

editor.on('cursorActivity', () => {
  const cursor = editor.getCursor();
  cursorManager.updateLocalCursor (cursor.line, cursor.ch);
});
\`\`\`

---

## AI Integration with Collaboration

### AI-Generated Suggestions

Integrating AI suggestions into collaborative environment:

\`\`\`typescript
/**
 * AI Suggestions in Collaborative Editor
 */

class CollaborativeAIAssistant {
  private ydoc: Y.Doc;
  private ytext: Y.Text;
  private provider: WebsocketProvider;
  private aiClient: OpenAI;

  constructor (ydoc: Y.Doc, provider: WebsocketProvider) {
    this.ydoc = ydoc;
    this.ytext = ydoc.getText('codemirror');
    this.provider = provider;
    this.aiClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // Listen to document changes
    this.ytext.observe (this.onDocumentChange.bind (this));
  }

  private debounceTimer: NodeJS.Timeout | null = null;

  private async onDocumentChange (event: Y.YTextEvent) {
    // Debounce AI calls (wait for user to stop typing)
    if (this.debounceTimer) {
      clearTimeout (this.debounceTimer);
    }

    this.debounceTimer = setTimeout (async () => {
      await this.generateSuggestion();
    }, 2000);  // 2 second delay
  }

  private async generateSuggestion() {
    const content = this.ytext.toString();
    
    // Don't generate if someone else is actively editing
    const activeUsers = Array.from (this.provider.awareness.getStates().values())
      .filter (state => state.user && state.cursor);
    
    if (activeUsers.length > 1) {
      console.log('Multiple users active, skipping AI suggestion');
      return;
    }

    try {
      // Generate AI suggestion
      const response = await this.aiClient.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful coding assistant. Suggest improvements to the code.'
          },
          {
            role: 'user',
            content: \`Current code:\\n\${content}\\n\\nSuggest one improvement.\`
          }
        ],
        temperature: 0.3
      });

      const suggestion = response.choices[0].message.content;

      // Broadcast suggestion to all users
      this.broadcastAISuggestion (suggestion);

    } catch (error) {
      console.error('AI suggestion error:', error);
    }
  }

  private broadcastAISuggestion (suggestion: string) {
    // Add suggestion as awareness state (doesn't modify document)
    this.provider.awareness.setLocalStateField('aiSuggestion', {
      content: suggestion,
      timestamp: Date.now(),
      id: Math.random().toString(36)
    });

    // Suggestion appears as a ghost text for all users
    // Users can accept (applies CRDT operation) or dismiss
  }

  // User accepts AI suggestion
  async acceptSuggestion (suggestionId: string) {
    const states = this.provider.awareness.getStates();
    let suggestion: any = null;

    // Find suggestion from any user's awareness state
    for (const state of states.values()) {
      if (state.aiSuggestion?.id === suggestionId) {
        suggestion = state.aiSuggestion;
        break;
      }
    }

    if (!suggestion) return;

    // Apply suggestion as CRDT operation
    // This ensures it syncs correctly with other users
    this.ydoc.transact(() => {
      this.ytext.insert (this.ytext.length, \`\\n\\n\${suggestion.content}\`);
    });

    // Clear suggestion from awareness
    this.provider.awareness.setLocalStateField('aiSuggestion', null);
  }
}

// Usage
const aiAssistant = new CollaborativeAIAssistant (doc, provider);

// UI: Show AI suggestions to all users
provider.awareness.on('change', () => {
  const states = provider.awareness.getStates();
  
  states.forEach((state, clientId) => {
    if (state.aiSuggestion) {
      showAISuggestion (state.aiSuggestion, () => {
        aiAssistant.acceptSuggestion (state.aiSuggestion.id);
      });
    }
  });
});
\`\`\`

---

## Scaling Collaboration

### Horizontal Scaling

Supporting 1000+ simultaneous editors:

\`\`\`typescript
/**
 * Horizontal Scaling with Redis PubSub
 */

import Redis from 'ioredis';
import { WebSocket, WebSocketServer } from 'ws';

class ScalableCollaborationServer {
  private wss: WebSocketServer;
  private redis: Redis;
  private redisSub: Redis;
  private documents: Map<string, Set<WebSocket>> = new Map();

  constructor() {
    this.wss = new WebSocketServer({ port: 4000 });
    this.redis = new Redis();
    this.redisSub = new Redis();

    // Subscribe to Redis channels
    this.redisSub.psubscribe('document:*');
    this.redisSub.on('pmessage', this.onRedisMessage.bind (this));

    this.wss.on('connection', this.onConnection.bind (this));
  }

  private onConnection (ws: WebSocket, req: any) {
    const docId = new URL(req.url, 'http://localhost').searchParams.get('doc');
    
    if (!docId) {
      ws.close();
      return;
    }

    // Add client to document's connection set
    if (!this.documents.has (docId)) {
      this.documents.set (docId, new Set());
    }
    this.documents.get (docId)!.add (ws);

    console.log(\`Client connected. Doc: \${docId}, Total: \${this.documents.get (docId)!.size}\`);

    // Handle messages from client
    ws.on('message', async (data: Buffer) => {
      // Broadcast to other servers via Redis
      await this.redis.publish(
        \`document:\${docId}\`,
        JSON.stringify({
          type: 'update',
          data: data.toString('base64'),
          sender: getClientId (ws)
        })
      );

      // Broadcast to local clients (except sender)
      this.broadcastToLocalClients (docId, data, ws);
    });

    ws.on('close', () => {
      this.documents.get (docId)?.delete (ws);
      
      // Cleanup empty document sets
      if (this.documents.get (docId)?.size === 0) {
        this.documents.delete (docId);
      }
    });
  }

  private onRedisMessage (pattern: string, channel: string, message: string) {
    const docId = channel.replace('document:', ');
    const payload = JSON.parse (message);

    // Broadcast to local clients
    const clients = this.documents.get (docId);
    if (!clients) return;

    const data = Buffer.from (payload.data, 'base64');
    
    clients.forEach (client => {
      // Don't send back to original sender
      if (getClientId (client) !== payload.sender) {
        client.send (data);
      }
    });
  }

  private broadcastToLocalClients (docId: string, data: Buffer, sender: WebSocket) {
    const clients = this.documents.get (docId);
    if (!clients) return;

    clients.forEach (client => {
      if (client !== sender && client.readyState === WebSocket.OPEN) {
        client.send (data);
      }
    });
  }
}

function getClientId (ws: WebSocket): string {
  // Store client ID on WebSocket object
  return (ws as any).clientId || ';
}

// Run multiple instances behind load balancer
const server = new ScalableCollaborationServer();
\`\`\`

---

## Operational Transformation (Alternative to CRDT)

### OT Basics

**Operational Transformation** is an alternative to CRDTs, used by Google Docs:

\`\`\`typescript
/**
 * Simple Operational Transformation Implementation
 */

type Operation = 
  | { type: 'insert', position: number, char: string }
  | { type: 'delete', position: number };

class OTDocument {
  private content: string = ';
  private version: number = 0;

  apply (op: Operation): void {
    if (op.type === 'insert') {
      this.content = 
        this.content.slice(0, op.position) +
        op.char +
        this.content.slice (op.position);
    } else {
      this.content =
        this.content.slice(0, op.position) +
        this.content.slice (op.position + 1);
    }
    this.version++;
  }

  // Transform operation against another operation
  transform (op1: Operation, op2: Operation): Operation {
    if (op1.type === 'insert' && op2.type === 'insert') {
      if (op1.position < op2.position) {
        return { ...op2, position: op2.position + 1 };
      }
      return op2;
    }

    if (op1.type === 'insert' && op2.type === 'delete') {
      if (op1.position <= op2.position) {
        return { ...op2, position: op2.position + 1 };
      }
      return op2;
    }

    if (op1.type === 'delete' && op2.type === 'insert') {
      if (op1.position < op2.position) {
        return { ...op2, position: op2.position - 1 };
      }
      return op2;
    }

    if (op1.type === 'delete' && op2.type === 'delete') {
      if (op1.position < op2.position) {
        return { ...op2, position: op2.position - 1 };
      } else if (op1.position > op2.position) {
        return op2;
      } else {
        // Both deleting same position - drop one
        return { type: 'delete', position: -1 };
      }
    }

    return op2;
  }
}

// Example: Concurrent operations
const doc = new OTDocument();
doc.apply({ type: 'insert', position: 0, char: 'H' });
doc.apply({ type: 'insert', position: 1, char: 'i' });
// Content: "Hi"

// User A and B make concurrent edits
const opA: Operation = { type: 'insert', position: 2, char: '!' };
const opB: Operation = { type: 'insert', position: 2, char: '?' };

// Transform opB against opA
const transformedB = doc.transform (opA, opB);
// transformedB: { type: 'insert', position: 3, char: '?' }

// Apply both operations
doc.apply (opA);      // "Hi!"
doc.apply (transformedB);  // "Hi!?"
// Both users see same result
\`\`\`

---

## Production Considerations

### Persistence Strategy

\`\`\`typescript
/**
 * Document Persistence with Snapshots
 */

import { Pool } from 'pg';
import * as Y from 'yjs';

class DocumentPersistence {
  private pool: Pool;

  constructor() {
    this.pool = new Pool({
      connectionString: process.env.DATABASE_URL
    });
  }

  async saveDocument (docId: string, ydoc: Y.Doc): Promise<void> {
    // Save full state
    const state = Y.encodeStateAsUpdate (ydoc);
    
    await this.pool.query(
      \`INSERT INTO documents (id, state, updated_at)
       VALUES ($1, $2, NOW())
       ON CONFLICT (id) DO UPDATE SET state = $2, updated_at = NOW()\`,
      [docId, state]
    );

    // Save operation log for debugging
    const operations = this.extractOperations (ydoc);
    await this.saveOperationLog (docId, operations);
  }

  async loadDocument (docId: string): Promise<Uint8Array | null> {
    const result = await this.pool.query(
      'SELECT state FROM documents WHERE id = $1',
      [docId]
    );

    return result.rows[0]?.state || null;
  }

  // Periodic snapshots (every 100 operations)
  private operationCount = 0;

  async maybeSnapshot (docId: string, ydoc: Y.Doc): Promise<void> {
    this.operationCount++;
    
    if (this.operationCount >= 100) {
      await this.saveDocument (docId, ydoc);
      this.operationCount = 0;
      console.log(\`Saved snapshot for \${docId}\`);
    }
  }

  private extractOperations (ydoc: Y.Doc): any[] {
    // Extract recent operations for debugging
    // Implementation depends on your needs
    return [];
  }

  private async saveOperationLog (docId: string, operations: any[]): Promise<void> {
    // Store operations for debugging and audit trail
    for (const op of operations) {
      await this.pool.query(
        'INSERT INTO operation_log (doc_id, operation, created_at) VALUES ($1, $2, NOW())',
        [docId, JSON.stringify (op)]
      );
    }
  }
}
\`\`\`

### Monitoring & Debugging

\`\`\`typescript
/**
 * Collaboration System Monitoring
 */

class CollaborationMetrics {
  private metrics: Map<string, number> = new Map();

  trackConnection (docId: string) {
    this.increment(\`connections.\${docId}\`);
    this.increment('connections.total');
  }

  trackDisconnection (docId: string) {
    this.decrement(\`connections.\${docId}\`);
    this.decrement('connections.total');
  }

  trackOperation (docId: string, opType: string) {
    this.increment(\`operations.\${docId}.\${opType}\`);
    this.increment(\`operations.total.\${opType}\`);
  }

  trackLatency (docId: string, latencyMs: number) {
    // Track average latency
    const key = \`latency.\${docId}\`;
    const current = this.metrics.get (key) || 0;
    this.metrics.set (key, (current + latencyMs) / 2);
  }

  private increment (key: string) {
    this.metrics.set (key, (this.metrics.get (key) || 0) + 1);
  }

  private decrement (key: string) {
    this.metrics.set (key, Math.max(0, (this.metrics.get (key) || 0) - 1));
  }

  getMetrics() {
    return Object.fromEntries (this.metrics);
  }

  // Export to Prometheus/DataDog
  async export() {
    // Send metrics to monitoring service
    console.log('Metrics:', this.getMetrics());
  }
}

// Usage
const metrics = new CollaborationMetrics();

setInterval(() => {
  metrics.export();
}, 10000);  // Export every 10 seconds
\`\`\`

---

## Conclusion

Building real-time collaborative AI applications requires:

1. **CRDT or OT**: Mathematical guarantees of convergence
2. **WebSocket Infrastructure**: Low-latency, persistent connections
3. **Presence Awareness**: Show where users are working
4. **AI Integration**: Suggestions that work with CRDT operations
5. **Horizontal Scaling**: Redis PubSub for multi-server deployments
6. **Persistence**: Snapshots and operation logs
7. **Monitoring**: Track connections, operations, latency

**Key Libraries**:
- **Yjs**: Best-in-class CRDT implementation
- **y-websocket**: WebSocket provider for Yjs
- **y-monaco/y-codemirror**: Editor bindings

**Challenges**:
- Complex to implement correctly
- Difficult to debug
- Performance at scale
- Conflict resolution edge cases

**Rewards**:
- Best-in-class user experience
- Competitive differentiation
- Viral growth through team features
- Higher user engagement

Real-time collaboration transforms your AI product from a single-player tool into a team platform—dramatically increasing value and stickiness.
`,
};
