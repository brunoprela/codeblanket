export const realTimeCollaborationQuiz = [
  {
    id: 'bcap-rtc-q-1',
    question:
      'Explain Operational Transformation (OT) vs CRDTs for real-time collaborative editing. When would you choose each? How do they handle concurrent edits, network partitions, and conflict resolution? Design a hybrid approach that combines the strengths of both for an AI code editor with real-time collaboration.',
    sampleAnswer:
      "OT: Transform operations relative to each other. Example: User A inserts \"x\" at position 5, User B deletes character at position 3. OT transforms A's insert to position 4 (adjusted for B's delete). Pros: Smaller operations, efficient. Cons: Complex transformation functions, central server required, doesn't handle network partitions well. CRDTs: Each operation is commutative (order doesn't matter). Example: Yjs uses list CRDT where each character has unique ID. Pros: Eventually consistent, works offline, no central authority, simpler. Cons: Larger data structures (metadata overhead), harder to implement custom behaviors. Choose OT when: Need central authority, online-only, small documents. Choose CRDT when: Offline support needed, p2p, large-scale. Hybrid for AI editor: Use Yjs (CRDT) for document structure + OT for AI suggestions. CRDT ensures users' edits merge correctly. When AI generates suggestion, treat as special operation with timestamp. If conflict (user edited same line), user edit wins. AI re-generates with new context. Network partition: CRDT allows offline work, syncs when online. AI features disabled offline (switch to local model). Presence: Separate WebSocket channel (doesn't need CRDT, ephemeral).",
    keyPoints: [
      'OT: transform operations, central server, complex but efficient',
      'CRDT: commutative operations, offline-first, simpler conflict resolution',
      'CRDTs better for modern apps (offline support, distributed)',
      'Hybrid: CRDT for user edits, special handling for AI suggestions',
      'User edits always win over AI suggestions in conflicts',
    ],
  },
  {
    id: 'bcap-rtc-q-2',
    question:
      "Design a real-time presence system for an AI code editor showing: cursors, selections, who's viewing which file, typing indicators, and AI activity (when AI is generating). How do you handle: 100+ concurrent users in same project, network efficiency, and privacy (not everyone should see everyone)? What happens when user goes offline?",
    sampleAnswer:
      'Architecture: (1) WebSocket server (Socket.io or custom) per region. (2) Redis pub/sub for cross-server presence sync. (3) Presence data: {userId, fileId, cursor: {line, col}, selection, isTyping, lastSeen}. (4) AI activity: {fileId, generating: true, type: "completion/refactor"}. Scalability (100+ users): (1) Only broadcast presence for users in same file (not entire project). (2) Throttle cursor updates (send max every 50ms). (3) Aggregate: Send bulk updates every 100ms instead of individual. (4) Compression: Use binary protocol (Protocol Buffers) not JSON. Privacy: (1) File-level permissions: Only see presence of users in files you have access to. (2) "Stealth mode": User can disable presence broadcast. (3) Anonymous presence: Show "5 others viewing" without names. Network efficiency: (1) WebSocket for real-time, fallback to SSE. (2) Differential updates: Only send changes (cursor moved), not full state. (3) Cleanup: Remove stale presence after 30s of no heartbeat. Offline: (1) Client detects offline, stops broadcasting. (2) Server removes presence after heartbeat timeout. (3) On reconnect, re-broadcast current state. Database: Store last-active timestamp in PostgreSQL for persistent "who viewed this file" analytics.',
    keyPoints: [
      'File-level presence: only broadcast to users in same file',
      'Throttle cursor updates (50ms), aggregate bulk updates (100ms)',
      'Binary protocol + differential updates for network efficiency',
      'Privacy: file-level permissions, stealth mode, anonymous counts',
      'Heartbeat system: remove stale presence after 30s timeout',
    ],
  },
  {
    id: 'bcap-rtc-q-3',
    question:
      'Your real-time collaborative editor has an AI assistant that makes suggestions. How do you handle: (1) AI suggestion appearing while multiple users are typing, (2) One user accepts, others need to see the change, (3) Undo/redo with AI changes, (4) Attribution (who made the change: user or AI)? How does this integrate with your CRDT/OT system?',
    sampleAnswer:
      'AI suggestions as special operations: (1) Generation: AI generates suggestion, assigned unique ID, stored as "pending" operation. Broadcasted to all users in file with metadata: {type: "ai_suggestion", id, text, position, requestedBy: userId}. (2) Preview: Each client shows ghost text (like GitHub Copilot). Doesn\'t affect CRDT until accepted. (3) Acceptance: User accepts → transforms to normal insert operation, broadcasted as CRDT change. Other users see change appear. (4) Rejection: Delete pending suggestion. Concurrent edits: If user types while AI generating, AI watches CRDT stream, adjusts suggestion position. If edit conflicts with suggestion location, cancel suggestion. Multiple users accepting: First acceptance wins, transforms to CRDT insert. Other clients\' pending suggestions become stale, marked invalid. Undo/redo: Track operation type in history. Undo AI change: Treat like normal delete, but annotate as "undo AI suggestion". Attribution: Store in operation metadata: {author: userId, isAI: true/false, aiModel: "claude-3-5-sonnet"}. Show in gutter: green indicator for AI changes, blue for users. History view: Filter by AI vs human changes. Version control: Git commits show AI attribution in message: "Co-authored-by: AI Assistant".',
    keyPoints: [
      'AI suggestions as pending operations (not applied until accepted)',
      'Ghost text preview, broadcast to all users, each decides to accept',
      'First acceptance wins, others marked stale',
      'Track attribution in operation metadata (user vs AI)',
      'Undo/redo treats AI changes like normal operations with AI flag',
    ],
  },
];
