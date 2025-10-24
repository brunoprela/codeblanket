import { MultipleChoiceQuestion } from '../../../types';

export const realTimeCollaborationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-rtc-mc-1',
    question:
      'What is the primary advantage of CRDTs over Operational Transformation for real-time collaborative editing?',
    options: [
      'CRDTs are faster than OT',
      "CRDTs work offline and don't require a central server, while OT needs central coordination",
      'CRDTs use less memory than OT',
      'OT cannot handle concurrent edits',
    ],
    correctAnswer: 1,
    explanation:
      'CRDTs (Conflict-free Replicated Data Types) are commutative - operations can be applied in any order and converge to the same state. This enables offline work and p2p collaboration without a central server. OT requires transforming operations relative to each other, needing a central authority. CRDTs are simpler to implement correctly and handle network partitions gracefully.',
  },
  {
    id: 'bcap-rtc-mc-2',
    question:
      'How should cursor positions be broadcast in a collaborative editor with 100+ concurrent users?',
    options: [
      'Broadcast to all users immediately on every mouse move',
      'Only broadcast to users in the same file, throttled to every 50ms',
      "Never show other users' cursors",
      'Broadcast once per second to all users',
    ],
    correctAnswer: 1,
    explanation:
      'Optimization: (1) File-level broadcasting - only users in same file see each other (not entire project), (2) Throttle to 50ms - balance between smoothness and network efficiency, (3) Aggregate bulk updates every 100ms instead of individual. Broadcasting to all 100 users on every mouse move would create 10,000 messages/sec (unsustainable). File-level + throttling reduces to ~200 messages/sec.',
  },
  {
    id: 'bcap-rtc-mc-3',
    question:
      'When an AI suggestion conflicts with a user edit in a collaborative editor, which should win?',
    options: [
      'AI suggestion always wins',
      'User edit always wins; AI re-generates with new context',
      'Show both and let user choose',
      'Whoever edited first wins',
    ],
    correctAnswer: 1,
    explanation:
      'User edits always take priority over AI suggestions. If user edits while AI is generating or after suggestion appears, their edit wins and AI suggestion is cancelled or marked stale. AI can re-generate with the new context. This ensures users never lose their work and maintains trust in the system. AI is assistive, not authoritative.',
  },
  {
    id: 'bcap-rtc-mc-4',
    question:
      'What is the recommended approach for handling network disconnections in real-time collaboration?',
    options: [
      'Lock the document until reconnected',
      'Continue allowing edits locally (CRDT), sync when reconnected, show offline indicator',
      'Discard all edits made while offline',
      'Reload the page immediately',
    ],
    correctAnswer: 1,
    explanation:
      "CRDTs enable offline-first editing: (1) User continues editing locally (no blocking), (2) Show clear offline indicator (yellow banner), (3) Buffer changes in local state, (4) On reconnect, sync buffered changes (CRDT guarantees convergence). This provides best UX - users don't lose work or productivity during temporary network issues.",
  },
  {
    id: 'bcap-rtc-mc-5',
    question:
      'How should presence information (cursors, selections) be stored?',
    options: [
      'In PostgreSQL with all other data',
      'In Redis (ephemeral, fast), not persistent storage',
      'In local storage only',
      'In a separate MongoDB cluster',
    ],
    correctAnswer: 1,
    explanation:
      "Presence is ephemeral (only matters while users are active) and high-frequency (updates every 50-100ms). Redis is perfect: (1) In-memory (fast reads/writes), (2) TTL support (auto-expire stale presence after 30s), (3) Pub/sub for broadcasting. Don't persist to PostgreSQL (wastes storage, slower). Keep presence separate from persistent data (documents, chat history).",
  },
];
