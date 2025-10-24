export const frontendDevelopmentQuiz = [
  {
    id: 'bcap-fd-q-1',
    question:
      'Design the streaming UI for an AI chat application. Requirements: (1) Show tokens as they arrive (smooth, no flashing), (2) Support markdown rendering during streaming, (3) Handle code blocks, (4) Show "thinking" state before first token, (5) Allow user to stop generation mid-stream. How do you handle partial markdown (e.g., incomplete code block)? Compare SSE vs WebSocket for streaming.',
    sampleAnswer:
      'SSE with incremental markdown rendering: (1) Architecture: Server streams tokens via SSE (data: {"token": "x"}\\\\n\\\\n), client appends to buffer. (2) Smooth rendering: Use React state with debounced updates (every 50ms) to avoid excessive re-renders. Append tokens to array, join on render. (3) Markdown: Use react-markdown with custom components. Parse incrementally: show partial markdown if valid, hold if incomplete. (4) Incomplete markdown: Buffer incomplete code blocks: "\\\\`\\\\`\\\\`python\\\\n" → wait for closing "\\\\`\\\\`\\\\`" before rendering. Show placeholder: "Code loading..." (5) Thinking state: Show animated dots before first token arrives (SSE connection open but no data). Timer: If >3s with no token, show "Still thinking..." (6) Stop generation: Cancel button sends DELETE /chat/{id}, server closes SSE stream, client stops rendering. (7) Code blocks: Detect language, use syntax highlighting (prism.js), copy button, line numbers. SSE vs WebSocket: SSE advantages: Simpler (HTTP), auto-reconnect, works through proxies, sufficient for unidirectional (server→client). WebSocket: Bidirectional (needed if client sends during generation), slightly lower latency. Recommendation: SSE for chat (simpler), WebSocket if need real-time collaboration features. Performance: Render in requestAnimationFrame, virtualize long responses (only render visible portion).',
    keyPoints: [
      'SSE for streaming (simpler than WebSocket, auto-reconnect, HTTP-based)',
      'Incremental markdown: parse and render as valid chunks arrive',
      'Buffer incomplete structures (code blocks) until closing tag',
      'Debounce renders (50ms) to avoid performance issues',
      'Thinking state with animated dots, "Still thinking..." after 3s',
    ],
  },
  {
    id: 'bcap-fd-q-2',
    question:
      'Your AI image generation feature takes 30-120 seconds. Users complain it feels too slow. Design the UX for: (1) Progress indication, (2) Time estimates, (3) Allowing users to leave and come back, (4) Showing low-res preview while generating, (5) Queue position visibility. How do you make long waits feel shorter?',
    sampleAnswer:
      'Progressive enhancement UX: (1) Immediate feedback: On submit, show: "Generating... You\'re 5th in queue, ~2 min wait." (2) Progress bar: 0-100% with stages: "Queued (0-10%)", "Generating (10-90%)", "Upscaling (90-100%)". Update every 2-5s via polling /jobs/{id}. (3) Time estimates: Calculate from historical data: p95 generation time = 90s. Adjust by queue depth: If 10 in queue, estimate = 10 × 15s (avg queue wait) + 90s. Show: "~3 minutes remaining". Update as progresses. (4) Low-res preview: Generate SDXL Turbo preview (512×512) in 5s, show immediately with label: "Preview (full quality generating...)". (5) Leave and return: Store job_id in URL + localStorage, email notification when ready. Show: "Safe to close this tab, we\'ll email you." (6) Entertainment: Show: "Did you know..." AI facts, user\'s previous generations, example prompts. Carousel auto-plays every 10s. (7) Reduce perceived wait: Show other users\' generations (with permission), "Popular right now" gallery. Make waiting feel productive: suggest prompt improvements while waiting. (8) Queue position: "You\'re 3rd in line" with countdown. Show when moved up: "You\'re now 2nd! 🎉" (gamification). (9) Skip queue: Offer: "Jump to front for $1" (monetization). Psychology: Break into micro-goals (queued→generating→upscaling→done), show progress frequently, provide distraction (gallery), give control (notifications, can close tab).',
    keyPoints: [
      'Immediate feedback: queue position + time estimate based on historical data',
      'Progressive enhancement: low-res preview in 5s while full-res generates',
      'Break into stages: queued → generating → upscaling (feels faster)',
      'Allow users to leave: job_id in URL, email notifications when ready',
      'Reduce perceived wait: entertainment (facts, gallery), show progress frequently',
    ],
  },
  {
    id: 'bcap-fd-q-3',
    question:
      'Design the state management for an AI application with: chat history, document uploads, generation queue, user settings, and real-time notifications. Requirements: (1) Persist across page refreshes, (2) Sync across tabs, (3) Offline support, (4) Undo/redo for actions. Compare: Redux, Zustand, Jotai, and local-first approaches (IndexedDB + sync).',
    sampleAnswer:
      'Zustand + IndexedDB + sync layer: (1) Why Zustand: Simpler than Redux (less boilerplate), faster than Context API, supports middleware (persist, devtools). (2) State structure: {user, conversations: {}, documents: {}, queue: [], settings, notifications: []}. (3) Persistence: Zustand persist middleware → saves to IndexedDB on every change. Hydrate on page load. Selective: Only persist conversations/settings, not temporary UI state. (4) Sync across tabs: Use BroadcastChannel API, emit events when state changes, other tabs listen and update. (5) Offline support: Queue mutations when offline (queue: [{type: "send_message", payload}]), sync when reconnected. Use service worker for offline detection. (6) Undo/redo: Maintain history stack: {past: [], present: {}, future: []}. On action: push present to past, update present. Undo: pop past, move present to future. Limit: Keep last 20 states (memory). (7) Real-time notifications: WebSocket for server push, append to notifications array, show toast, persist to IndexedDB. (8) Optimistic updates: Update UI immediately (send_message), rollback if API fails. Local-first: For collaborative features, use: Yjs (CRDT) + IndexedDB, sync to server. Overkill for simple apps. Redux comparison: Redux verbose (actions, reducers, dispatch), Zustand simpler. Jotai: Atomic state, good for complex derived state. Recommendation: Zustand for most apps (80% use case), local-first (Yjs) only if real-time collaboration needed.',
    keyPoints: [
      'Zustand: simpler than Redux, fast, supports middleware (persist, devtools)',
      'Persist to IndexedDB, selective (conversations/settings, not UI state)',
      'Sync across tabs: BroadcastChannel API for state synchronization',
      'Offline: queue mutations, sync when reconnected, service worker for detection',
      'Undo/redo: history stack (past, present, future), limit to 20 states',
    ],
  },
];
