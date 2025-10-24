import { MultipleChoiceQuestion } from '../../../types';

export const frontendDevelopmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-fd-mc-1',
    question:
      'What is the optimal debounce interval for rendering streaming LLM responses?',
    options: [
      'Render on every token (no debounce)',
      'Debounce renders every 50ms to avoid excessive re-renders',
      'Debounce for 5 seconds',
      'Only render when streaming completes',
    ],
    correctAnswer: 1,
    explanation:
      '50ms debounce optimal: (1) Smooth appearance to users (20 FPS), (2) Avoids excessive re-renders (React performance), (3) Still feels real-time. Rendering every token (can be 50-100 tokens/sec) causes lag and high CPU. 5s delay feels frozen. Use requestAnimationFrame + debounce for best performance while maintaining smooth streaming UX.',
  },
  {
    id: 'bcap-fd-mc-2',
    question:
      'How should incomplete markdown be handled during streaming (e.g., opened code block without closing)?',
    options: [
      'Render incomplete markdown as-is',
      'Buffer incomplete structures (code blocks, tables) until closing tag arrives, show placeholder',
      'Never render markdown during streaming',
      'Ignore markdown completely',
    ],
    correctAnswer: 1,
    explanation:
      'Buffer incomplete markdown: (1) Detect opened but unclosed structures (```python without closing ```), (2) Buffer and show placeholder ("Code loading..."), (3) Render when structure completes, (4) Parse incrementally for completed chunks. This prevents flickering/broken rendering of partial markdown while still showing progress. Once closing tag arrives, render the complete formatted content.',
  },
  {
    id: 'bcap-fd-mc-3',
    question:
      'What is the best way to make a 2-minute wait for image generation feel shorter?',
    options: [
      'Show nothing until complete',
      'Progress bar + low-res preview (5s) + time estimate + entertainment (gallery, facts)',
      'Only show spinning loader',
      'Disable page until complete',
    ],
    correctAnswer: 1,
    explanation:
      'Reduce perceived wait: (1) Immediate feedback: "Generating... You\'re 3rd in queue, ~2 min", (2) Low-res preview fast (SDXL Turbo in 5s), (3) Progress bar with stages (queued→generating→upscaling), (4) Entertainment: gallery of popular generations, AI facts carousel, (5) Allow leaving: "Safe to close tab, we\'ll email you". Psychology: Show progress frequently, provide distraction, break into micro-goals, give control.',
  },
  {
    id: 'bcap-fd-mc-4',
    question:
      'Which state management library is recommended for AI applications with chat, uploads, and real-time notifications?',
    options: [
      'Redux (always the best)',
      'Zustand with IndexedDB persistence and BroadcastChannel for cross-tab sync',
      'Only useState',
      'No state management needed',
    ],
    correctAnswer: 1,
    explanation:
      'Zustand + IndexedDB + BroadcastChannel: (1) Zustand: Simpler than Redux (less boilerplate), faster than Context, middleware support, (2) IndexedDB: Persist conversations/settings across refreshes, (3) BroadcastChannel: Sync state across tabs in real-time, (4) Selective persistence: Save important data (conversations), not UI state. This handles: offline support, cross-tab sync, persistence - critical for AI apps.',
  },
  {
    id: 'bcap-fd-mc-5',
    question: 'How should undo/redo be implemented for AI application state?',
    options: [
      'No undo/redo support',
      'History stack: {past: [], present: {}, future: []}, limit to last 20 states',
      'Store unlimited history',
      'Only undo, no redo',
    ],
    correctAnswer: 1,
    explanation:
      'History stack pattern: (1) Maintain: past (array of previous states), present (current state), future (for redo), (2) On action: push present to past, update present, clear future, (3) Undo: pop from past, move present to future, (4) Redo: pop from future, move to present, (5) Limit: Keep last 20 states (memory management). This provides familiar undo/redo UX without unbounded memory growth.',
  },
];
