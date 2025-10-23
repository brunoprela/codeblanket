/**
 * Quiz questions for Diff Generation & Patch Application section
 */

export const diffgenerationpatchapplicationQuiz = [
  {
    id: 'fpdu-diff-q-1',
    question:
      'Design the diff generation system for a Cursor-like code editor. How would you efficiently detect, display, and apply changes?',
    hint: 'Consider real-time detection, visual presentation, minimal diffs, and atomic application.',
    sampleAnswer:
      'Cursor-like diff system: (1) Use SequenceMatcher for efficient change detection. (2) Generate minimal diffs to show only what changed. (3) Display with color coding (red for deletions, green for additions). (4) Show line numbers for context. (5) Allow user to accept/reject changes individually. (6) Apply atomically with backup. (7) Support undo/redo. (8) Handle concurrent edits with conflict detection.',
    keyPoints: [
      'Use SequenceMatcher for detection',
      'Generate minimal diffs',
      'Color-code changes for clarity',
      'Atomic application with backup',
      'Support undo/redo',
    ],
  },
  {
    id: 'fpdu-diff-q-2',
    question:
      'Compare line-based diffs versus character-based diffs. When would you use each approach?',
    hint: 'Think about granularity, performance, readability, and use cases.',
    sampleAnswer:
      'Line-based diffs: Show changes per line. Fast, readable, standard for code. Use for most code editing, version control, code review. Character-based diffs: Show exact character changes within lines. More precise but verbose. Use for small text changes, inline editing, word-processing. Cursor likely uses line-based for code, character-based for inline suggestions.',
    keyPoints: [
      'Line-based: fast, readable, standard',
      'Character-based: precise, detailed',
      'Use line-based for code',
      'Use character-based for inline edits',
      'Combine approaches as needed',
    ],
  },
  {
    id: 'fpdu-diff-q-3',
    question:
      'How would you handle merge conflicts when applying patches? Design a conflict resolution system.',
    hint: 'Consider conflict detection, presentation, resolution strategies, and user interaction.',
    sampleAnswer:
      'Conflict resolution: (1) Detect conflicts using three-way merge (original, change1, change2). (2) Mark conflict regions in file with <<<<<<, ======, >>>>>>. (3) Present both versions to user side-by-side. (4) Allow manual resolution or auto-resolution strategies (take ours, take theirs, merge intelligently). (5) Validate resolved conflicts. (6) Test merged result. (7) Save resolution for similar future conflicts.',
    keyPoints: [
      'Three-way merge for detection',
      'Mark conflicts clearly',
      'Present options to user',
      'Support auto-resolution',
      'Validate merged result',
    ],
  },
];
