import { MultipleChoiceQuestion } from '../../../types';

export const buildingAiResearchAssistantMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bara-mc-1',
      question:
        'In a multi-agent research system, how do you prevent infinite loops between agents?',
      options: [
        'Agents can communicate unlimited times',
        'Set max depth (3 levels), visited URL tracking, and total timeout (5 min)',
        'Only allow one agent to run',
        'Agents never communicate with each other',
      ],
      correctAnswer: 1,
      explanation:
        'Prevent loops with multiple safeguards: (1) Max depth - agents can only delegate 3 levels deep (orchestrator → search → document → synthesis), (2) Visited tracking - never process same URL twice, (3) Global timeout - abort entire research after 5 minutes, (4) Circuit breaker - if agent fails 3x, skip it. These prevent runaway processes while allowing effective collaboration.',
    },
    {
      id: 'bcap-bara-mc-2',
      question:
        'For processing academic PDFs, when should you use vision models (GPT-4V) vs traditional parsing (PyPDF2)?',
      options: [
        'Always use vision models for best quality',
        'Use vision only for complex layouts, scanned documents, or figure analysis; parse text otherwise',
        'Never use vision models - too expensive',
        'Use vision for all text extraction',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid approach: Traditional parsing (PyPDF2/pdfplumber) for standard text (fast, cheap: $0.50 per 100 pages), vision models (GPT-4V) only for: (1) Complex layouts (multi-column, tables), (2) Scanned PDFs (OCR), (3) Figure/chart analysis (extracting data from images). Vision costs $10-20 per 100 pages but handles edge cases. Parse first, use vision only if needed - achieves 90% cost savings.',
    },
    {
      id: 'bcap-bara-mc-3',
      question:
        'How should a research agent handle conflicting information from different sources?',
      options: [
        'Always trust the first source',
        'Assign confidence scores, prefer primary sources, show multiple viewpoints when consensus lacking',
        'Ignore all conflicting information',
        'Only use one source',
      ],
      correctAnswer: 1,
      explanation:
        "Fact-checking agent handles conflicts: (1) Cross-reference claims across sources, (2) Assign confidence scores based on: source reliability (peer-reviewed > blog), recency, consistency, (3) Prefer primary sources over secondary, (4) When consensus lacking, present multiple viewpoints with evidence. Don't hide disagreement - show users the uncertainty with supporting evidence.",
    },
    {
      id: 'bcap-bara-mc-4',
      question:
        'What is the optimal way to show progress during a 10-minute research task?',
      options: [
        'Show nothing until complete',
        'Stream real-time updates: current task, sources found, insights, with progress bar',
        'Only show a simple loading spinner',
        'Email results when done',
      ],
      correctAnswer: 1,
      explanation:
        'Real-time progress via WebSocket: (1) Current task with animated indicator ("Searching for sources..."), (2) Sources as they\'re found (title + relevance), (3) Live insights panel (updates during synthesis), (4) Progress bar with estimated time, (5) Allow export at any point. Transparency reduces perceived wait time and builds trust. Users see value being created in real-time.',
    },
    {
      id: 'bcap-bara-mc-5',
      question: 'How should research sessions be saved and resumed?',
      options: [
        "Don't allow saving - users must complete in one session",
        'Store: query, plan, completed_tasks, intermediate_results; resume from last checkpoint',
        'Only save final results',
        'Require users to manually export and re-import',
      ],
      correctAnswer: 1,
      explanation:
        "Save complete research state: (1) Query and research plan, (2) Completed tasks and their results, (3) Intermediate findings, (4) Current progress (which step). On resume: Load state, continue from last step (don't restart). Generate session URL (shareable). This enables: long research (pause/resume), collaboration (share link), audit trail (review process).",
    },
  ];
