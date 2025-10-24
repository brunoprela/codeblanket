import { MultipleChoiceQuestion } from '../../../types';

export const buildingAiCodeEditorMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-bace-mc-1',
    question:
      'What is the most effective way to parse a codebase for semantic understanding in an AI code editor?',
    options: [
      'Split code into 500-character chunks',
      'Use Tree-sitter for AST parsing and extract complete functions/classes',
      'Use regex to find function definitions',
      'Parse only the currently open file',
    ],
    correctAnswer: 1,
    explanation:
      'Tree-sitter provides language-agnostic AST parsing that extracts complete semantic units (functions, classes) with their context (imports, docstrings). This is far superior to arbitrary text chunking (loses context) or regex (fragile, language-specific). Semantic chunks enable accurate code understanding and retrieval.',
  },
  {
    id: 'bcap-bace-mc-2',
    question:
      'When should you regenerate code embeddings in an AI code editor?',
    options: [
      'Every time the user types a character',
      'Only when the user manually requests',
      'When files are saved and their content hash changes',
      'Never - embeddings should be static',
    ],
    correctAnswer: 2,
    explanation:
      'Regenerate embeddings when files change (detected via content hash comparison). Regenerating on every keystroke wastes API calls and money. Hash-based change detection ensures embeddings stay fresh without redundant API calls. Typical workflow: parse file on save → compute hash → if hash changed, re-embed only modified functions/classes.',
  },
  {
    id: 'bcap-bace-mc-3',
    question:
      'What is the best approach for generating complex refactorings (e.g., rename across 50 files)?',
    options: [
      'Use LLM to process all 50 files',
      'Use AST for finding references + LLM for edge cases (hybrid approach)',
      'Use regex find-and-replace',
      'Manually edit each file',
    ],
    correctAnswer: 1,
    explanation:
      'Hybrid approach: AST finds 99% of references accurately and instantly (free), LLM handles edge cases like string literals or comments (1%). Pure LLM is slow (30s) and expensive ($2-5). Pure AST misses semantic references. Hybrid achieves 99%+ accuracy in 2s for $0.10, best of both worlds.',
  },
  {
    id: 'bcap-bace-mc-4',
    question: 'How should context be prioritized for code completions?',
    options: [
      'Include the entire codebase every time',
      'Current file > open tabs > dependencies > vector search results',
      'Only the current line',
      'Random selection of files',
    ],
    correctAnswer: 1,
    explanation:
      'Prioritization: (1) Current file (highest relevance), (2) Open tabs (user is actively working on), (3) Dependencies (imported files), (4) Vector search results (semantically relevant). This ensures the most relevant context fits in the token limit while maintaining quality. Dynamic adjustment based on completion complexity.',
  },
  {
    id: 'bcap-bace-mc-5',
    question:
      'What is the optimal strategy for handling very large files (>5000 lines) in context?',
    options: [
      'Exclude them entirely',
      'Include the full file',
      'Extract signatures, types, and key functions; omit function bodies',
      'Only include the first 1000 lines',
    ],
    correctAnswer: 2,
    explanation:
      'For large files, extract the important structural information: imports, type definitions, class/function signatures, while omitting function bodies. This provides the LLM with enough context to understand the API and structure without wasting tokens on implementation details. Reduces token usage by 80-90% while maintaining comprehension.',
  },
];
