/**
 * Quiz questions for Text File Processing section
 */

export const textfileprocessingQuiz = [
  {
    id: 'fpdu-text-proc-q-1',
    question:
      'When building a Cursor-like code editor that needs to track file changes, what strategies would you use for generating and applying diffs? Explain the trade-offs between different diff algorithms.',
    hint: 'Consider unified diff format, line-based vs character-based diffs, and how to handle conflicts.',
    sampleAnswer:
      "For a code editor, use line-based diffs (unified diff format) as they are human-readable and standard across tools like git. Python\'s difflib provides SequenceMatcher for computing diffs efficiently. Key strategies: (1) Use unified_diff for display - shows context lines and clear +/- indicators, (2) Use SequenceMatcher.get_opcodes() for programmatic change detection (equal, insert, delete, replace operations), (3) Store both old and new content for undo/redo, (4) Handle edge cases like empty files, trailing newlines, encoding differences. Trade-offs: Line-based diffs are intuitive but can be misleading for large line changes. Character-based diffs (Myers algorithm) are more precise but harder to read. Cursor likely uses line-based with syntax-aware diff highlighting for better code change visualization.",
    keyPoints: [
      'Line-based diffs (unified format) are standard and human-readable',
      'difflib.unified_diff for display, SequenceMatcher for processing',
      'Track operations: equal, insert, delete, replace',
      'Handle edge cases: encoding, line endings, trailing newlines',
      'Consider syntax-aware diffs for code (like Cursor)',
      'Store diffs for undo/redo functionality',
    ],
  },
  {
    id: 'fpdu-text-proc-q-2',
    question:
      'Explain different chunking strategies for processing large documents with LLMs. When would you use character-based vs paragraph-based vs sentence-based chunking?',
    hint: 'Consider context preservation, semantic boundaries, and token limits.',
    sampleAnswer:
      "Chunking strategies depend on the task and document type: (1) Character-based chunking: Simple but can break mid-word or mid-sentence. Use with overlap (e.g., 200 chars) to preserve context. Good for uniform processing when semantic boundaries don't matter. (2) Paragraph-based: Best for documents with clear structure. Respects semantic boundaries, ideal for summarization or Q&A. Combine paragraphs until reaching token limit. (3) Sentence-based: Natural units for reading comprehension. Use NLTK or spaCy for accurate splitting. Best for tasks requiring coherent text units. For production: prefer paragraph-based for documents, sentence-based for precise tasks, and always use overlap between chunks to prevent context loss. For code files (like Cursor processes), chunk by functions/classes using AST parsing for semantic boundaries.",
    keyPoints: [
      'Character-based: simple but breaks semantic boundaries',
      'Paragraph-based: respects document structure, best for documents',
      'Sentence-based: natural units, use NLTK/spaCy for accuracy',
      'Always include overlap between chunks (150-200 characters)',
      'For code: use AST-based chunking (functions/classes)',
      'Consider token limits: ~2000 chars ≈ 500 tokens',
    ],
  },
  {
    id: 'fpdu-text-proc-q-3',
    question:
      'How would you handle encoding issues when processing files from unknown sources? What is your error handling strategy for production systems?',
    hint: 'Think about encoding detection, fallback strategies, and data loss prevention.',
    sampleAnswer:
      'Production encoding strategy: (1) Always try UTF-8 first (modern standard), (2) Use chardet library to detect encoding from file sample (read first 10KB), (3) Implement fallback chain: UTF-8 → UTF-8-sig (BOM) → Latin-1/CP1252 → ASCII with errors="replace", (4) Never fail silently - log detected encoding and confidence level, (5) For critical applications, warn users about non-UTF-8 files and offer re-encoding, (6) Use errors="replace" or errors="ignore" as last resort but track that data may be lost. Tools like VS Code use similar strategies - detect encoding and show warning if confidence is low. For Cursor-like editors, store original encoding with file metadata to preserve it on save. Always test with files in multiple encodings during development.',
    keyPoints: [
      'Always try UTF-8 first, it is the modern standard',
      'Use chardet for automatic encoding detection',
      'Implement fallback chain with progressively permissive encodings',
      'Log encoding detection with confidence levels',
      'Use errors="replace" as last resort, not default',
      'Store original encoding to preserve it on save',
      'Test with various encodings in development',
    ],
  },
];
