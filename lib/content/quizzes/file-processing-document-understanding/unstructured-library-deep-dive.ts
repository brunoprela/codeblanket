/**
 * Quiz questions for Unstructured Library Deep Dive section
 */

export const unstructuredlibrarydeepdiveQuiz = [
  {
    id: 'fpdu-unstructured-q-1',
    question:
      'Design a production document processing pipeline using Unstructured that handles multiple file types, extracts content, and prepares it for LLM consumption.',
    hint: 'Consider file type detection, processing strategies, error handling, and output formatting.',
    sampleAnswer:
      'Document pipeline: (1) File upload/detection using Unstructured partition_auto. (2) Process with appropriate strategy (hi_res for PDFs with layout). (3) Extract elements by type (text, tables, titles). (4) Structure content with hierarchy preserved. (5) Convert tables to markdown or JSON. (6) Chunk text for LLM context limits. (7) Handle errors gracefully with fallback. (8) Cache processed results. (9) Provide metadata for context.',
    keyPoints: [
      'Use partition_auto for any file type',
      'Choose strategy based on content',
      'Extract and structure elements',
      'Convert tables appropriately',
      'Chunk for LLM limits',
      'Cache results',
      'Handle errors',
    ],
  },
  {
    id: 'fpdu-unstructured-q-2',
    question:
      'Compare Unstructured versus using specific libraries (pdfplumber, python-docx) for document processing. When would you use each?',
    hint: 'Think about flexibility, accuracy, performance, and specific needs.',
    sampleAnswer:
      'Comparison: Unstructured provides unified API for all formats, automatic detection, good for varied inputs. Specific libraries offer deeper control, better accuracy for specific formats, faster for single format. Use Unstructured when: processing mixed file types, need quick integration, want unified API. Use specific libraries when: single file type, need maximum accuracy, performance critical, advanced features needed (Excel formulas, Word styles).',
    keyPoints: [
      'Unstructured: unified, automatic, versatile',
      'Specific libraries: accurate, deep control',
      'Unstructured for mixed formats',
      'Specific for single format depth',
      'Consider accuracy vs convenience',
    ],
  },
  {
    id: 'fpdu-unstructured-q-3',
    question:
      'How would you optimize Unstructured processing for large-scale document pipelines?',
    hint: 'Consider parallelization, caching, strategy selection, and resource management.',
    sampleAnswer:
      'Optimization strategies: (1) Parallel processing with worker pools. (2) Cache results by file hash. (3) Use fast strategy for simple documents. (4) Queue-based architecture for async processing. (5) Batch similar file types together. (6) Monitor resource usage (CPU, memory). (7) Implement retry logic with exponential backoff. (8) Pre-filter files by type for optimal strategy.',
    keyPoints: [
      'Parallel processing',
      'Cache by file hash',
      'Choose fast strategy when possible',
      'Queue-based async processing',
      'Batch processing',
      'Monitor resources',
    ],
  },
];
