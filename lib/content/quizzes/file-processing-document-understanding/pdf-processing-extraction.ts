/**
 * Quiz questions for PDF Processing & Extraction section
 */

export const pdfprocessingextractionQuiz = [
  {
    id: 'fpdu-pdf-proc-q-1',
    question:
      'Design a PDF processing pipeline for a document Q&A system. How would you handle different types of PDFs (searchable vs scanned), extract tables, and prepare content for LLM processing?',
    hint: 'Consider PDF type detection, text extraction strategies, table handling, and chunking for LLM context limits.',
    sampleAnswer:
      'PDF processing pipeline: (1) PDF type detection: attempt text extraction with pdfplumber, if minimal text found (<100 chars per page), classify as scanned. (2) Content extraction strategy: For searchable PDFs, use pdfplumber for text and tables. For scanned PDFs, use OCR (pytesseract) with pdf2image. (3) Table extraction: use pdfplumber.extract_tables() with custom settings, convert to pandas DataFrames, clean and validate data. (4) Structure preservation: maintain page numbers, identify sections, extract headers. (5) Content chunking: split text by paragraphs or fixed chunks with overlap, keep table context intact. (6) Metadata extraction: capture title, author, page count for context. (7) LLM preparation: provide document structure summary, include relevant chunks based on query, embed tables as markdown for LLM understanding. (8) Caching: store extracted content to avoid reprocessing. Handle edge cases: encrypted PDFs, images within text, multi-column layouts, footnotes.',
    keyPoints: [
      'Detect PDF type: searchable vs scanned via text extraction test',
      'Use appropriate extraction method: pdfplumber vs OCR',
      'Extract tables separately with pdfplumber.extract_tables()',
      'Maintain structure: page numbers, sections, headers',
      'Chunk intelligently for LLM context limits with overlap',
      'Extract and provide metadata for context',
      'Cache processed content to avoid reprocessing',
    ],
  },
  {
    id: 'fpdu-pdf-proc-q-2',
    question:
      'Compare PyPDF2, pdfplumber, and PyMuPDF (fitz) for PDF processing. When would you use each library, and how would you combine them for optimal results?',
    hint: 'Consider speed, accuracy, table extraction, image handling, and API complexity.',
    sampleAnswer:
      'Library comparison: (1) PyPDF2: Pure Python, lightweight, good for basic text extraction and PDF manipulation (merge, split). Use for simple PDFs when you need basic text only. Cons: poor table handling, struggles with complex layouts. (2) pdfplumber: Built on pdfminer.six, excellent table extraction with customizable settings, layout-aware text extraction, visual debugging. Use for documents with tables and when accuracy matters more than speed. Best for structured documents like reports, invoices. (3) PyMuPDF (fitz): Fastest library, excellent image extraction, PDF rendering, comprehensive features. Use for image-heavy PDFs, when performance is critical, or need visual rendering. Combination strategy: (1) Use PyMuPDF for initial pass - fast metadata extraction, image extraction, determine if text-heavy. (2) Use pdfplumber for text and tables - more accurate layout preservation. (3) Fall back to PyPDF2 for PDF manipulation tasks. Example: extract images with PyMuPDF, extract tables with pdfplumber, process text with pdfplumber.',
    keyPoints: [
      'PyPDF2: Basic operations, lightweight, limited accuracy',
      'pdfplumber: Best for tables, accurate layout, slower',
      'PyMuPDF: Fastest, best for images, comprehensive features',
      'Combine libraries: PyMuPDF for images, pdfplumber for tables',
      'Choose based on: document type, performance needs, accuracy requirements',
      'Test with sample PDFs to determine best library for use case',
    ],
  },
  {
    id: 'fpdu-pdf-proc-q-3',
    question:
      'How would you handle table extraction from PDFs with complex layouts (merged cells, nested tables, irregular borders)? What validation would you perform on extracted tables?',
    hint: 'Think about extraction settings, data cleaning, validation strategies, and error handling.',
    sampleAnswer:
      'Complex table extraction: (1) Extraction settings: use pdfplumber with custom table_settings - try "lines" strategy first for bordered tables, fall back to "text" strategy for borderless. Adjust snap_tolerance and join_tolerance to handle irregular spacing. (2) Multiple extraction attempts: try different setting combinations, compare results, choose best match based on expected structure. (3) Post-processing: merge cells by detecting empty/null values in expected positions, clean whitespace and newlines, detect header rows (often bold or first row). (4) Validation: check row/column count consistency, validate data types (numeric columns should parse as numbers), check for duplicate headers, ensure no completely empty rows/columns, validate totals/sums if financial data. (5) Manual inspection support: provide visual representation of detected table boundaries, allow user correction of extraction settings. (6) Fallback strategies: if automated extraction fails, provide page image with OCR as fallback, allow manual table definition. Production approach: extract, validate, flag low-confidence extractions for human review, learn from corrections to improve settings.',
    keyPoints: [
      'Use custom table_settings in pdfplumber for accuracy',
      'Try multiple extraction strategies: lines vs text',
      'Post-process: merge cells, clean whitespace, detect headers',
      'Validate: check dimensions, data types, totals, consistency',
      'Provide visual debugging to inspect detected table boundaries',
      'Flag low-confidence extractions for human review',
      'Implement fallback: OCR or manual table definition',
    ],
  },
];
