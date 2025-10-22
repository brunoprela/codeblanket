/**
 * Quiz questions for Binary File Handling section
 */

export const binaryfilehandlingQuiz = [
    {
        id: 'fpdu-binary-q-1',
        question: 'How would you build a universal file processor that can handle any file type by detecting format and routing to appropriate handlers?',
        hint: 'Consider magic number detection, format identification, handler registry, and error handling.',
        sampleAnswer: 'Universal file processor: (1) Read first few bytes to check magic numbers. (2) Use python-magic for MIME type detection. (3) Maintain handler registry mapping MIME types to processors. (4) Route file to appropriate handler (PDF→pdfplumber, DOCX→python-docx, etc.). (5) Extract content with format-specific logic. (6) Normalize output to common format (text, structured data). (7) Handle unknown formats gracefully with binary analysis. (8) Provide fallback to generic text extraction.',
        keyPoints: ['Detect file type via magic numbers', 'Route to format-specific handlers', 'Normalize output format', 'Handle unknown formats gracefully', 'Use python-magic for detection'],
    },
    {
        id: 'fpdu-binary-q-2',
        question: 'Explain magic numbers and how they are used for file type detection. Why are they more reliable than file extensions?',
        hint: 'Think about file signatures, extension spoofing, and detection reliability.',
        sampleAnswer: 'Magic numbers are signature bytes at the start of files identifying format. Examples: PDF starts with %PDF, ZIP with PK\\x03\\x04, PNG with \\x89PNG. More reliable than extensions because: (1) Extensions can be changed/spoofed. (2) Magic numbers are intrinsic to file format. (3) Extensions may be missing or incorrect. (4) Security: detect malicious files disguised with wrong extensions. Always validate via magic numbers not just extension for security.',
        keyPoints: ['Magic numbers: signature bytes identifying format', 'More reliable than file extensions', 'Extensions can be spoofed', 'Critical for security validation', 'Use python-magic library'],
    },
    {
        id: 'fpdu-binary-q-3',
        question: 'How would you extract data from a SQLite database file for LLM processing?',
        hint: 'Consider table discovery, data extraction, relationship preservation, and format conversion.',
        sampleAnswer: 'SQLite extraction: (1) Connect with sqlite3 library. (2) List tables from sqlite_master. (3) Extract each table to pandas DataFrame. (4) Identify relationships via foreign keys. (5) Convert to LLM-friendly format (markdown tables or JSON). (6) Include schema information for context. (7) Handle large tables with chunking. (8) Preserve data types and nulls.',
        keyPoints: ['List tables from sqlite_master', 'Extract to pandas DataFrames', 'Convert to LLM-friendly format', 'Include schema for context', 'Handle large tables with chunking'],
    },
];

