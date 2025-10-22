/**
 * Quiz questions for CSV & Structured Data section
 */

export const csvstructureddataQuiz = [
    {
        id: 'fpdu-csv-q-1',
        question: 'Design a system for processing large CSV files (> 1GB) that need to be analyzed by an LLM. How would you chunk, validate, and prepare the data efficiently?',
        hint: 'Consider memory constraints, chunking strategies, validation, and LLM context limits.',
        sampleAnswer: 'Large CSV processing: (1) Use pandas chunksize parameter to read in chunks without loading entire file. (2) Validate each chunk for required columns and data types. (3) Clean and normalize data per chunk. (4) Convert chunks to text format for LLM (markdown tables or JSON). (5) Maintain context between chunks with overlap or summary. (6) Process chunks in parallel when possible. (7) Aggregate results from LLM responses. Handle edge cases: encoding errors, malformed rows, missing values.',
        keyPoints: ['Use pandas chunksize for memory efficiency', 'Validate each chunk independently', 'Convert to LLM-friendly format', 'Maintain context between chunks', 'Process in parallel for speed'],
    },
    {
        id: 'fpdu-csv-q-2',
        question: 'How would you automatically detect CSV delimiter, encoding, and data types for unknown CSV files?',
        hint: 'Think about sniffing, validation, and fallback strategies.',
        sampleAnswer: 'Auto-detection strategy: (1) Use csv.Sniffer to detect delimiter from sample. (2) Try UTF-8 first, fallback to latin-1, then chardet for detection. (3) Use pandas infer_objects() or pd.to_numeric() for type detection. (4) Validate detected settings produce sensible results. (5) Provide confidence scores for detections. (6) Allow manual override if auto-detection fails.',
        keyPoints: ['Use csv.Sniffer for delimiter detection', 'Try common encodings first', 'Infer data types from content', 'Validate detection results', 'Provide confidence scores'],
    },
    {
        id: 'fpdu-csv-q-3',
        question: 'Compare CSV, JSON, and JSONL for data exchange in LLM applications. When would you use each format?',
        hint: 'Consider structure, readability, streaming, and LLM processing.',
        sampleAnswer: 'Format comparison: CSV best for tabular data, simple structure, Excel compatibility. JSON best for nested/hierarchical data, complex structures. JSONL best for streaming, large datasets, append-only logs. Use CSV for simple tables, JSON for complex nested data, JSONL for log-style data and streaming. LLM processing: all work, but choose based on data structure and use case.',
        keyPoints: ['CSV: tabular, simple, Excel-friendly', 'JSON: nested, hierarchical, complex', 'JSONL: streaming, large files, logs', 'Choose based on data structure', 'All work with LLMs'],
    },
];

