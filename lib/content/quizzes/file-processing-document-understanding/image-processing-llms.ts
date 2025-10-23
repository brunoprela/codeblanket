/**
 * Quiz questions for Image Processing for LLMs section
 */

export const imageprocessingllmsQuiz = [
  {
    id: 'fpdu-image-proc-q-1',
    question:
      'Design a production system for processing receipts and invoices from images. How would you combine OCR, vision LLMs, and structured data extraction to achieve high accuracy?',
    hint: 'Consider preprocessing, OCR for text extraction, vision LLM for understanding, validation, and structured output.',
    sampleAnswer:
      'Receipt/invoice processing system: (1) Image preprocessing: convert to grayscale, increase contrast, denoise, resize to optimal dimensions for OCR, binarize for better text detection. (2) OCR extraction: use pytesseract to extract all text with positions and confidence scores, identify key-value pairs spatially (label on left, value on right), detect table-like structures. (3) Vision LLM analysis: send preprocessed image to GPT-4V or Claude 3 with structured schema, prompt: "Extract invoice data: number, date, vendor, items, total", request JSON format output with specific fields. (4) Data validation: cross-check OCR text against LLM extraction, validate total matches sum of items, check date format and ranges, verify required fields present. (5) Confidence scoring: combine OCR confidence + LLM confidence, flag low-confidence extractions for human review. (6) Structured output: parse LLM JSON response, normalize field names and formats, validate against schema. (7) Error handling: retry with different preprocessing if extraction fails, fall back to pure OCR if vision LLM unavailable, always provide raw OCR text for manual review. Best practice: use OCR for precise text, vision LLM for semantic understanding and structure.',
    keyPoints: [
      'Preprocess images: grayscale, contrast, denoise, binarize',
      'Use OCR (pytesseract) for text extraction with confidence',
      'Use vision LLM for semantic understanding and structure',
      'Provide structured schema to guide LLM extraction',
      'Validate extracted data: cross-check, totals, required fields',
      'Flag low-confidence extractions for human review',
      'Combine OCR precision with LLM semantic understanding',
    ],
  },
  {
    id: 'fpdu-image-proc-q-2',
    question:
      'Compare traditional OCR (pytesseract) versus vision LLMs (GPT-4V, Claude 3) for text extraction from images. When would you use each approach, and how would you combine them?',
    hint: 'Consider accuracy, cost, speed, understanding capability, and structured output.',
    sampleAnswer:
      'OCR vs Vision LLMs comparison: (1) Traditional OCR (pytesseract): Pros - free, fast, works offline, precise character recognition, provides character positions. Cons - struggles with poor image quality, needs preprocessing, no semantic understanding, outputs raw text only. Best for: clean documents, precise text extraction, bulk processing, cost-sensitive applications. (2) Vision LLMs (GPT-4V, Claude 3): Pros - understands context and layout, extracts structured data, handles poor image quality, interprets diagrams/charts. Cons - costs per API call, slower, requires internet, may hallucinate. Best for: complex documents, semantic understanding needed, structured data extraction, diagram interpretation. Combination strategy: (1) Use OCR first for clean text extraction - fast and free. (2) If OCR confidence is low or structured output needed, use vision LLM. (3) For invoices/forms: OCR for precise numbers, vision LLM for structure. (4) For diagrams/charts: vision LLM only. (5) Hybrid: OCR provides text, vision LLM provides structure and validation. Production pattern: tiered approach - try OCR first, escalate to vision LLM when needed, cache vision LLM results.',
    keyPoints: [
      'OCR: fast, free, precise, but no understanding',
      'Vision LLMs: semantic understanding, structured output, costlier',
      'Use OCR for clean documents and precise text',
      'Use vision LLMs for complex layouts and structured data',
      'Combine: OCR for text, vision LLM for structure',
      'Tiered approach: OCR first, escalate to vision LLM',
      'Cache expensive vision LLM results',
    ],
  },
  {
    id: 'fpdu-image-proc-q-3',
    question:
      'How would you optimize image processing costs when using vision LLMs at scale? What strategies would reduce API calls while maintaining quality?',
    hint: 'Think about preprocessing, caching, batching, resolution optimization, and selective processing.',
    sampleAnswer:
      'Cost optimization strategies: (1) Image preprocessing and optimization: resize images to minimum required resolution (1024x1024 often sufficient), compress without losing critical details, crop to relevant regions only, convert to optimal format (JPEG for photos, PNG for text). (2) Caching: cache vision LLM results by image hash, store structured extractions for reuse, cache common document types (standard invoice formats). (3) Selective processing: use OCR first to determine if vision LLM needed, only use vision LLM when OCR confidence < threshold or structured output required, route simple images to OCR, complex to vision LLM. (4) Batch processing: queue non-urgent requests, process during off-peak hours if pricing varies, batch similar images together. (5) Model selection: use smaller/cheaper models when possible (gpt-4-vision vs gpt-4o), Claude Haiku for simple tasks vs Opus for complex. (6) Prompt optimization: use concise prompts to reduce output tokens, request specific fields only not full description, use structured output formats. (7) Pre-validation: detect blank/corrupted images before API call, validate image contains expected content type. (8) Progressive enhancement: start with OCR, add vision LLM only for ambiguous cases. Monitor: track cost per document type, identify optimization opportunities, A/B test preprocessing techniques.',
    keyPoints: [
      'Resize and compress images before sending to APIs',
      'Cache vision LLM results by image hash',
      'Use OCR first, escalate to vision LLM when needed',
      'Batch non-urgent requests for efficiency',
      'Choose appropriate model: Haiku for simple, Opus for complex',
      'Optimize prompts to reduce output tokens',
      'Pre-validate images to avoid wasteful API calls',
    ],
  },
];
