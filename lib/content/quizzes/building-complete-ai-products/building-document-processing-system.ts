export const buildingDocumentProcessingSystemQuiz = [
  {
    id: 'bcap-bdps-q-1',
    question:
      'Design a universal document processing pipeline that handles: PDFs, DOCX, XLSX, images (with OCR), emails, and HTML. For each format, what challenges exist and how do you handle them? How do you: preserve structure, extract metadata, handle multi-column layouts, process tables, and maintain accuracy? Include error handling and quality validation.',
    sampleAnswer:
      'Universal pipeline: (1) Format detection (magic bytes/MIME type). (2) Format-specific extraction. (3) Normalization to common schema. (4) Quality validation. PDF: PyMuPDF for text + layout, handle: scanned (Tesseract OCR), multi-column (analyze bounding boxes), forms (extract field names + values). DOCX: python-docx for structure, preserve: headings (h1-h6), lists, tables, images. XLSX: openpyxl, detect tables (contiguous ranges with headers), preserve formulas as metadata. Images: Tesseract OCR + GPT-4V for complex layouts. Email: Parse headers, extract body (handle HTML + plain text), process attachments recursively. HTML: BeautifulSoup, extract main content (remove nav/footer), preserve semantic structure. Normalization: Convert all to {type, title, sections: [{heading, content, level}], tables: [{headers, rows}], metadata: {author, date}}. Structure preservation: Store original format + normalized version. Tables: Extract as JSON {headers, rows}, use Claude to detect table purpose. Multi-column: Detect layout with PyMuPDF, reorder text logically. Validation: (1) Check extracted text length vs expected. (2) Run spell check, flag if >10% errors (indicates OCR issues). (3) Compare table row counts. (4) Use LLM to validate coherence: "Does this extracted text make sense?" Error handling: Retry with alternative method, fallback to vision model, log failures for manual review.',
    keyPoints: [
      'Format-specific extraction then normalize to common schema',
      'PDF challenges: scanned (OCR), multi-column (layout analysis), forms',
      'Use vision models (GPT-4V) as fallback for complex layouts',
      'Preserve both original structure and normalized version',
      'Quality validation: length checks, spell check, LLM coherence check',
    ],
  },
  {
    id: 'bcap-bdps-q-2',
    question:
      'Your document processing system handles 10k documents/day. How do you architect for: (1) Processing throughput, (2) Cost optimization, (3) Error recovery, (4) Monitoring/alerting? Compare: synchronous processing, async workers (Celery), and serverless (Lambda). Which is best for high-volume processing?',
    sampleAnswer:
      "Architecture: Async workers (Celery) with Redis queue. Reasoning: (1) Throughput: Horizontal scaling (add workers), parallel processing, no cold start. (2) Cost: Workers on spot instances (70% cheaper), scale to zero during low traffic. (3) Error recovery: Built-in retries, dead letter queue. Lambda downsides: 15min timeout (too short for 100-page PDFs), cold starts, expensive at scale ($20 per 1M requests). Synchronous: Blocks API server, no parallelism, can't scale. Implementation: (1) API receives document → Store in S3 → Create job in Redis → Return job_id. (2) Worker pool (10-50 workers) polls queue → Downloads from S3 → Processes → Uploads result → Updates job status. (3) Client polls /jobs/{id} for status. Throughput: 10k docs/day = 7 docs/min. Average processing: 2min/doc. Need: (7 × 2) = 14 workers minimum, use 20 for buffer. Cost: 20 workers × $0.05/hr (spot) × 24hr = $24/day vs $200/day on Lambda. Optimization: (1) Batch small documents (process 10 at once). (2) Cache OCR results (many duplicate scans). (3) Skip processing for known documents (hash-based dedup). Error recovery: Retry 3x with exponential backoff, after failures send to manual review queue. Monitoring: Track: processing time (p95), success rate, queue depth, worker utilization. Alert if: queue depth >1000, success rate <95%, p95 latency >5min.",
    keyPoints: [
      'Async workers (Celery) best for high-volume: horizontal scaling, no cold start',
      'Spot instances for 70% cost savings vs on-demand',
      'Lambda: expensive at scale, 15min timeout insufficient for large docs',
      'Batch small docs, cache results, hash-based deduplication',
      'Monitor: queue depth, success rate, latency, worker utilization',
    ],
  },
  {
    id: 'bcap-bdps-q-3',
    question:
      'You need to extract structured data from invoices: vendor name, amount, line items, tax, due date. Invoices have inconsistent formats (PDFs, images, different layouts). Design a system using: (1) LLM structured output, (2) Template matching, (3) OCR + field detection. Which approach is most accurate? Most cost-effective? How do you validate extraction quality?',
    sampleAnswer:
      'Multi-approach system: (1) Template matching for known vendors (fast, free, 99% accurate). (2) LLM structured output for unknown layouts. (3) Human review for low-confidence extractions. Template matching: Store invoice templates (vendor ID → field locations). On new invoice, detect vendor (logo/name), apply template, extract fields using coordinates. Works for 60% of invoices (major vendors). LLM approach: Use Claude with function calling / structured output. Prompt: "Extract invoice data: {vendor, amount, line_items: [{description, quantity, price}], tax, due_date}". Use GPT-4V for image invoices. Accuracy: 95% for clear layouts, 80% for complex. Cost: $0.10-0.30 per invoice. Validation: (1) Run both approaches, compare results. If differ, flag for review. (2) Business rules: amount = sum(line items) + tax, due_date > invoice_date. (3) Confidence scores: If LLM confidence <0.8, send to review. (4) Human validation: Random sample 5% of extractions, measure accuracy, retrain if <98%. Workflow: Invoice → Detect vendor → Try template (if known vendor) → If template fails or unknown vendor → Use LLM → Validate → If low confidence → Human review. Cost: 60% templates (free) + 35% LLM ($0.20 each) + 5% human ($2 each) = $0.17 avg vs $0.30 pure LLM. Accuracy: 98% with validation vs 95% LLM-only.',
    keyPoints: [
      'Multi-approach: templates (fast, free) for known vendors, LLM for unknown',
      'Template matching covers 60% of invoices at major vendors (zero cost)',
      'LLM structured output (Claude/GPT-4V) for inconsistent formats',
      'Validation: business rules, confidence thresholds, human review for edge cases',
      'Cost optimization: hybrid approach 43% cheaper than pure LLM (80.17 vs $0.30)',
    ],
  },
];
