import { MultipleChoiceQuestion } from '../../../types';

export const buildingDocumentProcessingSystemMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bdps-mc-1',
      question: 'What is the best approach for handling scanned PDF documents?',
      options: [
        'Reject them - only accept text PDFs',
        'Use Tesseract OCR for text extraction, validate quality with spell check',
        'Manually transcribe each page',
        'Use regex to extract text',
      ],
      correctAnswer: 1,
      explanation:
        'Scanned PDFs require OCR: (1) Detect (if PyPDF2 extracts <50 characters, likely scanned), (2) Use Tesseract OCR (open-source, good quality), (3) Validate quality: run spell check, if >10% errors, flag for review or use GPT-4V as fallback, (4) Store both OCR text and confidence scores. Tesseract handles 90% of scanned docs, GPT-4V for complex layouts.',
    },
    {
      id: 'bcap-bdps-mc-2',
      question:
        'For high-volume document processing (10k docs/day), which architecture is most cost-effective?',
      options: [
        'Synchronous API processing',
        'AWS Lambda serverless',
        'Celery workers on spot instances',
        'Manual processing',
      ],
      correctAnswer: 2,
      explanation:
        'Celery workers on spot instances: (1) Horizontal scaling (add workers), (2) Spot instances (70% cheaper than on-demand), (3) No cold start (unlike Lambda), (4) 15min+ processing (Lambda has 15min limit). Cost: 20 workers × $0.05/hr spot × 24hr = $24/day vs $200/day on Lambda. Lambda good for <5min tasks, spot workers for long-running processing.',
    },
    {
      id: 'bcap-bdps-mc-3',
      question: 'How should you validate the quality of document extraction?',
      options: [
        'Trust all extraction is correct',
        'Check: text length vs expected, spell check, table row counts, LLM coherence check',
        'Only check manually',
        'Never validate',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-layer validation: (1) Length check: If extracted text <100 chars for 10-page doc, likely failed, (2) Spell check: >10% errors indicates OCR issues, (3) Table validation: Row counts match expected structure, (4) LLM coherence: Ask Claude "Does this text make sense?" for gibberish detection, (5) Sample validation: Human review 5% of extractions, measure accuracy. Automated validation catches 95% of errors.',
    },
    {
      id: 'bcap-bdps-mc-4',
      question:
        'What is the best approach for extracting structured data from invoices with inconsistent formats?',
      options: [
        'Template matching for all invoices',
        'Hybrid: template matching (60%) for known vendors + LLM (35%) for unknown + human review (5%)',
        'Pure LLM for all invoices',
        'Regex for all extractions',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid optimization: (1) Template matching for major vendors (60% of invoices, free, 99% accurate), (2) LLM structured output for unknown layouts (35%, $0.20 each), (3) Human review for low-confidence (5%, $2 each). Average cost: $0.17 vs $0.30 pure LLM (43% savings). Validate with business rules (amount = line items + tax), flag mismatches.',
    },
    {
      id: 'bcap-bdps-mc-5',
      question: 'How should multi-column PDF layouts be handled?',
      options: [
        'Ignore column structure',
        'Use PyMuPDF to analyze bounding boxes and reorder text logically',
        'Only process single-column documents',
        'Manually reformat each page',
      ],
      correctAnswer: 1,
      explanation:
        'PyMuPDF (or pdfplumber) provides bounding box coordinates for each text block. Algorithm: (1) Group blocks by vertical position (identify rows), (2) Within rows, sort by horizontal position (left-to-right), (3) Detect column boundaries (gaps >50px), (4) Read top-to-bottom within each column, then move to next column. This preserves logical reading order for multi-column layouts.',
    },
  ];
