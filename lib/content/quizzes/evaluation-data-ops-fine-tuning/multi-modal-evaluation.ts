/**
 * Discussion questions for Multi-Modal Evaluation section
 */

export const multiModalEvaluationQuiz = [
  {
    id: 'multimodal-eval-q-1',
    question:
      "You're building an image captioning system using GPT-4 Vision. Standard text metrics (BLEU, ROUGE) give you scores of 0.65, but human evaluators say 40% of captions are inaccurate. Why do automated metrics fail, and how would you design a better evaluation strategy combining automated and human evaluation?",
    hint: 'Consider that automated text metrics measure word overlap, not semantic correctness or visual grounding. Think about CLIPScore, human evaluation protocols, and error taxonomies.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Text metrics (BLEU/ROUGE) fail for vision: measure word overlap, not visual correctness',
      'CLIPScore: Image-text alignment using CLIP embeddings (0-1 similarity score)',
      'Object verification: Extract nouns from caption, check if detected in image with object detector',
      'Hallucination detection: Flag mentioned objects not present in image',
      'Human evaluation: Multi-dimensional (accuracy, completeness, fluency), error taxonomy',
      'Combined score: 40% CLIPScore + 40% object accuracy + 20% text metrics',
      'Targeted human review: Flag disagreements (high CLIP, low objects) or hallucinations (15-20% of cases)',
    ],
  },
  {
    id: 'multimodal-eval-q-2',
    question:
      'You\'re evaluating a video QA system (e.g., "What color shirt is the person wearing in the video?"). The model achieves 78% accuracy on short videos (<30 sec) but only 52% on long videos (>2 min). Diagnose why long videos are harder and design evaluation metrics specific to temporal understanding.',
    hint: 'Consider that long videos require understanding events across time, not just single frames. Think about temporal localization, action understanding, and temporal reasoning.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Long video challenges: Sparse relevance (5/60 frames relevant), context limits (180→50 frames), temporal reasoning needed',
      'Temporal localization: Measure if model identifies correct time segment (IoU >0.5)',
      'Question types: Single-frame (82%), duration (65%), sequence (48%), count (52%), causality (41%)',
      'Temporal coverage: Measure if model attends to full video (entropy/log (frames)), target >0.5',
      'Action segmentation: Test understanding of distinct actions in different segments',
      'Breakdown by length: Short (<30s) 78%, medium (30-120s) 65%, long (>120s) 52%',
      "Root cause: Model low temporal coverage (0.35), doesn't process full video, misses key segments",
    ],
  },
  {
    id: 'multimodal-eval-q-3',
    question:
      "You're evaluating a document understanding model (OCR + layout + content extraction from PDFs, invoices, forms). Traditional accuracy metrics show 92% on your test set, but production users report frequent errors. Design a comprehensive evaluation that catches real-world failure modes including layout errors, OCR mistakes, and logical extraction failures.",
    hint: 'Consider that documents have complex layouts, tables, multi-column text. Errors compound: OCR error → wrong text → wrong extraction. Think about component-level evaluation and end-to-end metrics.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Component-level evaluation: OCR (CER), layout (F1), tables (cell accuracy), extraction (exact match F1)',
      'Error compounding: Test with perfect OCR/layout to isolate bottlenecks (88% vs 83% vs 72% actual)',
      'OCR evaluation: Character Error Rate (CER), common confusions (O→0, l→1, S→5)',
      'Layout evaluation: Bounding box IoU >0.5, element type matching, precision/recall/F1',
      'Table evaluation: Row/column count, cell-level accuracy, perfect table match rate',
      'Information extraction: Exact match vs partial match (normalized), recall vs precision, hallucinations',
      'Stress tests: Low quality (68%), rotated (54%), handwritten (45%), complex tables (58%) reveal production failures',
    ],
  },
];
