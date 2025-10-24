/**
 * Multiple choice questions for Multi-Modal Evaluation section
 */

export const multiModalEvaluationMultipleChoice = [
  {
    id: 'multimodal-eval-mc-1',
    question:
      'Your image captioning model gets BLEU score of 0.70 but humans say 35% of captions are inaccurate. What is the BEST metric to detect this mismatch?',
    options: [
      'ROUGE score (better than BLEU)',
      'CLIPScore (image-text alignment)',
      'Perplexity of generated captions',
      'Caption length distribution',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (CLIPScore) is best for detecting visual-text misalignment. BLEU/ROUGE measure word overlap with reference, not visual correctness. Example: Image: [dog with red ball], Generated: "cat with blue ball", Reference: "dog with red ball" → BLEU: 0.67 (high word overlap), CLIPScore: 0.42 (low image-text alignment). CLIPScore uses CLIP model to measure similarity between image embeddings and text embeddings. High score = text describes image well. Low score = mismatch. Expected: BLEU can be high even when caption is visually wrong (wrong objects/colors but similar sentence structure). CLIPScore catches these errors. Use CLIPScore > 0.7 as threshold for visual faithfulness. Perplexity (C) measures language fluency, not visual accuracy. Length (D) is not reliability indicator.',
  },
  {
    id: 'multimodal-eval-mc-2',
    question:
      'For video QA, your model achieves 85% on short videos (<30s) but 58% on long videos (>2min). What is the MOST likely cause?',
    options: [
      'Longer videos are inherently harder questions',
      'Model can only process limited frames, misses key moments in long videos',
      'Long videos have more noise/distractions',
      'Questions about long videos are more ambiguous',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (limited frame processing) is most likely. Vision-language models have context limits (e.g., GPT-4V processes ~50-100 frames). Short video (30s @ 1 FPS): 30 frames → all fit in context, can see entire video. Long video (3min @ 1 FPS): 180 frames → must subsample to 50-60 frames. Subsampling risks: Missing key moments (relevant action happens in dropped frames), losing temporal continuity. Question: "What did person do AFTER picking up box?" If "picking up box" frame was dropped, model can\'t answer. Solution: Hierarchical processing (summarize chunks), adaptive sampling (sample more during motion/scene changes), explicit temporal localization (find relevant segment first). Options A, C, D could contribute but don\'t explain the sharp drop (85%→58%). The systematic degradation with length suggests a technical limitation, not inherent difficulty.',
  },
  {
    id: 'multimodal-eval-mc-3',
    question:
      'You evaluate document OCR and get 98% character accuracy. However, downstream extraction tasks only achieve 75% accuracy. What is the issue?',
    options: [
      'OCR errors are random and downstream task is just harder',
      'OCR errors compound: small errors in critical fields (numbers, IDs) cause extraction failures',
      '98% is actually a low OCR accuracy',
      'Extraction model is poorly trained',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (error compounding) is correct. Even 98% character accuracy means 2% errors, and these errors are NOT random. Critical fields matter more: Invoice number: "INV12345" → "INV1Z345" (one char wrong → entire number wrong → 0% extraction accuracy). Amount: "$1,234.56" → "$1,Z34.56" (OCR confusion: 2→Z → extraction parses as invalid → fails). Non-critical errors matter less: Company name: "Acme Corp" → "Acne Corp" (funny but often doesn\'t break extraction). Problem: Extraction depends on exact matches for IDs, amounts, dates. One character wrong = entire field wrong. 98% character accuracy could mean: 95% of words perfect, 5% of words have errors, but those 5% might be the critical fields you need to extract! Solution: Measure field-level accuracy (not character-level), use multiple OCR models and vote, add extraction-specific error correction (e.g., validate amounts against expected formats).',
  },
  {
    id: 'multimodal-eval-mc-4',
    question:
      'For image generation evaluation, which metric is MOST correlated with human preference?',
    options: [
      'FID (Fréchet Inception Distance)',
      'IS (Inception Score)',
      'CLIP similarity to prompt',
      'Human preference ratings (obviously)',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (CLIP similarity) correlates best with human preference for prompt adherence. How they compare: FID (Option A): Measures distribution distance between generated and real images. Good for overall quality but doesn\'t measure prompt alignment. Example: Can generate beautiful images that don\'t match prompt → low FID, low human rating. IS (Option B): Measures diversity and confidence. Similar issue—doesn\'t check prompt. CLIP similarity (Option C): Measures how well image matches text prompt using CLIP model. High correlation with human ratings for "did the image follow the prompt?" Research shows ~0.65-0.75 correlation with human preference. Example: Prompt: "red car on beach", Generated image of red car on beach → CLIP: 0.82, human rating: 8/10. Generated: Blue car in city → CLIP: 0.43, human rating: 2/10. Option D is circular (use humans to predict humans). Best practice: Use CLIP similarity for automated evaluation (0.7+ threshold), validate with human ratings on subset.',
  },
  {
    id: 'multimodal-eval-mc-5',
    question:
      'You evaluate speech recognition and get 95% word accuracy. The CEO listens to a demo and says "it gets my name wrong every time!" What evaluation metric would have caught this?',
    options: [
      'Overall word accuracy is sufficient',
      'Named entity recognition (NER) accuracy',
      'Per-user word error rate',
      'Phonetic similarity score',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (NER accuracy) would catch this. Problem with overall word accuracy: Common words dominate: "the", "a", "is" are frequent and easy → inflate accuracy. Rare named entities (people, companies, products) are few but important → errors hidden in aggregate. CEO\'s name: Appears 5 times in 1000-word speech. Wrong every time → 5 errors out of 1000 words → 99.5% accuracy (looks great!). But 0% accuracy on the important thing (CEO\'s name). Solution - NER accuracy: Extract named entities (people, orgs, locations), measure accuracy separately. Overall WER: 95%, but NER accuracy: 60% (catches the CEO name issue). Why other options fail: Option A ignores the problem. Option C (per-user WER) might help but doesn\'t specifically target named entities. Option D (phonetic similarity) is useful but doesn\'t directly measure if names are correct. Best practice: Report overall WER AND entity-level accuracy (especially person names for voice assistants, product names for customer service).',
  },
];
