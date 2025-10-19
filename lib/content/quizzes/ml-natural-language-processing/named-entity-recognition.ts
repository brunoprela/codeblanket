import { QuizQuestion } from '../../../types';

export const namedEntityRecognitionQuiz: QuizQuestion[] = [
  {
    id: 'ner-dq-1',
    question: 'Explain the IOB tagging scheme and why it is necessary for NER.',
    sampleAnswer: `IOB tags mark entity boundaries:
- B- (Begin): First token of entity  
- I- (Inside): Continuation tokens
- O (Outside): Non-entity tokens

**Why necessary:** Distinguishes adjacent entities of same type.
Example: "New York Times Square"
- Without IOB: "New York" and "Times Square" would be ambiguous
- With IOB: "New/B-LOC York/I-LOC Times/B-LOC Square/I-LOC"

Enables models to identify separate entities even when adjacent.`,
    keyPoints: [
      'B/I/O tags mark entity boundaries',
      'Critical for distinguishing adjacent entities',
      'Token-level classification task',
    ],
  },
  {
    id: 'ner-dq-2',
    question:
      'Why is label alignment necessary when tokenizing for NER with transformers?',
    sampleAnswer: `Transformer tokenizers may split words into subwords, but NER labels are per word. Must align subword tokens to original labels.

Example: "Apple's" → ["Apple", "'", "s"]
- Original label: B-ORG for "Apple's"
- Must handle: Which subword gets the label?
- Solution: First subword gets label, rest marked -100 (ignored)

Without alignment, training fails due to label/token mismatch.`,
    keyPoints: [
      'Subword tokenization splits words',
      'NER labels are per word, not subword',
      'First subword gets label, rest ignored (-100)',
    ],
  },
  {
    id: 'ner-dq-3',
    question: 'Compare token-level F1 vs entity-level F1 for evaluating NER.',
    sampleAnswer: `**Token-level F1:** Evaluates each token independently
- "New/B-LOC York/I-LOC" correctly tagged → 2/2 correct

**Entity-level F1:** Evaluates complete entities
- Must get ALL tokens right for entity to count
- "New/B-LOC York/O" → entity WRONG despite 50% tokens correct

**Entity-level is stricter and more meaningful:**
- Partial entities are useless in applications
- Better reflects real-world performance
- Standard metric (seqeval library)`,
    keyPoints: [
      'Token-level: per-token accuracy',
      'Entity-level: complete entity must be correct',
      'Entity-level stricter, more meaningful for applications',
    ],
  },
];
