import { QuizQuestion } from '../../../types';

export const textClassificationQuiz: QuizQuestion[] = [
  {
    id: 'text-classification-dq-1',
    question:
      'Explain the difference between multi-class and multi-label classification in NLP, with examples.',
    sampleAnswer: `Multi-class: one label per document. Multi-label: multiple labels per document.

**Multi-class:** Document belongs to exactly ONE class
- Sentiment: Positive, Negative, or Neutral (one only)
- Topic: Sports, Politics, Technology, etc. (one only)
- Output: Softmax (probabilities sum to 1)

**Multi-label:** Document can have MULTIPLE labels
- Tags: "Python", "Machine Learning", "Tutorial" (multiple)
- Emotions: "Happy" and "Excited" (multiple)
- Output: Sigmoid (independent probabilities)

**Implementation difference:**
- Multi-class: Cross-entropy loss, softmax activation
- Multi-label: Binary cross-entropy loss, sigmoid activation`,
    keyPoints: [
      'Multi-class: exactly one label per document',
      'Multi-label: multiple labels per document',
      'Multi-class uses softmax, multi-label uses sigmoid',
      'Different loss functions and evaluation metrics',
    ],
  },
  {
    id: 'text-classification-dq-2',
    question:
      'Your sentiment classifier has 95% accuracy but performs poorly in production. What could be wrong?',
    sampleAnswer: `High accuracy on test set doesn't guarantee production performance.

**Possible Issues:**

1. **Class Imbalance**: 95% positive reviews, model predicts all positive
   - Accuracy: 95% but useless
   - Solution: Use F1, precision, recall metrics

2. **Distribution Shift**: Training on formal text, production has slang
   - Model hasn't seen informal language
   - Solution: Train on diverse data

3. **Temporal Shift**: Training on old data, language evolves
   - New slang, topics not in training
   - Solution: Continuous retraining

4. **Domain Mismatch**: Trained on product reviews, used for movie reviews
   - Different language patterns
   - Solution: Fine-tune on target domain`,
    keyPoints: [
      'Accuracy can be misleading with class imbalance',
      'Distribution shift between training and production',
      'Use F1, precision, recall for imbalanced data',
      'Monitor production performance continuously',
    ],
  },
  {
    id: 'text-classification-dq-3',
    question:
      'Compare using BERT vs simpler models (Logistic Regression + TF-IDF) for text classification.',
    sampleAnswer: `Trade-off between accuracy and complexity.

**BERT:**
- Accuracy: 90-95%
- Inference: 100-500ms
- Training: Hours, requires GPU
- Model size: 110MB-1GB
- When: Accuracy critical, have compute

**Logistic Regression + TF-IDF:**
- Accuracy: 80-85%
- Inference: <10ms
- Training: Minutes, CPU fine
- Model size: <10MB
- When: Speed critical, limited resources

**Decision factors:**
- Accuracy needs vs latency requirements
- Available compute and budget
- Data size (BERT needs more data)`,
    keyPoints: [
      'BERT: higher accuracy but slower and resource-intensive',
      'Simple models: fast, lightweight, good for baselines',
      'Choose based on accuracy/speed/resource trade-offs',
    ],
  },
];
