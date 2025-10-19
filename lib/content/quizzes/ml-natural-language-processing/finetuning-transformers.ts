import { QuizQuestion } from '../../../types';

export const finetuningTransformersQuiz: QuizQuestion[] = [
  {
    id: 'finetuning-transformers-dq-1',
    question:
      'Compare full fine-tuning with parameter-efficient methods like LoRA. When would you choose each approach?',
    sampleAnswer: `Full fine-tuning updates all parameters while LoRA trains only small adapter matrices.

**Full Fine-tuning:**
- Updates all 110M+ parameters
- Better accuracy (typically +1-2%)
- Requires significant compute and memory
- Risk of catastrophic forgetting
- Each task needs separate full model

**LoRA (Low-Rank Adaptation):**
- Trains <1% of parameters (hundreds of thousands vs millions)
- Injects trainable rank decomposition matrices
- 10-100x less memory
- Faster training
- Can serve multiple tasks from one base model

**When to Use Full Fine-tuning:**
1. Large dataset (>100K examples)
2. Domain shift from pre-training
3. Critical accuracy requirements
4. Sufficient compute resources

**When to Use LoRA:**
1. Limited compute/memory
2. Multiple tasks to serve
3. Rapid iteration needed
4. Small datasets (<10K examples)
5. Edge deployment

**Practical Example:**
- Full fine-tuning: 16GB GPU, 6 hours, 110MB per task
- LoRA: 8GB GPU, 1 hour, 10MB adapter per task`,
    keyPoints: [
      'Full fine-tuning: all parameters, better accuracy, high resource cost',
      'LoRA: <1% parameters, 90%+ accuracy, 10-100x efficient',
      'Use full for large datasets, domain shift, critical accuracy',
      'Use LoRA for limited resources, multiple tasks, rapid iteration',
    ],
  },
  {
    id: 'finetuning-transformers-dq-2',
    question:
      'Explain why gradual unfreezing and discriminative learning rates improve fine-tuning performance.',
    sampleAnswer: `Lower layers learn general features; higher layers learn task-specific features. Different layers need different training strategies.

**Layer Hierarchy:**
- **Lower layers**: Syntax, grammar, basic patterns (general)
- **Middle layers**: Semantic relationships, context
- **Higher layers**: Task-specific patterns

**Gradual Unfreezing:**
1. Start: Train only classifier head
2. Then: Unfreeze top encoder layer
3. Finally: Unfreeze all layers

Benefits:
- Prevents catastrophic forgetting
- Stable training (small changes first)
- Better final performance
- Lower layers retain general knowledge

**Discriminative Learning Rates:**
\`\`\`
Embeddings: 1e-5 (barely change)
Lower encoders: 2e-5 (small updates)
Upper encoders: 5e-5 (moderate updates)
Classifier: 1e-4 (large updates)
\`\`\`

Why this works:
- General features already good from pre-training
- Large updates destroy pre-trained knowledge
- Task head needs most adaptation
- Balance preservation vs adaptation

**Empirical Results:**
- Uniform LR: 85% accuracy
- Discriminative LR: 87% accuracy
- +2% improvement from proper LR scheduling`,
    keyPoints: [
      'Lower layers: general features, need preservation',
      'Higher layers: task-specific, need adaptation',
      'Gradual unfreezing prevents catastrophic forgetting',
      'Discriminative LRs: smaller for lower layers, larger for upper',
      'Balances preserving pre-trained knowledge with task adaptation',
    ],
  },
  {
    id: 'finetuning-transformers-dq-3',
    question:
      'Your fine-tuned model overfits (95% train, 70% validation accuracy). Diagnose and propose solutions.',
    sampleAnswer: `Overfitting indicates model memorizes training data rather than learning generalizable patterns.

**Diagnosis:**
- Large train-validation gap (25% points)
- Likely causes: small dataset, high model capacity, insufficient regularization

**Solutions:**

**1. More Data:**
- Collect more labeled data (best solution)
- Data augmentation: back-translation, synonym replacement
- Semi-supervised: use unlabeled data

**2. Regularization:**
\`\`\`python
# Increase dropout
training_args = TrainingArguments(
    dropout=0.3,  # from 0.1
    attention_dropout=0.3,
    weight_decay=0.1,  # from 0.01
)
\`\`\`

**3. Early Stopping:**
\`\`\`python
EarlyStoppingCallback(early_stopping_patience=2)
# Stop when validation metric stops improving
\`\`\`

**4. Reduce Model Capacity:**
- Use DistilBERT instead of BERT
- Freeze more layers
- Smaller classifier head

**5. Learning Rate:**
- Lower LR (from 2e-5 to 5e-6)
- Shorter training (fewer epochs)
- More warmup steps

**6. Gradient Clipping:**
\`\`\`python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
\`\`\`

**Implementation Priority:**
1. Early stopping (immediate, no downside)
2. Dropout + weight decay (easy, effective)
3. Data augmentation (if possible)
4. Reduce model size (if accuracy allows)
5. Collect more data (if feasible)

**Expected Impact:**
- Early stopping: 70% → 75% val
- Regularization: 75% → 78% val
- More data: 78% → 82%+ val`,
    keyPoints: [
      'Overfitting: high train, low validation accuracy',
      'Solutions: more data, regularization, early stopping',
      'Increase dropout, weight decay, reduce model capacity',
      'Early stopping prevents memorization',
      'Data augmentation increases effective dataset size',
    ],
  },
];
