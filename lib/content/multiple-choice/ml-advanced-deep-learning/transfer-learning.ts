/**
 * Transfer Learning Multiple Choice Questions
 */

export const transferLearningMultipleChoice = [
  {
    id: 'mc-1',
    question:
      'What is the primary advantage of transfer learning over training from scratch?',
    options: [
      'Transfer learning always trains faster',
      'Transfer learning achieves high performance with much less labeled data',
      'Transfer learning uses fewer parameters',
      'Transfer learning works better on large datasets',
    ],
    correctAnswer: 1,
    explanation:
      "The key advantage is **data efficiency** - achieving high performance with limited labeled data: **From Scratch**: Needs massive labeled data (ImageNet: 1.4M images). With small dataset (e.g., 1000 images): Severe overfitting, 60-70% accuracy. **Transfer Learning**: Pre-trained on large dataset, fine-tuned on your small dataset. Same 1000 images: 85-95% accuracy (10-25% improvement!). **Why it works**: Pre-trained model learned general features (edges, textures, shapes for vision; grammar, semantics for NLP). Your task-specific training only needs to adapt these features, not learn from scratch. Analogy: Learning Spanish after knowing French (transfer grammar/vocabulary) vs. learning first language (everything new). Training time isn't always faster (fine-tuning still takes time), parameter count is similar, and transfer learning is MOST beneficial for SMALL datasets (large datasets can train from scratch effectively). The democratization of deep learning: You don't need Google-scale data anymore!",
  },
  {
    id: 'mc-2',
    question:
      'When fine-tuning a pre-trained model, why should you use a lower learning rate than when training from scratch?',
    options: [
      'To make training slower and more stable',
      'To prevent catastrophically forgetting the useful features learned during pre-training',
      'To reduce memory usage',
      'Because pre-trained models have fewer parameters',
    ],
    correctAnswer: 1,
    explanation:
      'Lower learning rate prevents **catastrophic forgetting**: **High LR (e.g., 1e-3)**: Large weight updates. Pre-trained weights: W_old = [valuable features learned on ImageNet]. After one update: W_new = W_old - 1e-3 × ∇L. Large changes can **destroy** useful pre-trained features! Model forgets what it learned, performance collapses. **Low LR (e.g., 1e-5)**: Small weight updates. W_new ≈ W_old (subtle adjustments). Preserves general features, adapts task-specific ones. Gradual fine-tuning vs. aggressive retraining. **Analogy**: Editing a masterpiece painting (gentle brush strokes) vs. painting over it (destruction). Typical LRs: From scratch: 1e-3 to 1e-2, Fine-tuning: 1e-5 to 1e-4 (10-100× smaller). Training stability is a side benefit, but the PRIMARY reason is preserving pre-trained knowledge. Memory usage and parameter count are unaffected by learning rate.',
  },
  {
    id: 'mc-3',
    question:
      'You have 200 labeled images for a new classification task. The task is very similar to ImageNet (both natural object photos). What is the best strategy?',
    options: [
      'Train from scratch with heavy data augmentation',
      'Use feature extraction (freeze backbone, train only classifier)',
      'Fine-tune all layers with high learning rate',
      'Use a smaller model architecture to prevent overfitting',
    ],
    correctAnswer: 1,
    explanation:
      '**Feature extraction** (frozen backbone) is optimal for this scenario: **Analysis**: (1) **Very small dataset** (200 images): Risk of overfitting if training too many parameters, (2) **High task similarity**: ImageNet → natural objects. Pre-trained features are HIGHLY relevant! **Strategy**: Feature extraction: Freeze pre-trained backbone (ResNet conv layers), Train only new classifier head (~20K parameters). Benefits: (1) Fast training (only classifier, 5-10 min), (2) Low overfitting risk (few parameters), (3) Leverages pre-trained features (edges, textures, objects), (4) Typically 85-90% accuracy. **Why not alternatives?** (A) From scratch: 200 images WAY too small, will overfit terribly (<60% accuracy). (C) Fine-tune all: Too many parameters for 200 images, will overfit. (D) Smaller model: Loses capacity, pre-trained features are the VALUE. **Rule of thumb**: <1K images + high similarity → feature extraction, 1K-10K images + medium similarity → fine-tune last layers, >10K images + low similarity → fine-tune all layers.',
  },
  {
    id: 'mc-4',
    question:
      'In discriminative learning rates for transfer learning, why do earlier layers get lower learning rates than later layers?',
    options: [
      'Earlier layers have more parameters, so they need smaller updates',
      'Earlier layers learn general features that transfer well, while later layers need more adaptation to the new task',
      'Earlier layers train faster and need to be slowed down',
      'This is just a convention with no performance benefit',
    ],
    correctAnswer: 1,
    explanation:
      "Layer specialization determines optimal learning rates: **Feature hierarchy** (vision example): Early layers (conv1, layer1): Edges, corners, textures - UNIVERSAL features, useful for ALL image tasks. Mid layers (layer2, layer3): Shapes, patterns, parts - Moderately general. Late layers (layer4): Objects, classes - TASK-SPECIFIC, need adaptation. **Learning rate strategy**: Early layers: Very low LR (1e-5) → Preserve general features, minimal changes. Mid layers: Medium LR (1e-4) → Moderate adaptation. Late layers: High LR (1e-3) → Significant adaptation to new task. New classifier: Highest LR (1e-3) → Learn from scratch. **Example**: ImageNet (cats, dogs) → Medical imaging (X-rays): Edges/textures (early): Still useful → small changes. Object detectors (late): Completely different → large changes. **Performance impact**: Discriminative LR: 92% accuracy, Uniform LR: 88% accuracy (all layers at 1e-4), 4% improvement! Not about parameter count (later layers often have MORE parameters), training speed, or convention - it's about feature transferability.",
  },
  {
    id: 'mc-5',
    question:
      'Adapter layers achieve 95% of full fine-tuning performance while updating only 2-3% of model parameters. How is this possible?',
    options: [
      'Adapters use quantization to reduce parameter count',
      'Adapters insert small bottleneck modules that learn task-specific transformations while keeping the powerful pre-trained representations frozen',
      'Adapters only work on small models with few parameters',
      'Adapters use knowledge distillation from the full model',
    ],
    correctAnswer: 1,
    explanation:
      "Adapters leverage **frozen representations + small task-specific transformations**: **Architecture**: Pre-trained layer (frozen): x → Pre-trained transformation → h (powerful features), Adapter (trainable): h → down_project (768→64) → ReLU → up_project (64→768) → + h. Only adapter parameters (2 × 768 × 64 ≈ 100K) trainable, not main model (100M parameters). **Why it works**: (1) **Pre-trained features are powerful**: Frozen representations already capture rich information, (2) **Bottleneck learning**: Small adapter learns task-specific refinements, (3) **Residual connection**: Adapter adds to frozen features, doesn't replace them, (4) **Parameter efficiency**: 64-dim bottleneck (vs. 768-dim) reduces parameters 12×. **Example** (BERT on text classification): Full fine-tuning: 110M parameters updated, 94% accuracy. Adapters: 2M parameters updated (2%), 93% accuracy (only 1% drop!). **Not quantization** (that's different compression), not model size (works on large models), not distillation (that's teacher-student training). The insight: Most information is in frozen features; adapters just learn how to USE them for your task.",
  },
];
