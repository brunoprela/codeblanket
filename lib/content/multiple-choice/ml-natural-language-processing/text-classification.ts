import { MultipleChoiceQuestion } from '../../../types';

export const textClassificationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'text-classification-mc-1',
    question:
      'What activation function should be used for multi-label classification?',
    options: ['Softmax', 'Sigmoid', 'ReLU', 'Tanh'],
    correctAnswer: 1,
    explanation:
      'Sigmoid is used for multi-label classification because each label needs an independent probability between 0 and 1. Softmax would force probabilities to sum to 1, preventing multiple labels. Multi-label uses binary cross-entropy loss with sigmoid activation.',
  },
  {
    id: 'text-classification-mc-2',
    question:
      'For a sentiment analysis task with imbalanced classes (90% positive, 10% negative), which metric is most informative?',
    options: ['Accuracy', 'F1 score', 'Loss', 'Perplexity'],
    correctAnswer: 1,
    explanation:
      'F1 score balances precision and recall, making it ideal for imbalanced datasets. A model predicting all positive would get 90% accuracy but 0% recall for negative class. F1 score would reveal this poor performance on the minority class.',
  },
  {
    id: 'text-classification-mc-3',
    question: 'What is the purpose of [CLS] token in BERT for classification?',
    options: [
      'To mark the end of sequence',
      'To provide a sequence-level representation for classification',
      'To separate sentences',
      'To handle unknown words',
    ],
    correctAnswer: 1,
    explanation:
      'The [CLS] (classification) token is added at the beginning of input and its final hidden state is used as the aggregate sequence representation for classification tasks. BERT is trained to encode sequence-level information in this token during pre-training.',
  },
  {
    id: 'text-classification-mc-4',
    question:
      'When fine-tuning BERT for classification on a small dataset (<1000 examples), what strategy helps prevent overfitting?',
    options: [
      'Increase batch size',
      'Freeze BERT layers and train only the classification head',
      'Use higher learning rate',
      'Remove dropout',
    ],
    correctAnswer: 1,
    explanation:
      'With small datasets, freezing BERT layers and training only the classification head prevents overfitting. The frozen BERT acts as a feature extractor using pre-trained knowledge, while only the task-specific head adapts to the small dataset.',
  },
  {
    id: 'text-classification-mc-5',
    question: 'What does "zero-shot classification" mean?',
    options: [
      'Classification without any training data',
      'Classification with zero accuracy',
      'Classification using only negative examples',
      'Classification without labels',
    ],
    correctAnswer: 0,
    explanation:
      'Zero-shot classification performs classification on classes not seen during training, using only class descriptions. Models like CLIP and large language models can classify text into new categories by understanding natural language descriptions of the classes.',
  },
];
