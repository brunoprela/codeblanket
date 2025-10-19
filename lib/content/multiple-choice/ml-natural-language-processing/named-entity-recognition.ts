import { MultipleChoiceQuestion } from '../../../types';

export const namedEntityRecognitionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ner-mc-1',
    question: 'In IOB tagging, what does the "B-" prefix indicate?',
    options: [
      'Background token (not an entity)',
      'Beginning of an entity',
      'Bold text',
      'Boundary marker',
    ],
    correctAnswer: 1,
    explanation:
      'B- (Begin) marks the first token of a named entity. This distinguishes it from I- (Inside) tokens that continue the entity. The B/I distinction is crucial for identifying boundaries between adjacent entities of the same type.',
  },
  {
    id: 'ner-mc-2',
    question: 'Why do we use -100 as a label for subword tokens in NER?',
    options: [
      'It represents the unknown label',
      'It tells PyTorch/TensorFlow to ignore these positions in loss computation',
      'It is a special entity type',
      'It improves model accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'In PyTorch and TensorFlow, -100 is a special value that tells the loss function to ignore these positions. This is necessary because when tokenizers split words into subwords, we only want to compute loss on the first subword token, not all pieces.',
  },
  {
    id: 'ner-mc-3',
    question:
      'What type of neural network layer is used for NER in transformer models?',
    options: [
      'Softmax over vocabulary',
      'Linear classification layer over entity types',
      'LSTM decoder',
      'Attention head',
    ],
    correctAnswer: 1,
    explanation:
      'NER uses a linear classification layer on top of transformer hidden states, with one output per entity type (including "O" for outside). Each token gets classified independently into one of the entity types using the representation from the transformer encoder.',
  },
  {
    id: 'ner-mc-4',
    question: 'Which evaluation metric is most commonly used for NER?',
    options: ['Accuracy', 'Perplexity', 'Entity-level F1 score', 'BLEU score'],
    correctAnswer: 2,
    explanation:
      'Entity-level F1 score is the standard NER metric. It requires the model to correctly identify the complete entity span AND type. This is stricter than token-level accuracy and better reflects real-world performance where partial entities are not useful.',
  },
  {
    id: 'ner-mc-5',
    question:
      'What is a common challenge when applying NER to domain-specific text?',
    options: [
      'NER cannot handle long sequences',
      'Pre-trained models may not recognize domain-specific entities',
      'NER only works on English',
      'Transformers are too small for NER',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-trained models are typically trained on general text (news, Wikipedia) and may not recognize domain-specific entities like medical terms, legal jargon, or specialized product names. Fine-tuning on domain-specific annotated data is usually necessary for good performance.',
  },
];
