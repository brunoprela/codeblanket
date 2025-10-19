import { MultipleChoiceQuestion } from '../../../types';

export const questionAnsweringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'qa-ir-mc-1',
    question: 'In extractive QA, what do the model outputs represent?',
    options: [
      'The full answer text',
      'The probability of each word being in the answer',
      'The start and end token positions of the answer span',
      'A confidence score for the question',
    ],
    correctAnswer: 2,
    explanation:
      'Extractive QA models (like BERT for QA) output two sets of logits: one for the start position and one for the end position of the answer span in the context. The model predicts which tokens mark the beginning and end of the answer, and the text between these positions is extracted as the answer.',
  },
  {
    id: 'qa-ir-mc-2',
    question:
      'What is the main advantage of BM25 over dense retrieval methods?',
    options: [
      'Better semantic understanding',
      'Handles synonyms automatically',
      'Exact term matching and no training required',
      'Works across languages',
    ],
    correctAnswer: 2,
    explanation:
      'BM25 is a sparse retrieval method that excels at exact term matching and requires no training. It works out of the box and is particularly good for queries with specific terms, IDs, or rare words. While dense methods handle semantics better, BM25 is reliable for lexical matching and is highly interpretable.',
  },
  {
    id: 'qa-ir-mc-3',
    question:
      'In a two-stage retrieval system, what is the purpose of the re-ranking stage?',
    options: [
      'To retrieve more documents',
      'To improve precision using more expensive models on fewer candidates',
      'To cache results for faster queries',
      'To generate answers from documents',
    ],
    correctAnswer: 1,
    explanation:
      'Re-ranking uses more expensive but accurate models (like cross-encoders) on a smaller set of candidate documents from the first stage. The first stage (bi-encoder or BM25) provides broad recall efficiently, then re-ranking improves precision. This two-stage approach balances speed and accuracy.',
  },
  {
    id: 'qa-ir-mc-4',
    question: 'What does FAISS (Facebook AI Similarity Search) optimize?',
    options: [
      'Training transformer models',
      'Efficient similarity search over large vector collections',
      'Text generation quality',
      'Question answering accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'FAISS is a library for efficient similarity search and clustering of dense vectors. When you have millions of document embeddings, FAISS enables fast approximate nearest neighbor search using techniques like product quantization and inverted file indices, making dense retrieval practical at scale.',
  },
  {
    id: 'qa-ir-mc-5',
    question:
      'In the SQuAD dataset format, what does the model need to learn during fine-tuning for extractive QA?',
    options: [
      'To generate natural language answers',
      'To predict start and end positions of answer spans',
      'To retrieve relevant documents',
      'To classify question types',
    ],
    correctAnswer: 1,
    explanation:
      'SQuAD (Stanford Question Answering Dataset) is an extractive QA dataset where answers are spans within provided paragraphs. During fine-tuning, the model learns to predict the start and end token positions of the answer span. The training loss combines the cross-entropy loss for both start and end position predictions.',
  },
];
