import { MultipleChoiceQuestion } from '../../../types';

export const wordEmbeddingsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'word-embeddings-mc-1',
    question:
      'What is the primary advantage of word embeddings over one-hot encoding or Bag of Words representations?',
    options: [
      'Word embeddings are faster to compute',
      'Word embeddings capture semantic similarity between words in dense vectors',
      'Word embeddings require less preprocessing',
      'Word embeddings work better with small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Word embeddings represent words as dense vectors (100-300 dimensions) where semantically similar words have similar vectors. Unlike sparse one-hot encoding where "king" and "queen" are orthogonal, embeddings place them close together in vector space, enabling models to understand semantic relationships.',
  },
  {
    id: 'word-embeddings-mc-2',
    question:
      'In Word2Vec, what is the key difference between Skip-gram and CBOW architectures?',
    options: [
      'Skip-gram is supervised; CBOW is unsupervised',
      'Skip-gram predicts context from target word; CBOW predicts target from context',
      'Skip-gram uses character n-grams; CBOW uses whole words',
      'Skip-gram requires labeled data; CBOW does not',
    ],
    correctAnswer: 1,
    explanation:
      'Skip-gram predicts surrounding context words given a target word, while CBOW predicts the target word given context words. Both are unsupervised. Skip-gram typically produces better embeddings but is slower, while CBOW is faster but generally lower quality.',
  },
  {
    id: 'word-embeddings-mc-3',
    question:
      'Which embedding method can generate vectors for out-of-vocabulary (OOV) words not seen during training?',
    options: ['Word2Vec Skip-gram', 'Word2Vec CBOW', 'GloVe', 'FastText'],
    correctAnswer: 3,
    explanation:
      'FastText represents words as sums of character n-gram embeddings, allowing it to generate embeddings for unseen words by composing their character n-grams. Word2Vec and GloVe assign vectors only to words in the training vocabulary and cannot handle OOV words.',
  },
  {
    id: 'word-embeddings-mc-4',
    question:
      'The famous analogy "king - man + woman ≈ queen" demonstrates that:',
    options: [
      'Word embeddings can solve any mathematical problem',
      'Semantic relationships are encoded as geometric relationships in vector space',
      'Word embeddings require labeled data for analogies',
      'Only royal words can participate in analogies',
    ],
    correctAnswer: 1,
    explanation:
      'This analogy shows that semantic relationships (like gender) are captured as directions in the embedding space. Vector arithmetic reveals that the model learned conceptual relationships from text patterns alone, encoding "gender" as a consistent vector direction across different word pairs.',
  },
  {
    id: 'word-embeddings-mc-5',
    question:
      'What is a major limitation of static word embeddings (Word2Vec, GloVe) compared to contextualized embeddings (BERT)?',
    options: [
      'Static embeddings are slower to train',
      'Static embeddings cannot handle rare words',
      'Static embeddings assign the same vector to a word regardless of context (polysemy)',
      'Static embeddings require more memory',
    ],
    correctAnswer: 2,
    explanation:
      'Static embeddings assign one fixed vector per word, so "bank" (financial) and "bank" (river) receive the same embedding averaged across all contexts. Contextualized embeddings like BERT generate different vectors for the same word based on surrounding context, handling polysemy effectively.',
  },
];
