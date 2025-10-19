import { MultipleChoiceQuestion } from '../../../types';

export const textRepresentationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'text-representation-mc-1',
    question:
      'Given a corpus of 1,000 documents where the word "machine" appears in 500 documents, what is the IDF (Inverse Document Frequency) value for "machine"?',
    options: [
      'log(1000 / 500) = 0.30',
      'log(500 / 1000) = -0.30',
      '1000 / 500 = 2.0',
      'log(1000 + 1 / 500 + 1) ≈ 0.29',
    ],
    correctAnswer: 0,
    explanation:
      'IDF = log(total_documents / documents_containing_term) = log(1000 / 500) = log(2) ≈ 0.30. This relatively low IDF indicates "machine" is common (appears in 50% of documents), so it will be down-weighted in TF-IDF. Rare words that appear in fewer documents would have higher IDF values.',
  },
  {
    id: 'text-representation-mc-2',
    question:
      'Why are sparse matrices preferred over dense matrices for BoW and TF-IDF representations in production systems?',
    options: [
      'Sparse matrices provide better accuracy',
      'Sparse matrices are faster to compute',
      'Sparse matrices save memory by storing only non-zero values',
      'Sparse matrices handle missing data better',
    ],
    correctAnswer: 2,
    explanation:
      'Sparse matrices store only non-zero values and their indices, dramatically reducing memory usage. With 99%+ zeros typical in text data, sparse matrices can reduce memory from GB to MB. Most scikit-learn algorithms work efficiently with sparse matrices, making them ideal for production systems with large vocabularies.',
  },
  {
    id: 'text-representation-mc-3',
    question:
      'For the text "not good", which representation would best capture the negative sentiment?',
    options: [
      'Unigrams only: ["not", "good"]',
      'Bigrams only: ["not good"]',
      'Unigrams + Bigrams: ["not", "good", "not good"]',
      'Character n-grams: ["not", " go", "ood"]',
    ],
    correctAnswer: 2,
    explanation:
      'Unigrams + Bigrams (1,2) captures both individual words and their combination as a phrase. "not good" as a bigram can learn negative sentiment, while also keeping individual tokens for other contexts. This is the standard approach for sentiment analysis where negations and modifiers are critical.',
  },
  {
    id: 'text-representation-mc-4',
    question:
      'A TfidfVectorizer is configured with max_features=1000, min_df=5, max_df=0.8 on a 10,000-document corpus. What is the purpose of these parameters?',
    options: [
      'Increase vocabulary size for better representation',
      'Limit vocabulary to control dimensionality and remove noise',
      'Speed up tokenization process',
      'Improve handling of unknown words',
    ],
    correctAnswer: 1,
    explanation:
      'These parameters limit vocabulary to control dimensionality: max_features=1000 keeps only the top 1000 terms, min_df=5 removes rare terms appearing in <5 documents, max_df=0.8 removes very common terms in >80% of documents. This reduces noise, prevents overfitting, and dramatically improves computational efficiency with minimal accuracy loss.',
  },
  {
    id: 'text-representation-mc-5',
    question:
      'What is the main limitation of Bag of Words and TF-IDF representations that word embeddings (covered in the next section) aim to solve?',
    options: [
      'They cannot handle large vocabularies',
      'They are too slow to compute',
      'They do not capture semantic similarity between words',
      'They require too much preprocessing',
    ],
    correctAnswer: 2,
    explanation:
      'BoW and TF-IDF treat "good" and "excellent" as completely different features with no relationship, failing to capture semantic similarity. Word embeddings create dense vector representations where semantically similar words have similar vectors, enabling better generalization and understanding of meaning beyond exact word matches.',
  },
];
