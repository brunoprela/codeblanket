import { MultipleChoiceQuestion } from '../../../types';

export const contextualizedEmbeddingsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'contextualized-embeddings-mc-1',
      question:
        'What is the fundamental difference between static embeddings (Word2Vec) and contextualized embeddings (ELMo)?',
      options: [
        'Static embeddings are trained faster',
        'Contextualized embeddings generate different vectors for the same word based on context',
        'Static embeddings have higher dimensions',
        'Contextualized embeddings cannot handle out-of-vocabulary words',
      ],
      correctAnswer: 1,
      explanation:
        'Contextualized embeddings generate different vectors for the same word depending on surrounding context. "bank" in "financial bank" receives a different embedding than "bank" in "river bank". Static embeddings assign one fixed vector per word regardless of context.',
    },
    {
      id: 'contextualized-embeddings-mc-2',
      question:
        'Why does ELMo use a bidirectional LSTM architecture rather than a forward-only LSTM?',
      options: [
        'Bidirectional LSTMs train faster',
        'Bidirectional LSTMs require less memory',
        'Bidirectional LSTMs capture context from both directions, essential for disambiguation',
        'Bidirectional LSTMs handle longer sequences better',
      ],
      correctAnswer: 2,
      explanation:
        'Bidirectional processing is crucial because meaning often depends on words appearing AFTER the target word. For example, in "The trophy didn\'t fit because it was too big", understanding "it" requires seeing "big" which comes later. Forward-only would miss this critical context.',
    },
    {
      id: 'contextualized-embeddings-mc-3',
      question: 'How does ELMo handle out-of-vocabulary (OOV) words?',
      options: [
        'ELMo replaces OOV words with a special <UNK> token',
        'ELMo uses a character-level CNN to generate embeddings for unseen words',
        'ELMo cannot handle OOV words',
        'ELMo uses the closest in-vocabulary word',
      ],
      correctAnswer: 1,
      explanation:
        'ELMo uses a character-level convolutional neural network (CNN) as its first layer, which processes words character-by-character. This allows ELMo to generate embeddings for words never seen during training, handling typos and rare words robustly.',
    },
    {
      id: 'contextualized-embeddings-mc-4',
      question:
        'ELMo typically uses multiple LSTM layers. What do different layers capture?',
      options: [
        'All layers capture the same information for redundancy',
        'Lower layers capture syntax; higher layers capture semantics',
        'Lower layers are for speed; higher layers for accuracy',
        'The number of layers only affects training time',
      ],
      correctAnswer: 1,
      explanation:
        'Studies show ELMo layers capture different linguistic information: lower layers encode syntax and part-of-speech tags, while higher layers capture semantic meaning and task-specific features. This hierarchical representation is why ELMo typically uses a weighted combination of all layers.',
    },
    {
      id: 'contextualized-embeddings-mc-5',
      question:
        'What major limitation of ELMo led to the development of transformer-based models like BERT?',
      options: [
        'ELMo cannot handle polysemy',
        "ELMo\'s LSTM architecture requires sequential processing and cannot parallelize effectively",
        'ELMo requires too much labeled data',
        'ELMo produces embeddings that are too large',
      ],
      correctAnswer: 1,
      explanation:
        'LSTMs must process tokens sequentially (word-by-word), preventing parallelization and making training slow. Transformers use self-attention which processes all tokens simultaneously, enabling massive parallelization and 10-100x faster training. This architectural advantage made transformers (BERT, GPT) dominant.',
    },
  ];
