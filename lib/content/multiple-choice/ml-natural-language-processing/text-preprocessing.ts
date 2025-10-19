import { MultipleChoiceQuestion } from '../../../types';

export const textPreprocessingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'text-preprocessing-mc-1',
    question:
      'Which tokenization method would best handle the text "Dr. Smith works at Apple Inc. in the U.S.A." while preserving entity boundaries?',
    options: [
      'Simple whitespace tokenization with text.split()',
      'NLTK word_tokenize()',
      'spaCy tokenizer',
      'Regular expression splitting on all punctuation',
    ],
    correctAnswer: 2,
    explanation:
      'spaCy tokenizer uses linguistic rules and handles abbreviations, company names, and multi-word entities intelligently. It recognizes "Dr.", "Inc.", and "U.S.A." as single tokens and understands entity boundaries, making it superior for complex text with named entities.',
  },
  {
    id: 'text-preprocessing-mc-2',
    question:
      'For a sentiment analysis task, you have the sentence "not good at all". What preprocessing would most likely DAMAGE model performance?',
    options: [
      'Converting to lowercase',
      'Removing stop words',
      'Removing punctuation',
      'Lemmatization',
    ],
    correctAnswer: 1,
    explanation:
      'Removing stop words would eliminate "not", changing "not good at all" to just "good at all", completely reversing the sentiment. For sentiment analysis, negation words (typically stopwords) are critical and should never be removed. This is a common pitfall in sentiment preprocessing.',
  },
  {
    id: 'text-preprocessing-mc-3',
    question: 'What is the primary advantage of lemmatization over stemming?',
    options: [
      'Lemmatization is significantly faster',
      'Lemmatization always reduces words to their root form',
      'Lemmatization returns actual dictionary words and considers context',
      'Lemmatization works better with social media text',
    ],
    correctAnswer: 2,
    explanation:
      'Lemmatization uses morphological analysis and considers part-of-speech to return actual dictionary words (lemmas). For example, "better" lemmatizes to "good" (correct), while stemming would produce "better" or "bett" (nonsense). This linguistic awareness makes lemmatization more accurate, though slower than stemming.',
  },
  {
    id: 'text-preprocessing-mc-4',
    question:
      'Which preprocessing approach is generally recommended for modern transformer models like BERT?',
    options: [
      'Aggressive: remove stopwords, punctuation, and lemmatize',
      'Moderate: lowercase and remove only HTML/URLs',
      'Minimal: keep original text with only basic cleaning',
      'Maximum: apply all available preprocessing techniques',
    ],
    correctAnswer: 2,
    explanation:
      'Transformer models like BERT benefit from minimal preprocessing because they learn contextualized representations from the full input. They can understand the importance of stopwords, punctuation, and word forms in context. Aggressive preprocessing actually removes valuable information that these models can leverage. Only remove clear noise like HTML tags and URLs.',
  },
  {
    id: 'text-preprocessing-mc-5',
    question:
      'A production model is experiencing OOV (out-of-vocabulary) errors in inference that did not occur during training. What is the most likely cause?',
    options: [
      'The training data was too small',
      'Preprocessing inconsistency between training and inference',
      'The model architecture is incorrect',
      'Learning rate was too high during training',
    ],
    correctAnswer: 1,
    explanation:
      'Preprocessing inconsistency is the most common cause of OOV errors appearing only at inference. For example, if training removed stopwords but inference does not, or if capitalization handling differs, tokens that appear in inference will not exist in the training vocabulary. This is a critical production bug that can silently degrade performance.',
  },
];
