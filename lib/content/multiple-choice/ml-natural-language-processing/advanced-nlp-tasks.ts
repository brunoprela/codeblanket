import { MultipleChoiceQuestion } from '../../../types';

export const advancedNlpTasksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'advanced-nlp-mc-1',
    question: 'What is the BLEU score used to evaluate?',
    options: [
      'Sentiment analysis accuracy',
      'Machine translation quality',
      'Named entity recognition performance',
      'Text classification F1 score',
    ],
    correctAnswer: 1,
    explanation:
      'BLEU (Bilingual Evaluation Understudy) measures machine translation quality by comparing n-gram overlap between generated translation and reference translations. It ranges from 0 to 1 (often shown as 0-100), with higher scores indicating better translation quality. BLEU-4 considers 1-grams through 4-grams.',
  },
  {
    id: 'advanced-nlp-mc-2',
    question:
      'In abstractive summarization, what is the main risk compared to extractive summarization?',
    options: [
      'Longer processing time',
      'Hallucination (generating facts not in the source)',
      'Lower ROUGE scores',
      'Limited vocabulary',
    ],
    correctAnswer: 1,
    explanation:
      'Abstractive summarization generates new text, which risks hallucination—creating plausible-sounding but factually incorrect information not present in the source document. This is particularly dangerous in financial and medical domains where accuracy is critical. Extractive summarization avoids this by only selecting existing sentences.',
  },
  {
    id: 'advanced-nlp-mc-3',
    question: 'What makes FinBERT specialized for financial NLP tasks?',
    options: [
      'It has more parameters than regular BERT',
      'It was pre-trained on financial text (earnings calls, news, filings)',
      'It uses a different architecture than BERT',
      'It can only process numbers',
    ],
    correctAnswer: 1,
    explanation:
      'FinBERT is BERT fine-tuned on large-scale financial text including earnings calls, financial news, and SEC filings. This domain-specific pre-training helps it understand financial terminology, sentiment, and context better than general BERT. The architecture remains the same; the key difference is the training data.',
  },
  {
    id: 'advanced-nlp-mc-4',
    question:
      'In few-shot learning with GPT-3, what is the purpose of providing examples in the prompt?',
    options: [
      'To fine-tune the model',
      'To enable in-context learning for the specific task',
      'To increase the model size',
      'To reduce inference time',
    ],
    correctAnswer: 1,
    explanation:
      "Few-shot learning leverages the model's ability to learn from context (in-context learning). By providing a few examples in the prompt, GPT-3 can infer the task pattern and apply it to new inputs without any parameter updates or fine-tuning. This emergent capability appears in large language models (>10B parameters).",
  },
  {
    id: 'advanced-nlp-mc-5',
    question:
      'When building a news sentiment trading system, why is entity extraction critical before sentiment analysis?',
    options: [
      'It reduces computation time',
      'It identifies which companies the sentiment applies to, enabling ticker-specific signals',
      'It improves translation accuracy',
      'It replaces the need for sentiment analysis',
    ],
    correctAnswer: 1,
    explanation:
      'Entity extraction (NER) identifies which companies are mentioned in news articles, enabling the system to generate ticker-specific trading signals. A single article might mention multiple companies with different sentiment ("Apple gains while Microsoft loses"), so accurate entity extraction and linking to stock tickers is essential for actionable trading signals.',
  },
];
