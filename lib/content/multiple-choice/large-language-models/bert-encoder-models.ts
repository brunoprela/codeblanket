export const bertEncoderModelsMC = {
  title: 'BERT & Encoder Models Quiz',
  id: 'bert-encoder-models-mc',
  sectionId: 'bert-encoder-models',
  questions: [
    {
      id: 1,
      question:
        'What training objective does BERT use that enables bidirectional context?',
      options: [
        'Next sentence prediction only',
        'Masked language modeling (MLM)',
        'Causal language modeling',
        'Contrastive predictive coding',
      ],
      correctAnswer: 1,
      explanation:
        "BERT uses Masked Language Modeling (MLM), randomly masking 15% of tokens and predicting them using bidirectional context. This allows the model to learn from both left and right context, unlike GPT's left-to-right approach.",
    },
    {
      id: 2,
      question: 'What key change did RoBERTa make to improve upon BERT?',
      options: [
        'Added more layers',
        'Removed Next Sentence Prediction task and trained longer',
        'Switched to decoder-only architecture',
        'Added explicit entity recognition tasks',
      ],
      correctAnswer: 1,
      explanation:
        'RoBERTa removed the Next Sentence Prediction (NSP) task, trained with dynamic masking, used larger batches, and trained much longer on more data. These changes, especially removing NSP and training longer, significantly improved performance.',
    },
    {
      id: 3,
      question:
        "How does DistilBERT achieve a 40% reduction in size while retaining 97% of BERT's performance?",
      options: [
        'Removing attention heads',
        'Knowledge distillation from BERT teacher model',
        'Quantization to 4-bit precision',
        'Pruning least important neurons',
      ],
      correctAnswer: 1,
      explanation:
        "DistilBERT uses knowledge distillation, training a smaller student model to match the outputs (and hidden states) of the larger BERT teacher model. This transfers BERT's knowledge to a more compact architecture.",
    },
    {
      id: 4,
      question:
        'Why are encoder-only models like BERT poorly suited for text generation tasks?',
      options: [
        'They are too small',
        'They use bidirectional attention, which would allow "cheating" during generation',
        "They don't have a language modeling head",
        'They can only process fixed-length sequences',
      ],
      correctAnswer: 1,
      explanation:
        'Bidirectional attention means each token sees future tokens, which would constitute cheating during autoregressive generation (where each token should only depend on previous tokens). BERT is designed for understanding, not generation.',
    },
    {
      id: 5,
      question:
        'What is the typical use case where BERT-style encoders still outperform GPT-style decoders?',
      options: [
        'Text generation',
        'Few-shot learning',
        'Semantic search and embedding tasks',
        'Long-form question answering',
      ],
      correctAnswer: 2,
      explanation:
        'Encoder models excel at creating dense embeddings for semantic search, classification, and clustering. Their bidirectional context creates rich representations ideal for understanding and similarity tasks, though GPT-style models have become competitive here too.',
    },
  ],
};
