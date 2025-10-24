export const bertEncoderModelsQuiz = {
  title: 'BERT & Encoder Models Discussion',
  id: 'bert-encoder-models-quiz',
  sectionId: 'bert-encoder-models',
  questions: [
    {
      id: 1,
      question:
        'BERT uses bidirectional context through masked language modeling, while GPT uses unidirectional context through causal language modeling. Explain why bidirectional context is crucial for understanding tasks but incompatible with generation tasks. Can we have both benefits simultaneously, and what approaches attempt to bridge this gap?',
      expectedAnswer:
        'Should cover: bidirectional context enabling richer representations for classification/NLU, "cheating" problem in generation with bidirectional context, masked LM training enabling bidirectional encoders, when you need full context (semantic similarity, classification), encoder-decoder architectures as compromise (T5, BART), prefix LM approaches, recent unified architectures attempting both, and practical guidance on architecture selection based on task requirements.',
    },
    {
      id: 2,
      question:
        'Discuss the evolution from BERT to RoBERTa, ALBERT, and DistilBERT. What specific limitations of BERT did each successor address? Analyze the tradeoffs between model size, training efficiency, inference speed, and performance. Which model would you choose for different production scenarios?',
      expectedAnswer:
        "Should analyze: BERT's undertraining discovery(RoBERTa), NSP task removal benefits, dynamic masking advantages, parameter sharing in ALBERT for memory efficiency, factorized embeddings, knowledge distillation in DistilBERT, 40% size reduction with 97 % performance retention, inference speed requirements, memory constraints in production, fine- tuning efficiency, and decision framework for model selection based on constraints.",
    },
    {
      id: 3,
      question:
        'Why have encoder-only models like BERT declined in popularity compared to decoder-only models (GPT family) despite their superior performance on many NLU benchmarks? Discuss the shift toward unified generative models and whether encoder-only models still have a place in modern NLP systems.',
      expectedAnswer:
        'Should discuss: versatility advantage of generative models, single model for multiple tasks, few-shot learning in GPT vs fine-tuning requirements in BERT, ease of prompting vs fine-tuning overhead, commercial preference for generative APIs, continued advantages of encoders for retrieval and embedding tasks (semantic search, clustering), efficiency benefits for pure understanding tasks, hybrid systems using both architectures, and future trajectory of specialized vs general-purpose models.',
    },
  ],
};
