import { MultipleChoiceQuestion } from '../../../types';

export const attentionNlpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'attention-nlp-mc-1',
    question:
      'In the attention mechanism formula Attention(Q,K,V) = softmax(QK^T/√d_k)V, what is the purpose of dividing by √d_k?',
    options: [
      'To normalize the output values',
      'To prevent the dot product from becoming too large before softmax',
      'To ensure the dimensions match',
      'To make the computation faster',
    ],
    correctAnswer: 1,
    explanation:
      'Dividing by √d_k prevents dot products from growing too large in magnitude, which would push softmax into regions with very small gradients. For large d_k, dot products grow in magnitude proportionally to √d_k, so scaling prevents saturation.',
  },
  {
    id: 'attention-nlp-mc-2',
    question: 'What is self-attention?',
    options: [
      'When a model attends only to its own predictions',
      'When Query, Key, and Value all come from the same input sequence',
      'When attention weights sum to 1',
      'When only one attention head is used',
    ],
    correctAnswer: 1,
    explanation:
      'Self-attention occurs when Q, K, and V are all derived from the same input sequence, allowing each position to attend to all positions in the same sequence. This is different from encoder-decoder attention where Q comes from decoder and K,V from encoder.',
  },
  {
    id: 'attention-nlp-mc-3',
    question:
      'In multi-head attention with 8 heads and d_model=512, what is the dimension of each head (d_k)?',
    options: ['512', '256', '64', '8'],
    correctAnswer: 2,
    explanation:
      'd_k = d_model / num_heads = 512 / 8 = 64. Each head operates in a lower-dimensional space, then outputs are concatenated back to d_model dimensions. This allows each head to specialize while maintaining overall model capacity.',
  },
  {
    id: 'attention-nlp-mc-4',
    question:
      'Why is a causal mask needed in decoder self-attention for text generation?',
    options: [
      'To improve training speed',
      'To prevent positions from attending to subsequent positions (prevent "cheating")',
      'To reduce memory usage',
      'To handle padding tokens',
    ],
    correctAnswer: 1,
    explanation:
      'Causal masking prevents positions from attending to future positions during training, ensuring the model learns to generate text autoregressively without seeing future tokens. Without this, the model would "cheat" by seeing the answer during training.',
  },
  {
    id: 'attention-nlp-mc-5',
    question:
      'What is the key advantage of attention mechanisms over LSTM encoder-decoder models?',
    options: [
      'Attention mechanisms have fewer parameters',
      'Attention provides direct connections between all positions, eliminating the fixed-size bottleneck',
      'Attention is always faster at inference',
      'Attention requires less training data',
    ],
    correctAnswer: 1,
    explanation:
      'Attention eliminates the fixed-size bottleneck of LSTM seq2seq where the entire source sequence must be compressed into a single hidden state. Instead, the decoder can attend to all encoder states directly, preventing information loss and enabling better handling of long sequences.',
  },
];
