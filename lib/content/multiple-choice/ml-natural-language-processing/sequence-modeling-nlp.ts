import { MultipleChoiceQuestion } from '../../../types';

export const sequenceModelingNlpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sequence-modeling-nlp-mc-1',
    question:
      'What is the primary problem that LSTMs solve compared to vanilla RNNs?',
    options: [
      'LSTMs are faster to train',
      'LSTMs prevent vanishing gradients through their cell state mechanism',
      'LSTMs require less memory',
      'LSTMs can process longer sequences in parallel',
    ],
    correctAnswer: 1,
    explanation:
      'LSTMs solve the vanishing gradient problem through their cell state, which acts as a "highway" for gradients to flow through many timesteps without diminishing. This allows LSTMs to learn long-term dependencies that vanilla RNNs cannot capture due to gradients vanishing after 10-20 timesteps.',
  },
  {
    id: 'sequence-modeling-nlp-mc-2',
    question: 'In an LSTM, what is the purpose of the forget gate?',
    options: [
      'To generate new candidate values for the cell state',
      'To control what information to remove from the cell state',
      'To determine the final output',
      'To handle out-of-vocabulary words',
    ],
    correctAnswer: 1,
    explanation:
      'The forget gate controls what information should be removed (forgotten) from the previous cell state. It outputs values between 0 (completely forget) and 1 (completely keep), allowing the LSTM to selectively maintain or discard information from long-term memory.',
  },
  {
    id: 'sequence-modeling-nlp-mc-3',
    question:
      'Why cannot bidirectional LSTMs be used for text generation tasks?',
    options: [
      'They are too slow',
      'They require too much memory',
      'They need to see future tokens which do not exist during generation',
      'They cannot handle variable-length sequences',
    ],
    correctAnswer: 2,
    explanation:
      'Bidirectional LSTMs process the sequence in both directions, requiring the complete sequence upfront. During text generation, future tokens do not yet exist—you are generating them one at a time. Therefore, the backward pass is impossible, making bidirectional LSTMs unsuitable for generation tasks.',
  },
  {
    id: 'sequence-modeling-nlp-mc-4',
    question: 'What is a key advantage of GRUs compared to LSTMs?',
    options: [
      'GRUs have better accuracy on all tasks',
      'GRUs have fewer parameters due to simpler gating mechanism',
      'GRUs can handle longer sequences',
      'GRUs solve the vanishing gradient problem better',
    ],
    correctAnswer: 1,
    explanation:
      'GRUs simplify the LSTM architecture by using only 2 gates (reset and update) instead of 3 (forget, input, output), and no separate cell state. This results in fewer parameters, faster training, and often similar performance to LSTMs, though sometimes slightly lower accuracy on complex tasks.',
  },
  {
    id: 'sequence-modeling-nlp-mc-5',
    question:
      'When processing a sequence of length 100 with an LSTM, what operation prevents parallelization?',
    options: [
      'The embedding lookup',
      'The final classification layer',
      'The recurrent computation where each hidden state depends on the previous',
      'The gradient computation',
    ],
    correctAnswer: 2,
    explanation:
      'LSTMs compute hidden states sequentially: h_t depends on h_{t-1}, which depends on h_{t-2}, etc. This sequential dependency means you must compute h_1 before h_2, preventing parallelization across the sequence. This is why transformers with self-attention, which can process all positions simultaneously, are much faster.',
  },
];
