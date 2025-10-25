/**
 * Recurrent Neural Networks (RNNs) Multiple Choice Questions
 */
export const recurrentNeuralNetworksMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question:
      'In a vanilla RNN, what is the primary role of the hidden state h_t?',
    options: [
      'It stores the final output of the network',
      'It serves as memory, carrying information from previous timesteps to influence current and future outputs',
      'It only determines the current output without affecting future timesteps',
      'It replaces the need for learnable parameters in the network',
    ],
    correctAnswer: 1,
    explanation:
      "The hidden state h_t is the **memory** of the RNN. It: (1) Summarizes all information seen so far in the sequence, (2) Is passed to the next timestep (h_t → h_{t+1}), influencing future processing, (3) Updates at each step: h_t = tanh(W_hh·h_{t-1} + W_xh·x_t), combining previous state with current input, (4) Enables the network to maintain context. For example, in 'The cat, which was black, __', h_t at the blank position carries information about 'cat' from earlier, helping predict 'was' (singular verb). Without this memory, RNN would be just independent predictions at each timestep.",
  },
  {
    id: 'cnn-mc-2',
    question:
      'Why does gradient clipping help with training RNNs, and what problem does it specifically address?',
    options: [
      'It solves the vanishing gradient problem by making gradients larger',
      'It prevents exploding gradients by capping gradient norms to a maximum value',
      'It eliminates the need for backpropagation through time',
      'It increases training speed by reducing the number of parameters',
    ],
    correctAnswer: 1,
    explanation:
      "Gradient clipping **prevents exploding gradients** (not vanishing). When gradients grow exponentially large during backpropagation through time, they cause: (1) NaN/Inf values in parameters, (2) Massive parameter updates that destabilize training, (3) Loss divergence. Clipping caps the gradient norm: if ||g|| > threshold, set g = threshold × g/||g||. Example: threshold=5, if gradient norm is 50, scale it down to 5. This prevents catastrophic updates while preserving gradient direction. Note: Clipping does NOT fix vanishing gradients (which require architectural changes like LSTM). It\'s a necessary but not sufficient technique for stable RNN training. Typical threshold values: 1-10.",
  },
  {
    id: 'cnn-mc-3',
    question:
      "What is 'truncated backpropagation through time' (truncated BPTT) and why is it used?",
    options: [
      'A technique to reduce sequence length by removing unimportant timesteps',
      'A method to split long sequences into chunks and compute gradients only within each chunk, reducing memory and computation',
      'An optimization that removes the recurrent connections from the RNN',
      'A way to skip certain timesteps during forward propagation',
    ],
    correctAnswer: 1,
    explanation:
      "Truncated BPTT handles long sequences by **chunking** them for gradient computation. Standard BPTT through 10,000 timesteps would: (1) Require storing all 10,000 hidden states (memory), (2) Backpropagate through all 10,000 steps (slow), (3) Suffer severe vanishing gradients. Truncated BPTT solution: Process sequence in chunks (e.g., 100 steps), backpropagate only within each chunk, but **carry forward** the hidden state between chunks. Example: 1000-step sequence → 10 chunks of 100 steps, backprop 100 steps at a time, h_100 passed to next chunk. Trade-off: Can't learn dependencies longer than chunk size, but practical and necessary. Modern frameworks do this automatically. Think: 'Forward pass is long, backward pass is chunked'.",
  },
  {
    id: 'cnn-mc-4',
    question:
      'In a many-to-one RNN architecture (e.g., sentiment analysis), which part of the RNN output is typically used for the final prediction?',
    options: [
      'The output at each timestep is averaged together',
      'Only the output from the first timestep',
      'The final hidden state h_T after processing the entire sequence',
      'A random timestep is selected during training',
    ],
    correctAnswer: 2,
    explanation:
      "In many-to-one architectures, the **final hidden state h_T** is used for prediction because it: (1) Has processed the entire sequence, (2) Summarizes all information from start to finish, (3) Represents the complete context. Example - Sentiment analysis: Input: 'This movie was terrible and boring' → Process all 6 words → h_6 contains aggregate sentiment → Classify h_6 as negative. The final hidden state is passed to a classification layer: logits = Linear (h_T). Why not earlier states? h_0, h_1, etc. haven't seen the full sequence yet. Why not average all outputs? Final state already aggregates information (and averaging is sometimes used as alternative). This pattern is used for: sentiment analysis, document classification, sequence summarization tasks.",
  },
  {
    id: 'cnn-mc-5',
    question:
      "What is the key difference between 'teacher forcing' and 'free running' during RNN training for sequence generation?",
    options: [
      'Teacher forcing uses larger batch sizes than free running',
      "Teacher forcing uses true previous outputs as inputs during training, while free running uses model's own predictions",
      'Teacher forcing requires multiple GPUs while free running works on CPU',
      'Teacher forcing is only used during inference, not training',
    ],
    correctAnswer: 1,
    explanation:
      "**Teacher Forcing** (training): At each timestep, feed the **true previous output** as input, regardless of what model predicted. Example - 'hello' generation: Step 1: input=START, target='h', predict 'h' → Step 2: input='h' (true), target='e' → Step 3: input='e' (true), target='l'. Even if model predicted 'x' at step 1, still feed 'h' at step 2. Benefits: (1) Faster convergence, (2) Stable training, (3) Each step learns from correct context. **Free Running** (inference): Feed model's **own predictions** as next inputs. Step 1: predict 'h' → Step 2: input='h' (predicted) → predict 'e' → Step 3: input='e'... Problem: **Exposure bias** - model never sees its own errors during training, struggles at inference. Solution: Scheduled sampling (gradually mix teacher forcing with free running during training). This mismatch between training/inference is a known RNN challenge.",
  },
];
