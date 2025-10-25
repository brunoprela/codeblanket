/**
 * LSTM & GRU Multiple Choice Questions
 */
export const lstmGruMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question: 'What is the primary purpose of the LSTM forget gate f_t?',
    options: [
      'To determine which neurons to drop out during training',
      'To control what information to remove from the cell state by outputting values between 0 and 1',
      'To initialize the cell state at the beginning of each sequence',
      'To prevent gradient exploding by clipping large values',
    ],
    correctAnswer: 1,
    explanation:
      "The forget gate **controls what information to discard** from the cell state. It: (1) Takes h_{t-1} and x_t as input, (2) Outputs values in [0,1] via sigmoid activation, (3) Multiplies element-wise with previous cell state C_{t-1}, (4) f_t ≈ 0 means 'forget this information', f_t ≈ 1 means 'keep it'. Example: After seeing sentence-ending period, forget gate might output low values to reset memory for new sentence. This is NOT dropout (which is a regularization technique), NOT initialization (happens once), and while it helps with gradients, clipping is a separate technique. The forget gate is learned during training to identify what's important to remember vs forget.",
  },
  {
    id: 'cnn-mc-2',
    question:
      "In the LSTM cell state update equation C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t, what does the addition operation (+) provide that's crucial for solving vanishing gradients?",
    options: [
      'It makes the model train faster by reducing computations',
      'It provides an additive path for gradients with derivative = 1, allowing gradients to flow unchanged',
      'It combines the forget and input gates into a single operation',
      'It prevents the cell state from growing too large',
    ],
    correctAnswer: 1,
    explanation:
      "The **addition operation is key to LSTM's gradient flow** solution. During backpropagation, the derivative of addition is 1: ∂(a+b)/∂a = 1. This means gradients can flow through the additive path **unchanged**. Compare to vanilla RNN which uses h_t = tanh(W·h_{t-1}) - this involves matrix multiplication whose gradients vanish exponentially. LSTM: C_t = f_t⊙C_{t-1} + i_t⊙C̃_t has ∂C_t/∂C_{t-1} = f_t (just the forget gate, typically 0.5-1.0). Through 100 timesteps: (0.9)^100 ≈ 0.00003 but still non-zero vs vanilla RNN's (0.25)^100 ≈ 10^{-60}. The addition creates a 'gradient highway' allowing information to flow over long sequences.",
  },
  {
    id: 'cnn-mc-3',
    question:
      'How many parameters does a single-layer LSTM have compared to a single-layer GRU, given input size d and hidden size h?',
    options: [
      'LSTM and GRU have exactly the same number of parameters',
      'LSTM has approximately 33% more parameters than GRU (4 vs 3 weight matrices)',
      'GRU has more parameters due to its reset gate mechanism',
      'LSTM has twice as many parameters as GRU',
    ],
    correctAnswer: 1,
    explanation:
      "LSTM has **approximately 33% more parameters** than GRU. Parameter count: **LSTM**: 4 weight matrices (forget, input, output, candidate) each (h+d)×h = 4(h²+dh) + 4h. **GRU**: 3 weight matrices (reset, update, candidate) each (h+d)×h = 3(h²+dh) + 3h. Ratio: 4/3 ≈ 1.33. **Concrete example**: d=512, h=1024 → LSTM: ~6.3M params, GRU: ~4.7M params (25% fewer). This translates to: (1) GRU trains 15-30% faster, (2) GRU uses less memory, (3) LSTM has more representational capacity. The difference comes from LSTM's separate output gate (GRU exposes entire hidden state) and LSTM maintaining both cell and hidden states.",
  },
  {
    id: 'cnn-mc-4',
    question:
      "What is the key structural difference between LSTM and GRU that reduces GRU's parameter count?",
    options: [
      'GRU uses only one gate while LSTM uses three',
      'GRU eliminates the separate cell state and combines forget/input gates into a single update gate',
      'GRU uses smaller hidden dimensions',
      "GRU doesn't have bias terms",
    ],
    correctAnswer: 1,
    explanation:
      "GRU's key simplification is **eliminating the separate cell state** and using only hidden state. Additionally, GRU combines LSTM's separate forget and input gates into a single **update gate**: z_t controls both forgetting old information (1-z_t)⊙h_{t-1} and accepting new information z_t⊙h̃_t. This reduces from 4 weight matrices (LSTM) to 3 (GRU). The update gate is elegant: high z_t means 'forget old, accept new'; low z_t means 'keep old, ignore new'. GRU also adds a reset gate r_t that controls how much previous hidden state influences the candidate. While simpler, GRU often performs similarly to LSTM, suggesting the extra complexity isn't always necessary. Think: GRU = 'streamlined LSTM' with ~75% of parameters but ~95% of performance.",
  },
  {
    id: 'cnn-mc-5',
    question:
      'When training an LSTM for text generation, which technique is essential to prevent exploding gradients?',
    options: [
      'Using very small learning rates (< 0.0001)',
      'Gradient clipping to cap gradient norms at a maximum threshold (e.g., 5.0)',
      'Adding more LSTM layers to distribute gradients',
      'Using only GRU instead of LSTM',
    ],
    correctAnswer: 1,
    explanation:
      "**Gradient clipping** is essential for stable LSTM training. While LSTM solves vanishing gradients, exploding gradients can still occur. Clipping works by: (1) Computing gradient norm: ||g|| = sqrt (sum of squared gradients), (2) If ||g|| > threshold (typically 1-10), rescale: g = threshold × g/||g||, (3) This caps the maximum update size while preserving direction. Implementation: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`. Why other options don't work: Small learning rates slow training without preventing explosions; More layers can worsen the problem; GRU has same issue. Without clipping, you'll see: NaN losses, training divergence, massive parameter updates. Gradient clipping is a standard practice for ALL RNN variants (vanilla RNN, LSTM, GRU) and should always be used.",
  },
];
