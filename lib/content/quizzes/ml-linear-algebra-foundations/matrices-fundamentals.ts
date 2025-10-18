/**
 * Quiz questions for Matrices Fundamentals section
 */

export const matricesfundamentalsQuiz = [
  {
    id: 'mat-fund-d1',
    question:
      'Explain how matrix multiplication enables efficient batch processing in neural networks. Why is this crucial for modern deep learning?',
    sampleAnswer:
      'Matrix multiplication allows neural networks to process multiple samples simultaneously through a single operation. Instead of computing y = Wx + b for each sample individually in a loop (which would require n forward passes for n samples), we organize samples into a batch matrix X of shape (batch_size, input_dim) and compute Y = XW^T + b in one operation, producing outputs for all samples at once. This is crucial for several reasons: (1) Computational efficiency—GPUs are optimized for matrix operations and can parallelize thousands of arithmetic operations simultaneously. A single matrix multiplication is vastly faster than many sequential vector-matrix multiplications. (2) Memory efficiency—Modern deep learning frameworks can optimize memory access patterns when working with batches. (3) Statistical benefits—Using mini-batches provides better gradient estimates than single samples (less noisy than SGD) while being more computationally efficient than full-batch gradient descent. (4) Hardware utilization—GPUs have thousands of cores that sit idle when processing single samples; batching keeps them busy. Without efficient batching via matrix operations, training modern deep learning models would be prohibitively slow. A model that takes hours with batching might take weeks or months without it.',
    keyPoints: [
      'Batch matrix multiplication: Y = XW^T processes all samples simultaneously',
      'GPU optimization: matrix ops enable massive parallelization (100x+ speedup)',
      'Batch size trade-off: computational efficiency vs gradient update frequency',
    ],
  },
  {
    id: 'mat-fund-d2',
    question:
      'The transpose operation reverses multiplication order: (AB)ᵀ = BᵀAᵀ. Explain why this property is important in backpropagation for training neural networks.',
    sampleAnswer:
      'This transpose property is fundamental to backpropagation because gradients flow backward through the network in reverse order of the forward pass. In the forward pass, we compute Y = XW + b (input X times weights W). During backpropagation, we receive gradients dL/dY and need to compute gradients with respect to X and W. Using the chain rule: dL/dX = dL/dY × dY/dX. Since Y = XW, we have dY/dX = W^T (transpose of W). So dL/dX = (dL/dY)W^T. Similarly, dL/dW = X^T(dL/dY). Notice how the matrices appear in reversed, transposed form compared to the forward pass. This mirrors the mathematical property (AB)^T = B^T A^T. The transpose reversal is why backprop involves transposed weight matrices—we are literally reversing the flow of information. This property ensures that gradient dimensions match correctly: if forward pass goes from (batch, input_dim) → (batch, output_dim) via W of shape (input_dim, output_dim), then backward pass must go (batch, output_dim) → (batch, input_dim) via W^T of shape (output_dim, input_dim). Understanding this transpose property is key to implementing backpropagation correctly and debugging gradient computation errors.',
    keyPoints: [
      'Forward: Y = XW; Backward: dL/dX = (dL/dY)W^T (transposed weights)',
      'Transpose reversal mirrors (AB)^T = B^T A^T mathematical property',
      'Ensures gradient dimensions match: information flows backward through transposed weights',
    ],
  },
  {
    id: 'mat-fund-d3',
    question:
      'Compare representing a dataset as a list of vectors versus a single matrix. What are the trade-offs in terms of operations, memory, and coding practices in machine learning?',
    sampleAnswer:
      "Representing a dataset as a matrix (2D array) rather than a list of vectors (list of 1D arrays) offers significant advantages in ML, though with some trade-offs. Matrix representation enables: (1) Vectorized operations—computing statistics (mean, std) or transformations (scaling, PCA) on all features at once using optimized linear algebra libraries, orders of magnitude faster than looping through vectors. (2) Consistent dimensions—matrices enforce uniform feature counts across samples, catching data inconsistencies early. (3) Straightforward model application—neural networks expect matrix inputs for batch processing; constantly converting lists to matrices adds overhead. (4) Memory layout—contiguous memory storage enables CPU/GPU cache optimization and efficient data transfer. However, list-of-vectors can be advantageous when: (1) Samples have variable lengths (though padding/masking or ragged tensors often work), (2) Data arrives sequentially and doesn't fit in memory (though streaming matrix operations exist), (3) You need flexibility to modify individual samples independently. In practice, use matrices for structured, fixed-size data (tabular, images, most ML tasks). Use lists when dealing with truly variable-length sequences (before padding), or when prototyping with small datasets where performance doesn't matter. Modern ML libraries (NumPy, PyTorch, TensorFlow) are built around matrix operations, so embracing matrix representation aligns with the ecosystem and unlocks performance. The rule: default to matrices, use lists only when necessary.",
    keyPoints: [
      'Matrix: vectorized ops, GPU optimization, enforced dimensions (orders of magnitude faster)',
      'List-of-vectors: flexibility for variable lengths, sequential processing',
      'Default to matrices for ML (aligns with NumPy/PyTorch/TensorFlow ecosystem)',
    ],
  },
];
