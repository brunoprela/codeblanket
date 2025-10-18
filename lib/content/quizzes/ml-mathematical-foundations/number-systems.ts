/**
 * Quiz questions for Number Systems & Properties section
 */

export const numbersystemsQuiz = [
  {
    id: 'dq1-float-precision',
    question:
      'In deep learning, why is float32 (single precision) preferred over float64 (double precision) for most applications, despite float64 having higher precision?',
    sampleAnswer: `Float32 is preferred in deep learning for several practical reasons:

**Memory Efficiency**: Float32 uses half the memory of float64 (4 bytes vs 8 bytes). With models containing millions or billions of parameters (e.g., GPT-3 has 175 billion parameters), this difference is significant:
- Float32: 175B parameters × 4 bytes = 700 GB
- Float64: 175B parameters × 8 bytes = 1,400 GB

**Computational Speed**: Modern GPUs are optimized for float32 operations (or even lower precision like float16). Float32 tensor cores provide 2-4x speedup compared to float64. For training that takes days or weeks, this matters substantially.

**Sufficient Precision**: The precision of float32 (~7 decimal digits) is sufficient for most ML applications. Neural networks are inherently noisy - we use stochastic gradient descent, dropout, and other stochastic processes. The additional precision of float64 doesn't meaningfully improve model performance.

**Gradient Descent Tolerance**: The optimization landscape of neural networks is complex and non-convex. The extra precision of float64 doesn't help us find better minima; we're often satisfied with "good enough" solutions.

**Trade-offs**: The only time float64 might be preferred is in scientific computing where numerical accuracy is critical, or in certain numerical stability situations. For standard deep learning, float32 or even mixed-precision training (float16 with float32 accumulation) is the standard.`,
    keyPoints: [
      'Memory efficiency: 50% reduction in model size',
      'GPU hardware optimization for float32',
      'Sufficient precision for ML optimization',
      'No significant accuracy gains from float64',
      'Faster training and inference',
    ],
  },
  {
    id: 'dq2-complex-numbers',
    question:
      'How are complex numbers used in machine learning and signal processing? Provide specific examples of where they are essential.',
    sampleAnswer: `Complex numbers are fundamental in several ML and signal processing applications:

**1. Fourier Transforms**:
The Discrete Fourier Transform (DFT) converts time-domain signals to frequency domain:
X(k) = Σ x(n) × e^(-i2πkn/N)

This is essential for:
- Audio processing (speech recognition, music analysis)
- Image processing (frequency filtering, compression)
- Time series analysis (detecting periodic patterns)

**2. Convolution Theorem**:
Complex numbers make convolution efficient via:
- Convolution in time domain = Multiplication in frequency domain
- FFT (Fast Fourier Transform) uses complex arithmetic
- Used in CNNs for efficient computation

**3. Eigenvalues and Eigenvectors**:
Many real matrices have complex eigenvalues:
- Stability analysis of systems
- PageRank algorithm
- Principal Component Analysis (PCA) in certain cases
- Markov chain analysis

**4. Quantum Machine Learning**:
Quantum computing fundamentally operates with complex numbers:
- Quantum states are complex-valued vectors
- Quantum gates are unitary matrices (complex)
- Emerging field of quantum neural networks

**5. Complex-Valued Neural Networks**:
Recently, research has explored networks with complex-valued weights and activations:
- Better suited for signals naturally represented as complex (radar, RF signals)
- Potential advantages in representing phase information
- Used in specific domains like magnetic resonance imaging (MRI)

**Example in Audio**:
When processing audio with a spectrogram, each point represents a complex number where the magnitude is the amplitude and the phase contains timing information. Both are crucial for reconstructing the original signal.`,
    keyPoints: [
      'Essential for Fourier transforms and frequency analysis',
      'Enable efficient convolution via FFT',
      'Required for eigenvalue computations',
      'Fundamental to quantum computing',
      'Emerging applications in complex-valued neural networks',
    ],
  },
  {
    id: 'dq3-numerical-trading',
    question:
      'In algorithmic trading systems, how can understanding number systems and numerical precision prevent costly bugs? Provide examples of numerical issues that could impact trading decisions.',
    sampleAnswer: `Numerical precision issues can cause severe problems in trading systems, potentially leading to significant financial losses:

**1. Price Precision and Tick Size**:
Stock prices are typically stored with limited decimal places (e.g., cents). Failing to account for this can cause issues:
- Rounding errors when calculating position sizes
- Impossible price targets (e.g., $10.125 when minimum tick is $0.01)
- Accumulation of small errors in high-frequency trading

**2. Floating-Point Arithmetic in P&L Calculations**:
\`\`\`python
# Dangerous
shares = 1000000
price_bought = 100.1
price_sold = 100.2
profit = shares * (price_sold - price_bought)  # Rounding errors
\`\`\`

Better approach: Use integer arithmetic with cents or basis points.

**3. Order Quantity Calculation**:
When calculating order quantities based on portfolio percentage:
\`\`\`python
# Bad
portfolio_value = 1000000.00
target_pct = 0.15
share_price = 123.45
shares = int((portfolio_value * target_pct) / share_price)
\`\`\`

Issues: Truncation errors, failure to account for lot sizes, minimum order quantities.

**4. Interest Rate Calculations**:
Compounding interest with daily rates:
- Direct multiplication accumulates floating-point errors
- Better: Use log space or exact rational arithmetic
- Critical for accurate bond pricing and yield calculations

**5. Cumulative Returns**:
\`\`\`python
# Unstable for long time series
cumulative_return = (1 + r1) * (1 + r2) * ... * (1 + rn)

# Better
log_cumulative = sum(log(1 + r) for r in returns)
cumulative_return = exp(log_cumulative)
\`\`\`

**6. Risk Metrics (VaR, CVaR)**:
When calculating Value at Risk:
- Sorting large arrays of returns requires stable precision
- Quantile calculations can be sensitive to numerical errors
- Covariance matrix computations can be ill-conditioned

**7. Stop-Loss Triggers**:
\`\`\`python
# Dangerous with floating point
if current_price <= stop_price:  # May not trigger due to precision
    execute_stop_loss()

# Better with tolerance
if current_price <= stop_price + EPSILON:
    execute_stop_loss()
\`\`\`

**Best Practices for Trading Systems**:
1. Use decimal.Decimal for money calculations
2. Store prices in smallest unit (cents, satoshis)
3. Implement tolerance-based comparisons
4. Use rational arithmetic for exact calculations
5. Thoroughly test with edge cases
6. Implement sanity checks and validation
7. Log all calculations for audit trails

**Real Example**:
The 2012 Knight Capital trading glitch lost $440 million in 45 minutes partly due to software bugs, including numerical handling issues in order execution logic.`,
    keyPoints: [
      'Price precision and tick size handling',
      'Use integer arithmetic or Decimal for money',
      'Floating-point errors in P&L calculations',
      'Numerical stability in risk calculations',
      'Tolerance-based comparisons for triggers',
      'Historical examples of costly bugs',
    ],
  },
];
