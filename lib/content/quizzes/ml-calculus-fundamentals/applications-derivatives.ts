/**
 * Quiz questions for Applications of Derivatives section
 */

export const applicationsderivativesQuiz = [
  {
    id: 'app-deriv-disc-1',
    question:
      "Compare gradient descent and Newton\'s method. When would you use each?",
    hint: 'Consider convergence speed, computational cost, and memory.',
    sampleAnswer: `**Gradient Descent:** Uses only first derivatives. Update: θ - α∇L. Pros: Low memory, scales well. Cons: Slower convergence. Use for: Large-scale ML (millions of parameters), non-smooth functions. **Newton\'s Method:** Uses second derivatives (Hessian). Update: θ - H⁻¹∇L. Pros: Quadratic convergence (fast). Cons: O(n²) memory, expensive Hessian computation. Use for: Small problems, when Hessian is available/approximated (L-BFGS).`,
    keyPoints: [
      'GD: first-order, scalable, slower',
      'Newton: second-order, fast convergence, expensive',
      'Quasi-Newton (L-BFGS) approximates Hessian',
      'Modern ML uses variants: Adam, RMSprop',
    ],
  },
  {
    id: 'app-deriv-disc-2',
    question:
      'Explain how Taylor series are used in second-order optimization methods.',
    hint: 'Consider quadratic approximation and the Hessian matrix.',
    sampleAnswer: `Second-order methods use Taylor expansion to order 2: L(θ) ≈ L(θ₀) + ∇L·(θ-θ₀) + ½(θ-θ₀)ᵀH(θ-θ₀). Minimizing this quadratic gives Newton update: θ = θ₀ - H⁻¹∇L. This is much faster than gradient descent because it accounts for curvature. The Hessian H captures second-order information. Trade-off: Computing/storing H is O(n²), limiting applicability to smaller problems or requiring approximations (L-BFGS, Fisher information matrix in natural gradient).`,
    keyPoints: [
      'Second-order Taylor approximation is quadratic',
      'Hessian captures curvature information',
      'Newton method minimizes quadratic approximation',
      'O(n²) cost limits scalability, requires approximations',
    ],
  },
  {
    id: 'app-deriv-disc-3',
    question:
      'How do critical points relate to loss landscape analysis in deep learning?',
    hint: 'Consider saddle points, local minima, and escaping suboptimal regions.',
    sampleAnswer: `In high dimensions, most critical points (∇L=0) are saddle points, not local minima. At saddle points, Hessian has both positive and negative eigenvalues. This has implications: (1) Gradient descent can escape saddles (noise helps), (2) Local minima in deep learning often have similar loss values (loss landscape is relatively flat), (3) Second-order information (Hessian) helps identify saddle vs minimum, (4) Modern understanding: SGD noise is beneficial for escaping saddles and finding flatter minima (better generalization).`,
    keyPoints: [
      'High-dimensional critical points usually saddles',
      'Hessian eigenvalues distinguish saddles from minima',
      'SGD noise helps escape saddles',
      'Flat minima generalize better',
    ],
  },
];
