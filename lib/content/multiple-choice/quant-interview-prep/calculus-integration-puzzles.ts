import { MultipleChoiceQuestion } from '@/lib/types';

export const calculusIntegrationPuzzlesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'cip-mc-1',
      question: 'Evaluate ∫₀¹ 3x² dx',
      options: ['1/3', '1', '3', '9'],
      correctAnswer: 1,
      explanation:
        '∫ 3x² dx = 3 · x³/3 = x³ + C. Evaluate from 0 to 1: [x³]₀¹ = 1³ - 0³ = 1. Alternatively, ∫₀¹ 3x² dx = 3∫₀¹ x² dx = 3 · (1/3) = 1.',
    },
    {
      id: 'cip-mc-2',
      question: 'Find the critical points of f(x) = x³ - 3x² + 2',
      options: ['x = 0, 2', 'x = 0, 3', 'x = 1, 2', 'x = 2 only'],
      correctAnswer: 0,
      explanation:
        "f'(x) = 3x² - 6x = 3x(x - 2). Critical points where f'(x) = 0: x = 0 or x = 2. Check second derivative: f''(x) = 6x - 6. At x=0: f''(0) = -6 < 0 (local max). At x=2: f''(2) = 6 > 0 (local min).",
    },
    {
      id: 'cip-mc-3',
      question: 'Using Taylor series, approximate e^0.05 (keep first 3 terms)',
      options: ['1.0500', '1.0513', '1.0525', '1.0538'],
      correctAnswer: 1,
      explanation:
        'eˣ ≈ 1 + x + x²/2 + ... For x=0.05: e^0.05 ≈ 1 + 0.05 + (0.05)²/2 = 1 + 0.05 + 0.00125 = 1.05125 ≈ 1.0513. Actual value: 1.05127... Error with 3 terms: 0.00002 (0.002%).',
    },
    {
      id: 'cip-mc-4',
      question:
        'Solve the differential equation dy/dx = 2y with initial condition y(0) = 3',
      options: ['y = 3e^(2x)', 'y = 2e^(3x)', 'y = 3 + 2x', 'y = 6e^x'],
      correctAnswer: 0,
      explanation:
        'Separable ODE: dy/y = 2dx. Integrate: ln|y| = 2x + C. Exponentiate: y = Ae^(2x). Initial condition y(0)=3: 3 = Ae⁰ = A. Therefore y = 3e^(2x). This represents exponential growth at rate 2.',
    },
    {
      id: 'cip-mc-5',
      question: 'If f(x,y) = x²y + y³, find ∂f/∂x at (2,1)',
      options: ['2', '4', '5', '7'],
      correctAnswer: 1,
      explanation:
        'Partial derivative with respect to x (treat y as constant): ∂f/∂x = 2xy + 0 = 2xy. At (2,1): ∂f/∂x = 2(2)(1) = 4. This represents the rate of change of f in the x-direction at point (2,1).',
    },
  ];
