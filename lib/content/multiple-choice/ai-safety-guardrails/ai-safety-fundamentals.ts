/**
 * Multiple choice questions for AI Safety Fundamentals section
 */

export const aisafetyfundamentalsMultipleChoice = [
  {
    id: 'safety-fund-mc-1',
    question:
      'Your AI system has a 2% false positive rate (blocks 2% of legitimate requests). What is the PRIMARY concern?',
    options: [
      'False positives waste computational resources',
      'False positives harm user experience and trust',
      'False positives increase operational costs',
      'False positives violate regulatory requirements',
    ],
    correctAnswer: 1,
    explanation:
      'The primary concern is user experience—blocking legitimate users frustrates them, reduces trust, and may cause them to abandon your service. While costs (A, C) matter, user trust is paramount. Option D is incorrect—regulations focus on false negatives (missing actual violations), not false positives.',
  },
  {
    id: 'safety-fund-mc-2',
    question:
      'When implementing defense-in-depth for AI safety, what does this principle mean?',
    options: [
      'Use the most expensive safety tools available',
      'Implement multiple independent layers of safety controls',
      'Only implement safety checks at the input layer',
      'Rely on a single comprehensive safety system',
    ],
    correctAnswer: 1,
    explanation:
      'Defense-in-depth means multiple independent layers—if one layer fails, others catch the issue. Example: Input validation + output validation + monitoring. Option D (single system) is the opposite of defense-in-depth. Options A and C are incorrect approaches.',
  },
  {
    id: 'safety-fund-mc-3',
    question:
      'A safety check detects a potential issue with 60% confidence. What is the BEST approach?',
    options: [
      'Always block—safety is more important than false positives',
      'Always allow—60% confidence is too low to act on',
      'Flag for human review and allow temporarily',
      'Run additional checks to increase confidence',
    ],
    correctAnswer: 3,
    explanation:
      'At medium confidence (60%), the best approach is to run additional checks to increase confidence before making a decision. Option C (flag for review) is good but allows potentially unsafe content. Option A creates too many false positives. Option B ignores a real signal.',
  },
  {
    id: 'safety-fund-mc-4',
    question:
      'You deploy a new safety feature that reduces violations by 80% but increases latency from 200ms to 800ms. What should you do?',
    options: [
      'Keep it—safety is worth the latency cost',
      'Remove it—users will not tolerate 800ms latency',
      'Optimize the feature to reduce latency while maintaining safety',
      'Make it optional for users who want extra safety',
    ],
    correctAnswer: 2,
    explanation:
      '800ms is too slow for most applications, but the safety benefit (80% reduction) is significant. The right approach is optimization—can you cache results, run checks in parallel, or use a faster implementation? Option D (optional) is dangerous—safety should not be optional.',
  },
  {
    id: 'safety-fund-mc-5',
    question:
      'Which safety principle is MOST important when you are uncertain about whether content is safe?',
    options: [
      'Transparency—explain the uncertainty to the user',
      'Fail-safe defaults—when in doubt, err on the side of caution',
      'Performance—process the request quickly',
      'Cost optimization—use the cheapest safety check',
    ],
    correctAnswer: 1,
    explanation:
      'Fail-safe defaults is the core principle—when uncertain, choose the safe option (block or flag). This prevents harm. Transparency (A) is important but secondary to safety. Options C and D prioritize wrong concerns over safety.',
  },
];
