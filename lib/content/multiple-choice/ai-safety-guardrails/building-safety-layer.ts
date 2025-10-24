/**
 * Multiple choice questions for Building a Safety Layer section
 */

export const buildingsafetylayerMultipleChoice = [
  {
    id: 'build-safety-mc-1',
    question:
      'Your safety layer runs input validation and output validation serially. Total latency: 150ms (70ms input + 80ms output). How can you reduce latency?',
    options: [
      'Run input and output validation in parallel',
      'Run multiple checks within each stage in parallel',
      'Cache validation results',
      'Skip less critical checks',
    ],
    correctAnswer: 1,
    explanation:
      "You can't run input and output validation in parallel—input must happen before generation, output after. But WITHIN each stage, run multiple checks in parallel. If input has 3 checks (20ms, 30ms, 40ms serial = 90ms), parallel = max(20,30,40) = 40ms. Option B is correct. Options C and D help but are secondary optimizations.",
  },
  {
    id: 'build-safety-mc-2',
    question: 'A validation service fails. What is the SAFEST approach?',
    options: [
      'Allow all requests (fail-open)',
      'Block all requests (fail-closed)',
      'Use cached results or degrade gracefully',
      'Retry the service 3 times',
    ],
    correctAnswer: 2,
    explanation:
      "Best practice: Use cached results or degrade gracefully. Block critical violations (PII, harmful content) with fallback rules. Allow non-critical checks to fail-open. Option A (fail-open entirely) is unsafe. Option B (fail-closed entirely) blocks legitimate users. Option D (retry) doesn't solve persistent outages.",
  },
  {
    id: 'build-safety-mc-3',
    question:
      'Your safety layer blocks 5% of requests: 3% true positives, 2% false positives. What should you prioritize?',
    options: [
      'Reduce false positives (improve user experience)',
      'Increase true positives (catch more violations)',
      'Reduce latency',
      'Both A and B equally',
    ],
    correctAnswer: 0,
    explanation:
      '2% false positive rate is high—blocking legitimate users harms UX and trust. Prioritize reducing false positives. 3% true positive rate is reasonable (depends on baseline violation rate). Option B (increase TP) risks more false positives. Balance is important (D), but A is higher priority here.',
  },
  {
    id: 'build-safety-mc-4',
    question:
      'You detect a new attack bypassing your filters 3 days after it starts. What is the PRIMARY improvement needed?',
    options: [
      'Better filters',
      'Continuous monitoring and anomaly detection',
      'More testing',
      'Faster deployment pipeline',
    ],
    correctAnswer: 1,
    explanation:
      "The primary issue is detection time (3 days). Continuous monitoring with anomaly detection (block rate changes, new patterns, user reports) would detect within 1 hour. Better filters (A) help but don't address detection speed. Options C and D are important but secondary to detection.",
  },
  {
    id: 'build-safety-mc-5',
    question: 'Your safety layer architecture should have:',
    options: [
      'Single layer: run all checks together',
      'Two layers: input validation and output validation',
      'Three layers: input, generation, output',
      'Defense in depth: multiple redundant safety mechanisms',
    ],
    correctAnswer: 3,
    explanation:
      'Defense in depth means multiple independent layers of security. If one layer fails, others catch violations. This includes: input validation, output validation, rate limiting, audit logging, human review, anomaly detection. Option D is most robust. Options B and C describe stages, not comprehensive defense.',
  },
];
