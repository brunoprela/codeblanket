/**
 * Multiple choice questions for Building Production Multi-Agent Systems section
 */

export const buildingproductionmultiagentsystemsMultipleChoice = [
  {
    id: 'maas-prod-mc-1',
    question:
      'In a blue-green deployment, you switch traffic from Blue (old) to Green (new) environment. After 10 minutes, error rates spike. What is the FASTEST rollback strategy?',
    options: [
      'Restart the Green environment and investigate',
      'Switch traffic back to Blue environment immediately',
      'Deploy a hotfix to Green and wait for it to propagate',
      'Gradually reduce traffic to Green (canary rollback)',
    ],
    correctAnswer: 1,
    explanation:
      "Blue-green deployments enable instant rollback—just switch traffic back to Blue (seconds to minutes). Option A doesn't fix the immediate issue. Option C takes too long. Option D (canary rollback) applies to canary deployments, not blue-green.",
  },
  {
    id: 'maas-prod-mc-2',
    question:
      'Your multi-agent system costs $270/hour due to GPT-4 calls. You implement caching with a 30% hit rate. What is the new cost?',
    options: [
      '$81/hour (30% of $270)',
      '$189/hour (70% of $270)',
      "$270/hour (caching doesn't reduce costs)",
      '$135/hour (50% of $270)',
    ],
    correctAnswer: 1,
    explanation:
      '30% hit rate means 30% of calls are cached (free), 70% call the API (full cost). New cost = 0.7 × $270 = $189/hour. Option A incorrectly calculates the cached portion instead of the remaining cost. Options C and D are incorrect.',
  },
  {
    id: 'maas-prod-mc-3',
    question:
      'A circuit breaker for the LLM API has a threshold of "5 failures in 1 minute" before opening. After opening, it waits 30 seconds before trying again (half-open). What is the PRIMARY purpose of the half-open state?',
    options: [
      'To permanently disable the API if it continues failing',
      'To test if the API has recovered before fully closing the circuit',
      'To allow some requests through while the API is failing',
      'To alert the on-call team that the API is down',
    ],
    correctAnswer: 1,
    explanation:
      "Half-open state is a test phase—allow one request through to check if the API has recovered. If it succeeds, close the circuit (resume normal operation). If it fails, reopen the circuit. Option A is incorrect—circuit breakers don't permanently disable. Option C misunderstands half-open (only test requests, not regular traffic). Option D is a side effect, not the primary purpose.",
  },
  {
    id: 'maas-prod-mc-4',
    question:
      'A production workflow has a 95% success rate, but 5% of failures are critical (data loss, security issues). What is the BEST monitoring strategy?',
    options: [
      'Alert on any failure (95% success rate is too low)',
      'Alert only on critical failures; log non-critical failures',
      'No alerts needed—95% success rate is acceptable',
      'Alert if success rate drops below 90%',
    ],
    correctAnswer: 1,
    explanation:
      "Differentiate between critical and non-critical failures. Alert immediately on critical issues (data loss, security). Log and monitor non-critical failures, but don't wake up the on-call team for every minor error. Option A causes alert fatigue. Option C is dangerous—critical failures need immediate attention. Option D might miss critical issues if they're rare but severe.",
  },
  {
    id: 'maas-prod-mc-5',
    question:
      'Your multi-agent system is deployed on Kubernetes. An agent pod crashes due to out-of-memory (OOM). What is the MOST likely cause?',
    options: [
      "The agent's code has a memory leak",
      'The agent is caching too much data without eviction',
      "The pod's memory limit is set too low for the agent's workload",
      'All of the above are possible causes',
    ],
    correctAnswer: 3,
    explanation:
      'OOM errors can have multiple causes: (A) memory leaks in code, (B) unbounded caches, (C) insufficient resource limits. All three are common in production. The correct debugging approach is to check memory usage patterns, caching strategy, and resource configuration.',
  },
];
