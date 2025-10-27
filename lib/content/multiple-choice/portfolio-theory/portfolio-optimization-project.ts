export const portfolioOptimizationProjectMC = {
  id: 'portfolio-optimization-project-mc',
  title: 'Module Project: Portfolio Optimization Platform - Multiple Choice',
  questions: [
    {
      id: 'pop-mc-1',
      type: 'multiple-choice' as const,
      question:
        'For 1000-stock portfolio, covariance matrix estimation with 5 years daily data requires storing:',
      options: [
        '500,000 data points',
        '1,000,000 data points',
        '1,250,000 data points',
        '5,000,000 data points',
      ],
      correctAnswer: 2,
      explanation:
        'Answer: C. 1000 stocks × 1250 trading days (5 years × ~250 days) = 1,250,000 return observations. Covariance matrix itself has N(N+1)/2 = 500,500 unique elements. Storage: 1.25M returns (10MB) + 500K covariances (4MB) ≈ 14MB. Manageable. NOT 500K (A), not 1M (B - close but incorrect math), not 5M (D - too high). The math: N stocks × T time periods = 1000 × 1250.',
    },
    {
      id: 'pop-mc-2',
      type: 'multiple-choice' as const,
      question:
        'CVXPY optimization with 1000 variables and 200 constraints typically solves in:',
      options: [
        '0.1-0.5 seconds (immediate)',
        '1-5 seconds (fast)',
        '10-30 seconds (moderate)',
        '60+ seconds (slow)',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Modern QP solvers (OSQP, ECOS) solve 1000-variable problems in 1-5 seconds on typical hardware. NOT immediate (A - that's for <100 variables), not 10-30s (C - that's 5000+ variables), not 60s+ (D - that's 10,000+ variables or complex conic constraints). 1-5 seconds is acceptable for production (can add caching). If slower, consider solver tuning or preconditioning.",
    },
    {
      id: 'pop-mc-3',
      type: 'multiple-choice' as const,
      question:
        'Caching covariance matrices for 1 hour with 100 clients provides:',
      options: [
        'No benefit (markets change continuously)',
        '10x efficiency (avoid recalculation)',
        '100x efficiency (serve all clients from cache)',
        'Infinite efficiency (never recompute)',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. Compute covariance once at 9am, serve all 100 clients requesting it from 9:00-10:00. 100 requests served with 1 computation = 100× efficiency. NOT no benefit (A - covariances don't change minute-to-minute), not just 10× (B - underestimate), not infinite (D - must recompute after 1 hour). Caching is CRITICAL for multi-client platforms. This is why shared infrastructure scales efficiently.",
    },
    {
      id: 'pop-mc-4',
      type: 'multiple-choice' as const,
      question: 'Production portfolio optimization platform should target:',
      options: [
        'p50 latency <5s, p95 latency <10s, 99% uptime',
        'p50 latency <1s, p95 latency <3s, 99.9% uptime',
        'p50 latency <0.1s, p95 latency <0.5s, 99.99% uptime',
        'p50 latency <10s, p95 latency <60s, 95% uptime',
      ],
      correctAnswer: 0,
      explanation:
        'Answer: A. Reasonable production targets for financial optimization: p50 <5s (most requests fast), p95 <10s (handle 95% within 10s), 99% uptime (8.7 hours downtime annually). Not <1s (B - too aggressive for 1000-stock optimization), not <0.1s (C - impossible for complex optimization), not 10s/60s (D - too slow, 95% uptime inadequate). These targets balance performance, cost, and reliability.',
    },
    {
      id: 'pop-mc-5',
      type: 'multiple-choice' as const,
      question: 'Integration test for portfolio optimization should verify:',
      options: [
        'Only that optimization completes without errors',
        'Results match pre-computed golden dataset within 0.01% tolerance',
        'Optimization runs in under 1 second',
        'All constraints are satisfied (sufficient test)',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Regression test against golden dataset ensures results are consistent over time. 0.01% tolerance allows for minor numerical differences while catching material changes. NOT just error-free (A - insufficient, could give wrong results), not <1s (C - that's performance test, not correctness), not just constraints (D - need to verify optimal solution value, weights, metrics). Golden dataset testing is gold standard for financial systems.",
    },
  ],
};
