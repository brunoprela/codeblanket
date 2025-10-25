export const horizontal_scalingMC = {
  title: 'Multiple Choice Quiz - Horizontal Scaling',
  questions: [
    {
      question:
        'What is the primary benefit of implementing horizontal scaling in a production LLM application?',
      options: [
        'Reduced development time',
        'Improved performance and cost efficiency',
        'Simplified codebase',
        'Easier debugging',
      ],
      correctAnswer: 1,
      explanation:
        'The primary benefit of horizontal scaling is improved performance and cost efficiency at scale. While other benefits may exist, the main goal is to handle larger workloads more effectively while controlling costs.',
    },
    {
      question:
        'When implementing horizontal scaling, what is typically the biggest challenge?',
      options: [
        'Writing the initial code',
        'Managing complexity and monitoring at scale',
        'Getting management buy-in',
        'Finding the right tools',
      ],
      correctAnswer: 1,
      explanation:
        'Managing complexity and monitoring at scale is typically the biggest challenge. As systems scale, they become more complex with more failure modes, making monitoring and management critical.',
    },
    {
      question:
        'In the context of horizontal scaling, what metric is most important to track?',
      options: [
        'Lines of code written',
        'Number of features shipped',
        'Cost per request and latency',
        'Team velocity',
      ],
      correctAnswer: 2,
      explanation:
        'Cost per request and latency are the most critical metrics for horizontal scaling. These directly measure the effectiveness of your scaling strategy and its impact on user experience and budget.',
    },
    {
      question:
        'What is a common mistake when first implementing horizontal scaling?',
      options: [
        'Not testing thoroughly enough',
        'Over-optimizing prematurely without measuring',
        'Using too many external libraries',
        'Not documenting the code',
      ],
      correctAnswer: 1,
      explanation:
        "Over-optimizing prematurely without measuring is a common mistake. It\'s important to establish baseline metrics first, identify actual bottlenecks through measurement, then optimize based on data rather than assumptions.",
    },
    {
      question:
        'For a production LLM application serving 100,000 users, what is a reasonable target for horizontal scaling?',
      options: [
        'Perfect optimization with zero overhead',
        '50-80% improvement in key metrics',
        'Doubling all performance metrics',
        'Minimal changes to keep things simple',
      ],
      correctAnswer: 1,
      explanation:
        "A 50-80% improvement in key metrics is a reasonable and achievable target. Perfect optimization (100%) is usually impossible and has diminishing returns, while minimal changes likely won't address scale challenges. Aim for significant, measurable improvements that justify the engineering investment.",
    },
  ],
};
