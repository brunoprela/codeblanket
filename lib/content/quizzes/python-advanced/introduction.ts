/**
 * Quiz questions for Advanced Python: Beyond the Basics section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what decorators are and why they are useful. Give a concrete example where decorators solve a real problem.',
    sampleAnswer:
      'Decorators are functions that modify or enhance other functions without changing their source code. They use the @ syntax and are a form of metaprogramming. For example, in a web API, I might have dozens of endpoints that need authentication. Instead of adding auth checking code to each function, I can create an @require_auth decorator that wraps functions and checks authentication before execution. This follows the DRY principle, makes the code cleaner, and centralizes authentication logic. If I need to change how auth works, I update one decorator instead of 50 functions.',
    keyPoints: [
      'Functions that modify other functions',
      'Applied with @ syntax',
      'Example: @require_auth for authentication',
      'Follows DRY principle',
      'Centralizes cross-cutting concerns',
    ],
  },
  {
    id: 'q2',
    question:
      'What are generators and how do they differ from regular functions? When should you use them?',
    sampleAnswer:
      'Generators are functions that use yield instead of return, creating iterators that produce values lazily on-demand. Unlike regular functions that compute all values at once, generators produce one value at a time and maintain their state between calls. Use generators when: (1) processing large datasets that would not fit in memory, (2) creating infinite sequences, or (3) building data pipelines. For example, reading a 10GB log file line by line with a generator uses constant memory, while loading it all at once would use 10GB.',
    keyPoints: [
      'Use yield instead of return',
      'Produce values lazily on-demand',
      'Maintain state between calls',
      'Memory efficient for large data',
      'Example: reading huge files line by line',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the with statement and context managers. Why are they important for resource management?',
    sampleAnswer:
      'Context managers handle setup and cleanup of resources automatically using the with statement. They implement __enter__ and __exit__ methods. This is crucial because it guarantees cleanup happens even if errors occur. For example: "with open (file) as f:" ensures the file is closed even if an exception is raised while reading. Without context managers, you need try/finally blocks everywhere, which is error-prone. Context managers are used for files, database connections, locks, and any resource that needs cleanup.',
    keyPoints: [
      'Automatic resource setup and cleanup',
      '__enter__ and __exit__ methods',
      'Guarantees cleanup even with errors',
      'Example: with open() ensures file closes',
      'Used for files, databases, locks',
    ],
  },
];
