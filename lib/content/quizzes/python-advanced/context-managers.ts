/**
 * Quiz questions for Context Managers & Resource Management section
 */

export const contextmanagersQuiz = [
  {
    id: 'q1',
    question:
      'Why are context managers critical for resource management? What problem do they solve?',
    sampleAnswer:
      'Context managers guarantee cleanup happens even when errors occur. Without them, if an exception is raised while using a resource like a file or database connection, you might forget to clean up, causing resource leaks. try/finally blocks work but are verbose and error-prone—developers forget them. Context managers centralize the cleanup logic and make it impossible to forget. For example, "with open (file)" ensures the file closes even if an exception occurs while reading. This prevents file descriptor exhaustion, database connection pool exhaustion, and other resource leak issues.',
    keyPoints: [
      'Guarantee cleanup even with exceptions',
      'Prevent resource leaks',
      'Centralize cleanup logic',
      'Example: file always closes',
      'Better than try/finally (less error-prone)',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the __exit__ method. What are its parameters and when should you return True vs False?',
    sampleAnswer:
      '__exit__ receives three parameters: exc_type (exception class or None), exc_val (exception instance), and exc_tb (traceback). It is called when exiting the with block, even if an exception occurred. Return False (default) to let exceptions propagate normally. Return True to suppress the exception—use this carefully only when you can properly handle the error. For example, a database context manager might rollback on exception and return False so the caller knows something failed. Only return True if the exception is expected and fully handled within __exit__.',
    keyPoints: [
      'Parameters: exc_type, exc_val, exc_tb',
      'Always called, even with exceptions',
      'Return False: let exception propagate (default)',
      'Return True: suppress exception (use carefully)',
      'Example: rollback on error, return False',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the @contextmanager decorator approach versus implementing __enter__/__exit__ directly. When would you use each?',
    sampleAnswer:
      "The @contextmanager decorator (from contextlib) lets you create context managers with a simple generator function: code before yield is __enter__, yield provides the value, code after yield is __exit__. This is much simpler for straightforward cases. Use it when: 1) cleanup logic is simple, 2) you don't need complex exception handling. Implement __enter__/__exit__ directly when: 1) you need fine-grained control over exception handling, 2) the context manager is a class with state and methods, 3) you want to reuse the same object multiple times. For example, a simple timer uses @contextmanager; a database connection pool with state uses __enter__/__exit__.",
    keyPoints: [
      '@contextmanager: simple, generator-based',
      'Direct implementation: more control, stateful',
      'Use decorator for simple cleanup',
      'Use class for complex state/exception handling',
      'Example: timer vs connection pool',
    ],
  },
];
