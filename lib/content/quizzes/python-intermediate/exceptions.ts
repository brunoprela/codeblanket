/**
 * Quiz questions for Exception Handling section
 */

export const exceptionsQuiz = [
  {
    id: 'pi-exceptions-q-1',
    question:
      'Explain the difference between catching specific exceptions versus using a bare except or except Exception. When is each appropriate?',
    hint: 'Think about debugging, KeyboardInterrupt, SystemExit, and error handling granularity.',
    sampleAnswer:
      'Always catch specific exceptions (like FileNotFoundError, ValueError) when you know what can go wrong and how to handle it. This makes code more maintainable and prevents hiding bugs. "except Exception" catches most errors but not system-critical ones like KeyboardInterrupt or SystemExit. Bare "except:" catches EVERYTHING including Ctrl+C, which is dangerous. Use specific exceptions for normal error handling, "except Exception" only for top-level logging, and bare except almost never (maybe for ensuring cleanup in critical systems).',
    keyPoints: [
      'Specific exceptions: best for known error handling',
      'except Exception: catches most, but not system exits',
      'Bare except: dangerous, catches KeyboardInterrupt too',
      'More specific = better debugging and maintenance',
    ],
  },
  {
    id: 'pi-exceptions-q-2',
    question:
      'What is the purpose of the finally block? How does it differ from putting code after the try-except block?',
    hint: 'Consider what happens when exceptions are raised or when return statements execute.',
    sampleAnswer:
      "The finally block ALWAYS executes, even if: 1) an exception is raised and not caught, 2) a return statement is executed in try/except, 3) break/continue is used in a loop. Code after try-except only runs if the exception was caught or didn't occur. Use finally for cleanup (closing files, releasing locks, database rollback) that must happen regardless of success or failure. However, context managers (with statement) are often cleaner than try-finally for resource management.",
    keyPoints: [
      'finally: always runs, even on return/break',
      'Code after try-except: only if exception caught',
      'Use for mandatory cleanup (files, locks, connections)',
      'Context managers often better than try-finally',
    ],
  },
  {
    id: 'pi-exceptions-q-3',
    question:
      'When should you create custom exceptions? What makes a good custom exception?',
    hint: 'Consider API design, error handling hierarchy, and what information exceptions should carry.',
    sampleAnswer:
      'Create custom exceptions for domain-specific errors that deserve special handling (like ValidationError, InsufficientFundsError, DatabaseConnectionError). Good custom exceptions: 1) Inherit from appropriate base (Exception or a more specific built-in), 2) Have descriptive names ending in "Error", 3) Can carry context (user_id, amount, etc.), 4) Form a logical hierarchy (APIError → HTTPError → NotFoundError). Don\'t create custom exceptions for every error - use built-ins when they fit (ValueError, TypeError). Custom exceptions make error handling more semantic and maintainable.',
    keyPoints: [
      'Use for domain-specific errors needing special handling',
      'Inherit from Exception or specific built-in',
      'Can carry context data for debugging',
      'Create hierarchy for related errors',
    ],
  },
];
