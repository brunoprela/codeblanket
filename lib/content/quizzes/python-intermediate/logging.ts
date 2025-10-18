/**
 * Quiz questions for Logging and Debugging section
 */

export const loggingQuiz = [
  {
    id: 'q1',
    question:
      'Why is logging better than print() for production applications? List at least 4 advantages.',
    hint: 'Think about control, formatting, output destinations, and performance.',
    sampleAnswer:
      'Logging is superior to print() for production code because: 1) **Levels**: You can control verbosity with DEBUG/INFO/WARNING/ERROR/CRITICAL and filter at runtime without code changes, 2) **Formatting**: Automatically includes timestamps, module names, line numbers, and exception tracebacks, 3) **Multiple outputs**: Can simultaneously log to console, files, network services, or cloud logging, 4) **Performance**: Can be disabled in production without removing code, and lazy evaluation means expensive string formatting only happens if the message will be logged, 5) **Thread-safe**: Safe for concurrent applications. For example, you can set level to ERROR in production to only log serious issues, then switch to DEBUG in development without changing any code.',
    keyPoints: [
      'Levels allow runtime verbosity control',
      'Automatic formatting with timestamps and context',
      'Can output to multiple destinations',
      'Lazy evaluation and can be disabled',
      'Thread-safe for concurrent applications',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between the five logging levels. Give an example use case for each.',
    hint: 'Consider severity and what action each level implies.',
    sampleAnswer:
      '**DEBUG**: Detailed diagnostic information, used during development to trace program flow. Example: "Entering function calculate_score with args: user_id=123". **INFO**: Confirmation that things are working as expected. Example: "Server started on port 8080" or "Processing batch 5 of 20". **WARNING**: Something unexpected happened but the program continues. Example: "Deprecated API endpoint used" or "Cache miss, loading from database". **ERROR**: A serious problem caused a function to fail, but the application continues. Example: "Failed to send email to user@example.com: SMTP timeout". **CRITICAL**: A serious error that may cause the program to crash or corrupt data. Example: "Database connection lost, shutting down" or "Out of memory". In production, typically log INFO and above; in development, use DEBUG.',
    keyPoints: [
      'DEBUG: detailed diagnostics for development',
      'INFO: confirmation of normal operation',
      'WARNING: unexpected but not breaking',
      'ERROR: function failed but app continues',
      'CRITICAL: may cause crash or data corruption',
    ],
  },
  {
    id: 'q3',
    question:
      'What is lazy evaluation in logging and why is it important? Show examples of correct and incorrect usage.',
    hint: 'Think about when string formatting happens and performance implications.',
    sampleAnswer:
      'Lazy evaluation means the log message string is only built if it will actually be logged (based on level). This matters for performance when logging is disabled or filtered. **Correct**: logging.debug("User %s has %d items", username, len(items)) - the string is only formatted if DEBUG is enabled. **Incorrect**: logging.debug(f"User {username} has {len(items)} items") - f-string always evaluates, even if DEBUG is disabled. For expensive operations like database queries or large JSON serialization, this difference is huge. Example: logging.debug(f"Data: {json.dumps(huge_dict)}") always serializes even if not logged, but logging.debug("Data: %s", json.dumps(huge_dict)) doesn\'t. Use % or comma-separated args for lazy evaluation.',
    keyPoints: [
      'Message only built if it will be logged',
      'Critical for performance with expensive formatting',
      'Use % formatting or comma args, not f-strings',
      'Example: logging.debug("Value: %s", expensive_func())',
      'Especially important for disabled DEBUG messages',
    ],
  },
];
