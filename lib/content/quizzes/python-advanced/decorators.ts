/**
 * Quiz questions for Decorators & Function Wrapping section
 */

export const decoratorsQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through how @lru_cache improves performance. What trade-offs does it make?',
    sampleAnswer:
      '@lru_cache memoizes function results in a dictionary, keyed by the function arguments. When the function is called again with the same arguments, it returns the cached result instead of recomputing. This trades memory for speed. For recursive functions like fibonacci, it turns O(2^n) into O(n) by eliminating redundant calculations. The trade-off is memory usage—the cache stores up to maxsize results. It only works for functions with hashable arguments and can consume lots of memory if results are large or if there are many unique argument combinations.',
    keyPoints: [
      'Stores results in a dictionary cache',
      'Returns cached result for same arguments',
      'Trades memory for speed',
      'Example: fibonacci O(2^n) to O(n)',
      'Requires hashable arguments',
    ],
  },
  {
    id: 'q2',
    question:
      'Why do we need functools.wraps when creating decorators? What problem does it solve?',
    sampleAnswer:
      'Without functools.wraps, the decorated function loses its original metadata like __name__, __doc__, and __module__. This breaks introspection and makes debugging harder. For example, if I decorate my_function, its __name__ would become "wrapper" instead of "my_function", and help (my_function) would show the wrapper docs, not the original docs. functools.wraps copies the metadata from the original function to the wrapper, preserving the function identity. This is critical for debugging, documentation generation, and any tools that rely on function introspection.',
    keyPoints: [
      'Preserves original function metadata',
      '__name__, __doc__, __module__ preserved',
      'Without it, all functions named "wrapper"',
      'Critical for debugging and introspection',
      'Copies metadata from original to wrapper',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how decorator chaining works and why order matters. What happens when you stack multiple decorators?',
    sampleAnswer:
      'When you stack decorators like @decorator1 @decorator2 @decorator3 def func(), they execute from bottom to top during decoration, meaning func is first wrapped by decorator3, that result is wrapped by decorator2, and finally by decorator1. However, at runtime, they execute top to bottom - decorator1 runs first, then decorator2, then decorator3, and finally func. Order matters because each decorator modifies what the next one sees. For example, if you have @auth @cache, auth runs first (good - no caching unauthorized requests). But @cache @auth would cache before auth checking (bad - security risk). Always consider the logical flow.',
    keyPoints: [
      'Decoration: bottom to top (innermost first)',
      'Execution: top to bottom (outermost first)',
      'Each decorator wraps the previous result',
      'Order affects behavior and can cause bugs',
      'Example: @auth @cache vs @cache @auth',
    ],
  },
];
