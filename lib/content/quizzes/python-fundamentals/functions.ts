/**
 * Quiz questions for Functions section
 */

export const functionsQuiz = [
  {
    id: 'pf-functions-q-1',
    question:
      'Explain the difference between *args and **kwargs. When would you use each, and can you use them together?',
    hint: 'Think about positional vs keyword arguments, and how they are unpacked.',
    sampleAnswer:
      '*args collects a variable number of positional arguments into a tuple, while **kwargs collects keyword arguments into a dictionary. Use *args when you want flexibility in the number of positional arguments (like print() or max()). Use **kwargs when you want to accept arbitrary named parameters (like for configuration options or flexible APIs). You can use them together - the order must be: regular args, *args, keyword-only args, **kwargs. For example: def func(a, b, *args, key=None, **kwargs). This allows maximum flexibility while maintaining clear function signatures.',
    keyPoints: [
      '*args: tuple of positional arguments',
      '**kwargs: dictionary of keyword arguments',
      'Can be combined in specific order',
      'Useful for flexible APIs and wrappers',
    ],
  },
  {
    id: 'pf-functions-q-2',
    question:
      'What is the difference between parameters and arguments? How do default parameters work, and what pitfall should you avoid?',
    hint: 'Think about function definition vs function call, and mutable default values.',
    sampleAnswer:
      'Parameters are the variables in the function definition; arguments are the actual values passed when calling the function. Default parameters provide fallback values if no argument is supplied. Critical pitfall: NEVER use mutable objects (like lists or dicts) as default parameters! Python evaluates defaults once at function definition, not at each call. So def func(items=[]): creates ONE shared list across all calls. If you append to it, all future calls see those changes. Use def func(items=None): followed by items = items or [] inside the function instead.',
    keyPoints: [
      'Parameters: in definition, Arguments: when calling',
      'Defaults evaluated once at definition time',
      'Never use mutable defaults (lists, dicts)',
      'Use None as default, then create new object inside',
    ],
  },
  {
    id: 'pf-functions-q-3',
    question:
      'When should you use a lambda function versus a regular function? What are the limitations of lambdas?',
    hint: 'Consider readability, debuggability, and complexity.',
    sampleAnswer:
      "Use lambdas for short, simple operations that are used once, typically as arguments to functions like map(), filter(), sorted(), or in list comprehensions. They're concise for simple transformations like `sorted(items, key=lambda x: x[1])`. However, lambdas are limited to single expressions (no statements), can't contain assignments, and are harder to debug (they don't have names in tracebacks). For anything more complex than a simple transformation, use a regular function with a descriptive name - readability trumps brevity. If you find yourself writing complex lambdas, define a regular function instead.",
    keyPoints: [
      'Lambdas: single expression, anonymous functions',
      'Best for simple, one-time transformations',
      'Cannot contain statements or assignments',
      'Regular functions are more readable and debuggable',
    ],
  },
];
