/**
 * Quiz questions for Property Decorators Deep-Dive section
 */

export const propertiesQuiz = [
  {
    id: 'q1',
    question:
      'When should you use a property versus a regular method? What are the design guidelines?',
    hint: 'Consider parameters, speed, side effects, and how the operation feels.',
    sampleAnswer:
      'Use properties when the operation feels like attribute access: no parameters needed, fast execution (< 0.1s), and minimal side effects beyond validation. Properties should act like attributes—getting a value should be cheap and idempotent. Use methods when: 1) the operation is expensive (database query, file I/O), 2) parameters are needed, 3) significant side effects occur (sending email, modifying external state), 4) the operation might fail frequently. For example, user.age is a property (fast, no parameters), but user.send_email (subject, body) must be a method (side effects, parameters). Think: "Would I be surprised if this had a getter/setter?" If yes, use a method.',
    keyPoints: [
      'Property: fast, no parameters, minimal side effects',
      'Method: expensive, needs parameters, side effects',
      'Properties feel like attribute access',
      'Methods feel like actions',
      'Example: user.age (property) vs user.calculate_taxes() (method)',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain lazy evaluation using properties. Why is this useful and what are the trade-offs?',
    hint: 'Consider when expensive operations happen and memory vs computation trade-offs.',
    sampleAnswer:
      "Lazy evaluation with properties defers expensive computation until the value is first accessed. The pattern: check if cached value exists, compute and cache if not, return cached value. This is useful when: 1) not all instances need the value, 2) the computation is expensive, 3) you want fast initialization. Trade-offs: saves computation if never accessed, but first access is slower. Use cached_property (Python 3.8+) for automatic caching. For example, loading a large dataset: don't load in __init__ (slow startup), load on first access (lazy). This is especially useful for optional features or computed statistics that might not be needed in every code path.",
    keyPoints: [
      'Defers expensive computation until first access',
      'Useful when value might not be needed',
      'Trade-off: fast init, slower first access',
      'Use cached_property for automatic caching',
      'Example: lazy loading large datasets',
    ],
  },
  {
    id: 'q3',
    question:
      'Why use properties for validation instead of validating in __init__? What advantage does this provide?',
    hint: 'Think about when validation happens and maintaining invariants over time.',
    sampleAnswer:
      "Properties provide continuous validation—they validate not just during initialization but every time the attribute is modified. This maintains class invariants throughout the object's lifetime. Without properties, you can set invalid values after creation: person._age = -5 bypasses validation. With properties, every assignment goes through the setter: person.age = -5 raises ValueError, whether in __init__ or later. This also provides a single source of truth for validation logic—one setter handles all assignments, not scattered validation. Additionally, properties allow adding validation to existing code without breaking the interface—change direct attribute to property without callers knowing.",
    keyPoints: [
      'Properties validate on every assignment, not just __init__',
      'Maintains invariants throughout lifetime',
      'Single source of truth for validation',
      'Can add validation without breaking interface',
      'Example: person.age = -5 always validated',
    ],
  },
];
