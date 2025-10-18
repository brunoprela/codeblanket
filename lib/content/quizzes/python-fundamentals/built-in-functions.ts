/**
 * Quiz questions for Essential Built-in Functions section
 */

export const builtinfunctionsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between map()/filter() and list comprehensions. Which is more Pythonic?',
    sampleAnswer:
      'map() and filter() are functional programming tools that return iterators, while list comprehensions directly create lists and are considered more Pythonic. For example, map(lambda x: x**2, nums) vs [x**2 for x in nums], and filter(lambda x: x>0, nums) vs [x for x in nums if x>0]. List comprehensions are generally preferred in Python because they are more readable and slightly faster. However, map() and filter() are useful when you already have a function defined (like map(int, strings)) rather than using a lambda.',
    keyPoints: [
      'map/filter return iterators',
      'List comprehensions create lists directly',
      'Comprehensions more Pythonic and readable',
      'map/filter useful with existing functions',
      'Comprehensions often faster',
    ],
  },
  {
    id: 'q2',
    question: 'When would you use enumerate() vs range(len())?',
    sampleAnswer:
      "enumerate() is preferred over range(len()) because it's more Pythonic and less error-prone. Compare: for i in range(len(items)): print(i, items[i]) vs for i, item in enumerate(items): print(i, item). enumerate() directly gives you both index and value, avoiding indexing errors and making code clearer. It's also more efficient and works with any iterable, not just sequences with indexing. Use enumerate() when you need both index and value; use plain for item in items when you only need values.",
    keyPoints: [
      'enumerate() more Pythonic',
      'Gives index and value directly',
      'Avoids indexing errors',
      'Works with any iterable',
      'More readable than range(len())',
    ],
  },
  {
    id: 'q3',
    question: 'Explain why all([]) returns True but any([]) returns False.',
    sampleAnswer:
      "all([]) returns True because of vacuous truth in logic: a statement about all elements of an empty set is considered true since there are no counterexamples. Think: 'all numbers in [] are positive' - technically true because there are no numbers to disprove it. any([]) returns False because there's no element to make it True. This matters in code: if all(validations): can pass with empty validations[], but if any(errors): won't trigger with no errors. Be careful with empty sequences - sometimes you want to explicitly check if not items: first.",
    keyPoints: [
      'all([]): True - vacuous truth, no counterexamples',
      'any([]): False - no element to be True',
      'Based on logical quantifiers',
      'Can cause bugs if not expected',
      'Check for empty sequences explicitly when it matters',
    ],
  },
];
