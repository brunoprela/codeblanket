import { MultipleChoiceQuestion } from '@/lib/types';

export const propertyBasedTestingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pbt-mc-1',
    question:
      'What is the main difference between example-based and property-based testing?',
    options: [
      'Property-based tests specific examples, example-based tests properties',
      'Property-based tests properties for many random inputs, example-based tests specific cases',
      'Property-based tests are faster than example-based',
      'Property-based tests replace all example-based tests',
    ],
    correctAnswer: 1,
    explanation:
      "Property-based tests properties for many inputs: Example-based: assert add(2, 3) == 5 (specific). Property-based: @given(st.integers(), st.integers()) assert add(a, b) == add(b, a) (all integers). Hypothesis generates 100 random test cases. Benefits: Finds edge cases (MAX_INT, negative, zero) you wouldn't think of. Not specific examples (opposite), not faster (runs more tests), doesn't replace (complements example-based). Use both: Property-based finds edge cases, example-based documents expected behavior.",
  },
  {
    id: 'pbt-mc-2',
    question: 'What does Hypothesis do when a property test fails?',
    options: [
      'Stops immediately and reports the failure',
      'Shrinks the failing input to the simplest example that still fails',
      'Generates more test cases to confirm the failure',
      'Automatically fixes the code',
    ],
    correctAnswer: 1,
    explanation:
      'Hypothesis shrinks to minimal failing example: Test with [1000, -500, 0, 42] fails → shrinks to [1, 0] (simplest). Process: Find failing input → try smaller versions → report minimal case. Example: @given(st.lists(st.integers())) finds bug in sort([3, 1, 2]) → shrinks to [1, 0]. Why useful: Easier to debug minimal case than random large input. Not just reports (shrinks first), not generates more (shrinks instead), not auto-fix (reports bug). Essential for diagnosing complex failures.',
  },
  {
    id: 'pbt-mc-3',
    question: 'What is st.integers(min_value=0, max_value=100) used for?',
    options: [
      'Tests exactly 100 integer values',
      'Generates random integers between 0 and 100 inclusive',
      'Sets the number of test examples to 100',
      'Creates a list of 100 integers',
    ],
    correctAnswer: 1,
    explanation:
      "st.integers(min_value=0, max_value=100) generates random int 0-100: @given(st.integers(min_value=0, max_value=100)) → Hypothesis picks random int in range for each test run. Example: Generates 42, 0, 100, 17, 3 across different test runs. Not exactly 100 values (random selection), not test count (that's max_examples=100), not list (that's st.lists()). Use for: Age validation, percentage checks, bounded inputs. Essential Hypothesis strategy for constrained integers.",
  },
  {
    id: 'pbt-mc-4',
    question: 'Which property is tested by: assert reverse(reverse(s)) == s?',
    options: [
      'Commutativity',
      'Associativity',
      'Idempotence (involution)',
      'Identity',
    ],
    correctAnswer: 2,
    explanation:
      'Idempotence/involution: Applying function twice returns to original. reverse(reverse(s)) == s tests reversing twice is identity. Other properties: Commutativity: f(a, b) == f(b, a) (e.g., add). Associativity: f(f(a, b), c) == f(a, f(b, c)) (e.g., +). Identity: f(a, identity) == a (e.g., 0 for +). Idempotence: f(f(a)) == f(a) (e.g., abs(abs(x)) == abs(x)). Similar examples: encrypt(decrypt(x)) == x, negate(negate(x)) == x.',
  },
  {
    id: 'pbt-mc-5',
    question: 'When is property-based testing most valuable?',
    options: [
      'For testing user interfaces and visual components',
      'For testing algorithms, parsers, and data transformations',
      'For testing database queries',
      'For testing third-party API integrations',
    ],
    correctAnswer: 1,
    explanation:
      "Property-based testing best for algorithms/data: Algorithms: Sorting (is sorted, same elements), search (found item is correct). Parsers: Round-trip (parse(build(x)) == x), valid output. Transformations: Reversibility, data preservation. Example: JSON parser: assert json.loads(json.dumps(obj)) == obj for all valid objects. Not ideal for: UIs (visual, not properties), databases (stateful, specific queries), API integrations (external, can't control inputs). Property-based excels at pure functions with clear properties.",
  },
];
