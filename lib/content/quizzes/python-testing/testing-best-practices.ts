export const testingBestPracticesQuiz = [
  {
    id: 'tbp-q-1',
    question:
      'Team debate: "Should we test private methods?" One engineer says YES (more coverage, catch internal bugs). Another says NO (brittle tests, couples to implementation). Who is right and why? Provide concrete examples and recommend testing strategy.',
    sampleAnswer:
      'Do NOT test private methods directly: (1) Brittle: Tests break when refactoring internal implementation (even if behavior unchanged). Example: Refactor _calculate_tax() → _compute_taxes(), tests break despite same behavior. (2) Implementation coupling: Tests verify HOW code works, not WHAT it does. Changes require updating tests unnecessarily. (3) Better: Test public methods that USE private methods. Example: test calculate_total() which calls _calculate_tax() internally. If _calculate_tax() has bug, calculate_total() test catches it. (4) Exception: Complex private method with critical logic might warrant extraction to separate public utility. Example: _validate_credit_card() → extract to validators.validate_credit_card(), test that. (5) Strategy: Test public API (behavior), private methods get covered implicitly. If private method feels complex enough to test directly, extract to public utility. Result: Robust tests that survive refactoring, focus on user-visible behavior.',
    keyPoints: [
      'Brittle: Private method tests break on refactoring (even if behavior unchanged)',
      'Implementation coupling: Tests verify HOW not WHAT, require updates on changes',
      'Better: Test public methods that use private methods, catches bugs implicitly',
      'Exception: Extract complex private logic to public utility if needs direct testing',
      'Strategy: Test public API (behavior), private methods covered implicitly',
    ],
  },
];
