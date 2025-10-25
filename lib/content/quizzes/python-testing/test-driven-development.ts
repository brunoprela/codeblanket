export const testDrivenDevelopmentQuiz = [
  {
    id: 'tdd-q-1',
    question:
      'New team member asks: "Why write tests before code? Isn\'t that slower?" Explain TDD benefits with concrete examples showing how test-first actually saves time and improves quality compared to test-after.',
    sampleAnswer:
      'TDD benefits vs test-after: (1) Faster debugging: Bug in 500-line feature? TDD: Last 10 lines (just wrote). Test-after: Entire feature (debugging hours). Example: TDD catches bug in 2 min, test-after: 2 hours debugging. (2) Better design: TDD forces simple interfaces. Test-first: "How do I test this?" → simple API. Test-after: Complex code → hard to test → skip tests. Example: TDD produces testable add (a, b), test-after produces complex Calculator with 5 dependencies. (3) No rework: TDD writes code once (guided by tests). Test-after writes code, realizes untestable, refactors (2× work). (4) Confidence: TDD: Refactor safely (tests pass = safe). Test-after: Fear to change (might break). (5) Real speed: TDD upfront 20% slower, saves 80% on debugging/rework. Example: Feature takes 8h TDD vs 10h test-after (2h debugging, 3h rework). TDD: Invest 20 min tests, save 2h later.',
    keyPoints: [
      'Faster debugging: TDD catches bugs in last 10 lines (2 min) vs entire feature (2h)',
      'Better design: Tests force simple interfaces, test-after produces complex, hard-to-test code',
      'No rework: TDD writes once, test-after writes → refactors untestable code (2× work)',
      'Confidence: TDD enables safe refactoring, test-after creates fear to change',
      'Real speed: TDD 8h vs test-after 10h (includes debugging/rework), invest 20 min saves 2h',
    ],
  },
];
