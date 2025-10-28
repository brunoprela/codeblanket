export default [
  {
    id: 'cp-m1-s15-q1',
    section: 'Contest-Day C++ Tips',
    question:
      'What is the recommended time budget for a typical 20-minute competitive programming problem?',
    options: [
      'Spend all 20 minutes coding',
      'Read 3 min, Plan 5 min, Code 10 min, Test 2 min',
      'Code immediately, test at the end',
      'Spend 15 minutes planning before coding',
    ],
    correctAnswer: 1,
    explanation:
      'Balanced time budget for 20-min problem: Reading 3 min (understand problem, constraints, samples), Planning 5 min (algorithm, complexity check), Coding 10 min (implementation), Testing 2 min (samples + edge cases). This prevents: rushing into wrong algorithm, missing edge cases, submitting untested code. Adjust for problem: easier problem = less planning, harder = more planning. Contest strategy: solve problems you CAN solve correctly, not just quickly. One AC submission > three WA submissions. Time saved testing locally > time lost on WA penalties. Common mistake: coding immediately without planning (leads to wrong approaches). Other extreme: over-planning delays submission.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s15-q2',
    section: 'Contest-Day C++ Tips',
    question:
      'What is the most important thing to check before submitting a solution?',
    options: [
      'Code formatting and comments',
      'That it passes all sample inputs',
      'Variable naming conventions',
      'Code length and elegance',
    ],
    correctAnswer: 1,
    explanation:
      "ALWAYS test on ALL sample inputs before submitting! Samples are free tests - use them. If failing samples, definitely won't pass judge tests. Also test: edge cases (n=1, n=max), special values. Pre-submission checklist: ✓ All samples pass, ✓ Edge cases tested, ✓ Used long long where needed, ✓ Output format correct, ✓ Cleared between test cases. Code formatting, comments, naming conventions don't affect correctness (judges don't care). Code length doesn't matter. Focus on CORRECTNESS. One bug-free submission > three buggy submissions. Test thoroughly locally to avoid WA penalties. Budget 2-3 minutes for testing - time well spent.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s15-q3',
    section: 'Contest-Day C++ Tips',
    question:
      'When should you skip a problem and move to the next one during a contest?',
    options: [
      'After 5 minutes if not solved',
      'After 15-20 minutes with no progress',
      'Never skip, always finish current problem',
      'Skip immediately if problem looks hard',
    ],
    correctAnswer: 1,
    explanation:
      "Skip after 15-20 minutes with NO progress. Signs to skip: (1) No algorithm idea after thinking, (2) Implementation seems very complex, (3) Tried 2 approaches, both wrong, (4) Other easier problems available. Don't skip at 5 min (too early). Don't never skip (wastes time on impossible problems). Don't skip immediately on looks (might be easier than it seems). Strategy: solve what you CAN solve. Better: 3 easy problems solved than 1 hard problem attempted unsuccessfully. Can return to skipped problem with fresh perspective. Contest is about maximizing points, not completing in order. Flexibility is key. Mark skipped problems mentally to return if time remains.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s15-q4',
    section: 'Contest-Day C++ Tips',
    question:
      'What is the most common cause of Wrong Answer after passing all samples?',
    options: [
      'Judge has different test cases',
      'Integer overflow, array out of bounds, or not clearing between test cases',
      'Compiler differences',
      'Random number generation',
    ],
    correctAnswer: 1,
    explanation:
      "Common WA causes: (1) Integer overflow - use long long for sums/products, (2) Array out of bounds - check loop bounds, (3) Not clearing global arrays between test cases, (4) Wrong output format, (5) Edge cases not tested (n=1, n=max). Judge DOES have different tests (that's the point). Compiler differences are rare (standard C++). Random numbers rarely used in CP. Prevention: Always check for overflow (multiply two large ints?), verify loop bounds (< n not <= n?), clear globals in solve(), test edge cases locally. Post-WA debugging: re-check these common issues systematically. Often simple fixes. Don't immediately assume algorithm is wrong - usually implementation bug.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s15-q5',
    section: 'Contest-Day C++ Tips',
    question:
      'How should you handle the psychological pressure of seeing others solve problems faster?',
    options: [
      'Panic and rush through your current problem',
      "Focus on your own progress, ignore others' rankings",
      "Give up if you're behind",
      "Copy others' approaches",
    ],
    correctAnswer: 1,
    explanation:
      "Mental game: Focus on YOUR progress, not others. Everyone has different backgrounds, experience, problem-solving styles. Seeing others ahead doesn't mean you're bad - might be solving different problems, or you'll solve harder problems they can't. Don't panic/rush (leads to bugs). Don't give up (contest isn't over!). Don't copy (cheating, and you won't learn). Strategy: compete with yourself, not others. Each contest is practice. Solve what you CAN solve correctly. One AC is progress. Bad contest? Learn and improve for next one. Pressure is normal - experienced competitors feel it too. Deep breath, focus, solve methodically. Rating comes from consistent improvement, not single contests.",
    difficulty: 'beginner',
  },
] as const;
