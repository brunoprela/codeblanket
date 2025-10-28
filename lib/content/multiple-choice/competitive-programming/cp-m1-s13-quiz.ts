export default [
  {
    id: 'cp-m1-s13-q1',
    section: 'Debugging in Competitive Environment',
    question:
      'What is the most effective debugging technique when you have limited time in a contest?',
    options: [
      'Using a debugger to step through code line by line',
      'Adding strategic print/cerr statements to trace execution',
      'Rewriting the entire solution from scratch',
      'Asking for help online',
    ],
    correctAnswer: 1,
    explanation:
      'Print statements (cerr) are fastest for debugging in contests. Strategy: (1) Print intermediate values, (2) Add checkpoints to find WHERE bug occurs, (3) Print variables before/after operations. Example: `cerr << "i = " << i << ", sum = " << sum << endl;`. Use cerr (goes to stderr) not cout (mixes with solution output). Remove before submission or use `#ifdef LOCAL` to auto-disable. Debuggers are powerful but slow - no time to set breakpoints and step through in a contest. Rewriting wastes time. Online help not available/allowed. Best practice: develop debugging style using print statements, become fast at isolating bugs. Budget: 2-5 minutes for debugging, not 20 minutes.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s13-q2',
    section: 'Debugging in Competitive Environment',
    question: 'When should you use the #ifdef LOCAL pattern for debug code?',
    options: [
      'Only for file I/O',
      'For all debug print statements and test-only code',
      'Never, comments are sufficient',
      'Only for advanced debugging techniques',
    ],
    correctAnswer: 1,
    explanation:
      '`#ifdef LOCAL` should wrap ALL debug code that shouldn\'t be in submission. Benefits: (1) Can add debug freely without risk, (2) Submit without modifying code, (3) Clean separation of debug vs production. Example: `#ifdef LOCAL\\n#define debug(x) cerr << #x << " = " << x << endl\\n#else\\n#define debug(x)\\n#endif`. Now `debug(x)` works locally (with -DLOCAL) but becomes no-op in submission. Use for: debug prints, file I/O, assertions, test-only code. Comments require manual toggle (error-prone). This pattern is standard practice in CP. Setup once in template, benefit forever. Eliminates accidental debug output in submissions (causes WA).',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s13-q3',
    section: 'Debugging in Competitive Environment',
    question: 'What is the "checkpoint" debugging technique?',
    options: [
      'Saving code versions periodically',
      'Adding print statements at key locations to find where code breaks',
      'Using version control during contests',
      'Creating backup copies of solutions',
    ],
    correctAnswer: 1,
    explanation:
      'Checkpoint debugging: add prints at key locations to binary-search for bugs. Example: `cerr << "Checkpoint 1" << endl;` after input reading, `cerr << "Checkpoint 2" << endl;` after processing, `cerr << "Checkpoint 3" << endl;` before output. If checkpoint 2 doesn\'t print, bug is in processing section. Then add more checkpoints in that section to narrow down. This quickly isolates WHERE the bug is (doesn\'t tell you WHAT, but narrows search). Alternative: print variable values at checkpoints. Fast and effective for finding crash location or infinite loops. Not about version control or backups. Combine with binary search approach: test middle of code, determine which half has bug, repeat.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s13-q4',
    section: 'Debugging in Competitive Environment',
    question: 'What is stress testing and when should you use it?',
    options: [
      'Testing under heavy server load',
      'Testing how long your program runs',
      'Generating random inputs and comparing your solution against a brute-force solution',
      'Testing with maximum memory usage',
    ],
    correctAnswer: 2,
    explanation:
      'Stress testing: generate random inputs, run both your optimized solution and a simple brute-force solution, compare outputs. When they differ, you found a failing test case! Use when: (1) Pass samples but get WA on judge, (2) Suspect edge case bugs, (3) Have time to be thorough. Script: `while true; do ./gen > input; ./solution < input > out1; ./brute < input > out2; diff out1 out2 || break; done`. Stops on first mismatch. Requirements: (1) Input generator, (2) Your solution, (3) Correct brute-force. Very powerful for finding bugs. Time investment: 10-15 min setup, but catches subtle bugs. Not about performance testing or load testing.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s13-q5',
    section: 'Debugging in Competitive Environment',
    question: 'What is the best way to debug "Wrong Answer" (WA) verdicts?',
    options: [
      'Resubmit multiple times hoping it passes',
      'Test on samples, then create and test edge cases',
      'Change algorithm completely',
      'Ignore it and move to next problem',
    ],
    correctAnswer: 1,
    explanation:
      "WA debugging process: (1) Re-test on ALL samples (did you break something?), (2) Test edge cases: n=1, n=max, all same values, sorted, reverse sorted, (3) Check for: integer overflow, array out of bounds, wrong output format, not clearing between test cases. Create minimal test cases to reproduce. Use stress testing if still stuck. DON'T resubmit without fixing (penalties!). DON'T change algorithm immediately (usually implementation bug, not algorithm). DON'T give up (might be simple fix). Systematic debugging saves time. Budget: 5-10 minutes debugging before considering different approach. Common bugs: ll vs int, <= vs <, not clearing global arrays.",
    difficulty: 'beginner',
  },
] as const;
