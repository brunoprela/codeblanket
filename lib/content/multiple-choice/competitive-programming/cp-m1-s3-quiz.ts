export default [
  {
    id: 'cp-m1-s3-q1',
    section: 'Environment Setup & Compilation',
    question:
      'Which compiler flag combination is standard for competitive programming to ensure optimization and C++17 features?',
    options: [
      'g++ -std=c++11 solution.cpp',
      'g++ -std=c++17 -O2 solution.cpp',
      'g++ -g -Wall solution.cpp',
      'g++ -std=c++20 -O3 -march=native solution.cpp',
    ],
    correctAnswer: 1,
    explanation:
      'The standard CP compilation is `g++ -std=c++17 -O2 solution.cpp`. `-std=c++17` enables C++17 features (structured bindings, fold expressions, etc.) which most judges support. `-O2` enables optimizations that can make code 2-5x faster. C++11 is outdated. `-g` is for debugging (adds overhead). `-O3` and `-march=native` are more aggressive but risky (may cause subtle bugs or not match judge environment). Most competitive programming judges use -O2 as standard. Additional useful flags for local testing: `-Wall -Wextra` (warnings) and `-DLOCAL` (for debug macros).',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s3-q2',
    section: 'Environment Setup & Compilation',
    question:
      'What is the purpose of #include <bits/stdc++.h> in competitive programming?',
    options: [
      'It includes only input/output libraries',
      'It includes all standard C++ libraries in one line',
      'It makes code run faster by precompiling headers',
      "It's required by competitive programming judges",
    ],
    correctAnswer: 1,
    explanation:
      "`#include <bits/stdc++.h>` is a GCC extension that includes all standard C++ libraries (iostream, vector, algorithm, map, etc.) in one line. This saves time in contests - no need to remember specific headers. However, it's non-standard (won't work with all compilers) and slower to compile. In CP, we prioritize speed of writing over compile time. It's NOT required by judges (you can include specific headers), doesn't make code run faster (just compilation convenience), and includes much more than just I/O. Example: instead of `#include <iostream>`, `#include <vector>`, `#include <algorithm>`, etc., just one line includes everything.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s3-q3',
    section: 'Environment Setup & Compilation',
    question: 'When should you use the -Wall and -Wextra compiler flags?',
    options: [
      'Only when submitting to the judge',
      'Never, they slow down compilation',
      'During local development to catch potential bugs',
      'Only for C++20 code',
    ],
    correctAnswer: 2,
    explanation:
      "`-Wall` and `-Wextra` enable compiler warnings that catch common bugs during development: uninitialized variables, unused variables, signed/unsigned comparisons, etc. These warnings help identify issues BEFORE submission. Example: `-Wall` catches `int x; cout << x;` (uninitialized). However, DON'T use them for judge submission (warnings aren't errors, and some judges might reject). Local development command: `g++ -std=c++17 -O2 -Wall -Wextra -DLOCAL solution.cpp`. Submission command: `g++ -std=c++17 -O2 solution.cpp`. Warnings don't slow down execution, only add to compilation output.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s3-q4',
    section: 'Environment Setup & Compilation',
    question:
      'What is the recommended way to set up file I/O for local testing without affecting judge submission?',
    options: [
      'Always use freopen() in your code',
      'Use #ifdef LOCAL with freopen() for conditional file I/O',
      'Manually comment/uncomment freopen() before submission',
      'Use command line redirection only',
    ],
    correctAnswer: 1,
    explanation:
      'The best practice is using `#ifdef LOCAL` with `freopen()` for conditional file I/O. Code: `#ifdef LOCAL\\nfreopen("input.txt", "r", stdin);\\n#endif`. Compile locally with `-DLOCAL` to enable file I/O. Submit without `-DLOCAL` and the freopen is automatically disabled. This avoids manual commenting (error-prone) while allowing convenient local testing. Command line redirection works (`./a.out < input.txt`) but file I/O is more convenient for iterative testing. Example full setup: `#ifdef LOCAL\\nfreopen("input.txt", "r", stdin);\\nfreopen("output.txt", "w", stdout);\\n#endif`. Clean and automatic!',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s3-q5',
    section: 'Environment Setup & Compilation',
    question: 'Which IDE feature is most valuable for competitive programming?',
    options: [
      'Advanced GUI designers',
      'Built-in compile-and-run with keyboard shortcuts',
      'Integrated database tools',
      'UML diagram generators',
    ],
    correctAnswer: 1,
    explanation:
      'Built-in compile-and-run with keyboard shortcuts (e.g., Ctrl+Shift+B in VS Code) is crucial for fast iteration. One keystroke to compile and test saves significant time over manual terminal commands. Good CP IDE setup includes: (1) Quick compile+run hotkey, (2) Syntax highlighting, (3) Auto-completion, (4) Multiple test files. GUI designers, database tools, and UML generators are irrelevant for CP. Popular choices: VS Code with custom tasks, CLion, Sublime Text with build systems, or even Vim with custom mappings. Key is minimizing friction between "write code" and "see results". Time saved: 5-10 seconds per test × 50 tests = 4-8 minutes saved per problem.',
    difficulty: 'beginner',
  },
] as const;
