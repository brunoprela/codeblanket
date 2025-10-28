export default [
  {
    id: 'cp-m1-s12-q1',
    section: 'Common Compilation Errors',
    question:
      'What is the most common cause of "expected \';\' before" compilation errors?',
    options: [
      'Forgetting to include necessary headers',
      'Missing semicolon on the previous line',
      'Using wrong variable types',
      'Incorrect function signatures',
    ],
    correctAnswer: 1,
    explanation:
      'The error "expected \';\' before X" usually means missing semicolon on the LINE BEFORE the error location. Example: `int x = 5` (missing semicolon), `return x;` (error reported here). Compiler sees `return` and expects semicolon first. Fix: add semicolon to previous line. This is one of the most common errors! Other causes: missing closing brace affects many lines, missing semicolon after struct/class definition. Always check the line BEFORE the error when you see this message. Auto-formatting can help catch these visually (indentation will look wrong). Quick fix: compile often to catch errors early, use editor with syntax highlighting.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s12-q2',
    section: 'Common Compilation Errors',
    question: 'What does "\'vector\' was not declared in this scope" mean?',
    options: [
      'Vector is not supported in C++',
      'Missing #include <vector> or using namespace std',
      'Vector variable name is reserved',
      'Vector is deprecated in modern C++',
    ],
    correctAnswer: 1,
    explanation:
      "This error means the compiler doesn't know what `vector` is. Causes: (1) Missing `#include <vector>`, (2) Missing `using namespace std;` (should use `std::vector`), (3) Typo in name. Fix: add `#include <vector>` at top, or use `#include <bits/stdc++.h>` (includes everything). If using namespace std is missing, either add it or use `std::vector<int>` explicitly. This applies to all STL types: map, set, queue, etc. In CP, `#include <bits/stdc++.h>` and `using namespace std;` prevent these errors. Similar error for other types: \"'string' was not declared\" = missing `#include <string>`.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s12-q3',
    section: 'Common Compilation Errors',
    question: 'How should you interpret template-heavy error messages in C++?',
    options: [
      'Read every line carefully from top to bottom',
      'Focus on the first error line, ignore template instantiation details',
      'Template errors always mean the template is wrong',
      'Skip the error and try different code',
    ],
    correctAnswer: 1,
    explanation:
      'Template errors can be 100+ lines long but most is noise. Strategy: (1) Read FIRST error line only - tells you what\'s wrong, (2) Ignore lines starting with "note:" or "in instantiation of" - these are template details, (3) Look for your code in error message - that\'s where the bug is. Example: "no match for operator< for type pair<int,int>" - first line tells you: need to define operator< for pair. Ignore the 50 lines of template instantiation below. Template errors don\'t mean template itself is wrong - usually means you\'re using it incorrectly (wrong type, missing operator, etc.). Clean up error by reading smart: first line + your code lines only.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s12-q4',
    section: 'Common Compilation Errors',
    question: 'What causes "undefined reference to X" linker errors?',
    options: [
      'Missing header file',
      'Function declared but not defined (implementation missing)',
      'Wrong variable type',
      'Syntax error in code',
    ],
    correctAnswer: 1,
    explanation:
      'Linker error "undefined reference to X" means function/variable is declared but definition (implementation) is missing. Example: `void solve();` declared but no `void solve() { /* body */ }` provided. This is a LINKER error (happens after compilation). Different from compiler errors. Common causes: (1) Forgot to implement declared function, (2) Typo in function name, (3) Missing library link (less common in CP). Fix: implement the function! Missing headers cause "not declared" errors (compiler, not linker). Syntax errors prevent compilation entirely (before linking). To distinguish: compilation errors = code syntax issues, linker errors = missing implementations.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s12-q5',
    section: 'Common Compilation Errors',
    question:
      'What is the best strategy when faced with multiple compilation errors?',
    options: [
      'Try to fix all errors simultaneously',
      'Fix the first error only, then recompile',
      'Skip errors and submit anyway',
      'Rewrite the entire solution',
    ],
    correctAnswer: 1,
    explanation:
      "Always fix the FIRST error only, then recompile. Why? First error often causes cascading errors. Example: missing semicolon causes 20 errors on subsequent lines, but fixing one semicolon fixes all. Strategy: (1) Read first error, (2) Fix it, (3) Recompile, (4) Repeat. Don't try to fix multiple errors at once - later errors might disappear after fixing first. Don't skip compilation (errors = code won't run). Don't rewrite unless absolutely necessary (debugging is faster). This systematic approach saves time. Exception: if you recognize first error as symptom of deeper issue (wrong algorithm), then might need larger fix. But usually, fix errors one at a time from top.",
    difficulty: 'beginner',
  },
] as const;
