export default [
  {
    id: 'cp-m1-s14-q1',
    section: "Reading Other People's C++ Code",
    question:
      'When encountering heavily macro-ed competitive programming code, what is the best first step to understand it?',
    options: [
      'Try to run the code first',
      'Check the macro definitions at the top of the file',
      'Skip the macros and focus on the algorithm',
      'Rewrite it without macros',
    ],
    correctAnswer: 1,
    explanation:
      "Always check macro definitions first! Top coders use heavy macros (ll, pii, pb, all, rep, etc.). Understanding these is KEY to reading the code. Example: seeing `rep(i,n)` means nothing until you know it's `for(int i=0;i<n;i++)`. Strategy: (1) Read macros at top, (2) Mentally expand them while reading code, (3) Focus on algorithm logic. Keep a \"macro dictionary\" of common abbreviations: ll=long long, vi=vector<int>, pb=push_back, etc. Don't skip macros (won't understand code). Don't rewrite (waste of time, just need to read). Running without understanding doesn't help learn. Once you know common macros, reading becomes much faster.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s14-q2',
    section: "Reading Other People's C++ Code",
    question:
      "What is the recommended order for reading someone else's competitive programming solution?",
    options: [
      'Read line by line from top to bottom',
      'Start with main(), then solve(), then helper functions as needed',
      'Read helper functions first',
      'Start with variable declarations',
    ],
    correctAnswer: 1,
    explanation:
      'Reading order: (1) main() - understand structure (single test? multiple tests?), (2) solve() - core logic, (3) Helper functions when referenced, (4) Macros/constants when confused. This gives context before details. Example: main shows `while(t--) solve();` tells you multiple test cases. solve() shows overall approach. Then dive into helpers. Reading top-to-bottom wastes time on unused code. Starting with helpers lacks context. Starting with variables misses the big picture. Best practice: understand high-level structure first, then zoom into details. Like reading a book: start with plot summary, then read chapters, then paragraphs.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s14-q3',
    section: "Reading Other People's C++ Code",
    question:
      'How do you quickly identify what algorithm is being used in unfamiliar code?',
    options: [
      'Read every line carefully',
      'Look for characteristic patterns (DFS recursion, BFS queue, DP arrays, etc.)',
      'Run the code and observe behavior',
      'Check the problem statement',
    ],
    correctAnswer: 1,
    explanation:
      'Pattern recognition is key! Look for: DFS = `bool vis[];`, `void dfs(int u)`, recursive calls. BFS = `queue<int> q;`, `while(!q.empty())`. DP = `int dp[N][M];`, nested loops building from smaller. Dijkstra = `priority_queue`, `dist[]` array. Segment tree = `tree[4*N]`, `2*v` indexing. DSU = `parent[]`, `find()`, `union()`. Once you recognize the pattern, you understand the approach. Reading line-by-line is slow. Running code doesn\'t explain algorithm. Problem statement tells WHAT to solve, not HOW (that\'s the code). Practice: study implementations of common algorithms to recognize their "fingerprints". With experience, instant recognition.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s14-q4',
    section: "Reading Other People's C++ Code",
    question:
      'What should you do when encountering code you want to learn from?',
    options: [
      'Copy it directly to your template',
      'Read, understand, then implement yourself without looking',
      'Memorize it line by line',
      'Just bookmark it for later',
    ],
    correctAnswer: 1,
    explanation:
      "Best learning: (1) Read and understand the code, (2) Close it, (3) Implement yourself from memory/understanding, (4) Compare with original. This ensures you truly understand, not just copy. Copying without understanding leads to bugs and inability to modify. Memorization doesn't build understanding. Bookmarking without studying doesn't help. Example: see a cool segment tree implementation → understand how it works → code your own → compare. Your version might differ (that's OK!) but you'll understand both. When to copy directly: well-tested utility functions you fully understand (like modpow, gcd). When to implement yourself: algorithms you're learning.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s14-q5',
    section: "Reading Other People's C++ Code",
    question:
      'Why do different competitive programming platforms (Codeforces vs AtCoder vs TopCoder) have different coding styles?',
    options: [
      'Different programming languages required',
      'Different competitive cultures and community standards',
      'Different compiler versions',
      'Different problem types',
    ],
    correctAnswer: 1,
    explanation:
      "Platform cultures shape coding styles: Codeforces = heavy macros, very terse (Russian/Eastern European influence, speed-focused). AtCoder = moderate macros, cleaner (Japanese influence, educational focus). TopCoder = minimal macros, class-based (American corporate influence, platform enforces structure). Google Code Jam = structured, case numbering. ICPC = very readable (team needs to understand). All use C++ (or allow multiple languages). Compiler versions similar. Problem types don't dictate style. Understanding these cultures helps when reading solutions: CF solutions are terse but fast, AtCoder more readable, TopCoder most structured. Learn from all platforms for well-rounded skills.",
    difficulty: 'intermediate',
  },
] as const;
