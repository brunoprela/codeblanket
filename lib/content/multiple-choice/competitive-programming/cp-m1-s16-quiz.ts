export default [
  {
    id: 'cp-m1-s16-q1',
    section: 'Building a Robust CP Starter Template',
    question:
      'What are the essential components that every competitive programming template should have?',
    options: [
      'Only #include and main function',
      'Fast I/O, common type aliases, basic macros, and solve() function structure',
      'Complete implementations of all data structures',
      'GUI components and database connectors',
    ],
    correctAnswer: 1,
    explanation:
      "Essential template components: (1) Fast I/O: `ios_base::sync_with_stdio(false); cin.tie(nullptr);`, (2) Common type aliases: `#define ll long long`, `#define vi vector<int>`, (3) Basic macros: `#define all(x)`, `#define pb`, loop macros, (4) solve() function structure, (5) Multiple test case handling, (6) Debug macros with #ifdef LOCAL. Optional: common utilities (gcd, modpow). Don't include: complete data structure implementations (add as needed), GUI (irrelevant for CP), databases (irrelevant). Template should be: minimal but useful, personalized to YOUR style, tested and reliable. Start minimal, add what YOU use frequently over time.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s16-q2',
    section: 'Building a Robust CP Starter Template',
    question: 'Why should debug code be wrapped in #ifdef LOCAL blocks?',
    options: [
      'To make code compile faster',
      'To automatically disable debug output when submitting without code changes',
      'To prevent memory leaks',
      "It's required by judges",
    ],
    correctAnswer: 1,
    explanation:
      '`#ifdef LOCAL` automatically disables debug code on submission without manual changes. Benefits: (1) Write debug freely, (2) Submit without modifications, (3) Zero risk of debug output in submission (causes WA), (4) Clean workflow. Example: `#ifdef LOCAL\\n#define debug(x) cerr << #x << " = " << x << endl\\n#else\\n#define debug(x)\\n#endif`. Compile locally with -DLOCAL to enable debug. Submit without -DLOCAL and debug is automatically removed. Doesn\'t affect compile speed significantly. Doesn\'t prevent memory leaks. Not required by judges (but prevents accidental debug output). This pattern is standard practice - set up once, benefit forever. Alternative (manual commenting) is error-prone.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s16-q3',
    section: 'Building a Robust CP Starter Template',
    question:
      'How should you organize multiple template variants (basic, graph, geometry, etc.)?',
    options: [
      'Keep everything in one massive template file',
      'Have separate template files for different problem types',
      "Don't use templates at all",
      'Memorize all code instead',
    ],
    correctAnswer: 1,
    explanation:
      "Organize specialized templates: ~/cp/templates/basic.cpp (default), graph.cpp (DFS/BFS/Dijkstra), geometry.cpp (point/line operations), string.cpp (KMP/Z-algorithm), number_theory.cpp (primes/modular arithmetic). Benefits: (1) Start with appropriate template for problem type, (2) Don't clutter basic template with unused code, (3) Faster compilation, (4) Easier to maintain. Quick copy script: `cptemplate graph A.cpp`. One massive template is slow to compile and hard to navigate. No templates wastes time rewriting. Memorization is unrealistic for complex implementations. Best practice: minimal basic template + specialized templates for specific domains. Choose based on problem requirements.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s16-q4',
    section: 'Building a Robust CP Starter Template',
    question:
      'What is the best way to evolve and improve your template over time?',
    options: [
      'Never change it once created',
      'Copy random code from the internet constantly',
      'Review after each contest, test new features, use version control',
      'Rewrite it completely every month',
    ],
    correctAnswer: 2,
    explanation:
      'Template evolution process: (1) Use in contests, (2) Note what\'s missing or buggy, (3) Test improvements locally, (4) Add to template, (5) Document changes, (6) Use version control (Git). Post-contest review: "What did I reimplement?", "What template feature had bugs?", "What was missing?". Test new features thoroughly before adding (create test file, verify correctness). Use Git to track changes and versions. Don\'t never change (template should grow with skills). Don\'t copy untested code (leads to bugs). Don\'t constantly rewrite (waste of time). Gradual, tested improvements are best. Template should be YOUR template, refined over time.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s16-q5',
    section: 'Building a Robust CP Starter Template',
    question:
      'What should you do if you encounter a bug in your template during a contest?',
    options: [
      'Fix it quickly during the contest',
      'Abandon the template and code from scratch',
      'Work around it for now, fix properly after contest',
      'Panic and give up',
    ],
    correctAnswer: 2,
    explanation:
      "Template bug during contest: (1) Quick workaround for current problem, (2) Note the bug, (3) Fix properly after contest with thorough testing. Don't spend 20 minutes debugging template mid-contest (time pressure). Don't abandon template (most of it works). Don't panic (bugs happen). Example: if fast I/O breaks on interactive problem, just remove it for that problem. After contest: test fix thoroughly, update template, document. Prevention: thoroughly test template components before contests (create test suite). Use version control to revert if needed. Template should be reliable but not perfect - minor bugs are fixable. Main point: during contest focus on solving problems, after contest improve infrastructure.",
    difficulty: 'intermediate',
  },
] as const;
