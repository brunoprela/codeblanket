export default [
  {
    id: 'cp-m1-s4-q1',
    section: 'Modern CP Tool Ecosystem',
    question:
      'What is the primary benefit of using a competitive programming helper tool like Competitive Companion browser extension?',
    options: [
      'It automatically solves problems for you',
      'It parses sample test cases from problem pages and sends them to your editor',
      'It submits solutions directly without manual copy-paste',
      'It provides hints for unsolved problems',
    ],
    correctAnswer: 1,
    explanation:
      "Competitive Companion parses sample test cases from problem pages (Codeforces, AtCoder, etc.) and automatically sends them to your local environment. This eliminates manual copy-paste of inputs/outputs, saving 1-2 minutes per problem. It works with tools like cp-editor, Hightail, or custom scripts. You click the extension, and test files are instantly created. It does NOT solve problems (that's your job!), submit solutions (manual or via cf-tool), or provide hints. Setup: Install browser extension → Configure to work with your local tool → Click extension on problem page → Test cases automatically loaded. Time saved: 1-2 min × 5 problems = 5-10 minutes per contest.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s4-q2',
    section: 'Modern CP Tool Ecosystem',
    question: 'What is cf-tool and when is it useful?',
    options: [
      'A compiler specifically optimized for competitive programming',
      'A command-line tool for interacting with Codeforces (submitting, testing, etc.)',
      'A debugging tool for finding runtime errors',
      'An IDE for competitive programming',
    ],
    correctAnswer: 1,
    explanation:
      "cf-tool is a command-line tool for Codeforces that automates common tasks: parsing problems, running tests, and submitting solutions. Commands: `cf race` (start virtual contest), `cf test` (test on samples), `cf submit` (submit solution). This saves time over manual browser submission. Example workflow: `cf race 1234` → parse all problems → write solution → `cf test` → `cf submit`. It's NOT a compiler (uses your system's g++), not a debugger (use gdb/print statements), not an IDE (use with your existing editor). Other platforms have similar tools: atcoder-tools for AtCoder. Benefits: faster workflow, less context switching.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s4-q3',
    section: 'Modern CP Tool Ecosystem',
    question:
      'What is the main advantage of using stress testing tools in competitive programming?',
    options: [
      'They help you write code faster',
      'They automatically generate random test cases to find bugs',
      'They provide hints for solving problems',
      'They make your code run faster',
    ],
    correctAnswer: 1,
    explanation:
      'Stress testing tools generate random test cases and compare your solution against a brute-force "correct" solution to find bugs. This catches edge cases you might miss. Workflow: (1) Write optimized solution, (2) Write simple brute-force solution, (3) Generate random inputs, (4) Compare outputs. When outputs differ, you found a bug! Example script: `while true; do gen > input; ./solution < input > out1; ./brute < input > out2; diff out1 out2 || break; done`. This finds the first failing case. Stress testing doesn\'t write code for you, provide hints, or make code faster - it helps you find and fix bugs before submission, saving Wrong Answer penalties.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s4-q4',
    section: 'Modern CP Tool Ecosystem',
    question:
      'Which tool is most useful for visualizing algorithm execution during debugging?',
    options: [
      'Valgrind for memory leaks',
      'GDB for step-by-step execution',
      'Custom visualization scripts or tools like VisuAlgo',
      'Compiler optimization flags',
    ],
    correctAnswer: 2,
    explanation:
      "Custom visualization scripts or web tools like VisuAlgo help visualize algorithm execution (graph traversals, sorting, tree operations). For debugging, adding print statements to output intermediate states, then visualizing with scripts is most practical in CP. Example: for graph problems, output edges in DOT format and render with Graphviz. GDB is powerful but slow for CP (stepping through code takes time). Valgrind is for memory debugging (less common in CP). Compiler flags don't visualize anything. Practical approach: `#ifdef LOCAL` blocks with visualization output. Tools: VisuAlgo (online visualizations), Graphviz (graph visualization), custom Python scripts for array/tree visualization.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s4-q5',
    section: 'Modern CP Tool Ecosystem',
    question:
      'What is the benefit of using a code snippet manager or template system?',
    options: [
      'It automatically optimizes your code',
      'It stores reusable code blocks (templates, algorithms) for quick access',
      'It submits code to multiple judges simultaneously',
      'It translates code between programming languages',
    ],
    correctAnswer: 1,
    explanation:
      "A snippet manager stores reusable code blocks for instant access: template file, segment tree implementation, Dijkstra's algorithm, etc. This saves rewriting common code. VS Code has built-in snippets, Sublime Text has them, or use external tools. Example: type `segtre` → auto-expands to full segment tree implementation. Create snippets for: (1) Template header, (2) Common algorithms (DFS, BFS, binary search), (3) Data structures (DSU, fenwick tree), (4) Math functions (modpow, gcd). Time saved: 2-5 minutes per problem implementing common structures. It doesn't optimize code, submit anywhere, or translate languages - purely for quick code insertion. Setup once, benefit forever.",
    difficulty: 'beginner',
  },
] as const;
