import { MultipleChoiceQuestion } from '@/lib/types';

export const precommitHooksCICDMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pch-mc-1',
    question: 'What is the main purpose of pre-commit hooks?',
    options: [
      'Run comprehensive test suite before every commit',
      'Catch common issues locally before code reaches CI/CD',
      'Replace CI/CD pipelines entirely',
      'Automatically fix all code issues',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-commit hooks catch issues locally before CI/CD: Formatting (Black), linting (Ruff), basic checks run in <10s. Benefits: Fast feedback (seconds vs CI minutes), saves CI resources, prevents "fix linting" commits. Not comprehensive (tests too slow), not replacement (CI still needed for thorough checks), not auto-fix all (mypy requires manual fixes). Pattern: Pre-commit (80% issues, fast) + CI (20% issues, thorough). Essential for team productivity.',
  },
  {
    id: 'pch-mc-2',
    question: 'How do you skip pre-commit hooks in an emergency?',
    options: [
      'Delete .pre-commit-config.yaml temporarily',
      'Use git commit --no-verify',
      'Disable git hooks permanently',
      'Pre-commit hooks cannot be skipped',
    ],
    correctAnswer: 1,
    explanation:
      'Skip hooks with --no-verify: git commit --no-verify -m "Emergency fix". Bypasses all pre-commit hooks. Use sparingly: Only for critical hotfixes, production issues, hook bugs. Not permanent solution (fix underlying issue). Example: Production down, need quick fix, pre-commit hook failing. Commit with --no-verify, fix in production, then fix hook. Better: Fix issue and commit normally. --no-verify should be rare in git history.',
  },
  {
    id: 'pch-mc-3',
    question: 'What is the purpose of caching in CI/CD?',
    options: [
      'Store test results between runs',
      'Speed up pipeline by reusing dependencies from previous runs',
      'Cache code to prevent changes',
      'Backup code automatically',
    ],
    correctAnswer: 1,
    explanation:
      "Caching reuses dependencies: actions/setup-python with cache: 'pip' saves/restores pip packages. Without cache: pip install (60s every run). With cache: 5s restore. Example: 1000 pip packages, 50MB. First run: 60s download. Subsequent: 5s restore. Savings: 55s per run. Also cache: mypy cache (.mypy_cache, 20s savings), pytest cache (.pytest_cache, 10s), node_modules. Not test results (tests rerun), not code (checked out fresh), not backup. Essential for fast CI (5 min vs 8 min).",
  },
  {
    id: 'pch-mc-4',
    question: 'What is GitHub branch protection used for?',
    options: [
      'Encrypt sensitive code in specific branches',
      'Prevent direct pushes and require CI checks before merging',
      'Create backup copies of important branches',
      'Automatically fix failing tests',
    ],
    correctAnswer: 1,
    explanation:
      'Branch protection enforces quality standards: Settings → Branches → main: Require PR (no direct push), require status checks (CI must pass), require review (1+ approval), require up-to-date branch. Prevents: Direct push to main, merging failing PRs, merging without review. Example: Developer tries git push origin main → rejected (requires PR). PR with failing tests → cannot merge (status checks required). Enforces: Code review + CI pass before merge. Not encryption, not backup, not auto-fix. Essential for quality control.',
  },
  {
    id: 'pch-mc-5',
    question: 'How can you optimize CI/CD pipeline speed?',
    options: [
      'Run all jobs sequentially to avoid conflicts',
      'Use parallel jobs, caching, and pytest-xdist for faster execution',
      'Skip tests in CI to save time',
      'Reduce test coverage to run fewer tests',
    ],
    correctAnswer: 1,
    explanation:
      'Optimize with parallelism and caching: (1) Parallel jobs: quality + security + test jobs run simultaneously (3 min vs 7 min sequential). (2) Caching: pip cache (saves 60s), mypy cache (20s), pytest cache (10s). (3) pytest-xdist: pytest -n auto runs tests on multiple CPUs (2 min vs 8 min). (4) Matrix: test on py3.10, py3.11, py3.12 in parallel. Result: 8 min → 3 min pipeline. Not sequential (slower), not skip tests (defeats purpose), not reduce coverage (loses quality). Essential for developer productivity (fast feedback).',
  },
];
