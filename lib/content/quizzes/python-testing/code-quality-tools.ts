export const codeQualityToolsQuiz = [
  {
    id: 'cqt-q-1',
    question:
      'Design a code quality pipeline for a new Python microservice project. Include: (1) which tools to use and why, (2) order of execution, (3) CI/CD integration strategy, (4) how to handle false positives, (5) performance optimization (tools run in < 2 minutes).',
    sampleAnswer:
      'Code quality pipeline: (1) Tools: Black (formatting, fast), Ruff (linting, 100× faster than alternatives), mypy (type safety), bandit (security, critical), pytest (testing). Skip pylint (slow, covered by Ruff). (2) Order: Black first (format), then Ruff (lint formatted code), mypy (types), bandit (security), pytest (tests with coverage). Why: Format first prevents style lint failures, types after lint. (3) CI/CD: GitHub Actions matrix: separate jobs for speed. Black + Ruff (1 min), mypy (30s), bandit (20s), pytest (2 min). Parallel execution: 2 min total vs 4 min sequential. (4) False positives: Black: none (deterministic). Ruff: --extend-ignore for specific cases, per-file ignores in pyproject.toml. mypy: # type: ignore with explanation. bandit: #nosec with comment. (5) Performance: Cache dependencies (actions/cache), Ruff instead of Flake8+isort (10× faster), mypy cache (--cache-dir), pytest-xdist (-n auto). Result: < 2 min pipeline, comprehensive quality checks.',
    keyPoints: [
      'Tools: Black, Ruff, mypy, bandit, pytest (skip slow pylint, use fast Ruff)',
      'Order: Format → lint → types → security → tests (logical dependency)',
      'CI/CD: Parallel jobs, GitHub Actions matrix, 2 min total with caching',
      'False positives: --extend-ignore, # type: ignore, #nosec with comments',
      'Performance: actions/cache, Ruff (10× faster), mypy cache, pytest-xdist',
    ],
  },
];
