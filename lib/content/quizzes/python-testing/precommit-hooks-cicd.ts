export const precommitHooksCICDQuiz = [
  {
    id: 'pch-q-1',
    question:
      'Design a pre-commit and CI/CD strategy balancing speed and thoroughness. Requirements: (1) pre-commit runs in <10s, (2) CI pipeline completes in <5 min, (3) comprehensive checks (formatting, linting, types, security, tests), (4) works for team of 10 developers. Detail hooks, pipeline jobs, caching strategy.',
    sampleAnswer:
      'Speed-optimized quality pipeline: (1) Pre-commit (<10s): Black (1s, auto-format), Ruff --fix (2s, lint+fix), trailing whitespace (0.1s), check-yaml (0.5s). Total: ~4s on changed files. Skip mypy/tests (too slow). (2) CI/CD (<5 min) parallel jobs: Job 1: Quality checks (Black --check, Ruff, mypy) - 90s with cache. Job 2: Security (bandit) - 30s. Job 3: Unit tests (pytest -n auto, SQLite) - 2 min. Job 4: Integration (PostgreSQL, conditional on paths) - 3 min parallel. (3) Caching: pip (actions/cache), mypy cache (.mypy_cache), pytest cache (.pytest_cache). Saves 60s. (4) Team workflow: Pre-commit catches style (fast local feedback). CI catches types/tests (thorough). Developers bypass hooks rarely (--no-verify). Branch protection requires all CI jobs. Result: 4s pre-commit + 3 min CI (parallel) = developers wait 3 min for feedback.',
    keyPoints: [
      'Pre-commit <10s: Black (1s), Ruff --fix (2s), basic checks. Skip slow mypy/tests.',
      'CI <5 min: Parallel jobs (quality 90s, security 30s, tests 2 min, integration 3 min).',
      'Caching: pip, mypy cache, pytest cache. Saves 60s per run.',
      'Team workflow: Pre-commit style/lint, CI types/tests. Branch protection enforces.',
      'Result: 4s pre-commit (local), 3 min CI (parallel). Fast feedback loop.',
    ],
  },
];
