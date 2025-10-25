export const precommitHooksCICD = {
  title: 'Pre-commit Hooks & CI/CD Integration',
  id: 'precommit-hooks-cicd',
  content: `
# Pre-commit Hooks & CI/CD Integration

## Introduction

**Pre-commit hooks run quality checks before code is committed**—catching issues locally before they reach CI/CD. **CI/CD pipelines enforce quality standards** on every pull request—preventing bad code from merging. Together, they create a robust quality gate system.

This section covers configuring pre-commit hooks, designing CI/CD pipelines, and optimizing for speed while maintaining thoroughness.

---

## Pre-commit Hooks

**pre-commit is a framework for managing git hooks**—automating quality checks before commit.

### Installation & Setup

\`\`\`bash
# Install pre-commit
pip install pre-commit

# Create config file
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
EOF

# Install hooks
pre-commit install
\`\`\`

### How It Works

\`\`\`
1. Developer runs: git commit -m "Add feature"
2. pre-commit intercepts commit
3. Runs all configured hooks
4. If any hook fails:
   - Commit is aborted
   - Developer fixes issues
   - Tries again
5. If all hooks pass:
   - Commit succeeds
\`\`\`

### Running Pre-commit

\`\`\`bash
# Run on staged files (automatic on commit)
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black

# Skip hooks (emergency only!)
git commit --no-verify -m "Emergency fix"

# Update hooks to latest versions
pre-commit autoupdate
\`\`\`

### Advanced Configuration

\`\`\`yaml
repos:
  - repo: local
    hooks:
      # Custom script
      - id: check-migrations
        name: Check database migrations
        entry: python scripts/check_migrations.py
        language: system
        pass_filenames: false
      
      # Run tests on commit
      - id: pytest-fast
        name: Fast tests
        entry: pytest tests/unit -x
        language: system
        pass_filenames: false
        stages: [commit]
  
  # Black formatting
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        args: [--line-length=100]
        exclude: ^(migrations/|generated/)
  
  # Ruff linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        args: [--strict]
        additional_dependencies:
          - types-requests
          - types-redis
          - sqlalchemy[mypy]
  
  # Security scanning
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ['bandit[toml]']
  
  # Commit message formatting
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.12.0
    hooks:
      - id: commitizen
        stages: [commit-msg]
\`\`\`

### Performance Optimization

Pre-commit can be slow. Optimize:

\`\`\`yaml
# Only run on changed files (default)
# vs running on all files

# Skip slow checks for quick commits
repos:
  - repo: local
    hooks:
      - id: mypy
        name: Type check
        entry: mypy
        language: system
        types: [python]
        stages: [push]  # Only on push, not commit
\`\`\`

\`\`\`bash
# Set up fast vs thorough hooks
# Commit: Fast checks (Black, Ruff)
# Push: Thorough checks (mypy, bandit, tests)

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        stages: [commit]  # Run on commit (fast)
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        stages: [push]  # Run on push (slower)
\`\`\`

---

## CI/CD Pipeline Design

### GitHub Actions Example

\`\`\`yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  # Job 1: Code Quality (fast checks)
  quality:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install black ruff mypy
      
      - name: Black (formatting)
        run: black --check myapp/ tests/
      
      - name: Ruff (linting)
        run: ruff check myapp/ tests/
      
      - name: mypy (type checking)
        run: mypy myapp/
  
  # Job 2: Security Scan
  security:
    runs-on: ubuntu-latest
    timeout-minutes: 3
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install bandit
        run: pip install bandit[toml]
      
      - name: Bandit scan
        run: bandit -r myapp/ -c pyproject.toml
  
  # Job 3: Tests (slower, runs in parallel)
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python \${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: \${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Run tests with coverage
        env:
          DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/ \\
            -n auto \\
            --cov=myapp \\
            --cov-report=xml \\
            --cov-report=term-missing \\
            --cov-fail-under=80
      
      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
  
  # Job 4: Integration Tests (even slower)
  integration:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: [quality, security, test]  # Only run if others pass
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Start services with Docker Compose
        run: docker-compose -f docker-compose.test.yml up -d
      
      - name: Wait for services
        run: |
          timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml exec -T app pg_isready; do sleep 1; done'
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -m integration --maxfail=3
      
      - name: Cleanup
        if: always()
        run: docker-compose -f docker-compose.test.yml down -v
\`\`\`

### Optimization Strategies

**1. Caching Dependencies**

\`\`\`yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'
    cache: 'pip'  # Caches pip dependencies
\`\`\`

**2. Parallel Jobs**

\`\`\`yaml
jobs:
  quality:   # Runs in parallel
  security:  # Runs in parallel
  test:      # Runs in parallel
\`\`\`

**3. Matrix Testing**

\`\`\`yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]
# Runs 9 jobs in parallel (3 versions × 3 OS)
\`\`\`

**4. Conditional Execution**

\`\`\`yaml
- name: Upload coverage
  if: matrix.python-version == '3.11'  # Only one job uploads
  uses: codecov/codecov-action@v3

- name: Deploy to staging
  if: github.ref == 'refs/heads/develop'  # Only on develop branch
  run: ./deploy_staging.sh
\`\`\`

**5. Fail Fast**

\`\`\`yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
  matrix:
    python-version: ['3.10', '3.11']

# Or use --maxfail in pytest
pytest tests/ --maxfail=3  # Stop after 3 failures
\`\`\`

---

## Branch Protection Rules

### GitHub Branch Protection

\`\`\`
Repository Settings → Branches → Add rule

Branch name pattern: main

☑ Require pull request before merging
  ☑ Require approvals: 1
  ☑ Dismiss stale reviews when new commits are pushed

☑ Require status checks before merging
  ☑ Require branches to be up to date
  Status checks:
    - quality
    - security
    - test
    - integration

☑ Require conversation resolution before merging

☑ Include administrators (enforce for everyone)
\`\`\`

This prevents:
- Direct pushes to main
- Merging failing PRs
- Merging without code review
- Merging with unresolved comments

---

## Advanced CI/CD Patterns

### Pattern 1: Differential Testing

**Only test changed files** for speed:

\`\`\`yaml
- name: Get changed files
  id: changed-files
  uses: tj-actions/changed-files@v40
  with:
    files: |
      myapp/**/*.py
      tests/**/*.py

- name: Run tests for changed modules
  if: steps.changed-files.outputs.any_changed == 'true'
  run: |
    pytest \$(echo "\${{ steps.changed-files.outputs.all_changed_files }}" | tr ' ' '\\n' | grep test_)
\`\`\`

### Pattern 2: Scheduled Full Test Suite

\`\`\`yaml
# Run full test suite nightly
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  full-test-suite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run all tests
        run: pytest tests/ --slow --integration
\`\`\`

### Pattern 3: Performance Regression Testing

\`\`\`yaml
- name: Run benchmarks
  run: pytest tests/benchmarks/ --benchmark-only --benchmark-save=pr_\${{ github.event.number }}

- name: Compare with main
  run: |
    git checkout main
    pytest tests/benchmarks/ --benchmark-only --benchmark-save=main
    pytest-benchmark compare main pr_\${{ github.event.number }} --group-by=name
\`\`\`

### Pattern 4: Deployment Pipeline

\`\`\`yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  # Runs all quality/test jobs first
  quality:
    # ... (same as CI)
  
  test:
    # ... (same as CI)
  
  # Deploy only after all tests pass
  deploy-staging:
    needs: [quality, test]
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to staging
        run: |
          ./scripts/deploy_staging.sh
      
      - name: Run smoke tests
        run: |
          pytest tests/smoke/ --env=staging
  
  deploy-production:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          ./scripts/deploy_production.sh
      
      - name: Run smoke tests
        run: |
          pytest tests/smoke/ --env=production
\`\`\`

---

## Monitoring & Notifications

### Slack Notifications

\`\`\`yaml
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "❌ CI Failed: \${{ github.event.pull_request.html_url }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*CI Pipeline Failed*\\n*PR:* \${{ github.event.pull_request.title }}\\n*Author:* \${{ github.event.pull_request.user.login }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: \${{ secrets.SLACK_WEBHOOK_URL }}
\`\`\`

### GitHub Status Checks

\`\`\`python
# scripts/update_github_status.py
import requests
import os

def update_status (state, description):
    """Update GitHub commit status"""
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/statuses/{os.environ['GITHUB_SHA']}"
    
    requests.post(
        url,
        json={
            "state": state,  # pending, success, failure, error
            "description": description,
            "context": "custom-check",
        },
        headers={
            "Authorization": f"token {os.environ['GITHUB_TOKEN']}"
        }
    )
\`\`\`

---

## Best Practices

### Pre-commit

1. **Run fast checks** (Black, Ruff) on commit
2. **Run slow checks** (mypy, tests) on push
3. **Use \`--all-files\`** occasionally to catch global issues
4. **Keep hooks updated** with \`pre-commit autoupdate\`
5. **Document bypass** (--no-verify) only for emergencies

### CI/CD

1. **Parallel jobs** for speed (quality + security + tests)
2. **Cache dependencies** (pip, npm, docker layers)
3. **Matrix testing** for multiple Python versions/OS
4. **Branch protection** to enforce checks
5. **Fast feedback** (<5 min for basic checks)
6. **Scheduled full tests** (nightly comprehensive suite)
7. **Monitor pipeline performance** (optimize slow steps)

### Team Workflow

1. **Pre-commit catches** 80% of issues locally
2. **CI catches** remaining 20% + integration issues
3. **Code review** focuses on logic, not style
4. **Branch protection** enforces all checks
5. **Clear error messages** help developers fix issues

---

## Troubleshooting

### Pre-commit Hook Fails

\`\`\`bash
# See what failed
pre-commit run --all-files --verbose

# Skip hook temporarily (NOT recommended)
git commit --no-verify

# Fix and retry
black myapp/
git add -u
git commit
\`\`\`

### CI/CD Takes Too Long

**Analyze slow steps**:
\`\`\`yaml
- name: Slow step
  run: |
    time pytest tests/  # Shows execution time
\`\`\`

**Optimize**:
- Add caching (pip, mypy, pytest)
- Use pytest-xdist (-n auto)
- Run integration tests conditionally
- Split into more parallel jobs

### Flaky Tests in CI

\`\`\`yaml
# Retry flaky tests
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: pytest tests/
\`\`\`

---

## Summary

**Pre-commit hooks**:
- Fast local checks (Black, Ruff)
- Prevent bad commits
- 80% of issues caught locally

**CI/CD pipeline**:
- Comprehensive checks (types, security, tests)
- Parallel execution (<5 min)
- Branch protection enforcement
- 20% of issues caught in CI

**Together**: Robust quality gate system preventing bad code from reaching production.
`,
};
