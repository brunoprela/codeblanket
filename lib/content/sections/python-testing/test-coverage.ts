export const testCoverage = {
  title: 'Test Coverage',
  id: 'test-coverage',
  content: `
# Test Coverage

## Introduction

**Test coverage measures which code is executed during tests**—identifying untested code that may harbor bugs. Coverage is a valuable metric but frequently misunderstood. Having 100% coverage doesn't guarantee quality, and obsessing over perfect coverage can waste time testing trivial code. Professional teams target **80-90% coverage**, focusing effort on critical paths while accepting that some code (like \`__repr__\` methods) doesn't warrant tests.

This section covers measuring coverage, understanding metrics, enforcing standards, and using coverage wisely to improve test quality.

---

## Understanding Coverage Metrics

### Line Coverage

**Line coverage** measures which lines of code execute during tests.

\`\`\`python
def calculate_discount(price, discount_percent):
    if discount_percent > 100:        # Line 1
        raise ValueError("Invalid")    # Line 2
    discount = price * (discount_percent / 100)  # Line 3
    return price - discount            # Line 4
\`\`\`

**Test 1** (normal case):
\`\`\`python
def test_calculate_discount():
    result = calculate_discount(100, 20)
    assert result == 80
\`\`\`

**Coverage**: 75% (Lines 1, 3, 4 executed; Line 2 never executed)

**Test 2** (error case):
\`\`\`python
def test_calculate_discount_invalid():
    with pytest.raises(ValueError):
        calculate_discount(100, 150)
\`\`\`

**Coverage**: 100% (all lines executed across both tests)

### Branch Coverage

**Branch coverage** measures which decision paths are taken.

\`\`\`python
def process_payment(amount):
    if amount > 0:        # Branch point
        return "success"  # Branch 1: True path
    else:
        return "error"    # Branch 2: False path
\`\`\`

**Line coverage**: 100% if both lines 2 and 4 execute
**Branch coverage**: 100% if both True and False paths tested

**Why it matters**:
\`\`\`python
def withdraw(account, amount):
    if account.balance >= amount:    # Branch point
        account.balance -= amount
        return True
    # Missing else—what if balance < amount?
    
# Test with sufficient balance
def test_withdraw_success():
    account = Account(balance=100)
    assert withdraw(account, 50)
    assert account.balance == 50

# Line coverage: 100% (all lines execute)
# Branch coverage: 50% (only True branch tested)
\`\`\`

**Missing test** for False branch:
\`\`\`python
def test_withdraw_insufficient():
    account = Account(balance=100)
    result = withdraw(account, 150)
    assert result is False  # This will fail! No return value for False branch
    assert account.balance == 100
\`\`\`

---

## Measuring Coverage with pytest-cov

### Installation

\`\`\`bash
pip install pytest-cov
\`\`\`

### Basic Usage

\`\`\`bash
# Measure coverage for myapp package
pytest --cov=myapp tests/

# Output:
# Name                    Stmts   Miss  Cover
# -------------------------------------------
# myapp/__init__.py           2      0   100%
# myapp/models.py            45      5    89%
# myapp/services.py          67      2    97%
# -------------------------------------------
# TOTAL                     114      7    94%
\`\`\`

### Detailed Reports

\`\`\`bash
# Show missing lines
pytest --cov=myapp --cov-report=term-missing tests/

# Output:
# Name                    Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# myapp/models.py            45      5    89%   23-27, 89
# myapp/services.py          67      2    97%   103, 156
# -------------------------------------------------------

# Generate HTML report (interactive, detailed)
pytest --cov=myapp --cov-report=html tests/
# Open htmlcov/index.html in browser

# Generate XML report (for CI/CD tools like CodeCov, Coveralls)
pytest --cov=myapp --cov-report=xml tests/

# Generate JSON report
pytest --cov=myapp --cov-report=json tests/
\`\`\`

### HTML Report Features

The HTML report provides:
- **Per-file coverage** with exact line-by-line highlighting
- **Green lines**: Executed during tests
- **Red lines**: Never executed
- **Yellow lines**: Partially covered (branches)
- **Click-through navigation** between files
- **Coverage trends** over time

---

## Branch Coverage

### Enabling Branch Coverage

\`\`\`bash
# Enable branch coverage analysis
pytest --cov=myapp --cov-branch --cov-report=term-missing tests/
\`\`\`

### Example: Branch Coverage Analysis

\`\`\`python
def categorize_age(age):
    if age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"

# Test 1
def test_minor():
    assert categorize_age(10) == "minor"

# Test 2
def test_adult():
    assert categorize_age(30) == "adult"

# Test 3
def test_senior():
    assert categorize_age(70) == "senior"
\`\`\`

**Without branch coverage**:
- Line coverage: 100% (all lines execute)

**With branch coverage**:
- Branch coverage: 100% (all branches tested: <18, 18-64, ≥65)

**Missing test** (edge cases):
\`\`\`python
def test_edge_cases():
    assert categorize_age(18) == "adult"  # Boundary: exactly 18
    assert categorize_age(65) == "senior"  # Boundary: exactly 65
\`\`\`

---

## Configuration

### .coveragerc Configuration File

\`\`\`.coveragerc
[run]
# Source code to measure
source = myapp

# Files to exclude from measurement
omit =
    */tests/*
    */migrations/*
    */__init__.py
    */config.py
    */settings.py
    */manage.py

# Enable branch coverage by default
branch = True

# Parallel execution mode (for pytest-xdist)
parallel = True

[report]
# Exclude lines from coverage
exclude_lines =
    # Standard pragma
    pragma: no cover
    
    # Don't complain about missing debug code
    def __repr__
    def __str__
    
    # Don't complain if tests don't hit defensive assertion code
    raise AssertionError
    raise NotImplementedError
    
    # Don't complain about script entry points
    if __name__ == .__main__.:
    
    # Don't complain about type checking code
    if TYPE_CHECKING:
    if typing.TYPE_CHECKING:
    
    # Don't complain about abstract methods
    @abstractmethod
    @abc.abstractmethod

# Precision for coverage percentage
precision = 2

# Show line numbers of missing coverage
show_missing = True

# Skip empty files
skip_empty = True

[html]
# Directory for HTML report
directory = htmlcov

# Title for HTML report
title = MyApp Coverage Report

[xml]
# File for XML report
output = coverage.xml
\`\`\`

### pytest.ini Configuration

\`\`\`ini
[pytest]
# Run coverage by default
addopts = 
    --cov=myapp
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
    --cov-branch
\`\`\`

---

## Enforcing Coverage Thresholds

### Fail Build on Low Coverage

\`\`\`bash
# Fail if coverage < 80%
pytest --cov=myapp --cov-fail-under=80 tests/

# Exit code 1 if coverage is 79%, exit code 0 if 80%+
\`\`\`

### Per-Package Thresholds

\`\`\`.coveragerc
[report]
# Global threshold
fail_under = 80

# Per-package thresholds (advanced, requires additional tooling)
[coverage:report:myapp/services]
fail_under = 95  # Critical services need higher coverage

[coverage:report:myapp/utils]
fail_under = 70  # Utilities can have lower coverage
\`\`\`

### CI/CD Integration

\`\`\`yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests with coverage
        run: |
          pytest --cov=myapp --cov-report=xml --cov-fail-under=80
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
\`\`\`

---

## What Coverage Tells You (and Doesn't)

### Coverage Shows: What Code Ran

✅ **Coverage identifies untested code**

\`\`\`python
def process_user(user):
    if user.is_active:
        send_welcome_email(user)
    else:
        log_inactive_user(user)  # 0% coverage—never tested!

# Only test active users
def test_process_active_user():
    user = User(is_active=True)
    process_user(user)
\`\`\`

**Coverage report**: Shows line 5 (log_inactive_user) has 0% coverage, prompting you to add test:

\`\`\`python
def test_process_inactive_user():
    user = User(is_active=False)
    process_user(user)
    # Should verify logging occurred
\`\`\`

### Coverage Doesn't Show: Test Quality

❌ **100% coverage ≠ correct code**

\`\`\`python
def add(a, b):
    return a + b

def test_add_bad():
    add(2, 3)  # NO ASSERTION!
    # 100% coverage but doesn't verify correctness
\`\`\`

❌ **High coverage can give false confidence**

\`\`\`python
def withdraw(account, amount):
    if account.balance >= amount:
        account.balance -= amount
        return True
    return False

def test_withdraw_weak():
    account = Account(balance=100)
    withdraw(account, 50)
    # 100% coverage but no assertions!
    # Doesn't verify balance changed or return value
\`\`\`

✅ **Proper test**:
\`\`\`python
def test_withdraw_proper():
    account = Account(balance=100)
    result = withdraw(account, 50)
    
    assert result is True
    assert account.balance == 50  # Verify side effect
\`\`\`

---

## The 100% Coverage Debate

### Arguments For 100%

**Benefits**:
- Complete confidence: Every line tested
- No "hidden" code paths
- Prevents "this will never happen" scenarios
- Forces testing edge cases

**Example where 100% matters**:
\`\`\`python
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# Without testing b==0, production encounters unhandled exception
\`\`\`

### Arguments Against 100%

**Diminishing returns**:
- Last 10% (90% → 100%) takes 50% of effort
- Trivial code not worth testing
- Maintenance burden increases
- Team focus shifts from quality to metrics

**Example of wasteful 100% coverage**:
\`\`\`python
class User:
    def __repr__(self):
        return f"User({self.username})"
    
    def __str__(self):
        return self.username

# Testing these is low value:
def test_repr():  # Trivial, unlikely to break
    user = User(username="alice")
    assert "alice" in repr(user)
\`\`\`

### Professional Recommendation: 80-90%

**Target coverage** by component:

| Component | Target | Rationale |
|-----------|--------|-----------|
| **Payment processing** | 95-100% | Critical, money involved |
| **Authentication** | 95-100% | Security critical |
| **Business logic** | 90-95% | Core functionality |
| **API endpoints** | 85-90% | User-facing |
| **Utilities** | 70-80% | Lower risk |
| **__repr__, __str__** | 0-50% | Trivial, low value |
| **Configuration** | 0-30% | Simple, rarely changes |

**Overall target**: 80-90% coverage

---

## Coverage with pytest-xdist (Parallel Tests)

### Combining Coverage from Multiple Processes

pytest-cov automatically handles parallel execution:

\`\`\`bash
# Run tests in parallel with coverage
pytest -n auto --cov=myapp --cov-report=html tests/
\`\`\`

**What happens**:
1. Each worker measures coverage independently
2. pytest-cov collects data from all workers
3. Coverage data is combined into single report

### Configuration for Parallel Coverage

\`\`\`.coveragerc
[run]
# Enable parallel mode
parallel = True

# Directory for parallel data files
data_file = .coverage
\`\`\`

---

## Excluding Code from Coverage

### Using # pragma: no cover

\`\`\`python
def debug_helper():  # pragma: no cover
    """Development-only debugging function"""
    import pdb; pdb.set_trace()

def main():  # pragma: no cover
    """Script entry point, not tested"""
    if __name__ == "__main__":
        run_application()

class BaseModel:
    def __repr__(self):  # pragma: no cover
        return f"<{self.__class__.__name__}>"
\`\`\`

### Excluding Files

\`\`\`.coveragerc
[run]
omit =
    */tests/*           # Don't measure test files
    */migrations/*      # Don't measure database migrations
    */__init__.py      # Don't measure empty __init__ files
    */config.py        # Don't measure configuration
    */settings.py
    venv/*             # Don't measure virtual environment
    */.venv/*
\`\`\`

### Excluding Patterns

\`\`\`.coveragerc
[report]
exclude_lines =
    # Exclude abstract methods
    @abstractmethod
    @abc.abstractmethod
    
    # Exclude type checking code
    if TYPE_CHECKING:
    if typing.TYPE_CHECKING:
    
    # Exclude debug/development code
    if DEBUG:
    if settings.DEBUG:
    
    # Exclude unreachable code
    raise NotImplementedError
    raise AssertionError
    
    # Exclude __main__ blocks
    if __name__ == .__main__.:
\`\`\`

---

## Advanced Coverage Analysis

### Differential Coverage (Changes Only)

\`\`\`bash
# Install diff-cover
pip install diff-cover

# Run tests with XML coverage
pytest --cov=myapp --cov-report=xml

# Show coverage for changed lines only
diff-cover coverage.xml

# Fail if changed lines have < 80% coverage
diff-cover coverage.xml --fail-under=80
\`\`\`

**Use case**: In CI/CD, ensure new code is well-tested without worrying about legacy code coverage.

### Coverage Trends

\`\`\`bash
# Track coverage over time
# Commit 1
pytest --cov=myapp --cov-report=json -o coverage.json

# Commit 2 (after changes)
pytest --cov=myapp --cov-report=json -o coverage_new.json

# Compare
coverage_diff coverage.json coverage_new.json
\`\`\`

---

## Best Practices

1. **Target 80-90%** overall coverage
2. **95%+ for critical paths** (payment, auth, security)
3. **Use branch coverage** for decision-heavy code
4. **Enforce in CI** with --cov-fail-under
5. **Focus on test quality**, not just coverage
6. **Exclude trivial code** (__repr__, config) with pragma: no cover
7. **Review uncovered code** regularly—some might need tests
8. **Use HTML reports** for detailed analysis
9. **Differential coverage** for code reviews
10. **Coverage is a guide**, not a goal

---

## Common Pitfalls

❌ **Chasing 100% coverage** on trivial code
❌ **Tests with no assertions** (coverage but no validation)
❌ **Ignoring branch coverage** (missing paths)
❌ **Testing implementation** instead of behavior
❌ **Coverage without code review** (quality matters more than quantity)

✅ **Good practices**:
- Meaningful tests with assertions
- Focus on critical paths
- Use coverage to find gaps
- Combine with code review
- Balance speed and thoroughness

---

## Coverage in CI/CD

### GitHub Actions Example

\`\`\`yaml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests with coverage
        run: |
          pytest --cov=myapp --cov-report=xml --cov-report=html --cov-fail-under=80
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
      
      - name: Upload HTML report as artifact
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: htmlcov/
\`\`\`

### Coverage Badges

\`\`\`markdown
# README.md
[![codecov](https://codecov.io/gh/username/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo)
\`\`\`

---

## Tools & Services

| Tool | Purpose | Cost |
|------|---------|------|
| **pytest-cov** | Local coverage measurement | Free |
| **Codecov** | Coverage tracking, trends, reports | Free (open source) |
| **Coveralls** | Coverage tracking for GitHub/GitLab | Free (open source) |
| **SonarQube** | Code quality + coverage | Free/Paid |
| **diff-cover** | Coverage for changed lines only | Free |

---

## Summary

**Coverage essentials**:
- **80-90% target**: Professional standard, balances effort and value
- **Branch coverage**: Tests all decision paths
- **pytest-cov**: Easy measurement, multiple report formats
- **Enforce in CI**: --cov-fail-under prevents regression
- **HTML reports**: Detailed, interactive analysis
- **Coverage ≠ quality**: 80% with good assertions > 100% with weak tests
- **Focus on critical paths**: Payment, auth, security deserve 95%+

Use coverage to **find gaps**, not as sole measure of quality.
`,
};
