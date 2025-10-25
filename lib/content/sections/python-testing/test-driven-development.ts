export const testDrivenDevelopment = {
  title: 'Test-Driven Development (TDD)',
  id: 'test-driven-development',
  content: `
# Test-Driven Development (TDD)

## Introduction

**TDD is a development methodology where tests are written before code**—Red (write failing test), Green (make it pass), Refactor (improve code). TDD leads to better design, fewer bugs, and higher confidence.

---

## The TDD Cycle: Red-Green-Refactor

\`\`\`
1. RED: Write failing test
   └─> Test fails (code doesn't exist yet)

2. GREEN: Write minimal code to pass
   └─> Test passes (code works)

3. REFACTOR: Improve code
   └─> Tests still pass (safety net)

Repeat for next feature
\`\`\`

---

## Example: TDD Workflow

### Step 1: RED - Write Failing Test

\`\`\`python
# test_calculator.py
def test_add():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5

# Run: FAILS (Calculator doesn't exist)
\`\`\`

### Step 2: GREEN - Make It Pass

\`\`\`python
# calculator.py
class Calculator:
    def add(self, a, b):
        return a + b  # Minimal implementation

# Run: PASSES
\`\`\`

### Step 3: REFACTOR - Improve

\`\`\`python
# Add type hints, docstrings
class Calculator:
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

# Run: Still PASSES
\`\`\`

---

## Benefits of TDD

1. **Better Design**: Tests force simple, testable interfaces
2. **Fewer Bugs**: Code is tested from start
3. **Confidence**: Refactor safely with test safety net
4. **Documentation**: Tests show how code should be used
5. **Focus**: Write only code needed to pass tests

---

## TDD for Bug Fixes

**Bug found in production?** Write test first:

\`\`\`python
# 1. Write test that reproduces bug
def test_divide_by_zero_bug():
    calc = Calculator()
    with pytest.raises(ValueError):
        calc.divide(10, 0)

# FAILS (bug exists)

# 2. Fix code
def divide(self, a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# PASSES (bug fixed, regression prevented)
\`\`\`

---

## TDD Best Practices

1. **Write smallest possible test** first
2. **Write only enough code** to pass
3. **Refactor only when tests pass**
4. **One test at a time**
5. **Run tests frequently** (after each change)

---

## Common Mistakes

❌ **Writing multiple tests before code**  
✅ Write one test, make it pass, repeat

❌ **Over-engineering in first pass**  
✅ Simplest code to pass test

❌ **Skipping refactor step**  
✅ Always improve code after green

---

## TDD vs Test-After

| Aspect | TDD (Test-First) | Test-After |
|--------|------------------|------------|
| **Design** | Forces simple design | Design may be untestable |
| **Coverage** | High (every feature tested) | Variable (may skip tests) |
| **Confidence** | High (tests guide development) | Lower (tests retrofit) |
| **Bugs** | Fewer (caught early) | More (found later) |

---

## Summary

**TDD workflow**: Red (failing test) → Green (pass) → Refactor (improve)

**Benefits**:
- Better design
- Higher confidence
- Fewer bugs
- Living documentation

**When to use**: New features, bug fixes, refactoring

TDD requires discipline but **produces higher quality code**.
`,
};
