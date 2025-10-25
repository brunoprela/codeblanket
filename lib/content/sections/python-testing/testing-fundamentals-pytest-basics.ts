export const testingFundamentalsPytestBasics = {
  title: 'Testing Fundamentals & pytest Basics',
  id: 'testing-fundamentals-pytest-basics',
  content: `
# Testing Fundamentals & pytest Basics

## Introduction

Testing is not optional—it's the foundation of professional software development. Yet many developers treat tests as an afterthought. **In production Python applications, your test suite is as important as your production code.** A comprehensive test suite catches bugs before customers do, enables confident refactoring, and serves as living documentation.

### Why Testing Matters in Production

Consider this real scenario: A payment processing service deployed without comprehensive tests. A developer refactored the decimal rounding logic. The tests passed (only 3 simple tests existed). In production, **$47,000 was lost in 2 hours** due to incorrect rounding before the bug was caught.

**Cost of bugs increases exponentially**:
- Bug caught in tests: 5 minutes to fix
- Bug caught in code review: 30 minutes
- Bug caught in QA: 2 hours (includes deployment, testing, fix)
- Bug in production: 2 days (includes incident response, hotfix, postmortem)
- Bug causing data corruption: Weeks (includes data recovery, customer communication)

### The Testing Pyramid

\`\`\`
           /\\
          /E2E\\       Few (5-10%)
         /------\\     Slow, brittle, expensive
        /Integration\\ Medium (20-30%)
       /------------\\ Test component interactions
      /    Unit      \\ Many (60-75%)
     /----------------\\ Fast, reliable, cheap
    
    Speed:  Fast → Slow
    Cost:   Cheap → Expensive
    Count:  Many → Few
\`\`\`

**Unit Tests** (60-75% of tests):
- Test individual functions/methods in isolation
- Fast (<100ms per test)
- No external dependencies (databases, APIs, files)
- Use mocking for dependencies
- Example: Testing a function that calculates interest

**Integration Tests** (20-30% of tests):
- Test multiple components working together
- Medium speed (100ms-1s per test)
- May use test database, Redis, etc.
- Example: Testing API endpoint with database

**E2E Tests** (5-10% of tests):
- Test complete user workflows
- Slow (1s-10s+ per test)
- Full stack (UI, API, database)
- Brittle (many points of failure)
- Example: User registration flow through UI

### Why pytest Over unittest

Python includes \`unittest\` in the standard library, but \`pytest\` has become the industry standard:

| Feature | unittest | pytest |
|---------|----------|--------|
| Syntax | Verbose (classes, setUp) | Simple (functions, fixtures) |
| Assertions | self.assertEqual() | assert x == y |
| Fixtures | setUp/tearDown | @pytest.fixture decorator |
| Parametrization | Clumsy | Built-in, elegant |
| Plugins | Limited | 800+ plugins |
| Test discovery | Requires specific naming | Flexible |
| Output | Basic | Beautiful, detailed |

**unittest example** (verbose):

\`\`\`python
import unittest

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_addition(self):
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def tearDown(self):
        self.calc = None

if __name__ == '__main__':
    unittest.main()
\`\`\`

**pytest example** (simple):

\`\`\`python
from calculator import Calculator

def test_addition():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5
\`\`\`

**30% less code, 100% more readable.**

---

## Setting Up pytest

### Installation

\`\`\`bash
# Install pytest
pip install pytest

# Install with common plugins
pip install pytest pytest-cov pytest-mock pytest-asyncio pytest-xdist

# Or in requirements.txt
pytest==7.4.3
pytest-cov==4.1.0  # Coverage reporting
pytest-mock==3.12.0  # Mocking helpers
pytest-asyncio==0.21.1  # Async test support
pytest-xdist==3.5.0  # Parallel execution
\`\`\`

### Project Structure

\`\`\`
my_project/
├── src/
│   ├── __init__.py
│   ├── calculator.py
│   ├── payment.py
│   └── database.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── test_calculator.py
│   ├── test_payment.py
│   └── integration/
│       ├── __init__.py
│       └── test_payment_flow.py
├── pytest.ini               # pytest configuration
└── requirements.txt
\`\`\`

### pytest Configuration

**pytest.ini** (project root):

\`\`\`ini
[pytest]
# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Directories to search
testpaths = tests

# Minimum Python version
minversion = 3.10

# Additional options
addopts = 
    -v                          # Verbose output
    --strict-markers            # Require marker definitions
    --tb=short                  # Shorter tracebacks
    --cov=src                   # Coverage for src directory
    --cov-report=html           # HTML coverage report
    --cov-report=term-missing   # Show missing lines
    -n auto                     # Parallel execution (requires pytest-xdist)

# Custom markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    db: marks tests that require database
\`\`\`

---

## Writing Your First Tests

### Simple Test Example

**calculator.py**:

\`\`\`python
"""Calculator module for testing examples"""

class Calculator:
    """Simple calculator for demonstration"""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to exponent"""
        return base ** exponent
\`\`\`

**test_calculator.py**:

\`\`\`python
"""Tests for calculator module"""
import pytest
from calculator import Calculator


def test_add():
    """Test addition"""
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5


def test_add_negative():
    """Test addition with negative numbers"""
    calc = Calculator()
    result = calc.add(-5, 3)
    assert result == -2


def test_subtract():
    """Test subtraction"""
    calc = Calculator()
    result = calc.subtract(10, 4)
    assert result == 6


def test_multiply():
    """Test multiplication"""
    calc = Calculator()
    result = calc.multiply(3, 7)
    assert result == 21


def test_divide():
    """Test division"""
    calc = Calculator()
    result = calc.divide(10, 2)
    assert result == 5.0


def test_divide_by_zero():
    """Test division by zero raises exception"""
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)


def test_power():
    """Test exponentiation"""
    calc = Calculator()
    result = calc.power(2, 8)
    assert result == 256
\`\`\`

### Running Tests

\`\`\`bash
# Run all tests
pytest

# Run specific file
pytest tests/test_calculator.py

# Run specific test
pytest tests/test_calculator.py::test_add

# Run tests matching pattern
pytest -k "divide"

# Run with verbose output
pytest -v

# Run with detailed output
pytest -vv

# Stop at first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff
\`\`\`

---

## Assert Statements

pytest uses plain Python \`assert\` statements with introspection for detailed error messages.

### Basic Assertions

\`\`\`python
def test_assertions():
    """Demonstrate various assertion patterns"""
    # Equality
    assert 2 + 2 == 4
    assert "hello" == "hello"
    
    # Inequality
    assert 5 != 3
    
    # Comparison
    assert 10 > 5
    assert 3 < 7
    assert 5 >= 5
    assert 4 <= 10
    
    # Boolean
    assert True
    assert not False
    
    # Identity
    x = [1, 2, 3]
    y = x
    assert x is y
    
    # Membership
    assert 3 in [1, 2, 3]
    assert "hello" in "hello world"
    
    # Type checking
    assert isinstance(42, int)
    assert isinstance("hello", str)
\`\`\`

### Advanced Assertions

\`\`\`python
import pytest

def test_advanced_assertions():
    """Advanced assertion patterns"""
    # Approximate equality (floats)
    result = 0.1 + 0.2
    assert result == pytest.approx(0.3)
    assert result == pytest.approx(0.3, abs=0.001)
    
    # Multiple items
    results = [1, 2, 3, 4]
    assert 2 in results
    assert all(x > 0 for x in results)
    assert any(x == 3 for x in results)
    
    # String matching
    message = "Error: File not found"
    assert "Error" in message
    assert message.startswith("Error")
    assert message.endswith("found")
    
    # Dict assertions
    user = {"name": "Alice", "age": 30}
    assert "name" in user
    assert user["age"] == 30
    assert user.get("email") is None


def test_collection_equality():
    """Test collection assertions"""
    # Lists (order matters)
    assert [1, 2, 3] == [1, 2, 3]
    assert [1, 2, 3] != [3, 2, 1]
    
    # Sets (order doesn't matter)
    assert {1, 2, 3} == {3, 2, 1}
    
    # Dicts
    assert {"a": 1, "b": 2} == {"b": 2, "a": 1}


def test_exception_assertions():
    """Test exception handling"""
    # Simple exception check
    with pytest.raises(ValueError):
        int("not a number")
    
    # Exception with message matching
    with pytest.raises(ValueError, match="invalid literal"):
        int("not a number")
    
    # Capture exception for inspection
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("Custom error message")
    
    assert str(exc_info.value) == "Custom error message"
    assert exc_info.type is ValueError
\`\`\`

### Custom Error Messages

\`\`\`python
def test_with_custom_messages():
    """Add custom error messages to assertions"""
    user_age = 15
    
    # Basic message
    assert user_age >= 18, f"User must be 18+, got {user_age}"
    
    # Detailed message
    balance = 50
    cost = 100
    assert balance >= cost, (
        f"Insufficient funds. "
        f"Balance: \${balance}, Required: \${cost}"
    )
\`\`\`

---

## Test Discovery

pytest automatically discovers tests based on naming conventions:

### Discovery Rules

**Files**: \`test_*.py\` or \`*_test.py\`
- ✅ \`test_calculator.py\`
- ✅ \`calculator_test.py\`
- ❌ \`calculator.py\`

**Classes**: \`Test*\` (no \`__init__\` method)
- ✅ \`class TestCalculator:\`
- ✅ \`class TestUserAuthentication:\`
- ❌ \`class Calculator:\`

**Functions**: \`test_*\`
- ✅ \`def test_addition():\`
- ✅ \`def test_user_can_login():\`
- ❌ \`def addition():\`

### Test Organization Patterns

**Pattern 1: Flat Structure** (simple projects)

\`\`\`
tests/
├── test_calculator.py
├── test_payment.py
└── test_user.py
\`\`\`

**Pattern 2: Mirrored Structure** (recommended for larger projects)

\`\`\`
src/
├── calculator.py
├── payment.py
└── user.py

tests/
├── test_calculator.py  # Mirrors src/calculator.py
├── test_payment.py     # Mirrors src/payment.py
└── test_user.py        # Mirrors src/user.py
\`\`\`

**Pattern 3: Test Type Separation** (large projects)

\`\`\`
tests/
├── unit/
│   ├── test_calculator.py
│   ├── test_payment.py
│   └── test_user.py
├── integration/
│   ├── test_payment_flow.py
│   └── test_user_registration.py
└── e2e/
    └── test_complete_workflow.py
\`\`\`

---

## Test Organization with Classes

While pytest works great with simple functions, classes help organize related tests:

\`\`\`python
"""Organized tests with classes"""
import pytest
from calculator import Calculator


class TestCalculatorBasicOps:
    """Group related calculator tests"""
    
    def test_add(self):
        """Test addition"""
        calc = Calculator()
        assert calc.add(2, 3) == 5
    
    def test_subtract(self):
        """Test subtraction"""
        calc = Calculator()
        assert calc.subtract(10, 4) == 6


class TestCalculatorAdvancedOps:
    """Advanced operations"""
    
    def test_power(self):
        """Test exponentiation"""
        calc = Calculator()
        assert calc.power(2, 8) == 256
    
    def test_divide_by_zero(self):
        """Test error handling"""
        calc = Calculator()
        with pytest.raises(ValueError):
            calc.divide(10, 0)
\`\`\`

**Benefits**:
- Logical grouping of related tests
- Shared setup using class-level fixtures
- Better test organization and navigation
- Clear test reports

---

## Practical Example: Testing a Payment Service

Let's test a realistic payment processing module:

**payment.py**:

\`\`\`python
"""Payment processing module"""
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional


class PaymentStatus(Enum):
    """Payment status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Payment:
    """Payment model"""
    amount: Decimal
    currency: str
    status: PaymentStatus = PaymentStatus.PENDING
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate payment"""
        if self.amount <= 0:
            raise ValueError("Amount must be positive")
        if self.currency not in ["USD", "EUR", "GBP"]:
            raise ValueError(f"Unsupported currency: {self.currency}")


class PaymentProcessor:
    """Process payments"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.processed_payments: list[Payment] = []
    
    def process_payment(self, payment: Payment) -> bool:
        """
        Process a payment
        
        Returns:
            True if successful, False otherwise
        """
        if not self.api_key:
            raise ValueError("API key required")
        
        if payment.status != PaymentStatus.PENDING:
            raise ValueError("Payment already processed")
        
        # Simulate processing
        try:
            # In real implementation, call payment gateway API
            payment.status = PaymentStatus.COMPLETED
            self.processed_payments.append(payment)
            return True
        except Exception:
            payment.status = PaymentStatus.FAILED
            return False
    
    def refund_payment(self, payment: Payment) -> bool:
        """Refund a payment"""
        if payment.status != PaymentStatus.COMPLETED:
            raise ValueError("Can only refund completed payments")
        
        payment.status = PaymentStatus.REFUNDED
        return True
    
    def get_total_processed(self) -> Decimal:
        """Get total amount processed"""
        return sum(
            p.amount for p in self.processed_payments
            if p.status == PaymentStatus.COMPLETED
        )
\`\`\`

**test_payment.py**:

\`\`\`python
"""Tests for payment processing"""
import pytest
from decimal import Decimal
from payment import Payment, PaymentProcessor, PaymentStatus


class TestPayment:
    """Test Payment model"""
    
    def test_create_payment(self):
        """Test creating valid payment"""
        payment = Payment(
            amount=Decimal("99.99"),
            currency="USD",
            description="Test payment"
        )
        assert payment.amount == Decimal("99.99")
        assert payment.currency == "USD"
        assert payment.status == PaymentStatus.PENDING
    
    def test_payment_negative_amount(self):
        """Test payment with negative amount fails"""
        with pytest.raises(ValueError, match="Amount must be positive"):
            Payment(amount=Decimal("-10.00"), currency="USD")
    
    def test_payment_zero_amount(self):
        """Test payment with zero amount fails"""
        with pytest.raises(ValueError, match="Amount must be positive"):
            Payment(amount=Decimal("0.00"), currency="USD")
    
    def test_payment_invalid_currency(self):
        """Test payment with invalid currency fails"""
        with pytest.raises(ValueError, match="Unsupported currency"):
            Payment(amount=Decimal("100.00"), currency="JPY")


class TestPaymentProcessor:
    """Test PaymentProcessor"""
    
    def test_process_payment_success(self):
        """Test successful payment processing"""
        processor = PaymentProcessor(api_key="test_key")
        payment = Payment(amount=Decimal("50.00"), currency="USD")
        
        result = processor.process_payment(payment)
        
        assert result is True
        assert payment.status == PaymentStatus.COMPLETED
        assert len(processor.processed_payments) == 1
    
    def test_process_payment_no_api_key(self):
        """Test processing fails without API key"""
        processor = PaymentProcessor(api_key="")
        payment = Payment(amount=Decimal("50.00"), currency="USD")
        
        with pytest.raises(ValueError, match="API key required"):
            processor.process_payment(payment)
    
    def test_process_payment_already_processed(self):
        """Test cannot process payment twice"""
        processor = PaymentProcessor(api_key="test_key")
        payment = Payment(amount=Decimal("50.00"), currency="USD")
        payment.status = PaymentStatus.COMPLETED
        
        with pytest.raises(ValueError, match="already processed"):
            processor.process_payment(payment)
    
    def test_refund_payment_success(self):
        """Test successful refund"""
        processor = PaymentProcessor(api_key="test_key")
        payment = Payment(amount=Decimal("50.00"), currency="USD")
        processor.process_payment(payment)
        
        result = processor.refund_payment(payment)
        
        assert result is True
        assert payment.status == PaymentStatus.REFUNDED
    
    def test_refund_pending_payment(self):
        """Test cannot refund pending payment"""
        processor = PaymentProcessor(api_key="test_key")
        payment = Payment(amount=Decimal("50.00"), currency="USD")
        
        with pytest.raises(ValueError, match="only refund completed"):
            processor.refund_payment(payment)
    
    def test_get_total_processed(self):
        """Test calculating total processed amount"""
        processor = PaymentProcessor(api_key="test_key")
        
        payment1 = Payment(amount=Decimal("50.00"), currency="USD")
        payment2 = Payment(amount=Decimal("75.50"), currency="USD")
        
        processor.process_payment(payment1)
        processor.process_payment(payment2)
        
        total = processor.get_total_processed()
        assert total == Decimal("125.50")
\`\`\`

---

## Best Practices for pytest

### 1. Test One Thing Per Test

❌ **Bad**: Multiple assertions testing different things

\`\`\`python
def test_user():
    user = User("Alice", 30)
    assert user.name == "Alice"  # Tests name
    assert user.age == 30  # Tests age
    assert user.is_adult() is True  # Tests business logic
\`\`\`

✅ **Good**: Separate tests

\`\`\`python
def test_user_name():
    user = User("Alice", 30)
    assert user.name == "Alice"

def test_user_age():
    user = User("Alice", 30)
    assert user.age == 30

def test_user_is_adult():
    user = User("Alice", 30)
    assert user.is_adult() is True
\`\`\`

### 2. Use Descriptive Test Names

❌ **Bad**: Unclear what's being tested

\`\`\`python
def test_user():
    ...

def test_payment_1():
    ...
\`\`\`

✅ **Good**: Clear, descriptive names

\`\`\`python
def test_user_with_negative_age_raises_error():
    ...

def test_payment_processing_succeeds_with_valid_card():
    ...
\`\`\`

### 3. Follow AAA Pattern

**Arrange-Act-Assert**: Clear test structure

\`\`\`python
def test_payment_processing():
    # Arrange: Set up test data
    processor = PaymentProcessor(api_key="test")
    payment = Payment(amount=Decimal("100.00"), currency="USD")
    
    # Act: Execute the code being tested
    result = processor.process_payment(payment)
    
    # Assert: Verify the results
    assert result is True
    assert payment.status == PaymentStatus.COMPLETED
\`\`\`

### 4. Keep Tests Independent

Each test should be able to run independently and in any order:

❌ **Bad**: Tests depend on each other

\`\`\`python
# Global state shared between tests
user = None

def test_create_user():
    global user
    user = User("Alice", 30)
    assert user is not None

def test_user_name():
    # Depends on previous test running first!
    assert user.name == "Alice"
\`\`\`

✅ **Good**: Independent tests

\`\`\`python
def test_create_user():
    user = User("Alice", 30)
    assert user is not None

def test_user_name():
    user = User("Alice", 30)  # Create new instance
    assert user.name == "Alice"
\`\`\`

### 5. Test Edge Cases and Error Conditions

Don't just test the happy path:

\`\`\`python
def test_divide_normal():
    """Test normal division"""
    calc = Calculator()
    assert calc.divide(10, 2) == 5.0

def test_divide_by_zero():
    """Test edge case: division by zero"""
    calc = Calculator()
    with pytest.raises(ValueError):
        calc.divide(10, 0)

def test_divide_negative_numbers():
    """Test edge case: negative numbers"""
    calc = Calculator()
    assert calc.divide(-10, 2) == -5.0

def test_divide_very_small_numbers():
    """Test edge case: precision"""
    calc = Calculator()
    result = calc.divide(1, 3)
    assert result == pytest.approx(0.333333, abs=0.00001)
\`\`\`

---

## Running Tests Effectively

### Useful pytest Options

\`\`\`bash
# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (4 workers)
pytest -n 4

# Run only failed tests
pytest --lf

# Run failed first, then others
pytest --ff

# Stop after 3 failures
pytest --maxfail=3

# Show print statements
pytest -s

# Run tests matching pattern
pytest -k "payment"

# Run tests with specific marker
pytest -m "unit"

# Skip tests with marker
pytest -m "not slow"

# Very verbose output
pytest -vv

# Show locals on failure
pytest -l --tb=long
\`\`\`

### Markers for Test Organization

\`\`\`python
import pytest

@pytest.mark.slow
def test_large_dataset_processing():
    """This test takes 10+ seconds"""
    ...

@pytest.mark.integration
def test_database_integration():
    """Requires database"""
    ...

@pytest.mark.unit
def test_calculator_add():
    """Pure unit test"""
    ...

# Skip test
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    ...

# Skip conditionally
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_new_syntax():
    ...

# Expected to fail
@pytest.mark.xfail(reason="Known bug #123")
def test_buggy_feature():
    ...
\`\`\`

Run tests by marker:

\`\`\`bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run unit and integration, but not slow
pytest -m "unit or integration and not slow"
\`\`\`

---

## Summary

**Testing is essential for production Python**:
- Catches bugs early (saves time and money)
- Enables confident refactoring
- Serves as living documentation
- Ensures code quality

**pytest advantages**:
- Simple syntax (plain assert, no boilerplate)
- Powerful features (fixtures, parametrization, plugins)
- Great ecosystem (800+ plugins)
- Beautiful output and detailed errors

**Best practices**:
1. Test one thing per test
2. Use descriptive names
3. Follow AAA pattern (Arrange-Act-Assert)
4. Keep tests independent
5. Test edge cases and errors
6. Organize tests logically

**Next steps**:
- Learn fixtures for test setup (next section)
- Master parametrization for data-driven tests
- Understand mocking for isolating dependencies
- Integrate testing into CI/CD pipeline

Your test suite is an investment that pays dividends throughout the project lifecycle. Start testing today—your future self will thank you.
`,
};
