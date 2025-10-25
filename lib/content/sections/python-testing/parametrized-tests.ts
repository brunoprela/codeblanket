export const parametrizedTests = {
  title: 'Parametrized Tests',
  id: 'parametrized-tests',
  content: `
# Parametrized Tests

## Introduction

**Parametrized tests eliminate code duplication when testing the same logic with different inputs.** Instead of writing 10 nearly identical tests, write one parametrized test that runs with 10 different parameter sets.

### The Problem: Test Duplication

Testing the same function with different inputs leads to duplication:

\`\`\`python
def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-2, -3) == -5

def test_add_mixed_numbers():
    assert add(-2, 3) == 1

def test_add_zeros():
    assert add(0, 0) == 0

def test_add_large_numbers():
    assert add(1000, 2000) == 3000

# 5 tests, 20 lines, 80% duplication
\`\`\`

**Problem**: Violates DRY (Don't Repeat Yourself), hard to maintain.

### Solution: Parametrization

\`\`\`python
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-2, -3, -5),
    (-2, 3, 1),
    (0, 0, 0),
    (1000, 2000, 3000),
])
def test_add(a, b, expected):
    assert add(a, b) == expected

# 1 test, 7 lines, 0% duplication
# Runs 5 times with different parameters
\`\`\`

**Result**: 65% less code, easier to maintain, easier to add new test cases.

---

## Basic Parametrization

### Syntax

\`\`\`python
@pytest.mark.parametrize("parameter_name", [value1, value2, value3])
def test_function(parameter_name):
    # Test logic using parameter_name
    ...
\`\`\`

### Single Parameter Example

\`\`\`python
import pytest

@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_is_positive(number):
    """Test runs 5 times, once for each number"""
    assert number > 0

# Output:
# test_is_positive[1] PASSED
# test_is_positive[2] PASSED
# test_is_positive[3] PASSED
# test_is_positive[4] PASSED
# test_is_positive[5] PASSED
\`\`\`

### Multiple Parameters

\`\`\`python
@pytest.mark.parametrize("a, b, expected", [
    (1, 1, 2),
    (2, 3, 5),
    (10, 20, 30),
    (-1, 1, 0),
])
def test_add(a, b, expected):
    """Test with multiple parameters"""
    result = add(a, b)
    assert result == expected

# Output:
# test_add[1-1-2] PASSED
# test_add[2-3-5] PASSED
# test_add[10-20-30] PASSED
# test_add[-1-1-0] PASSED
\`\`\`

---

## Custom Test IDs

Make test output more readable with custom IDs:

### Without IDs (Default)

\`\`\`python
@pytest.mark.parametrize("currency, amount", [
    ("USD", 100.0),
    ("EUR", 85.0),
    ("GBP", 75.0),
])
def test_payment(currency, amount):
    assert amount > 0

# Output (hard to read):
# test_payment[USD-100.0] PASSED
# test_payment[EUR-85.0] PASSED
# test_payment[GBP-75.0] PASSED
\`\`\`

### With Custom IDs

\`\`\`python
@pytest.mark.parametrize(
    "currency, amount",
    [
        ("USD", 100.0),
        ("EUR", 85.0),
        ("GBP", 75.0),
    ],
    ids=["US Dollars", "Euros", "British Pounds"]
)
def test_payment(currency, amount):
    assert amount > 0

# Output (readable):
# test_payment[US Dollars] PASSED
# test_payment[Euros] PASSED
# test_payment[British Pounds] PASSED
\`\`\`

### Dynamic IDs with Function

\`\`\`python
def payment_id(payment):
    """Generate readable ID from payment data"""
    currency, amount = payment
    return f"{currency}-\${amount:.2f}"

@pytest.mark.parametrize(
    "currency, amount",
    [
        ("USD", 100.0),
        ("EUR", 85.0),
        ("GBP", 75.0),
    ],
    ids = payment_id
)
def test_payment(currency, amount):
    assert amount > 0

# Output:
# test_payment[USD - $100.00]PASSED
# test_payment[EUR - $85.00]PASSED
# test_payment[GBP - $75.00]PASSED
\`\`\`

---

## pytest.param for Advanced Control

Use \`pytest.param\` for per-parameter configuration:

### Skip Specific Cases

\`\`\`python
@pytest.mark.parametrize("value", [
    1,
    2,
    pytest.param(3, marks=pytest.mark.skip(reason="Known bug #123")),
    4,
])
def test_process(value):
    assert process(value) is not None

# Output:
# test_process[1] PASSED
# test_process[2] PASSED
# test_process[3] SKIPPED (Known bug #123)
# test_process[4] PASSED
\`\`\`

### Expected Failures

\`\`\`python
@pytest.mark.parametrize("value", [
    pytest.param(1, id="valid"),
    pytest.param(0, marks=pytest.mark.xfail(reason="Zero not supported"), id="zero"),
    pytest.param(-1, marks=pytest.mark.xfail, id="negative"),
])
def test_divide(value):
    result = 10 / value
    assert result > 0

# Output:
# test_divide[valid] PASSED
# test_divide[zero] XFAIL (expected to fail)
# test_divide[negative] XFAIL
\`\`\`

### Custom IDs with pytest.param

\`\`\`python
@pytest.mark.parametrize("a, b, expected", [
    pytest.param(2, 3, 5, id="positive numbers"),
    pytest.param(-2, -3, -5, id="negative numbers"),
    pytest.param(0, 0, 0, id="zeros"),
    pytest.param(1000000, 2000000, 3000000, id="large numbers"),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
\`\`\`

---

## Multiple Parametrize Decorators

Stack multiple \`@pytest.mark.parametrize\` for combinations:

\`\`\`python
@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
@pytest.mark.parametrize("amount", [10.0, 50.0, 100.0])
def test_payment(currency, amount):
    """Test all combinations: 3 currencies × 3 amounts = 9 tests"""
    payment = Payment(currency=currency, amount=amount)
    assert payment.is_valid()

# Output (9 tests):
# test_payment[10.0-USD] PASSED
# test_payment[10.0-EUR] PASSED
# test_payment[10.0-GBP] PASSED
# test_payment[50.0-USD] PASSED
# test_payment[50.0-EUR] PASSED
# test_payment[50.0-GBP] PASSED
# test_payment[100.0-USD] PASSED
# test_payment[100.0-EUR] PASSED
# test_payment[100.0-GBP] PASSED
\`\`\`

**Matrix expansion**: N decorators with M1, M2, ..., MN values = M1 × M2 × ... × MN tests

---

## Parametrizing Fixtures

Fixtures can also be parametrized:

\`\`\`python
@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database(request):
    """Fixture parametrized by database type"""
    db_type = request.param
    
    if db_type == "sqlite":
        db = SQLiteDB(":memory:")
    elif db_type == "postgresql":
        db = PostgreSQLDB("localhost")
    elif db_type == "mysql":
        db = MySQLDB("localhost")
    
    db.connect()
    yield db
    db.disconnect()

def test_query(database):
    """Test runs 3 times (once per database)"""
    result = database.query("SELECT 1")
    assert result is not None

def test_insert(database):
    """Also runs 3 times"""
    database.insert("users", {"name": "Alice"})
    result = database.query("SELECT * FROM users")
    assert len(result) == 1

# Output:
# test_query[sqlite] PASSED
# test_query[postgresql] PASSED
# test_query[mysql] PASSED
# test_insert[sqlite] PASSED
# test_insert[postgresql] PASSED
# test_insert[mysql] PASSED
# (6 tests total: 2 test functions × 3 database types)
\`\`\`

**Power**: Every test using \`database\` fixture runs with all database types.

---

## Indirect Parametrization

Use \`indirect=True\` to pass parameters through fixtures:

\`\`\`python
@pytest.fixture
def user(request):
    """Fixture that creates user from parameter"""
    name = request.param
    user = User(name=name)
    db.add(user)
    db.commit()
    return user

@pytest.mark.parametrize("user", ["Alice", "Bob", "Charlie"], indirect=True)
def test_user_creation(user):
    """user parameter goes through user() fixture"""
    assert user.id is not None
    assert user.name in ["Alice", "Bob", "Charlie"]

# Execution flow:
# 1. pytest calls user("Alice") fixture → creates User("Alice")
# 2. Test runs with User("Alice")
# 3. pytest calls user("Bob") fixture → creates User("Bob")
# 4. Test runs with User("Bob")
# ... etc
\`\`\`

### Partial Indirect

Mix direct and indirect parameters:

\`\`\`python
@pytest.fixture
def database(request):
    db_type = request.param
    return create_database(db_type)

@pytest.mark.parametrize(
    "database, query",
    [
        ("sqlite", "SELECT 1"),
        ("postgresql", "SELECT NOW()"),
        ("mysql", "SELECT VERSION()"),
    ],
    indirect=["database"]  # Only database goes through fixture
)
def test_query(database, query):
    """database indirect, query direct"""
    result = database.execute(query)
    assert result is not None
\`\`\`

---

## Real-World Examples

### Example 1: Testing HTTP Status Codes

\`\`\`python
@pytest.mark.parametrize("endpoint, expected_status", [
    ("/", 200),
    ("/users", 200),
    ("/users/1", 200),
    ("/users/999999", 404),
    ("/admin", 401),
    ("/api/data", 200),
    ("/nonexistent", 404),
])
def test_endpoint_status(api_client, endpoint, expected_status):
    """Test API endpoints return correct status codes"""
    response = api_client.get(endpoint)
    assert response.status_code == expected_status
\`\`\`

### Example 2: Input Validation

\`\`\`python
@pytest.mark.parametrize("email, is_valid", [
    ("alice@example.com", True),
    ("bob@test.co.uk", True),
    ("invalid", False),
    ("@example.com", False),
    ("alice@", False),
    ("alice@.com", False),
    ("", False),
    ("alice+tag@example.com", True),
])
def test_email_validation(email, is_valid):
    """Test email validation logic"""
    result = validate_email(email)
    assert result == is_valid
\`\`\`

### Example 3: Edge Cases

\`\`\`python
@pytest.mark.parametrize("amount, currency, should_succeed", [
    (100.0, "USD", True),      # Valid
    (0.01, "USD", True),       # Minimum
    (0.00, "USD", False),      # Zero (invalid)
    (-100.0, "USD", False),    # Negative (invalid)
    (999999.99, "USD", True),  # Large (valid)
    (1000000.00, "USD", False),# Too large (invalid)
    (100.0, "XXX", False),     # Invalid currency
])
def test_payment_validation(amount, currency, should_succeed):
    """Test payment validation edge cases"""
    if should_succeed:
        payment = Payment(amount=amount, currency=currency)
        assert payment.validate() is True
    else:
        with pytest.raises(ValueError):
            Payment(amount=amount, currency=currency)
\`\`\`

### Example 4: Mathematical Properties

\`\`\`python
@pytest.mark.parametrize("a, b", [
    (1, 2),
    (5, 10),
    (-3, 7),
    (0, 0),
    (1000, -500),
])
def test_addition_commutative(a, b):
    """Test addition is commutative: a + b == b + a"""
    assert add(a, b) == add(b, a)

@pytest.mark.parametrize("a, b, c", [
    (1, 2, 3),
    (5, 10, 15),
    (-1, 0, 1),
])
def test_addition_associative(a, b, c):
    """Test addition is associative: (a + b) + c == a + (b + c)"""
    assert add(add(a, b), c) == add(a, add(b, c))
\`\`\`

### Example 5: Date/Time Testing

\`\`\`python
from datetime import datetime

@pytest.mark.parametrize("date_str, is_valid", [
    ("2024-01-01", True),
    ("2024-12-31", True),
    ("2024-02-29", True),  # Leap year
    ("2023-02-29", False), # Not leap year
    ("2024-13-01", False), # Invalid month
    ("2024-01-32", False), # Invalid day
    ("invalid", False),
])
def test_date_parsing(date_str, is_valid):
    """Test date parsing with various formats"""
    if is_valid:
        result = parse_date(date_str)
        assert isinstance(result, datetime)
    else:
        with pytest.raises(ValueError):
            parse_date(date_str)
\`\`\`

### Example 6: File Processing

\`\`\`python
@pytest.mark.parametrize("filename, expected_lines", [
    ("empty.txt", 0),
    ("single_line.txt", 1),
    ("multi_line.txt", 5),
    ("large_file.txt", 10000),
])
def test_file_line_count(tmp_path, filename, expected_lines):
    """Test counting lines in files"""
    # Create test file
    file_path = tmp_path / filename
    file_path.write_text("\\n" * expected_lines)
    
    # Test
    result = count_lines(file_path)
    assert result == expected_lines
\`\`\`

---

## Best Practices

### 1. Meaningful Test IDs

✅ **Good**: Clear what's being tested

\`\`\`python
@pytest.mark.parametrize("value, expected", [
    pytest.param(100, True, id="valid amount"),
    pytest.param(0, False, id="zero amount"),
    pytest.param(-100, False, id="negative amount"),
])
def test_amount_validation(value, expected):
    ...
\`\`\`

❌ **Bad**: Unclear IDs

\`\`\`python
@pytest.mark.parametrize("value, expected", [
    (100, True),   # test[100-True] - what does this mean?
    (0, False),
    (-100, False),
])
\`\`\`

### 2. Group Related Test Cases

✅ **Good**: Organized by category

\`\`\`python
valid_emails = [
    "alice@example.com",
    "bob@test.co.uk",
    "charlie+tag@example.com",
]

invalid_emails = [
    "invalid",
    "@example.com",
    "alice@",
]

@pytest.mark.parametrize("email", valid_emails)
def test_valid_emails(email):
    assert validate_email(email) is True

@pytest.mark.parametrize("email", invalid_emails)
def test_invalid_emails(email):
    assert validate_email(email) is False
\`\`\`

### 3. Don't Over-Parametrize

❌ **Bad**: Too many parameters, hard to understand

\`\`\`python
@pytest.mark.parametrize("a, b, c, d, e, f, g, expected", [
    (1, 2, 3, 4, 5, 6, 7, 28),
    (2, 3, 4, 5, 6, 7, 8, 35),
    # What are these parameters? Hard to read!
])
\`\`\`

✅ **Good**: Use dataclasses or dicts for many parameters

\`\`\`python
from dataclasses import dataclass

@dataclass
class TestCase:
    name: str
    inputs: dict
    expected: int

@pytest.mark.parametrize("test_case", [
    TestCase("case 1", {"a": 1, "b": 2, "c": 3}, 6),
    TestCase("case 2", {"a": 2, "b": 3, "c": 4}, 9),
], ids=lambda tc: tc.name)
def test_with_dataclass(test_case):
    result = process(**test_case.inputs)
    assert result == test_case.expected
\`\`\`

### 4. Test One Thing Per Parametrization

✅ **Good**: Each parametrized test tests one aspect

\`\`\`python
@pytest.mark.parametrize("amount", [10.0, 50.0, 100.0])
def test_payment_amount_accepted(amount):
    """Test various amounts are accepted"""
    payment = Payment(amount=amount)
    assert payment.validate()

@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
def test_payment_currency_accepted(currency):
    """Test various currencies are accepted"""
    payment = Payment(amount=100.0, currency=currency)
    assert payment.validate()
\`\`\`

❌ **Bad**: Parametrizing unrelated aspects together

\`\`\`python
@pytest.mark.parametrize("amount, currency, status", [
    (10.0, "USD", "pending"),
    (50.0, "EUR", "completed"),
    # What's being tested? Amount? Currency? Status? Unclear!
])
\`\`\`

### 5. Document Expected Behavior

\`\`\`python
@pytest.mark.parametrize("age, can_vote", [
    (17, False),  # Below voting age
    (18, True),   # Minimum voting age
    (65, True),   # Above voting age
    (150, True),  # Edge case: extremely high age
])
def test_voting_eligibility(age, can_vote):
    """Test voting eligibility by age"""
    result = is_eligible_to_vote(age)
    assert result == can_vote
\`\`\`

---

## Combining Parametrization Techniques

### Example: Complex Payment Testing

\`\`\`python
import pytest
from decimal import Decimal

# Test data
valid_amounts = [
    pytest.param(Decimal("10.00"), id="small"),
    pytest.param(Decimal("100.00"), id="medium"),
    pytest.param(Decimal("10000.00"), id="large"),
]

invalid_amounts = [
    pytest.param(Decimal("0.00"), id="zero"),
    pytest.param(Decimal("-100.00"), id="negative"),
    pytest.param(Decimal("1000000.00"), id="too_large"),
]

currencies = ["USD", "EUR", "GBP"]

# Parametrized fixture
@pytest.fixture(params=currencies)
def currency(request):
    """Parametrized currency fixture"""
    return request.param

# Tests
@pytest.mark.parametrize("amount", valid_amounts)
def test_valid_payment(currency, amount):
    """Test valid payments for all currencies"""
    payment = Payment(amount=amount, currency=currency)
    assert payment.validate() is True

@pytest.mark.parametrize("amount", invalid_amounts)
def test_invalid_payment(currency, amount):
    """Test invalid payments for all currencies"""
    with pytest.raises(ValueError):
        Payment(amount=amount, currency=currency)

# Runs:
# test_valid_payment[USD-small] PASSED
# test_valid_payment[USD-medium] PASSED
# test_valid_payment[USD-large] PASSED
# test_valid_payment[EUR-small] PASSED
# test_valid_payment[EUR-medium] PASSED
# test_valid_payment[EUR-large] PASSED
# test_valid_payment[GBP-small] PASSED
# test_valid_payment[GBP-medium] PASSED
# test_valid_payment[GBP-large] PASSED
# (3 amounts × 3 currencies = 9 tests for valid)
# (3 amounts × 3 currencies = 9 tests for invalid)
# Total: 18 tests from 2 test functions
\`\`\`

---

## Performance Considerations

### Problem: Slow Parametrized Tests

\`\`\`python
@pytest.mark.parametrize("value", range(1000))  # 1000 tests!
def test_slow_operation(value):
    """Each test takes 1 second"""
    time.sleep(1)  # Simulate slow operation
    assert process(value) is not None

# Total time: 1000 seconds (16.7 minutes) serial
\`\`\`

### Solution 1: Reduce Test Cases

\`\`\`python
# Use representative sample instead of all values
@pytest.mark.parametrize("value", [1, 10, 100, 500, 999])
def test_slow_operation(value):
    """Test with representative values"""
    time.sleep(1)
    assert process(value) is not None

# Total time: 5 seconds (200× faster)
\`\`\`

### Solution 2: Parallelize with pytest-xdist

\`\`\`bash
# Run tests in parallel
pytest -n 8  # Use 8 CPU cores

# 1000 tests / 8 cores = 125 seconds (vs 1000 seconds)
\`\`\`

### Solution 3: Mark Slow Tests

\`\`\`python
@pytest.mark.slow
@pytest.mark.parametrize("value", range(1000))
def test_comprehensive(value):
    """Comprehensive test suite (run nightly)"""
    ...

# Run fast tests only (for development)
pytest -m "not slow"

# Run all tests (for CI)
pytest
\`\`\`

---

## Summary

**Parametrized tests eliminate duplication**:
- **Single decorator**: Test same logic with different inputs
- **Multiple decorators**: Test all combinations (matrix)
- **Custom IDs**: Make test output readable
- **pytest.param**: Skip/xfail specific cases
- **Parametrized fixtures**: All tests using fixture run with all parameters
- **Indirect parametrization**: Pass parameters through fixtures

**Benefits**:
- 60-90% less code duplication
- Easier to add new test cases (add one line vs. write new function)
- Better test coverage (more cases tested)
- Clearer test intent (all variations in one place)

**Best practices**:
- Use meaningful IDs
- Group related cases
- Don't over-parametrize (use dataclasses for many parameters)
- Test one thing per parametrization
- Document expected behavior with comments

**Impact**:
- Write 5 test cases in 7 lines instead of 20 lines
- Add new test case: 1 line instead of 4 lines
- Maintain one test function instead of 10+

Master parametrization, and you'll write **concise, maintainable, comprehensive test suites**.
`,
};
