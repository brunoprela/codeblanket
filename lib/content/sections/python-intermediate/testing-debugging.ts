/**
 * Testing & Debugging Section
 */

export const testingdebuggingSection = {
  id: 'testing-debugging',
  title: 'Testing & Debugging',
  content: `# Testing & Debugging

Writing tests and debugging effectively are critical skills for production code and interview success.

## Why Testing Matters

1. **Catches bugs early** - Find issues before they reach production
2. **Enables refactoring** - Change code confidently
3. **Documents behavior** - Tests serve as examples
4. **Interview advantage** - Shows professionalism and thoroughness

## Unit Testing with \`unittest\`

Python's built-in testing framework.

\`\`\`python
import unittest

def add(a, b):
    return a + b

class TestMathOperations(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)
    
    def test_add_zero(self):
        self.assertEqual(add(5, 0), 5)
    
    def test_add_floats(self):
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=7)

if __name__ == '__main__':
    unittest.main()
\`\`\`

### Common Assertions

\`\`\`python
# Equality
self.assertEqual(a, b)
self.assertNotEqual(a, b)

# Truth
self.assertTrue(x)
self.assertFalse(x)

# Identity
self.assertIs(a, b)       # Same object
self.assertIsNot(a, b)

# Membership
self.assertIn(item, collection)
self.assertNotIn(item, collection)

# Exceptions
with self.assertRaises(ValueError):
    function_that_raises()

# Floating point
self.assertAlmostEqual(a, b, places=7)

# None
self.assertIsNone(x)
self.assertIsNotNone(x)
\`\`\`

### Setup and Teardown

\`\`\`python
class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Run before each test"""
        self.db = Database('test.db')
        self.db.connect()
    
    def tearDown(self):
        """Run after each test"""
        self.db.close()
        os.remove('test.db')
    
    def test_insert(self):
        self.db.insert('key', 'value')
        self.assertEqual(self.db.get('key'), 'value')
\`\`\`

## \`pytest\` - Modern Testing

More concise and powerful than unittest.

\`\`\`python
# test_math.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(5, 0) == 5

# Parametrized tests
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, -1, -2),
    (0, 0, 0),
    (100, 200, 300),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
\`\`\`

### Fixtures (pytest)

\`\`\`python
@pytest.fixture
def sample_list():
    """Reusable test data"""
    return [1, 2, 3, 4, 5]

def test_length(sample_list):
    assert len(sample_list) == 5

def test_sum(sample_list):
    assert sum(sample_list) == 15

# Fixture with cleanup
@pytest.fixture
def temp_file():
    f = open('temp.txt', 'w')
    yield f  # Test runs here
    f.close()
    os.remove('temp.txt')
\`\`\`

## Debugging Techniques

### 1. Print Debugging

\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        print(f"Debug: left={left}, right={right}, mid={mid}, arr[mid]={arr[mid]}")
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
\`\`\`

### 2. Logging Module

Better than print statements for production code.

\`\`\`python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_data(data):
    logging.debug(f"Processing {len(data)} items")
    
    for item in data:
        try:
            result = expensive_operation(item)
            logging.info(f"Processed {item}: {result}")
        except Exception as e:
            logging.error(f"Failed to process {item}: {e}")

# Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
\`\`\`

### 3. Python Debugger (pdb)

Interactive debugging.

\`\`\`python
import pdb

def problematic_function(data):
    result = []
    for item in data:
        pdb.set_trace()  # Breakpoint here
        processed = item * 2
        result.append(processed)
    return result

# In pdb:
# n - next line
# s - step into function
# c - continue
# p variable - print variable
# l - list code
# q - quit
\`\`\`

### 4. Assert Statements

Check assumptions during development.

\`\`\`python
def divide(a, b):
    assert b != 0, "Division by zero!"
    return a / b

def binary_search(arr, target):
    assert len(arr) > 0, "Array must not be empty"
    assert all(arr[i] <= arr[i+1] for i in range(len(arr)-1)), "Array must be sorted"
    # ... implementation
\`\`\`

## Test-Driven Development (TDD)

Write tests before code!

\`\`\`python
# Step 1: Write test (it fails)
def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True

# Step 2: Write minimal code to pass
def is_palindrome(s):
    return s == s[::-1]

# Step 3: Refactor if needed
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
\`\`\`

## Edge Cases to Test

Always test:

1. **Empty input:** \`[], "", {}, None\`
2. **Single element:** \`[1], "a"\`
3. **Duplicates:** \`[1, 1, 1]\`
4. **Negative numbers:** \`[-1, -5]\`
5. **Large inputs:** Stress test
6. **Boundary values:** \`0, MAX_INT, MIN_INT\`
7. **Invalid input:** Wrong types, out of range

\`\`\`python
def test_find_max():
    # Normal case
    assert find_max([1, 5, 3]) == 5
    
    # Edge cases
    assert find_max([1]) == 1              # Single element
    assert find_max([-5, -1, -10]) == -1   # All negative
    assert find_max([5, 5, 5]) == 5        # All same
    
    # Error cases
    with pytest.raises(ValueError):
        find_max([])  # Empty list
\`\`\`

## Debugging Strategies

1. **Read error messages carefully**
   - Line number
   - Error type
   - Traceback

2. **Binary search debugging**
   - Comment out half the code
   - Find which half has the bug
   - Repeat

3. **Rubber duck debugging**
   - Explain code line-by-line to someone (or a rubber duck)
   - Often reveals the bug

4. **Minimal reproducible example**
   - Reduce to smallest code that shows the bug
   - Easier to understand and fix

5. **Check assumptions**
   - Is input what you expect?
   - Are variables the right type?
   - Is the algorithm correct?

## Interview Testing Tips

During interviews:

1. **Ask about testing expectations**
   - Should I write tests?
   - How thorough?

2. **Test as you code**
   - Test each piece immediately
   - Don't wait until the end

3. **Vocalize edge cases**
   - "What if the array is empty?"
   - "Should I handle negative numbers?"

4. **Manual test cases**
   - Walk through your code with example inputs
   - Trace variable values

5. **Think about failure modes**
   - What could go wrong?
   - What inputs would break this?

## Code Coverage

Measure how much code is tested.

\`\`\`bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m pytest

# View report
coverage report

# HTML report
coverage html
\`\`\`

Aim for >80% coverage, but quality > quantity!`,
};
