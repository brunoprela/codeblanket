export const propertyBasedTesting = {
  title: 'Property-Based Testing',
  id: 'property-based-testing',
  content: `
# Property-Based Testing

## Introduction

**Property-based testing verifies code properties hold for ALL inputs**, not just specific examples. Instead of testing add(2, 3) == 5, test "add is commutative for all integers". **Hypothesis** is Python\'s property-based testing library.

---

## Example-Based vs Property-Based

**Example-based (traditional)**:
\`\`\`python
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
# Tests 3 specific cases
\`\`\`

**Property-based**:
\`\`\`python
from hypothesis import given
import hypothesis.strategies as st

@given (st.integers(), st.integers())
def test_add_commutative (a, b):
    assert add (a, b) == add (b, a)
# Tests 100 random integer pairs
\`\`\`

---

## Installing Hypothesis

\`\`\`bash
pip install hypothesis
\`\`\`

---

## Basic Property Tests

\`\`\`python
from hypothesis import given
import hypothesis.strategies as st

@given (st.integers())
def test_absolute_value_positive (n):
    """abs() always returns positive"""
    assert abs (n) >= 0

@given (st.text())
def test_reverse_twice_is_identity (s):
    """Reversing twice returns original"""
    assert reverse (reverse (s)) == s
\`\`\`

---

## Common Strategies

\`\`\`python
import hypothesis.strategies as st

# Built-in types
st.integers()           # Any integer
st.integers (min_value=0, max_value=100)  # 0-100
st.floats()             # Floats
st.text()               # Unicode strings
st.booleans()           # True/False

# Collections
st.lists (st.integers())           # List of integers
st.dictionaries (st.text(), st.integers())  # Dict[str, int]

# Custom
st.builds(User, username=st.text(), age=st.integers (min_value=0))
\`\`\`

---

## Finding Bugs with Hypothesis

**Hypothesis shrinks failing examples**:

\`\`\`python
@given (st.lists (st.integers()))
def test_sort_is_sorted (lst):
    sorted_lst = sorted (lst)
    for i in range (len (sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]

# Hypothesis tries 100 random lists
# If one fails, shrinks to minimal example: [1, 0]
\`\`\`

---

## Properties to Test

1. **Commutativity**: f (a, b) == f (b, a)
2. **Associativity**: f (f(a, b), c) == f (a, f (b, c))
3. **Identity**: f (a, identity) == a
4. **Idempotence**: f (f(a)) == f (a)
5. **Inverse**: f (inverse (a)) == identity

---

## Realistic Example: Testing Sorting

\`\`\`python
@given (st.lists (st.integers()))
def test_sort_properties (lst):
    sorted_lst = my_sort (lst)
    
    # Property 1: Same length
    assert len (sorted_lst) == len (lst)
    
    # Property 2: Same elements
    assert sorted (sorted_lst) == sorted (lst)
    
    # Property 3: Is sorted
    for i in range (len (sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]
\`\`\`

---

## Best Practices

1. **Start with simple properties** (length preservation, commutativity)
2. **Use example-based tests too** (property-based complements, doesn't replace)
3. **Fix shrunk examples** when found
4. **Set reasonable limits** (max_examples=1000 for thorough testing)
5. **Test properties, not implementations**

---

## Summary

**Property-based testing**:
- **Tests properties** for many inputs
- **Hypothesis library**: @given decorator with strategies
- **Finds edge cases** you wouldn't think of
- **Shrinks failures** to minimal examples

**Use for**: Algorithms, data transformations, parsers, business rules

Property-based testing **catches bugs example-based tests miss**.
`,
};
