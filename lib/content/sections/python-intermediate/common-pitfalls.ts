/**
 * Common Python Pitfalls Section
 */

export const commonpitfallsSection = {
  id: 'common-pitfalls',
  title: 'Common Python Pitfalls',
  content: `# Common Python Pitfalls

Understanding common mistakes helps you avoid them and debug faster.

## 1. Mutable Default Arguments

### The Problem

\`\`\`python
# ❌ BAD - Mutable default argument
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2]  # Unexpected!
print(add_item(3))  # [1, 2, 3]  # Same list!

# Why? Default [] is created once when function is defined,
# not each time function is called
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Use None and create new list inside
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [2]  # New list each time
print(add_item(3))  # [3]
\`\`\`

**Rule:** Never use mutable objects (\`list\`, \`dict\`, \`set\`) as default arguments. Use \`None\` instead.

---

## 2. Late Binding Closures

### The Problem

\`\`\`python
# ❌ BAD - All functions use final value of i
functions = []
for i in range(3):
    functions.append(lambda: i)

print([f() for f in functions])  # [2, 2, 2]
# Expected [0, 1, 2] but all return 2!

# Why? Lambda captures i by reference, not value
# By the time lambda is called, loop has finished and i=2
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Capture i by value using default argument
functions = []
for i in range(3):
    functions.append(lambda x=i: x)  # x=i captures current value

print([f() for f in functions])  # [0, 1, 2]

# Alternative: Use functools.partial
from functools import partial
functions = [partial(lambda x: x, i) for i in range(3)]
\`\`\`

---

## 3. Modifying List While Iterating

### The Problem

\`\`\`python
# ❌ BAD - Modifying list during iteration
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 4, 5]  # 4 still there!
# Iterator gets confused when list changes
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Iterate over copy
numbers = [1, 2, 3, 4, 5]
for num in numbers[:]:  # Slice creates copy
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 5]

# ✅ BETTER - List comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
\`\`\`

---

## 4. Integer Division Gotchas

\`\`\`python
# Python 2 vs Python 3
# ❌ In Python 2: 5 / 2 = 2 (integer division)
# ✅ In Python 3: 5 / 2 = 2.5 (float division)

# Use // for integer division in both versions
print(5 // 2)   # 2
print(5 / 2)    # 2.5

# Be careful with negative numbers
print(-7 // 2)  # -4 (not -3!)
# Python rounds toward negative infinity, not toward zero

# For ceiling division
import math
print(math.ceil(7 / 2))  # 4
\`\`\`

---

## 5. Shallow vs Deep Copy

\`\`\`python
# ❌ Assignment doesn't copy, just creates reference
list1 = [[1, 2], [3, 4]]
list2 = list1
list2[0][0] = 99
print(list1)  # [[99, 2], [3, 4]]  # Original changed!

# ✅ Shallow copy (copies outer list only)
import copy
list1 = [[1, 2], [3, 4]]
list2 = copy.copy(list1)  # or list1.copy() or list1[:]
list2[0][0] = 99
print(list1)  # [[99, 2], [3, 4]]  # Inner lists still shared!

# ✅ Deep copy (copies everything)
list1 = [[1, 2], [3, 4]]
list2 = copy.deepcopy(list1)
list2[0][0] = 99
print(list1)  # [[1, 2], [3, 4]]  # Original unchanged
\`\`\`

---

## 6. Name Shadowing

\`\`\`python
# ❌ BAD - Shadowing built-in names
list = [1, 2, 3]  # Shadows built-in list!
# list([1, 2, 3])  # TypeError! Can't use list() anymore

# ✅ GOOD - Use different names
my_list = [1, 2, 3]

# Common names NOT to shadow:
# list, dict, set, str, int, float, bool, type, id, sum, min, max, all, any
\`\`\`

---

## 7. String Concatenation in Loops

\`\`\`python
# ❌ BAD - O(n²) due to string immutability
result = ""
for i in range(10000):
    result += str(i)  # Creates new string each time!

# ✅ GOOD - O(n) using list and join
parts = []
for i in range(10000):
    parts.append(str(i))
result = '.join(parts)

# ✅ BEST - List comprehension
result = '.join([str(i) for i in range(10000)])
\`\`\`

---

## 8. Forgetting to Return

\`\`\`python
# ❌ BAD - Forgot return statement
def add(a, b):
    result = a + b  # Calculated but not returned!

print(add(2, 3))  # None

# ✅ GOOD - Return the result
def add(a, b):
    return a + b

print(add(2, 3))  # 5
\`\`\`

---

## 9. Using \`is\` Instead of \`==\`

\`\`\`python
# is checks identity (same object), == checks equality (same value)

a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True (same values)
print(a is b)  # False (different objects)

# ❌ Common mistake with integers
a = 1000
b = 1000
print(a is b)  # False! (CPython caches -5 to 256 only)

# ✅ Use == for value comparison
print(a == b)  # True

# ✅ Use is only for None, True, False
if value is None:
    pass
\`\`\`

---

## 10. Catching All Exceptions

\`\`\`python
# ❌ BAD - Catches everything, even KeyboardInterrupt
try:
    risky_operation()
except:  # Bare except catches ALL exceptions!
    pass

# ✅ GOOD - Catch specific exceptions
try:
    risky_operation()
except ValueError:
    handle_value_error()
except KeyError:
    handle_key_error()

# ✅ If you must catch all, use Exception
try:
    risky_operation()
except Exception as e:
    log_error(e)
    # Still allows KeyboardInterrupt, SystemExit to pass through
\`\`\`

---

## 11. Mixing Tabs and Spaces

\`\`\`python
# ❌ BAD - Invisible but causes IndentationError
def func():
    if True:
→   print("tab")    # Uses tab
        print("spaces")  # Uses spaces
# IndentationError: inconsistent use of tabs and spaces

# ✅ GOOD - Use spaces consistently (4 spaces per PEP 8)
def func():
    if True:
        print("spaces")
        print("spaces")
\`\`\`

---

## 12. Circular Imports

\`\`\`python
# module_a.py
from module_b import func_b

def func_a():
    return func_b()

# module_b.py
from module_a import func_a  # Circular import!

def func_b():
    return func_a()

# ❌ ImportError: cannot import name 'func_a'

# ✅ FIX: Import inside function (lazy import)
# module_b.py
def func_b():
    from module_a import func_a  # Import when called
    return func_a()
\`\`\`

---

## 13. Global Variables

\`\`\`python
# ❌ BAD - Modifying global without declaration
count = 0

def increment():
    count = count + 1  # UnboundLocalError!
    # Python sees assignment, treats count as local

# ✅ GOOD - Declare global
count = 0

def increment():
    global count
    count = count + 1

# ✅ BETTER - Avoid globals, use return values
def increment(count):
    return count + 1

count = increment(count)
\`\`\`

---

## How to Avoid Pitfalls

1. **Use linters:** \`pylint\`, \`flake8\`, \`mypy\`
2. **Follow PEP 8:** Python style guide
3. **Write tests:** Catch bugs early
4. **Code reviews:** Learn from others
5. **Read error messages:** They're usually clear
6. **Use IDE warnings:** They catch many issues
7. **Keep learning:** Python has many gotchas

## Interview Relevance

Interviewers may:
- Introduce these bugs in code review questions
- Ask "what's wrong with this code?"
- Test if you can spot issues quickly

Know these pitfalls to debug faster and write cleaner code!`,
};
