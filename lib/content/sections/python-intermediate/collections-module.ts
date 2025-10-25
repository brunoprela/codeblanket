/**
 * Collections Module - Advanced Data Structures Section
 */

export const collectionsmoduleSection = {
  id: 'collections-module',
  title: 'Collections Module - Advanced Data Structures',
  content: `# Collections Module - Advanced Data Structures

The \`collections\` module provides specialized container datatypes beyond the built-in list, dict, tuple, and set. These are essential for interviews and production code.

## Counter - Frequency Counting

\`Counter\` is a dict subclass for counting hashable objects. **Extremely common in coding interviews!**

### Basic Usage

\`\`\`python
from collections import Counter

# Count elements in list
fruits = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counts = Counter (fruits)
print(counts)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Access counts
print(counts['apple'])  # 3
print(counts['grape'])  # 0 (no KeyError!)

# Count characters in string
text = 'hello world'
char_counts = Counter (text)
print(char_counts)  # Counter({'l': 3, 'o': 2, 'h': 1, ...})
\`\`\`

### Most Common Elements

\`\`\`python
# Get most common
numbers = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
counter = Counter (numbers)

# Top 2 most common
print(counter.most_common(2))  # [(4, 4), (1, 3)]

# All elements ordered by frequency
print(counter.most_common())  # [(4, 4), (1, 3), (2, 2), (3, 1)]
\`\`\`

### Counter Operations

\`\`\`python
c1 = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = Counter(['a', 'b', 'b', 'd'])

# Addition
print(c1 + c2)  # Counter({'b': 5, 'a': 3, 'c': 1, 'd': 1})

# Subtraction (keeps only positive)
print(c1 - c2)  # Counter({'b': 1, 'c': 1, 'a': 0})

# Intersection (min of counts)
print(c1 & c2)  # Counter({'b': 2, 'a': 1})

# Union (max of counts)
print(c1 | c2)  # Counter({'b': 3, 'a': 2, 'c': 1, 'd': 1})
\`\`\`

### Interview Applications

\`\`\`python
# Check if two strings are anagrams
def is_anagram (s1, s2):
    return Counter (s1) == Counter (s2)

print(is_anagram('listen', 'silent'))  # True

# Find first non-repeating character
def first_unique_char (s):
    counts = Counter (s)
    for char in s:
        if counts[char] == 1:
            return char
    return None

print(first_unique_char('leetcode'))  # 'l'

# Top K frequent elements
def top_k_frequent (nums, k):
    counter = Counter (nums)
    return [num for num, count in counter.most_common (k)]

print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]
\`\`\`

---

## defaultdict - No More KeyError

\`defaultdict\` is a dict subclass that provides default values for missing keys.

### Basic Usage

\`\`\`python
from collections import defaultdict

# Regular dict - KeyError
regular = {}
# regular['key'].append('value')  # KeyError!

# defaultdict with list
d = defaultdict (list)
d['key'].append('value')  # Works! Auto-creates empty list
print(d)  # defaultdict(<class 'list'>, {'key': ['value']})

# defaultdict with int (default 0)
counts = defaultdict (int)
counts['a'] += 1  # Works! Starts from 0
counts['b'] += 1
print(counts)  # defaultdict(<class 'int'>, {'a': 1, 'b': 1})

# defaultdict with set
groups = defaultdict (set)
groups['team1'].add('Alice')
groups['team1'].add('Bob')
print(groups)  # defaultdict(<class 'set'>, {'team1': {'Alice', 'Bob'}})
\`\`\`

### Common Use Cases

**1. Grouping items:**
\`\`\`python
# Group words by first letter
words = ['apple', 'apricot', 'banana', 'blueberry', 'cherry']
groups = defaultdict (list)

for word in words:
    groups[word[0]].append (word)

print(dict (groups))
# {'a': ['apple', 'apricot'], 
#  'b': ['banana', 'blueberry'], 
#  'c': ['cherry']}
\`\`\`

**2. Graph adjacency list:**
\`\`\`python
# Build graph
graph = defaultdict (list)
edges = [(1, 2), (1, 3), (2, 4), (3, 4)]

for u, v in edges:
    graph[u].append (v)
    graph[v].append (u)  # Undirected

print(dict (graph))
# {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
\`\`\`

**3. Counting with categories:**
\`\`\`python
# Track scores by player
scores = defaultdict (list)
scores['Alice'].append(10)
scores['Bob'].append(15)
scores['Alice'].append(20)

# Calculate averages
for player, score_list in scores.items():
    avg = sum (score_list) / len (score_list)
    print(f"{player}: {avg}")
\`\`\`

### Factory Functions

\`\`\`python
# Default to specific value
d = defaultdict (lambda: 'N/A')
print(d['missing'])  # 'N/A'

# Default to 0
counts = defaultdict (int)

# Default to empty dict
nested = defaultdict (dict)
nested['level1']['level2'] = 'value'
\`\`\`

---

## deque - Double-Ended Queue

\`deque\` (pronounced "deck") is optimized for fast appends/pops from both ends. **Essential for queues and sliding windows!**

### Why deque?

\`\`\`python
# List: O(n) to pop from front
my_list = [1, 2, 3, 4, 5]
my_list.pop(0)  # O(n) - shifts all elements!

# deque: O(1) for both ends
from collections import deque
my_deque = deque([1, 2, 3, 4, 5])
my_deque.popleft()  # O(1) - efficient!
\`\`\`

### Basic Operations

\`\`\`python
from collections import deque

# Create deque
dq = deque([1, 2, 3])

# Add to right
dq.append(4)  # [1, 2, 3, 4]

# Add to left
dq.appendleft(0)  # [0, 1, 2, 3, 4]

# Remove from right
dq.pop()  # Returns 4, deque: [0, 1, 2, 3]

# Remove from left
dq.popleft()  # Returns 0, deque: [1, 2, 3]

# Extend both ends
dq.extend([4, 5])  # [1, 2, 3, 4, 5]
dq.extendleft([0, -1])  # [-1, 0, 1, 2, 3, 4, 5]
\`\`\`

### Queue Implementation

\`\`\`python
# Perfect for BFS queues
queue = deque()
queue.append(1)  # Enqueue
queue.append(2)
first = queue.popleft()  # Dequeue - O(1)!
\`\`\`

### Sliding Window Maximum

\`\`\`python
def max_sliding_window (nums, k):
    """Find max in each sliding window"""
    dq = deque()  # Store indices
    result = []
    
    for i in range (len (nums)):
        # Remove out-of-window indices
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append (i)
        
        if i >= k - 1:
            result.append (nums[dq[0]])
    
    return result

print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
\`\`\`

### Rotation

\`\`\`python
dq = deque([1, 2, 3, 4, 5])

# Rotate right
dq.rotate(2)  # [4, 5, 1, 2, 3]

# Rotate left
dq.rotate(-2)  # [1, 2, 3, 4, 5]
\`\`\`

### Max Length

\`\`\`python
# Limited-size deque (circular buffer)
dq = deque (maxlen=3)
dq.append(1)
dq.append(2)
dq.append(3)
dq.append(4)  # Removes 1 automatically
print(dq)  # deque([2, 3, 4], maxlen=3)
\`\`\`

---

## OrderedDict - Remembers Insertion Order

**Note:** Python 3.7+ dicts maintain insertion order, but OrderedDict has extra features.

\`\`\`python
from collections import OrderedDict

# Maintains order
od = OrderedDict()
od['b'] = 2
od['a'] = 1
od['c'] = 3
print(list (od.keys()))  # ['b', 'a', 'c']

# Move to end
od.move_to_end('a')  # a moved to end
print(list (od.keys()))  # ['b', 'c', 'a']

# Move to beginning
od.move_to_end('a', last=False)
print(list (od.keys()))  # ['a', 'b', 'c']

# Pop last item
od.popitem (last=True)  # Remove from end
od.popitem (last=False)  # Remove from beginning
\`\`\`

### LRU Cache Implementation

\`\`\`python
class LRUCache:
    """Least Recently Used Cache"""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get (self, key):
        if key not in self.cache:
            return -1
        # Move to end (most recently used)
        self.cache.move_to_end (key)
        return self.cache[key]
    
    def put (self, key, value):
        if key in self.cache:
            self.cache.move_to_end (key)
        self.cache[key] = value
        if len (self.cache) > self.capacity:
            self.cache.popitem (last=False)  # Remove LRU
\`\`\`

---

## namedtuple - Lightweight Objects

Create simple classes without defining full class.

\`\`\`python
from collections import namedtuple

# Define structure
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', 'name age city')  # Space-separated

# Create instances
p1 = Point(10, 20)
person = Person('Alice', 30, 'NYC')

# Access by name or index
print(p1.x, p1.y)  # 10 20
print(p1[0], p1[1])  # 10 20

# Unpack
x, y = p1

# Immutable (like tuples)
# p1.x = 15  # AttributeError

# Convert to dict
print(person._asdict())
# {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Replace values (creates new instance)
person2 = person._replace (age=31)
\`\`\`

### Use Cases

\`\`\`python
# Function returns
def get_stats():
    Stats = namedtuple('Stats', 'mean median mode')
    return Stats (mean=10, median=9, mode=8)

stats = get_stats()
print(f"Mean: {stats.mean}")

# CSV/Database rows
Employee = namedtuple('Employee', 'id name department salary')
employees = [
    Employee(1, 'Alice', 'Engineering', 100000),
    Employee(2, 'Bob', 'Sales', 80000),
]

for emp in employees:
    print(f"{emp.name} in {emp.department}")
\`\`\`

---

## ChainMap - Combined Views

Combine multiple dicts into single view.

\`\`\`python
from collections import ChainMap

# Multiple dicts
defaults = {'color': 'red', 'user': 'guest'}
config = {'user': 'admin'}
cli_args = {'debug': True}

# Chain them (first dict takes priority)
combined = ChainMap (cli_args, config, defaults)

print(combined['color'])  # 'red' (from defaults)
print(combined['user'])   # 'admin' (from config, overrides defaults)
print(combined['debug'])  # True (from cli_args)

# Update
combined['user'] = 'root'  # Updates first dict (cli_args)

# Add new dict to front
combined = combined.new_child({'temp': 'value'})
\`\`\`

---

## Quick Reference

| Collection | Use Case | Key Feature |
|------------|----------|-------------|
| **Counter** | Frequency counting | \`most_common()\`, math operations |
| **defaultdict** | Auto-initialize missing keys | No KeyError |
| **deque** | Queue, stack, both ends | O(1) append/pop from both ends |
| **OrderedDict** | Order matters + operations | \`move_to_end()\`, \`popitem()\` |
| **namedtuple** | Lightweight objects | Named access, immutable |
| **ChainMap** | Multiple dicts | Layered lookups |

## Performance Comparison

\`\`\`python
import timeit

# List vs deque for queue operations
def list_queue():
    q = []
    for i in range(1000):
        q.append (i)
    for i in range(1000):
        q.pop(0)  # O(n) each time!

def deque_queue():
    from collections import deque
    q = deque()
    for i in range(1000):
        q.append (i)
    for i in range(1000):
        q.popleft()  # O(1) each time!

# deque is ~100x faster for this!
print(timeit.timeit (list_queue, number=100))
print(timeit.timeit (deque_queue, number=100))
\`\`\`

---

## Interview Patterns

**1. Use Counter for:**
- Anagrams
- Top K frequent elements
- Character frequency

**2. Use defaultdict for:**
- Grouping
- Graph adjacency lists
- Nested structures

**3. Use deque for:**
- BFS queues
- Sliding window
- Both-ends operations

**4. Use namedtuple for:**
- Coordinate pairs
- Return multiple values
- Lightweight objects

**Remember:** These are in Python standard library - no pip install needed!`,
};
