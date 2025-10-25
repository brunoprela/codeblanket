/**
 * Advanced Techniques Section
 */

export const advancedSection = {
  id: 'advanced',
  title: 'Advanced Techniques',
  content: `**Technique 1: Multiple Hash Tables**

Use multiple hash tables to track different properties:

\`\`\`python
def find_intersection (arr1, arr2):
    """Find common elements in O(n) time."""
    set1 = set (arr1)
    set2 = set (arr2)
    return list (set1 & set2)  # Set intersection
\`\`\`

**Technique 2: Hash Table as Visited Tracker**

Track what you've seen in O(1) space per element:

\`\`\`python
def has_cycle (arr):
    """Detect cycle using hash set."""
    visited = set()
    current = arr[0]
    
    while current not in visited:
        if current == END:
            return False
        visited.add (current)
        current = next_value (current)
    
    return True
\`\`\`

**Technique 3: Rolling Hash (Rabin-Karp)**

Efficient string matching using hash:

\`\`\`python
def rabin_karp (text, pattern):
    """Find pattern in text using rolling hash."""
    BASE = 256
    MOD = 10**9 + 7
    m, n = len (pattern), len (text)
    
    # Compute hash of pattern
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * BASE + ord (char)) % MOD
    
    # Rolling hash for text
    text_hash = 0
    for i in range (n):
        # Add new character
        text_hash = (text_hash * BASE + ord (text[i])) % MOD
        
        # Remove old character if window full
        if i >= m:
            text_hash = (text_hash - ord (text[i-m]) * pow(BASE, m, MOD)) % MOD
        
        # Check match
        if i >= m - 1 and text_hash == pattern_hash:
            if text[i-m+1:i+1] == pattern:
                return i - m + 1
    
    return -1
\`\`\`

**Technique 4: Coordinate Compression**

Map large values to small indices:

\`\`\`python
def compress_coordinates (arr):
    """Compress large values to 0, 1, 2, ..."""
    sorted_unique = sorted (set (arr))
    compress = {val: i for i, val in enumerate (sorted_unique)}
    return [compress[x] for x in arr]
\`\`\`

**Technique 5: Hashable Custom Keys**

Use tuples or strings as keys:

\`\`\`python
# For 2D points
seen = set()
seen.add((x, y))

# For lists (convert to tuple)
key = tuple (sorted (lst))
\`\`\``,
};
