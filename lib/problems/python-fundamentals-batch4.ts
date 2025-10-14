/**
 * Python Fundamentals - Batch 4 (Problems 61-70)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch4: Problem[] = [
  {
    id: 'fundamentals-convert-base',
    title: 'Convert Number Base',
    difficulty: 'Medium',
    description: `Convert a number from one base to another (base 2-36).

Bases use digits 0-9 and letters A-Z for values 10-35.

**Example:** Convert "1010" from base 2 to base 10 → "10"

This tests:
- Number system conversion
- String manipulation
- Base arithmetic`,
    examples: [
      {
        input: 'num = "1010", from_base = 2, to_base = 10',
        output: '"10"',
      },
      {
        input: 'num = "FF", from_base = 16, to_base = 10',
        output: '"255"',
      },
    ],
    constraints: ['2 <= base <= 36', 'Valid input for given base'],
    hints: [
      'Convert to decimal first',
      'Then convert decimal to target base',
      'Use int(num, base) and string building',
    ],
    starterCode: `def convert_base(num, from_base, to_base):
    """
    Convert number between bases.
    
    Args:
        num: Number as string
        from_base: Source base (2-36)
        to_base: Target base (2-36)
        
    Returns:
        Converted number as string
        
    Examples:
        >>> convert_base("1010", 2, 10)
        "10"
    """
    pass


# Test
print(convert_base("FF", 16, 10))
`,
    testCases: [
      {
        input: ['1010', 2, 10],
        expected: '10',
      },
      {
        input: ['FF', 16, 10],
        expected: '255',
      },
    ],
    solution: `def convert_base(num, from_base, to_base):
    # Convert to decimal
    decimal = int(num, from_base)
    
    # Convert decimal to target base
    if decimal == 0:
        return "0"
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    
    while decimal > 0:
        result.append(digits[decimal % to_base])
        decimal //= to_base
    
    return ''.join(reversed(result))`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',
    order: 61,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-group-anagrams-simple',
    title: 'Group Anagrams (Simple)',
    difficulty: 'Medium',
    description: `Group strings that are anagrams of each other.

Anagrams are words made by rearranging letters.

**Example:** ["eat","tea","tan","ate","nat","bat"]
→ [["bat"],["nat","tan"],["ate","eat","tea"]]

This tests:
- String sorting
- Dictionary grouping
- List manipulation`,
    examples: [
      {
        input: 'strs = ["eat","tea","tan","ate","nat","bat"]',
        output: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
      },
    ],
    constraints: ['1 <= len(strs) <= 10^4', '0 <= len(strs[i]) <= 100'],
    hints: [
      'Sort each string to get signature',
      'Group strings with same signature',
      'Use defaultdict(list)',
    ],
    starterCode: `def group_anagrams(strs):
    """
    Group anagram strings together.
    
    Args:
        strs: List of strings
        
    Returns:
        List of grouped anagrams
        
    Examples:
        >>> group_anagrams(["eat","tea","tan"])
        [["eat","tea"],["tan"]]
    """
    pass


# Test
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
`,
    testCases: [
      {
        input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
        expected: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']],
      },
    ],
    solution: `def group_anagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Sort string to get signature
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())`,
    timeComplexity: 'O(n * k log k) where k is max string length',
    spaceComplexity: 'O(n * k)',
    order: 62,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-nim-game',
    title: 'Nim Game',
    difficulty: 'Easy',
    description: `You and your friend play Nim game with n stones.

Rules:
- Players take turns removing 1, 2, or 3 stones
- Player who removes the last stone wins
- You go first

Can you win if both play optimally?

**Key insight:** You lose if n is divisible by 4!

This tests:
- Game theory
- Mathematical pattern
- Modulo operation`,
    examples: [
      {
        input: 'n = 4',
        output: 'False',
        explanation: 'Any move leaves 1-3 stones for opponent to win',
      },
      {
        input: 'n = 5',
        output: 'True',
        explanation: 'Remove 1 stone, leave 4 for opponent',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'Think about losing positions',
      'n=4 is losing, why?',
      'Simple modulo check',
    ],
    starterCode: `def can_win_nim(n):
    """
    Check if you can win Nim game.
    
    Args:
        n: Number of stones
        
    Returns:
        True if you can guarantee win
        
    Examples:
        >>> can_win_nim(4)
        False
        >>> can_win_nim(5)
        True
    """
    pass


# Test
print(can_win_nim(4))
`,
    testCases: [
      {
        input: [4],
        expected: false,
      },
      {
        input: [5],
        expected: true,
      },
      {
        input: [1],
        expected: true,
      },
    ],
    solution: `def can_win_nim(n):
    return n % 4 != 0`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 63,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-bulls-and-cows',
    title: 'Bulls and Cows',
    difficulty: 'Medium',
    description: `Calculate hint for Bulls and Cows game.

- Bull: digit in correct position
- Cow: digit exists but wrong position

**Example:** secret="1807", guess="7810"
→ "1A3B" (1 bull: 8, 3 cows: 1,7,0)

This tests:
- Character counting
- Position matching
- String formatting`,
    examples: [
      {
        input: 'secret = "1807", guess = "7810"',
        output: '"1A3B"',
      },
      {
        input: 'secret = "1123", guess = "0111"',
        output: '"1A1B"',
      },
    ],
    constraints: ['1 <= len(secret) <= 1000', 'Only digits'],
    hints: [
      'Count bulls first (exact matches)',
      'Count remaining digit frequencies',
      'Cows = min(secret_count, guess_count)',
    ],
    starterCode: `def get_hint(secret, guess):
    """
    Get Bulls and Cows hint.
    
    Args:
        secret: Secret number string
        guess: Guessed number string
        
    Returns:
        Hint in format "xAyB"
        
    Examples:
        >>> get_hint("1807", "7810")
        "1A3B"
    """
    pass


# Test
print(get_hint("1807", "7810"))
`,
    testCases: [
      {
        input: ['1807', '7810'],
        expected: '1A3B',
      },
      {
        input: ['1123', '0111'],
        expected: '1A1B',
      },
    ],
    solution: `def get_hint(secret, guess):
    bulls = 0
    secret_counts = [0] * 10
    guess_counts = [0] * 10
    
    # Count bulls and non-bull digits
    for i in range(len(secret)):
        if secret[i] == guess[i]:
            bulls += 1
        else:
            secret_counts[int(secret[i])] += 1
            guess_counts[int(guess[i])] += 1
    
    # Count cows
    cows = 0
    for i in range(10):
        cows += min(secret_counts[i], guess_counts[i])
    
    return f"{bulls}A{cows}B"`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 64,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-zigzag-string',
    title: 'ZigZag Conversion',
    difficulty: 'Medium',
    description: `Convert string to zigzag pattern with given number of rows.

**Example:** "PAYPALISHIRING", 3 rows
\`\`\`
P   A   H   N
A P L S I I G
Y   I   R
\`\`\`
Read row by row: "PAHNAPLSIIGYIR"

This tests:
- Pattern recognition
- Array manipulation
- String building`,
    examples: [
      {
        input: 's = "PAYPALISHIRING", numRows = 3',
        output: '"PAHNAPLSIIGYIR"',
      },
      {
        input: 's = "PAYPALISHIRING", numRows = 4',
        output: '"PINALSIGYAHRPI"',
      },
    ],
    constraints: ['1 <= len(s) <= 1000', '1 <= numRows <= 1000'],
    hints: [
      'Create array of strings for each row',
      'Track current row and direction',
      'Change direction at top/bottom',
    ],
    starterCode: `def convert_zigzag(s, num_rows):
    """
    Convert string to zigzag pattern.
    
    Args:
        s: Input string
        num_rows: Number of rows
        
    Returns:
        String read row by row
        
    Examples:
        >>> convert_zigzag("PAYPALISHIRING", 3)
        "PAHNAPLSIIGYIR"
    """
    pass


# Test
print(convert_zigzag("PAYPALISHIRING", 3))
`,
    testCases: [
      {
        input: ['PAYPALISHIRING', 3],
        expected: 'PAHNAPLSIIGYIR',
      },
      {
        input: ['AB', 1],
        expected: 'AB',
      },
    ],
    solution: `def convert_zigzag(s, num_rows):
    if num_rows == 1 or num_rows >= len(s):
        return s
    
    rows = [''] * num_rows
    current_row = 0
    going_down = False
    
    for char in s:
        rows[current_row] += char
        
        if current_row == 0 or current_row == num_rows - 1:
            going_down = not going_down
        
        current_row += 1 if going_down else -1
    
    return ''.join(rows)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 65,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-reverse-bits',
    title: 'Reverse Bits',
    difficulty: 'Easy',
    description: `Reverse bits of a 32-bit unsigned integer.

**Example:** 43261596 (00000010100101000001111010011100)
→ 964176192 (00111001011110000010100101000000)

This tests:
- Bit manipulation
- Binary representation
- Bit shifting`,
    examples: [
      {
        input: 'n = 43261596',
        output: '964176192',
      },
    ],
    constraints: ['Input is 32-bit unsigned integer'],
    hints: [
      'Build result bit by bit',
      'Extract rightmost bit with n & 1',
      'Shift n right, result left',
    ],
    starterCode: `def reverse_bits(n):
    """
    Reverse bits of 32-bit integer.
    
    Args:
        n: 32-bit unsigned integer
        
    Returns:
        Integer with reversed bits
        
    Examples:
        >>> reverse_bits(43261596)
        964176192
    """
    pass


# Test
print(reverse_bits(43261596))
`,
    testCases: [
      {
        input: [43261596],
        expected: 964176192,
      },
    ],
    solution: `def reverse_bits(n):
    result = 0
    
    for i in range(32):
        # Extract rightmost bit
        bit = n & 1
        # Shift result left and add bit
        result = (result << 1) | bit
        # Shift n right
        n >>= 1
    
    return result


# Alternative using string conversion
def reverse_bits_string(n):
    binary = bin(n)[2:].zfill(32)
    reversed_binary = binary[::-1]
    return int(reversed_binary, 2)`,
    timeComplexity: 'O(1) - fixed 32 iterations',
    spaceComplexity: 'O(1)',
    order: 66,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-number-of-1-bits',
    title: 'Number of 1 Bits (Hamming Weight)',
    difficulty: 'Easy',
    description: `Count the number of 1 bits in a binary representation.

Also known as the Hamming weight.

**Example:** 11 = 1011 → 3 ones

**Trick:** n & (n-1) removes rightmost 1 bit

This tests:
- Bit manipulation
- Counting
- Binary representation`,
    examples: [
      {
        input: 'n = 11',
        output: '3',
        explanation: '1011 has three 1s',
      },
      {
        input: 'n = 128',
        output: '1',
        explanation: '10000000 has one 1',
      },
    ],
    constraints: ['0 <= n <= 2^31 - 1'],
    hints: [
      'Use n & 1 to check last bit',
      'Or use n & (n-1) trick',
      'Or use bin(n).count("1")',
    ],
    starterCode: `def hamming_weight(n):
    """
    Count number of 1 bits.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Count of 1 bits
        
    Examples:
        >>> hamming_weight(11)
        3
    """
    pass


# Test
print(hamming_weight(11))
`,
    testCases: [
      {
        input: [11],
        expected: 3,
      },
      {
        input: [128],
        expected: 1,
      },
      {
        input: [0],
        expected: 0,
      },
    ],
    solution: `def hamming_weight(n):
    count = 0
    
    while n:
        count += 1
        n &= n - 1  # Remove rightmost 1 bit
    
    return count


# Alternative checking each bit
def hamming_weight_simple(n):
    count = 0
    
    while n:
        count += n & 1
        n >>= 1
    
    return count


# One-liner
def hamming_weight_oneliner(n):
    return bin(n).count('1')`,
    timeComplexity: 'O(1) - at most 32 bits',
    spaceComplexity: 'O(1)',
    order: 67,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-count-primes',
    title: 'Count Primes Below N',
    difficulty: 'Medium',
    description: `Count the number of prime numbers less than n.

Use Sieve of Eratosthenes for efficient solution.

**Example:** n = 10 → 4 primes (2, 3, 5, 7)

This tests:
- Sieve algorithm
- Boolean array
- Prime generation`,
    examples: [
      {
        input: 'n = 10',
        output: '4',
        explanation: '2, 3, 5, 7',
      },
      {
        input: 'n = 0',
        output: '0',
      },
    ],
    constraints: ['0 <= n <= 5*10^6'],
    hints: [
      'Use Sieve of Eratosthenes',
      'Mark multiples of each prime',
      'Count unmarked numbers',
    ],
    starterCode: `def count_primes(n):
    """
    Count primes less than n.
    
    Args:
        n: Upper bound (exclusive)
        
    Returns:
        Count of primes < n
        
    Examples:
        >>> count_primes(10)
        4
    """
    pass


# Test
print(count_primes(10))
`,
    testCases: [
      {
        input: [10],
        expected: 4,
      },
      {
        input: [0],
        expected: 0,
      },
      {
        input: [1],
        expected: 0,
      },
    ],
    solution: `def count_primes(n):
    if n <= 2:
        return 0
    
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            # Mark multiples as not prime
            for j in range(i * i, n, i):
                is_prime[j] = False
    
    return sum(is_prime)`,
    timeComplexity: 'O(n log log n)',
    spaceComplexity: 'O(n)',
    order: 68,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-valid-perfect-square',
    title: 'Valid Perfect Square',
    difficulty: 'Easy',
    description: `Check if a number is a perfect square without using sqrt function.

Perfect square: n = k² for some integer k

**Example:** 16 = 4² → true, 14 → false

Use binary search for O(log n) solution.

This tests:
- Binary search
- Integer arithmetic
- Square calculation`,
    examples: [
      {
        input: 'num = 16',
        output: 'True',
      },
      {
        input: 'num = 14',
        output: 'False',
      },
    ],
    constraints: ['1 <= num <= 2^31 - 1'],
    hints: [
      'Use binary search',
      'Check if mid * mid == num',
      'Search range: 1 to num//2 + 1',
    ],
    starterCode: `def is_perfect_square(num):
    """
    Check if number is perfect square.
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect square
        
    Examples:
        >>> is_perfect_square(16)
        True
        >>> is_perfect_square(14)
        False
    """
    pass


# Test
print(is_perfect_square(16))
`,
    testCases: [
      {
        input: [16],
        expected: true,
      },
      {
        input: [14],
        expected: false,
      },
      {
        input: [1],
        expected: true,
      },
    ],
    solution: `def is_perfect_square(num):
    if num < 2:
        return True
    
    left, right = 2, num // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


# Alternative using math property
def is_perfect_square_math(num):
    x = num
    while x * x > num:
        x = (x + num // x) // 2
    return x * x == num`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 69,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-guess-number',
    title: 'Guess Number Higher or Lower',
    difficulty: 'Easy',
    description: `Guess a number from 1 to n using binary search.

API available: guess(num) returns:
- -1: my number is lower
- 1: my number is higher
- 0: correct!

Minimize number of guesses using binary search.

This tests:
- Binary search
- API interaction
- Search strategy`,
    examples: [
      {
        input: 'n = 10, pick = 6',
        output: '6',
        explanation: 'Binary search finds it',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1', '1 <= pick <= n'],
    hints: [
      'Use binary search',
      'Update left/right based on guess result',
      'Classic binary search pattern',
    ],
    starterCode: `# The guess API is predefined
def guess(num):
    """Mock API - compare num with picked number"""
    pick = 6  # This would be set in real game
    if num > pick:
        return -1
    elif num < pick:
        return 1
    else:
        return 0


def guess_number(n):
    """
    Find the number I picked.
    
    Args:
        n: Upper bound
        
    Returns:
        The picked number
        
    Examples:
        >>> guess_number(10)  # pick = 6
        6
    """
    pass


# Test
print(guess_number(10))
`,
    testCases: [
      {
        input: [10],
        expected: 6,
      },
    ],
    solution: `def guess_number(n):
    left, right = 1, n
    
    while left <= right:
        mid = (left + right) // 2
        result = guess(mid)
        
        if result == 0:
            return mid
        elif result == -1:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1  # Should never reach here`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 70,
    topic: 'Python Fundamentals',
  },
];
