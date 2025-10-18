/**
 * Quiz questions for Substring Search & Pattern Matching section
 */

export const substringsearchQuiz = [
  {
    id: 'q-sub1',
    question:
      'Explain the Rabin-Karp rolling hash technique for pattern matching.',
    sampleAnswer:
      'Rabin-Karp uses a rolling hash function to compute hash values for all substrings of length m in O(1) per substring after initial O(m) computation. Hash formula: hash = (c1*d^(m-1) + c2*d^(m-2) + ... + cm) % q, where d is base (e.g., 256) and q is prime. To "roll" from position i to i+1: remove first character contribution, shift, add new character. If hashes match, verify with character-by-character check (avoid hash collisions). Average O(n+m), worst O(nm).',
    keyPoints: [
      'Rolling hash: O(1) to compute next substring hash',
      'Hash formula: polynomial with base d, modulo prime q',
      'Remove old char: hash = (hash - c1*d^(m-1)) / d',
      'Add new char: hash = hash * d + new_c',
      'Verify matches to handle collisions',
    ],
  },
  {
    id: 'q-sub2',
    question:
      'What is the LPS (Longest Proper Prefix which is also Suffix) array in KMP and how is it used?',
    sampleAnswer:
      'LPS[i] stores length of longest proper prefix of pattern[0...i] that is also a suffix. Used to avoid re-comparing characters after mismatch. When mismatch at pattern[j], we know pattern[0...j-1] matched, so we can skip to position LPS[j-1] instead of starting over. This is because pattern[0...LPS[j-1]] equals pattern[j-LPS[j-1]...j-1]. Example: pattern="ABABC", LPS=[0,0,1,2,0]. Guarantees O(n+m) time.',
    keyPoints: [
      'LPS[i] = length of longest proper prefix = suffix',
      'Proper prefix: excludes full string',
      'Used to skip redundant comparisons',
      'On mismatch at j, jump to LPS[j-1]',
      'Preprocessing: O(m), Matching: O(n)',
    ],
  },
  {
    id: 'q-sub3',
    question:
      'Why is naive substring search O(nm) and when is it acceptable to use?',
    sampleAnswer:
      'Naive search checks every position in text (n positions), and for each position compares up to m characters of the pattern. In worst case (e.g., text="aaaaa", pattern="aaab"), all m characters are compared at each of n positions, giving O(nm). It\'s acceptable when: pattern and text are short, pattern has no repeating structure (making worst case rare), or simplicity is prioritized over performance. For most real-world text (English), average case is much better than worst case.',
    keyPoints: [
      'Checks n positions in text',
      'Compares up to m chars per position',
      'Worst case: O(nm) with repeating characters',
      'Average case better for random text',
      'Use when simplicity matters or inputs are small',
    ],
  },
];
