/**
 * Quiz questions for Common Trie Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the word search pattern. How does Trie make it more efficient than checking each word individually?',
    sampleAnswer:
      'Word search checks if string exists in dictionary. Naive approach: iterate through all words, compare each. For N words of length M, this is O(N×M) per search. With Trie: insert all words once in O(N×M) total, then each search is O(M) regardless of N. For example, dictionary with 100K words: naive search checks all 100K words every time. Trie search just traverses M nodes. The efficiency comes from shared prefixes: multiple words share path, so we do not redundantly check same prefix. If searching "apple", we traverse A→P→P→L→E once, check is_end_of_word. Dictionary operations (spell check, autocorrect) need many searches, so Trie amortizes cost across all words.',
    keyPoints: [
      'Naive: O(N×M) per search (check all words)',
      'Trie: O(N×M) build once, O(M) per search',
      'Shared prefixes avoid redundant checks',
      'Example: 100K words, Trie searches only M nodes',
      'Perfect for: many searches on same dictionary',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the word break pattern. How does Trie help with checking if segments are valid words?',
    sampleAnswer:
      'Word break determines if string can be segmented into dictionary words. Example: "leetcode" → "leet" + "code". Use DP with Trie for validation. DP: dp[i] = can we break s[0..i]. For each i, try all possible last words ending at i. Trie helps: from each position j, traverse Trie character by character toward i. If we hit end-of-word marker, s[j..i] is valid word, so dp[i] = dp[i] or dp[j]. Without Trie, checking if s[j..i] is valid word takes O(word_count × word_length). With Trie, traverse once character by character, finding all valid words simultaneously. The Trie acts as oracle: "is this prefix valid? is this word complete?" enabling efficient DP.',
    keyPoints: [
      'Segment string into dictionary words',
      'DP: dp[i] = can break s[0..i]',
      'Trie validates: is substring valid word?',
      'Traverse Trie from each position, find valid words',
      'More efficient than checking each word separately',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the prefix matching pattern for IP routing. Why is Trie ideal for longest prefix match?',
    sampleAnswer:
      'IP routing finds longest matching prefix for destination IP. Example: routes for 192.168.*, 192.168.1.*, 192.168.1.100. For IP 192.168.1.100, longest match is 192.168.1.100 (most specific). Build Trie of IP prefixes (each bit is node). Search by traversing IP bits. Track last end-of-word node seen (last valid route). At end, return longest match found. Trie is ideal because: naturally finds longest match by traversing deep as possible, shared prefixes stored once (192.168 shared by all), O(bits) lookup regardless of route count. In practice, 32-bit IPv4 gives O(32) lookup for millions of routes. The tree structure mirrors IP hierarchy: general to specific.',
    keyPoints: [
      'Find longest matching prefix for IP',
      'Build Trie of IP prefixes (bits as nodes)',
      'Traverse IP, track last valid route seen',
      'Naturally finds longest by going deep as possible',
      'O(bits) lookup for millions of routes',
    ],
  },
];
