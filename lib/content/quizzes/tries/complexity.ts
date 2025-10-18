/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare Trie vs Hash Table vs BST for dictionary operations. When would you choose Trie?',
    sampleAnswer:
      'For exact word lookup: Hash Table is O(m) average case (same as Trie), BST is O(m log n). For prefix operations: Trie excels with O(m), Hash Table cannot do prefix search efficiently, BST is O(m log n + k) for k results. Space: Trie uses most space (26 pointers per node), Hash Table is moderate, BST is least (two pointers per node). Choose Trie when: prefix operations needed (autocomplete, word games), many words share prefixes (amortizes space), need ordered traversal by prefix. Choose Hash Table when: only exact lookup, space matters, no prefix operations. Choose BST when: need ordered iteration, space critical, fewer words. The killer feature of Trie is prefix operations - nothing else does them efficiently.',
    keyPoints: [
      'Lookup: Trie O(m), Hash O(m) avg, BST O(m log n)',
      'Prefix: Trie O(m), Hash inefficient, BST O(m log n + k)',
      'Space: Trie highest, Hash moderate, BST lowest',
      'Choose Trie: prefix operations crucial',
      'Trie killer feature: efficient prefix queries',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain Trie space complexity. Why is it O(ALPHABET_SIZE × m × n) worst case?',
    sampleAnswer:
      'Each node has array of ALPHABET_SIZE pointers (26 for lowercase). For n words of average length m, worst case has n × m nodes (no shared prefixes, all words completely different). Each node has 26 pointers, so total space is 26 × m × n pointers. For example, 1000 words of length 10: worst case is 26 × 10 × 1000 = 260K pointers. In practice, much better due to shared prefixes. If all words start with "a", first node is shared. Best case: all words identical, only m nodes for shared path. Average case depends on prefix overlap. Using HashMap children reduces to actual children count, not fixed 26. The alphabet size dominates: English (26) vs Unicode (100K+) makes huge difference.',
    keyPoints: [
      'Each node: ALPHABET_SIZE pointers',
      'Worst case: n words, m length, no sharing',
      'Space: ALPHABET_SIZE × m × n',
      'Best case: O(m) when all words identical',
      'HashMap children: only actual children, more efficient',
    ],
  },
  {
    id: 'q3',
    question: 'When should you NOT use a Trie? What are its limitations?',
    sampleAnswer:
      'Do NOT use Trie when: only exact lookups needed (Hash Table simpler and uses less space), space is critical constraint (Trie is space-heavy), alphabet is huge like Unicode (26^1000 becomes unreasonable), words are very long with no shared prefixes (defeats purpose of Trie), need additional operations like range queries (better suited for other structures). For example, storing random UUIDs: no shared prefixes, no prefix queries needed - Hash Table is better. Storing full DNA sequences: 4-letter alphabet is good, but sequences are millions long with little sharing - other structures might be better. Trie shines when: many prefix queries, moderate alphabet size, significant prefix overlap, moderate word lengths. Know when NOT to use it is as important as knowing when to use it.',
    keyPoints: [
      'Not for: only exact lookups, space critical',
      'Not for: huge alphabet, no prefix sharing',
      'Example: random UUIDs better in Hash Table',
      'Trie shines: prefix queries, moderate alphabet, sharing',
      'Wrong tool for job: wastes space, adds complexity',
    ],
  },
];
