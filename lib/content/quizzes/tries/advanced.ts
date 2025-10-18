/**
 * Quiz questions for Advanced Trie Techniques section
 */

export const advancedQuiz = [
  {
    id: 'q1',
    question:
      'Explain compressed Trie (Radix Tree). How does it save space compared to standard Trie?',
    sampleAnswer:
      'Compressed Trie merges chains of single-child nodes into one node storing edge label (string). Standard Trie for "test" and "testing" has nodes T→E→S→T→I→N→G with "test" marked at first T. Compressed Trie stores edge "test" from root, then edge "ing" for "testing". Saves space by eliminating intermediate nodes that have only one child. Standard uses O(total characters), compressed uses O(number of words). For sparse dictionary where words share few prefixes, compressed Trie is much more space efficient. Tradeoff: implementation complexity increases (store strings on edges, not chars in nodes). Used in practice: Git uses Radix trees, file systems for path lookup. Best when: long strings, few shared prefixes.',
    keyPoints: [
      'Merge single-child chains into one node',
      'Store strings on edges, not chars',
      'Space: O(words) vs O(total chars)',
      'Best: sparse dictionary, long strings, few prefixes',
      'Tradeoff: space savings vs implementation complexity',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe Trie with counts. What additional problems does it enable?',
    sampleAnswer:
      'Add count field to each node tracking number of words passing through. During insert, increment count on each node traversed. This enables: prefix frequency (how many words start with prefix = count at prefix node), word frequency (store count at end node), autocomplete with popularity (sort suggestions by count), stream problems (maintain counts as words arrive). For example, insert "cat" twice, "car" once. Node C has count 3, A has count 3, T has count 2 (end), R has count 1 (end). Query "ca" prefix: count at A is 3 words start with "ca". This augmentation adds minimal overhead (one integer per node) but enables frequency-based queries crucial for real-world applications like search suggestions.',
    keyPoints: [
      'Add count field: words passing through node',
      'Increment during insert, decrement during delete',
      'Enables: prefix frequency, word frequency',
      'Autocomplete with popularity ranking',
      'Minimal overhead, major functionality gain',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the XOR Trie for maximum XOR problem. Why use binary Trie?',
    sampleAnswer:
      'Maximum XOR finds two numbers with maximum XOR result. XOR maximized when bits differ. Build binary Trie: each node has children 0 and 1, insert numbers bit by bit (32 bits for integers). For each query number, traverse Trie trying opposite bit at each level (if number has 0, try to go 1; if has 1, try to go 0). This greedily maximizes XOR from most significant bit. If opposite bit path exists, take it. Otherwise, take same bit. At leaf, we have the number that gives maximum XOR with query. Example: numbers [3,10,5,25], query 2. Binary: 00010. Try to go opposite at each bit to maximize XOR. Binary Trie perfect because it organizes numbers by bit patterns, enabling greedy bit-by-bit maximization.',
    keyPoints: [
      'Binary Trie: nodes have children 0 and 1',
      'Insert numbers bit by bit (32 bits)',
      'Query: try opposite bit at each level',
      'Greedy maximizes XOR from MSB',
      'Perfect for: bit manipulation, XOR problems',
    ],
  },
];
