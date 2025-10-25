/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the basic Trie template. What are the essential components?',
    sampleAnswer:
      'Basic Trie has two classes: TrieNode and Trie. TrieNode has children (dict or array) and is_end_of_word flag. Trie has root node and three methods. Insert: start at root, for each char, get or create child node, move to child, mark last as end. Search: traverse chars, return False if any missing, check is_end_of_word at end. StartsWith: same as search but ignore is_end_of_word. The pattern: all operations traverse character by character. Children dict is most flexible (any alphabet). Array (size 26) is faster but limited. The is_end_of_word flag is crucial for distinguishing complete words from prefixes. This template is foundation for all Trie problems - understand it deeply.',
    keyPoints: [
      'Two classes: TrieNode (children, is_end_of_word) and Trie (root, methods)',
      'Insert: traverse/create path, mark end',
      'Search: traverse, check exists and is_end_of_word',
      'StartsWith: traverse, ignore end flag',
      'Pattern: all ops traverse char by char',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the autocomplete template. How do you collect all words with a prefix?',
    sampleAnswer:
      'Autocomplete is two-phase. Phase 1: Navigate to prefix node using starts_with logic. If prefix does not exist, return empty. Phase 2: DFS from prefix node to collect all words. DFS function: takes current node and path built so far. If node is_end_of_word, add prefix + path to results. Recurse on all children, adding child char to path. This DFS explores entire subtree under prefix node. For example, prefix "ca", navigate to node at A (after C), then DFS finds all paths: R (car), R→P→E→T (carpet), T (cat). Return results. The key insight: all words with prefix are descendants of prefix node. DFS naturally collects them all.',
    keyPoints: [
      'Phase 1: navigate to prefix node',
      'Phase 2: DFS from prefix node',
      'DFS: add word if is_end, recurse on children',
      'All prefix words are in subtree',
      'Example: "ca" → DFS finds car, carpet, cat',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the Trie delete template. Why is it more complex than insert?',
    sampleAnswer:
      'Delete is complex because we must avoid breaking other words. Use recursive approach: delete (node, word, index). Base case: index equals word length, unmark is_end_of_word. Recursive: get child for current char, recursively delete from child. After recursion, check if child should be removed: child has no children and is not end of other word. Return whether current node should be deleted (no children, not end of word). The complexity: cannot just remove nodes blindly. If deleting "car" but "carpet" exists, cannot remove R node (has children). If deleting "carpet" but "car" exists, cannot remove R node (is end of word). Must carefully check each node bottom-up. This is why delete is rarely asked in interviews - too complex for 45 minutes.',
    keyPoints: [
      'Recursive: delete (node, word, index)',
      'Base: unmark is_end_of_word',
      'Recursive: delete from child, check if remove child',
      'Cannot blindly remove: might break other words',
      'Check: node has no children and not end of word',
    ],
  },
];
