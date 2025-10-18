/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize that a problem needs a Trie? What keywords or patterns signal this?',
    sampleAnswer:
      'Several signals indicate Trie. Explicit: "prefix", "autocomplete", "dictionary", "word search", "spell check". Implicit: multiple words need efficient lookup, need to find all words with prefix, matching patterns in strings. Problem types: implement dictionary with prefix query, word break, design search autocomplete, replace words, stream of characters. If problem involves many words and prefix operations, Trie is likely. For example, "design search autocomplete system" screams Trie. "Check if word exists in dictionary" could be hash table unless prefix operations mentioned. Ask yourself: do I need prefix matching? Do multiple words share prefixes? Is dictionary static or changing? Trie excels at: changing dictionary with prefix queries.',
    keyPoints: [
      'Keywords: prefix, autocomplete, dictionary, spell check',
      'Multiple words with prefix operations',
      'Problems: autocomplete, word break, replace words',
      'Ask: prefix matching needed? Words share prefixes?',
      'Trie excels: dynamic dictionary with prefix queries',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your complete Trie interview approach from recognition to implementation.',
    sampleAnswer:
      'First, recognize Trie from keywords (prefix, autocomplete). Second, clarify: character set (lowercase only?), will words be deleted, max word length, number of words. Third, explain approach: build Trie, insert all words, for queries traverse Trie. Fourth, state complexity: insert O(m) per word, search O(m), space O(ALPHABET × m × n) worst case, better with shared prefixes. Fifth, discuss implementation: use dict for children (flexible) vs array (faster), is_end_of_word flag essential. Sixth, draw small example: insert "cat", "car", show shared prefix. Seventh, code clearly with TrieNode class and Trie class. Test with example. Finally, optimize if needed: compressed Trie for space, add counts for frequency queries.',
    keyPoints: [
      'Recognize from keywords, clarify requirements',
      'Explain: build Trie, traverse for queries',
      'State complexity with reasoning',
      'Discuss: dict vs array, is_end_of_word flag',
      'Draw example showing shared prefix',
      'Code clearly, test, optimize if needed',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common Trie mistakes in interviews and how do you avoid them?',
    sampleAnswer:
      'First: forgetting is_end_of_word flag, causing failure when words are prefixes of others ("car" and "carpet"). Second: using fixed array without checking char is in range (crashes on non-lowercase). Third: not handling empty string edge case. Fourth: in autocomplete, forgetting to add prefix to collected words (returning "r" instead of "car" for prefix "ca"). Fifth: delete implementation breaking other words (removing shared nodes). Sixth: thinking Trie needed when Hash Table sufficient (if no prefix operations). My strategy: always include is_end_of_word, validate character range if using array, test with prefix words ("a", "ab", "abc"), draw tree to visualize structure, verify sharing works correctly.',
    keyPoints: [
      'Forgetting is_end_of_word (fails on prefix words)',
      'Array without range check (crashes)',
      'Empty string edge case',
      'Autocomplete: remember to add prefix',
      'Test with: prefix words, draw tree, verify sharing',
    ],
  },
];
