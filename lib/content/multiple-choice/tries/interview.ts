/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What keywords in a problem description signal that a Trie might be needed?',
    options: [
      'Sort, binary, search',
      'Prefix, autocomplete, dictionary, spell check',
      'Array, list, queue',
      'Graph, cycle, path',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "prefix", "autocomplete", "dictionary", "spell check", "word search", or "type-ahead" strongly suggest Trie-based solutions. Trie excels at prefix operations on dictionaries.',
  },
  {
    id: 'mc2',
    question:
      'What should you clarify before implementing a Trie in an interview?',
    options: [
      "The interviewer's name",
      'Character set, word deletions, max word length, number of words',
      'Programming language only',
      'Nothing, just start coding',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: character set (lowercase only?), whether words will be deleted, max word length (affects depth), and number of words (affects space). These affect implementation choices.',
  },
  {
    id: 'mc3',
    question: 'What is the most common mistake with Tries in interviews?',
    options: [
      'Using too much memory',
      'Forgetting the is_end_of_word flag, causing failures with prefix words',
      'Making it too fast',
      'Using wrong variable names',
    ],
    correctAnswer: 1,
    explanation:
      'Forgetting is_end_of_word causes failures when words are prefixes of others (e.g., "car" and "carpet"). Just having a node doesn\'t mean a word exists - must check the flag.',
  },
  {
    id: 'mc4',
    question: 'When explaining Trie complexity, what should you mention?',
    options: [
      'Only time complexity',
      'Insert/Search O(m) per word, Space O(ALPHABET × m × n) worst case but better with shared prefixes',
      'Only space complexity',
      "It\'s always fast",
    ],
    correctAnswer: 1,
    explanation:
      "Explain time O(m) for all operations where m is word length, and space O(ALPHABET × m × n) worst case, but emphasize it's better in practice due to shared prefixes.",
  },
  {
    id: 'mc5',
    question: 'What is a good practice progression for mastering Tries?',
    options: [
      'Start with the hardest problems',
      'Week 1: Basics (implement Trie), Week 2: Applications (Word Search), Week 3: Advanced (XOR)',
      'Only practice one problem',
      'Skip practice',
    ],
    correctAnswer: 1,
    explanation:
      'Progress from basic implementation to applications (autocomplete, word search) to advanced techniques (XOR, compressed Trie). Build understanding incrementally.',
  },
];
