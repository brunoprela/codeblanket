/**
 * String Algorithms Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introSection } from '../sections/string-algorithms/intro';
import { palindromesSection } from '../sections/string-algorithms/palindromes';
import { anagramsSection } from '../sections/string-algorithms/anagrams';
import { substringsearchSection } from '../sections/string-algorithms/substring-search';

// Import quizzes
import { introQuiz } from '../quizzes/string-algorithms/intro';
import { palindromesQuiz } from '../quizzes/string-algorithms/palindromes';
import { anagramsQuiz } from '../quizzes/string-algorithms/anagrams';
import { substringsearchQuiz } from '../quizzes/string-algorithms/substring-search';

// Import multiple choice
import { introMultipleChoice } from '../multiple-choice/string-algorithms/intro';
import { palindromesMultipleChoice } from '../multiple-choice/string-algorithms/palindromes';
import { anagramsMultipleChoice } from '../multiple-choice/string-algorithms/anagrams';
import { substringsearchMultipleChoice } from '../multiple-choice/string-algorithms/substring-search';

export const stringAlgorithmsModule: Module = {
  id: 'string-algorithms',
  title: 'String Algorithms',
  description:
    'Master essential string manipulation algorithms including pattern matching, string processing, and common interview patterns.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📝',
  keyTakeaways: [
    'Strings are immutable—avoid repeated concatenation in loops, use list + join instead',
    'Two pointers effective for palindromes and comparisons',
    'Hash maps (Counter) essential for anagrams and character frequency problems',
    'Sliding window for substring problems: maintains state while moving through string',
    'Rabin-Karp uses rolling hash for O(n+m) average pattern matching',
    'KMP guarantees O(n+m) by using LPS array to avoid backtracking',
    'Always clarify: case sensitivity, spaces/punctuation, empty string handling',
    'For anagrams: sorted string as key (O(k log k)) or count tuple (O(k))',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introSection,
      quiz: introQuiz,
      multipleChoice: introMultipleChoice,
    },
    {
      ...palindromesSection,
      quiz: palindromesQuiz,
      multipleChoice: palindromesMultipleChoice,
    },
    {
      ...anagramsSection,
      quiz: anagramsQuiz,
      multipleChoice: anagramsMultipleChoice,
    },
    {
      ...substringsearchSection,
      quiz: substringsearchQuiz,
      multipleChoice: substringsearchMultipleChoice,
    },
  ],
};
