/**
 * Multiple choice questions for Application Designs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const applicationdesignsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'In Design Twitter, what is the time complexity of getNewsFeed?',
    options: [
      'O(1)',
      'O(F) where F is number of followees',
      'O(F log K) where K is feed size',
      'O(N log N) where N is all tweets',
    ],
    correctAnswer: 2,
    explanation:
      "getNewsFeed is O(F log K) where F = number of followees and K = feed size (typically 10). We look at last 10 tweets from each followee (~10F tweets), build heap, extract top K. The log K factor comes from heap operations. We don't need to look at all tweets ever, just recent ones.",
  },
  {
    id: 'mc2',
    question: 'Why does Browser History use two stacks instead of one?',
    options: [
      'One stack is not enough memory',
      'Two stacks allow both back and forward navigation efficiently',
      'Two stacks are faster',
      'It does not need two stacks',
    ],
    correctAnswer: 1,
    explanation:
      'Two stacks naturally represent back and forward navigation. Current page is at top of back_stack. Going back pops from back_stack and pushes to forward_stack. Going forward does the reverse. Single stack cannot efficiently support both directions.',
  },
  {
    id: 'mc3',
    question:
      'In Autocomplete, what is the time complexity of input(c) if we precompute top K at each Trie node?',
    options: [
      'O(1)',
      'O(p) where p is prefix length',
      'O(p + m log m) where m is matching queries',
      'O(N) where N is all queries',
    ],
    correctAnswer: 1,
    explanation:
      'With precomputed top K at each node, input(c) only needs to navigate to the prefix in O(p) time, then return the cached top_k list. No need to collect all matches and sort - that work was done during insertion. This optimization makes autocomplete very fast.',
  },
  {
    id: 'mc4',
    question: 'What data structure does Twitter use to track followees?',
    options: ['List', 'Set', 'Array', 'Stack'],
    correctAnswer: 1,
    explanation:
      'Set is used because: (1) O(1) follow/unfollow. (2) Prevents duplicate follows. (3) O(1) check if following. (4) No need for ordering. List would have O(N) for unfollow (need to find and remove).',
  },
  {
    id: 'mc5',
    question:
      'In Design Twitter, why do we check "if followerId != followeeId" in follow()?',
    options: [
      'To save memory',
      'To prevent users from following themselves',
      'To make it faster',
      'It is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      "We prevent users from following themselves because: (1) It doesn't make logical sense. (2) Would cause duplicate tweets in news feed (own tweets appear twice). (3) Matches real Twitter behavior. This is an edge case that should be handled explicitly.",
  },
];
