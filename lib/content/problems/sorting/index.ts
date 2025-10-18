/**
 * sorting Problems
 * 6 problems total
 */

import { merge_sorted_arraysProblem } from './merge-sorted-arrays';
import { sort_array_parityProblem } from './sort-array-parity';
import { insertion_sort_listProblem } from './insertion-sort-list';
import { sort_listProblem } from './sort-list';
import { wiggle_sortProblem } from './wiggle-sort';
import { count_smallerProblem } from './count-smaller';

export const sortingProblems = [
  merge_sorted_arraysProblem, // 1. Merge Two Sorted Arrays
  sort_array_parityProblem, // 2. Sort Array By Parity
  insertion_sort_listProblem, // 3. Insertion Sort List
  sort_listProblem, // 4. Sort List (Merge Sort)
  wiggle_sortProblem, // 5. Wiggle Sort
  count_smallerProblem, // 6. Count of Smaller Numbers After Self
];
