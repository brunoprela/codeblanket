/**
 * What is Binary Search? Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'What is Binary Search?',
  content: `Binary Search is one of the most fundamental and efficient algorithms in computer science. It\'s a **divide-and-conquer** algorithm that finds the position of a target value within a **sorted array** by repeatedly dividing the search interval in half.

**The Core Insight:**
When dealing with a sorted array, we can determine which half contains our target by comparing it with the middle element. This eliminates half of the remaining elements with each comparison.

**Why "Binary"?**
At each step, we make a binary (yes/no) decision: is our target in the left half or the right half? This binary decision tree is what gives the algorithm its name and its logarithmic efficiency.

**Real-World Analogy:**
Think of finding a word in a dictionary. You don't start from 'A' and flip through every page. You open the dictionary roughly in the middle, check if your word comes before or after that page, then repeat the process with the appropriate half. That\'s binary search!

**Key Prerequisites:**
- The array MUST be sorted (ascending or descending)
- You need random access to elements (arrays work great, linked lists don't)
- The comparison operation must be well-defined`,
};
