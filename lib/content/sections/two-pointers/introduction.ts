/**
 * What is the Two Pointers Technique? Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'What is the Two Pointers Technique?',
  content: `The Two Pointers technique is a powerful algorithmic pattern that uses **two pointers** to traverse a data structure (typically an array or string) in a clever way to solve problems efficiently. Instead of using nested loops that lead to O(n²) time complexity, two pointers can often reduce this to O(n).

**The Core Concept:**
Use two references (pointers) to traverse your data structure. The pointers can:
- Start at opposite ends and move toward each other
- Both start at the beginning but move at different speeds
- Define a window that slides through the data

**Why Two Pointers?**
Many problems that seem to require checking all pairs (O(n²)) can be solved more efficiently by maintaining two positions and making smart decisions about which pointer to move.

**Real-World Analogy:**
Imagine two people searching through a sorted list of prices to find two items that sum to your budget. One starts from the cheapest items, the other from the most expensive. If the sum is too high, the person at expensive items moves down. If too low, the person at cheap items moves up. They meet in the middle with the answer - much faster than checking every possible pair!

**When Does Two Pointers Work?**
- When dealing with sorted or sortable data
- When you need to find pairs/triplets with certain properties
- When working with palindromes or symmetric patterns
- When you need to partition or reorder array elements
- When tracking a window of elements with specific constraints`,
};
