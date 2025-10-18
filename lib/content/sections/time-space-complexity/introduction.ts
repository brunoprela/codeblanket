/**
 * What is Complexity Analysis? Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'What is Complexity Analysis?',
  content: `Complexity analysis is the process of determining how the runtime and memory usage of an algorithm scale as the input size grows. It's one of the most fundamental skills in computer science and is crucial for writing efficient code.

**Why Does It Matter?**
- **Scalability:** An algorithm that works for 100 items might fail for 1 million items
- **Resource Management:** Understanding memory usage prevents crashes and optimizes performance
- **Interview Success:** Virtually every technical interview asks about complexity
- **Design Decisions:** Helps you choose the right algorithm and data structure

**The Core Question:**
As the input size \`n\` grows, how does the number of operations (time) or the amount of memory (space) change?

**Real-World Analogy:**
Imagine you need to find a friend's phone number:
- **O(1):** You have them on speed dial - instant access
- **O(log n):** You binary search through a sorted phonebook - very fast
- **O(n):** You scan through your contacts list one by one - grows linearly
- **O(n²):** You check every contact against every other contact - gets slow fast!

**Key Principles:**
- We care about **growth rate**, not exact numbers
- We focus on the **worst case** unless stated otherwise
- We ignore **constants and lower-order terms** (O(2n + 5) → O(n))
- We measure both **time complexity** (operations) and **space complexity** (memory)`,
};
