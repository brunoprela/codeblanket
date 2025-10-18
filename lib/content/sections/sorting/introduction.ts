/**
 * Why Sorting Matters Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Why Sorting Matters',
  content: `Sorting is one of the most fundamental operations in computer science. It's the process of arranging data in a specific order (usually ascending or descending), and it serves as the foundation for many other algorithms.

**Why Learn Sorting?**
- **Ubiquitous:** Used everywhere - databases, search engines, operating systems
- **Prerequisite:** Many algorithms require sorted data (binary search, merge operations)
- **Interview Favorite:** Nearly every coding interview includes sorting questions
- **Algorithmic Thinking:** Teaches divide-and-conquer, recursion, and optimization

**Real-World Applications:**
- **Search Engines:** Ranking search results
- **Databases:** ORDER BY queries, index maintenance
- **E-commerce:** Sorting products by price, rating, popularity
- **Operating Systems:** Process scheduling, memory management
- **Data Analysis:** Finding top-k elements, percentiles

**The Sorting Landscape:**

Sorting algorithms can be categorized by:

1. **Time Complexity:**
   - Simple sorts: O(n²) - Bubble, Selection, Insertion
   - Efficient sorts: O(n log n) - Merge, Quick, Heap
   - Special cases: O(n) - Counting, Radix, Bucket

2. **Space Complexity:**
   - In-place: O(1) extra space - Quick, Heap, Insertion
   - Not in-place: O(n) extra space - Merge

3. **Stability:**
   - Stable: Equal elements maintain relative order - Merge, Insertion
   - Unstable: May change relative order - Quick, Heap

4. **Adaptive:**
   - Adaptive: Faster on nearly-sorted data - Insertion
   - Non-adaptive: Same speed regardless - Selection

**Key Questions to Ask:**
1. How large is the dataset? (Small → Insertion, Large → Quick/Merge)
2. Is it already partially sorted? (Yes → Insertion, Timsort)
3. Do I have extra memory? (Yes → Merge, No → Quick/Heap)
4. Must equal elements stay in order? (Yes → Stable sorts only)
5. What's the nature of the data? (Integers in range → Counting sort)

**The Classic Tradeoff:**
- **Simple algorithms:** Easy to understand, bad performance (O(n²))
- **Advanced algorithms:** Complex to understand, great performance (O(n log n))
- **Specialized algorithms:** Only work for specific data types, can be O(n)`,
};
