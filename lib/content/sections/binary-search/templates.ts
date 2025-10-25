/**
 * Code Templates & Patterns Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates & Patterns',
  content: `**Template 1: Classic Binary Search**
Find exact match or return -1

**Template 2: Find First Occurrence**
When duplicates exist, find leftmost occurrence

\`\`\`python
def find_first (nums: List[int], target: int) -> int:
    left, right = 0, len (nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
\`\`\`

**Template 3: Find Last Occurrence**
Find rightmost occurrence when duplicates exist

\`\`\`python
def find_last (nums: List[int], target: int) -> int:
    left, right = 0, len (nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
\`\`\`

**Template 4: Find Insert Position**
Where to insert target to maintain sorted order

\`\`\`python
def search_insert (nums: List[int], target: int) -> int:
    left, right = 0, len (nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Insertion position
\`\`\`

**When to Use Each Template:**
- Use Template 1 for simple existence checks
- Use Template 2 for finding ranges (start boundary)
- Use Template 3 for finding ranges (end boundary)
- Use Template 4 for insertion/floor/ceiling problems`,
};
