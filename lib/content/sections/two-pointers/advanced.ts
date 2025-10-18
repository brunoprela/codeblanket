/**
 * Advanced Techniques & Variations Section
 */

export const advancedSection = {
  id: 'advanced',
  title: 'Advanced Techniques & Variations',
  content: `**Three Pointers (3Sum Problem)**

Finding three numbers that sum to target:

\`\`\`python
def three_sum(nums: List[int], target: int) -> List[List[int]]:
    """Find all unique triplets that sum to target."""
    nums.sort()  # O(n log n)
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two pointers for remaining two numbers
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current = nums[i] + nums[left] + nums[right]
            
            if current == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                    
                left += 1
                right -= 1
            elif current < target:
                left += 1
            else:
                right -= 1
    
    return result
\`\`\`

**Cycle Detection (Floyd's Algorithm)**

Detect cycle in linked list using fast & slow pointers:

\`\`\`python
def has_cycle(head: ListNode) -> bool:
    """Detect if linked list has a cycle."""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True
\`\`\`

**Why it works:** If there's a cycle, fast will eventually catch up to slow.

**Dutch National Flag (Three-Way Partitioning)**

Partition array into three parts:

\`\`\`python
def sort_colors(nums: List[int]) -> None:
    """Sort array with values 0, 1, 2 in-place."""
    left = 0  # Next position for 0
    curr = 0  # Current element being examined
    right = len(nums) - 1  # Next position for 2
    
    while curr <= right:
        if nums[curr] == 0:
            nums[left], nums[curr] = nums[curr], nums[left]
            left += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[right] = nums[right], nums[curr]
            right -= 1
            # Don't move curr - need to examine swapped element
        else:  # nums[curr] == 1
            curr += 1
\`\`\`

**Container With Most Water**

Find maximum area between two vertical lines:

\`\`\`python
def max_area(height: List[int]) -> int:
    """Find container that holds the most water."""
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        # Height limited by shorter line
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)
        
        # Move pointer at shorter line
        # (might find taller line)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
\`\`\`

**Key Insight:** Always move the pointer at the shorter line - moving the taller line can only decrease area.`,
};
