/**
 * Code Templates & Patterns Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates & Patterns',
  content: `**Template 1: Opposite Direction - Pair with Target Sum**

\`\`\`python
def pair_with_sum(nums: List[int], target: int) -> List[int]:
    """Find pair that sums to target in sorted array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current = nums[left] + nums[right]
        
        if current == target:
            return [left, right]
        elif current < target:
            left += 1  # Need larger value
        else:
            right -= 1  # Need smaller value
    
    return [-1, -1]  # Not found
\`\`\`

**Template 2: Same Direction - Remove Elements**

\`\`\`python
def remove_element(nums: List[int], val: int) -> int:
    """Remove all instances of val in-place."""
    slow = 0  # Position for next kept element
    
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow  # New length
\`\`\`

**Template 3: Sliding Window - Fixed Size**

\`\`\`python
def max_sum_subarray(nums: List[int], k: int) -> int:
    """Find maximum sum of subarray of size k."""
    # Initialize first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide window
    for right in range(k, len(nums)):
        window_sum += nums[right]  # Add new element
        window_sum -= nums[right - k]  # Remove old element
        max_sum = max(max_sum, window_sum)
    
    return max_sum
\`\`\`

**Template 4: Sliding Window - Variable Size**

\`\`\`python
def longest_substring_k_distinct(s: str, k: int) -> int:
    """Longest substring with at most k distinct characters."""
    left = 0
    max_len = 0
    char_count = {}
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if constraint violated
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update max length
        max_len = max(max_len, right - left + 1)
    
    return max_len
\`\`\`

**Template 5: Partition Array**

\`\`\`python
def partition(nums: List[int], pivot: int) -> int:
    """Partition array around pivot value."""
    left = 0  # Next position for elements < pivot
    
    for right in range(len(nums)):
        if nums[right] < pivot:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
    
    return left  # Pivot position
\`\`\`

**Choosing the Right Template:**
1. Sorted array + pair/triplet? → Opposite direction
2. Remove/partition in-place? → Same direction
3. Subarray/substring problems? → Sliding window
4. Need to reorder elements? → Partition template`,
};
