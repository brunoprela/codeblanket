/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: Fixed-Size Window**
\`\`\`python
def fixed_window (arr: List[int], k: int) -> int:
    """
    Generic fixed-size sliding window.
    Adjust the logic for your specific problem.
    """
    if len (arr) < k:
        return 0  # or appropriate default
    
    # Initialize window with first k elements
    window_sum = sum (arr[:k])  # or other initialization
    result = window_sum
    
    # Slide the window
    for i in range (k, len (arr)):
        # Add new element on the right
        window_sum += arr[i]
        
        # Remove old element on the left
        window_sum -= arr[i - k]
        
        # Update result
        result = max (result, window_sum)  # or min, etc.
    
    return result
\`\`\`

**Template 2: Variable Window - Find Maximum/Longest**
\`\`\`python
def variable_window_max (arr: List[int]) -> int:
    """
    Variable-size window to find maximum/longest.
    Expand when invalid, track max when valid.
    """
    left = 0
    max_length = 0
    window_state = {}  # Hash map/set to track window
    
    for right in range (len (arr)):
        # Add arr[right] to window
        # Update window_state
        
        # Shrink window while condition violated
        while window_violates_condition (window_state):
            # Remove arr[left] from window
            # Update window_state
            left += 1
        
        # Update result (window is now valid)
        max_length = max (max_length, right - left + 1)
    
    return max_length
\`\`\`

**Template 3: Variable Window - Find Minimum/Shortest**
\`\`\`python
def variable_window_min (arr: List[int], target) -> int:
    """
    Variable-size window to find minimum/shortest.
    Expand until valid, then shrink to find minimum.
    """
    left = 0
    min_length = float('inf')
    window_state = {}
    
    for right in range (len (arr)):
        # Add arr[right] to window
        # Update window_state
        
        # Shrink window while condition MET
        while window_meets_condition (window_state, target):
            # Update result (current window is valid)
            min_length = min (min_length, right - left + 1)
            
            # Remove arr[left] from window
            # Update window_state
            left += 1
    
    return min_length if min_length != float('inf') else 0
\`\`\`

**Template 4: Sliding Window with Frequency Map**
\`\`\`python
from collections import defaultdict

def sliding_window_with_freq (s: str) -> int:
    """
    Sliding window tracking character frequencies.
    """
    left = 0
    max_length = 0
    freq = defaultdict (int)
    
    for right in range (len (s)):
        # Add character to window
        freq[s[right]] += 1
        
        # Shrink if condition violated
        while condition_violated (freq):
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]  # Clean up
            left += 1
        
        max_length = max (max_length, right - left + 1)
    
    return max_length
\`\`\`

**Template 5: Sliding Window with Two Pointers (Distinct Elements)**
\`\`\`python
def sliding_window_set (arr: List[int], k: int) -> int:
    """
    Track unique elements using a set.
    """
    left = 0
    window = set()
    
    for right in range (len (arr)):
        # Add to window, handle duplicates
        while arr[right] in window:
            window.remove (arr[left])
            left += 1
        
        window.add (arr[right])
        
        # Process window
        if len (window) == k:
            # Found a valid window
            pass
    
    return result
\`\`\``,
};
