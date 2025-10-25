/**
 * Common Pitfalls Section
 */

export const commonpitfallsSection = {
  id: 'common-pitfalls',
  title: 'Common Pitfalls',
  content: `**Pitfall 1: Off-by-One Errors in Window Size**

❌ **Wrong:**
\`\`\`python
# Window size is right - left, missing the +1
max_length = max (max_length, right - left)
\`\`\`

✅ **Correct:**
\`\`\`python
# Window size is right - left + 1 (inclusive)
max_length = max (max_length, right - left + 1)
\`\`\`

---

**Pitfall 2: Not Cleaning Up Hash Map**

❌ **Wrong:**
\`\`\`python
freq[s[left]] -= 1
left += 1
# Leaves 0 counts in map, affecting len (freq)
\`\`\`

✅ **Correct:**
\`\`\`python
freq[s[left]] -= 1
if freq[s[left]] == 0:
    del freq[s[left]]  # Clean up to maintain accurate distinct count
left += 1
\`\`\`

---

**Pitfall 3: Forgetting to Initialize First Window**

❌ **Wrong (Fixed Window):**
\`\`\`python
for i in range (len (arr)):  # Starts from 0, recalculating
    window_sum = sum (arr[i:i+k])
\`\`\`

✅ **Correct:**
\`\`\`python
window_sum = sum (arr[:k])  # Calculate first window once
for i in range (k, len (arr)):  # Start sliding from index k
    window_sum += arr[i] - arr[i-k]
\`\`\`

---

**Pitfall 4: Moving Both Pointers in Same Iteration**

❌ **Wrong:**
\`\`\`python
for right in range (len (arr)):
    # Add arr[right]
    while invalid:
        left += 1
    right += 1  # Don't manually increment right!
\`\`\`

✅ **Correct:**
\`\`\`python
for right in range (len (arr)):  # for loop handles right
    # Add arr[right]
    while invalid:
        left += 1  # Only manually move left
\`\`\`

---

**Pitfall 5: Using Wrong Condition for Min vs Max Window**

**For Maximum/Longest** (want largest valid window):
\`\`\`python
while window_is_INVALID:  # Shrink when invalid
    left += 1
max_length = max (max_length, right - left + 1)  # Update outside while
\`\`\`

**For Minimum/Shortest** (want smallest valid window):
\`\`\`python
while window_is_VALID:  # Keep shrinking while still valid
    min_length = min (min_length, right - left + 1)  # Update inside while
    left += 1
\`\`\``,
};
