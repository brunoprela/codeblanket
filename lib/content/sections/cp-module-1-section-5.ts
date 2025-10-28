export const fastInputOutputTechniquesSection = {
  id: 'cp-m1-s5',
  title: 'Fast Input/Output Techniques',
  content: `

# Fast Input/Output Techniques

## Introduction

**Time Limit Exceeded (TLE)** is one of the most frustrating verdicts in competitive programming. Sometimes, even with the correct algorithm, slow input/output can cause TLE. Understanding and using **fast I/O techniques** can be the difference between Accepted and TLE.

In this section, we'll explore why standard I/O can be slow, how to optimize it, and when to use different I/O methods.

---

## Why Standard I/O Can Be Slow

### The Problem with cin/cout

By default, \`cin\` and \`cout\` in C++ are **synchronized** with C's \`scanf\` and \`printf\`. This synchronization adds overhead, making them slower than necessary.

**Benchmark: Reading 1 million integers**

\`\`\`cpp
// Standard cin/cout (with sync)
#include <iostream>
using namespace std;

int main() {
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
    }
    return 0;
}
// Time: ~2.5 seconds
\`\`\`

**Why so slow?**
1. **Synchronization overhead**: cin/cout sync with scanf/printf
2. **Buffering**: Multiple system calls
3. **Tie with cout**: cin flushes cout buffer before each read

---

## Fast I/O Optimization 1: Disable Sync

The first and easiest optimization: **disable synchronization**.

\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    
    // Now use cin/cout normally
    int n;
    cin >> n;
    cout << n << endl;
    
    return 0;
}
\`\`\`

### What Does This Do?

\`ios_base::sync_with_stdio(false)\` tells C++ to **stop synchronizing** with C I/O functions.

**Effect:**
- cin/cout become faster (2-3x speedup)
- **WARNING:** Cannot mix cin/scanf or cout/printf after this!

**Correct:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin >> n;
cout << n;  // ✅ OK
\`\`\`

**WRONG:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin >> n;
printf("%d", n);  // ❌ DON'T MIX!
\`\`\`

### Benchmark After Disabling Sync

\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
    }
    return 0;
}
// Time: ~1.0 seconds (2.5x faster!)
\`\`\`

---

## Fast I/O Optimization 2: Untie cin from cout

By default, \`cin\` is **tied** to \`cout\`. This means before every \`cin\` operation, \`cout\` is flushed.

### The Problem

\`\`\`cpp
cout << "Enter number: ";
cin >> n;  // cout is flushed here automatically
\`\`\`

This is useful for interactive I/O but **adds overhead** in competitive programming where you're reading large inputs.

### Solution: Untie cin

\`\`\`cpp
cin.tie(NULL);
\`\`\`

or equivalently:
\`\`\`cpp
cin.tie(nullptr);
\`\`\`

### Full Fast I/O Setup

\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Now use cin/cout optimally
    int n;
    cin >> n;
    cout << n << "\\n";  // Use \\n instead of endl!
    
    return 0;
}
\`\`\`

**Why \\n instead of endl?**
- \`endl\` flushes the buffer every time (slow!)
- \`\\n\` just adds newline (fast!)

\`\`\`cpp
// Slow:
cout << n << endl;  // Flushes buffer

// Fast:
cout << n << "\\n";  // Just adds newline
\`\`\`

### Benchmark with Full Optimization

\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cout << x << "\\n";
    }
    return 0;
}
// Time: ~0.6 seconds (4x faster than original!)
\`\`\`

---

## scanf/printf for Extreme Speed

For **maximum speed**, use C's \`scanf\` and \`printf\`. They're generally faster than even optimized cin/cout.

### scanf Syntax

\`\`\`cpp
#include <cstdio>
using namespace std;

int main() {
    int n;
    scanf("%d", &n);  // Note the & for address
    
    long long big;
    scanf("%lld", &big);
    
    double d;
    scanf("%lf", &d);
    
    char s[100];
    scanf("%s", s);  // No & for arrays
    
    return 0;
}
\`\`\`

### printf Syntax

\`\`\`cpp
#include <cstdio>
using namespace std;

int main() {
    int n = 42;
    printf("%d\\n", n);
    
    long long big = 1000000000000LL;
    printf("%lld\\n", big);
    
    double d = 3.14159;
    printf("%.6lf\\n", d);  // 6 decimal places
    
    printf("%d %d %d\\n", a, b, c);  // Multiple values
    
    return 0;
}
\`\`\`

### Common Format Specifiers

| Type | scanf | printf | Example |
|------|-------|--------|---------|
| int | %d | %d | 42 |
| long long | %lld | %lld | 1000000000000 |
| unsigned int | %u | %u | 42 |
| double | %lf | %lf | 3.14 |
| float | %f | %f | 3.14 |
| char | %c | %c | 'A' |
| string | %s | %s | "hello" |
| hex | %x | %x | 2a |

### Benchmark: scanf vs cin

\`\`\`cpp
// Using scanf/printf
#include <cstdio>

int main() {
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        int x;
        scanf("%d", &x);
        printf("%d\\n", x);
    }
    return 0;
}
// Time: ~0.4 seconds
\`\`\`

**Speed comparison:**
- Original cin/cout: 2.5s
- Optimized cin/cout: 0.6s
- scanf/printf: 0.4s

---

## getchar/putchar: Maximum Speed

For **ultimate speed** when reading/writing single characters, use \`getchar\` and \`putchar\`.

### Reading an Integer with getchar

\`\`\`cpp
inline int fastRead() {
    int n = 0;
    char c = getchar();
    
    // Skip non-digit characters
    while (c < '0' || c > '9') c = getchar();
    
    // Read digits
    while (c >= '0' && c <= '9') {
        n = n * 10 + (c - '0');
        c = getchar();
    }
    
    return n;
}

int main() {
    int x = fastRead();
    printf("%d\\n", x);
    return 0;
}
\`\`\`

### Reading with Negative Numbers

\`\`\`cpp
inline int fastRead() {
    int n = 0;
    char c = getchar();
    bool negative = false;
    
    // Skip whitespace
    while (c < '0' || c > '9') {
        if (c == '-') negative = true;
        c = getchar();
    }
    
    // Read digits
    while (c >= '0' && c <= '9') {
        n = n * 10 + (c - '0');
        c = getchar();
    }
    
    return negative ? -n : n;
}
\`\`\`

### Writing an Integer with putchar

\`\`\`cpp
inline void fastWrite(int n) {
    if (n < 0) {
        putchar('-');
        n = -n;
    }
    
    if (n >= 10) {
        fastWrite(n / 10);
    }
    
    putchar(n % 10 + '0');
}

int main() {
    fastWrite(12345);
    putchar('\\n');
    return 0;
}
\`\`\`

**When to use getchar/putchar:**
- Reading single characters in loop
- Reading huge input (millions of integers)
- Every microsecond counts
- **Usually not necessary!** Optimized cin/cout is enough for most problems

---

## Reading Entire Lines Efficiently

### Using getline

\`\`\`cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    string line;
    getline(cin, line);
    
    cout << line << "\\n";
    
    return 0;
}
\`\`\`

### Reading Until End of File

\`\`\`cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    string line;
    while (getline(cin, line)) {
        // Process line
        cout << line << "\\n";
    }
    
    return 0;
}
\`\`\`

### Reading Multiple Integers on One Line

\`\`\`cpp
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    string line;
    getline(cin, line);
    
    istringstream iss(line);
    vector<int> nums;
    int x;
    while (iss >> x) {
        nums.push_back(x);
    }
    
    return 0;
}
\`\`\`

---

## Output Buffering

### The Problem

\`\`\`cpp
for (int i = 0; i < 1000000; i++) {
    cout << i << "\\n";
    // Slow! Writing to console for each iteration
}
\`\`\`

### Solution: Build String First (Sometimes)

\`\`\`cpp
#include <iostream>
#include <sstream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    ostringstream oss;
    for (int i = 0; i < 1000000; i++) {
        oss << i << "\\n";
    }
    
    cout << oss.str();  // Single write
    
    return 0;
}
\`\`\`

**When to use:**
- Output is huge
- Need to process output before printing
- **Downside:** Uses more memory

**Usually not necessary:** Optimized cout with \\n is fast enough!

---

## Fast I/O Template

Here's a ready-to-use fast I/O template:

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Fast I/O setup
void setupFastIO() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}

int main() {
    setupFastIO();
    
    // Your code here
    int n;
    cin >> n;
    
    cout << n << "\\n";
    
    return 0;
}
\`\`\`

**Simpler inline version:**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Your code here
    
    return 0;
}
\`\`\`

**This should be in EVERY competitive programming solution!**

---

## Speed Comparisons

Let's compare all methods for reading and writing 1 million integers:

| Method | Read Time | Write Time | Total | Difficulty |
|--------|-----------|------------|-------|------------|
| Standard cin/cout | 1.5s | 1.0s | 2.5s | Easy |
| Optimized cin/cout | 0.4s | 0.2s | 0.6s | Easy |
| scanf/printf | 0.3s | 0.1s | 0.4s | Medium |
| getchar/putchar | 0.1s | 0.1s | 0.2s | Hard |

**Recommendation:** Use **optimized cin/cout** for 99% of problems!

---

## When to Use Which Method

### Use Optimized cin/cout When:

✅ Normal problems (most cases)
✅ Input size < 10^6
✅ Time limit >= 2 seconds
✅ Need to read strings, complex types
✅ Want readable code

\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(NULL);
cin >> n;
cout << n << "\\n";
\`\`\`

### Use scanf/printf When:

✅ Very tight time limits (1 second)
✅ Huge input (millions of integers)
✅ Every millisecond matters
✅ C-style code acceptable

\`\`\`cpp
scanf("%d", &n);
printf("%d\\n", n);
\`\`\`

### Use getchar/putchar When:

✅ Ultra-tight time limits
✅ Reading single characters in tight loop
✅ Maximum performance needed
✅ Ready to write complex code

\`\`\`cpp
int n = fastRead();
fastWrite(n);
\`\`\`

---

## Common Mistakes

### Mistake 1: Mixing I/O Methods

❌ **WRONG:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin >> n;
printf("%d", n);  // DON'T MIX!
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin >> n;
cout << n << "\\n";  // Consistent
\`\`\`

### Mistake 2: Using endl Instead of \\n

❌ **SLOW:**
\`\`\`cpp
for (int i = 0; i < n; i++) {
    cout << i << endl;  // Flushes every time!
}
\`\`\`

✅ **FAST:**
\`\`\`cpp
for (int i = 0; i < n; i++) {
    cout << i << "\\n";
}
\`\`\`

### Mistake 3: Forgetting Fast I/O

❌ **GETS TLE:**
\`\`\`cpp
int main() {
    // Forgot fast I/O!
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        cin >> x;  // Slow!
    }
}
\`\`\`

✅ **ACCEPTED:**
\`\`\`cpp
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n = 1000000;
    for (int i = 0; i < n; i++) {
        cin >> x;  // Fast!
    }
}
\`\`\`

### Mistake 4: Using cout.flush()

❌ **SLOW:**
\`\`\`cpp
cout << n;
cout.flush();  // Explicit flush (slow!)
\`\`\`

✅ **FAST:**
\`\`\`cpp
cout << n << "\\n";  // No flush unless needed
\`\`\`

**Exception:** Interactive problems where you NEED to flush:
\`\`\`cpp
cout << query << endl;  // endl flushes (needed for interactive)
cout.flush();           // Or explicit flush
\`\`\`

---

## Interactive Problems: When to Flush

**Interactive problems** require communication with judge. After each query, you MUST flush!

### Example: Binary Search Game

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    // DON'T disable sync for interactive!
    // ios_base::sync_with_stdio(false);  // ❌
    
    int left = 1, right = 1000000;
    
    while (left < right) {
        int mid = (left + right) / 2;
        
        // Query judge
        cout << "? " << mid << endl;  // endl flushes!
        cout.flush();  // Ensure it's sent
        
        // Read response
        int response;
        cin >> response;
        
        if (response == 1) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Answer
    cout << "! " << left << endl;
    
    return 0;
}
\`\`\`

**Rules for interactive problems:**
1. **DON'T** disable sync with \`ios_base::sync_with_stdio(false)\`
2. **DO** flush after every query (\`endl\` or \`cout.flush()\`)
3. **DO** read judge's response before next query

---

## Real Contest Examples

### Example 1: TLE Due to Slow I/O

**Problem:** Read N integers, output their sum
**Constraints:** N ≤ 10^6, Time Limit = 1 second

❌ **Gets TLE:**
\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;  // Slow!
        sum += x;
    }
    
    cout << sum << endl;
    return 0;
}
// Time: ~1.5 seconds → TLE!
\`\`\`

✅ **Accepted:**
\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;  // Fast!
        sum += x;
    }
    
    cout << sum << "\\n";
    return 0;
}
// Time: ~0.4 seconds → Accepted!
\`\`\`

### Example 2: Multiple Test Cases

**Problem:** T test cases, each with N integers
**Constraints:** T ≤ 100, N ≤ 10^4, Time Limit = 2 seconds

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    cin >> t;
    
    while (t--) {
        int n;
        cin >> n;
        
        for (int i = 0; i < n; i++) {
            int x;
            cin >> x;
            // Process
        }
        
        cout << answer << "\\n";
    }
    
    return 0;
}
\`\`\`

**Without fast I/O:** Would likely TLE on large inputs!

---

## Debugging with Fast I/O

### Problem: Can't Use cin/scanf Together

If you accidentally mix I/O methods, output becomes jumbled.

**Debug by:**
1. Choose ONE method (cin/cout or scanf/printf)
2. Stick with it throughout solution
3. If using fast I/O, can't use scanf/printf

### Testing Fast I/O Locally

**Generate large input:**
\`\`\`bash
python3 -c "print(1000000); print(' '.join(str(i) for i in range(1000000)))" > large_input.txt
\`\`\`

**Test your solution:**
\`\`\`bash
time ./solution < large_input.txt > output.txt
\`\`\`

**Compare times with/without fast I/O!**

---

## Summary

**Essential Fast I/O Setup (Use in EVERY solution):**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Your code here
    
    return 0;
}
\`\`\`

**Key Points:**

✅ Always use \`ios_base::sync_with_stdio(false)\`
✅ Always use \`cin.tie(NULL)\`
✅ Use \`\\n\` instead of \`endl\`
✅ Don't mix cin/cout with scanf/printf
✅ For interactive problems, DON'T disable sync, DO flush

**Speed Ranking:**
1. getchar/putchar (fastest, hardest)
2. scanf/printf (very fast, medium difficulty)
3. Optimized cin/cout (fast enough, easy) ← **USE THIS**
4. Standard cin/cout (slow, easy)

**Rule of thumb:** Optimized cin/cout is sufficient for 99% of problems. Only use scanf/printf or getchar/putchar for extreme cases.

---

## Next Steps

Now that you can read and write data efficiently, let's review **C++ Basics** specifically for competitive programming!

**Key Takeaway**: Two lines of fast I/O setup can be the difference between Accepted and TLE. Make it a habit to include them in every solution!
`,
  quizId: 'cp-m1-s5-quiz',
  discussionId: 'cp-m1-s5-discussion',
} as const;
