export default {
    id: 'cp-m1-s12-discussion',
    title: 'Common Compilation Errors - Discussion Questions',
    questions: [
        {
            question: 'Compilation errors can waste significant time in contests. Describe a systematic approach to reading and fixing C++ compiler errors, including how to interpret cryptic template error messages.',
            answer: `Compiler errors are inevitable, but handling them efficiently separates fast coders from slow ones. Here's the complete system:

**The Golden Rule: Read from TOP to BOTTOM**

When you see this:
\`\`\`
solution.cpp:10: error: 'vector' was not declared
solution.cpp:15: error: 'v' was not declared
solution.cpp:20: error: 'v' was not declared
solution.cpp:25: error: 'v' was not declared
... 50 more errors
\`\`\`

**DON'T PANIC!** The first error caused all others (cascading errors).

**Fix the FIRST error only, then recompile.**

**Systematic Error-Fixing Process:**

**Step 1: Look at LINE NUMBER**
\`\`\`
solution.cpp:15:23: error: expected ';' before 'return'
\`\`\`
- Error detected on line 15
- BUT bug might be on line 14! (missing semicolon)

**Step 2: Read ERROR MESSAGE**
\`\`\`
error: expected ';' before 'return'
\`\`\`
- "expected ';'" = missing semicolon
- "before 'return'" = on previous line

**Step 3: Check CONTEXT**
\`\`\`
   14 |     int x = 5
      |              ^
      |              ;
   15 |     return x;
\`\`\`
Compiler shows where and suggests fix!

**Step 4: FIX and RECOMPILE**
Don't try to fix all errors at once!

**Common Error Patterns:**

**Pattern 1: Missing Semicolon**

Error:
\`\`\`
error: expected ';' before 'X'
\`\`\`

Location: Line BEFORE the error

Fix:
\`\`\`cpp
// Before:
int x = 5
return x;

// After:
int x = 5;
return x;
\`\`\`

**Pattern 2: Undeclared Identifier**

Error:
\`\`\`
error: 'vector' was not declared in this scope
\`\`\`

Causes:
1. Missing include
2. Wrong namespace
3. Typo in name

Fix:
\`\`\`cpp
// Missing include:
#include <vector>

// Or use bits/stdc++.h:
#include <bits/stdc++.h>
using namespace std;
\`\`\`

**Pattern 3: Template Arguments Missing**

Error:
\`\`\`
error: missing template arguments before 'v'
\`\`\`

Fix:
\`\`\`cpp
// Before:
vector v;

// After:
vector<int> v;
\`\`\`

**Pattern 4: Type Mismatch**

Error:
\`\`\`
error: cannot convert 'string' to 'int' in assignment
\`\`\`

Fix:
\`\`\`cpp
// Before:
int x = "hello";

// After:
string x = "hello";
\`\`\`

**Pattern 5: Wrong Number of Arguments**

Error:
\`\`\`
error: too few arguments to function 'void foo(int, int)'
\`\`\`

Fix: Check function signature and call

**Interpreting Template Errors:**

Template errors are VERBOSE and SCARY. Don't read all of it!

Example error:
\`\`\`
error: no match for 'operator<' (operand types are 'std::pair<int, int>' and 'std::pair<int, int>')
note: candidate: template<class T1, class T2> bool operator<(const std::pair<T1, T2>&, const std::pair<T1, T2>&)
note:   template argument deduction/substitution failed:
note:   cannot convert 'std::pair<int, std::__cxx11::basic_string<char>>' to 'std::pair<int, int>'
[... 50 more lines ...]
\`\`\`

**How to read it:**

1. **Read FIRST line only:**
   \`error: no match for 'operator<'\`
   
2. **Identify the problem:**
   Missing operator< for your type

3. **Ignore the rest** (template instantiation details)

4. **Fix:** Define operator< or use different comparison

**Common Template Error Causes:**

**Cause 1: Type Deduction Failure**
\`\`\`cpp
template<typename T>
T max(T a, T b) { return (a > b) ? a : b; }

max(5, 3.14);  // Error: T is int or double?
\`\`\`

Fix:
\`\`\`cpp
max<double>(5, 3.14);  // Explicit type
// Or:
max(5.0, 3.14);  // Both double
\`\`\`

**Cause 2: Missing Operator**
\`\`\`cpp
struct Point { int x, y; };

vector<Point> v;
sort(v.begin(), v.end());  // Error: no operator< for Point
\`\`\`

Fix:
\`\`\`cpp
bool operator<(const Point& a, const Point& b) {
    return a.x < b.x;  // Define comparison
}
\`\`\`

**Cause 3: Wrong Template Arguments**
\`\`\`cpp
priority_queue<int, greater<int>> pq;  // Error: Missing middle argument
\`\`\`

Fix:
\`\`\`cpp
priority_queue<int, vector<int>, greater<int>> pq;
\`\`\`

**Quick Error Reference:**

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| expected ';' | Missing semicolon | Add ; on previous line |
| was not declared | Missing include/typo | Add include or fix name |
| missing template arguments | Forgot <type> | Add template argument |
| cannot convert | Type mismatch | Fix variable type |
| no matching function | Wrong arguments | Check function signature |
| expected '}' | Missing brace | Add closing brace |
| undefined reference | No function definition | Define the function |

**Pre-Submission Error Prevention:**

**Use these compiler flags:**
\`\`\`bash
g++ -std=c++17 -O2 -Wall -Wextra -Wshadow solution.cpp
\`\`\`

**Warnings to never ignore:**
- unused variables
- uninitialized variables  
- comparison between signed/unsigned
- missing return statement

**These ARE bugs waiting to happen!**

**Advanced: Understanding Linker Errors**

Different from compiler errors!

**Linker error:**
\`\`\`
undefined reference to \`solve()'
\`\`\`

**Means:** Function declared but not defined

Fix:
\`\`\`cpp
void solve();  // Declaration

int main() {
    solve();  // Call
}

void solve() {  // Add definition!
    // Implementation
}
\`\`\`

**IDE Integration:**

Modern IDEs show errors in real-time:
- Red underlines = errors
- Yellow underlines = warnings
- Click for details

**Use this BEFORE compiling!**

**Debugging Strategy When Stuck:**

If error message is unclear:

1. **Google the error** (first line only)
2. **Simplify the code** (remove parts until it compiles)
3. **Compare with working code** (your template)
4. **Ask compiler for details:**
   \`\`\`bash
   g++ -v solution.cpp  # Verbose output
   \`\`\`

**Time-Saving Tips:**

1. **Fix errors one at a time**
   - Don't try to fix 50 errors at once
   - Recompile after each fix

2. **Most errors are simple**
   - Missing semicolon: 30%
   - Missing include: 20%
   - Typos: 20%
   - Type errors: 15%
   - Everything else: 15%

3. **Check the obvious first**
   - Semicolons
   - Braces
   - Includes
   - Typos

4. **Use auto-formatting**
   - Reveals missing braces
   - Makes code readable

**Practice Exercise:**

Find the errors:
\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    vector<int> v
    for(int i = 0; i < 10; i++) {
        v.push_back(i)
    }
    sort(v.begin(), v.end())
    return 0
}
\`\`\`

Errors:
1. Line 1: Missing \`<vector>\` and \`<algorithm>\` includes
2. Line 4: Missing \`<int>\` template argument
3. Line 4: Missing semicolon
4. Line 6: Missing semicolon
5. Line 8: Missing semicolon
6. Line 9: Missing semicolon

**Bottom Line:**

Compilation errors:
- ✅ Read first error only
- ✅ Check line before error too
- ✅ Fix one at a time
- ✅ Recompile after each fix
- ✅ Ignore cascading errors
- ✅ Template errors: read first line only

**With practice, you'll fix most errors in <30 seconds!**`,
        },
        {
            question: 'Many competitive programmers have compilation shortcuts and error-handling workflows. Describe an efficient compile-test-debug cycle that minimizes time wasted on errors.',
            answer: `An efficient workflow can save 5-10 minutes per contest. Here's the complete system:

**The Fast Compile-Test-Debug Cycle:**

\`\`\`
1. WRITE code
   ↓
2. COMPILE (with warnings)
   ↓ (if errors)
3. FIX first error → back to 2
   ↓ (if success)
4. TEST on samples
   ↓ (if wrong)
5. DEBUG → back to 1
   ↓ (if correct)
6. SUBMIT
\`\`\`

**Step 1: Setup (One-Time)**

Create compilation script \`compile.sh\`:
\`\`\`bash
#!/bin/bash
g++ -std=c++17 -O2 -Wall -Wextra -Wshadow "$1" -o "\${1 %.cpp}" && echo "✓ Compiled successfully"
\`\`\`

Make executable:
\`\`\`bash
chmod +x compile.sh
\`\`\`

Usage:
\`\`\`bash
./compile.sh solution.cpp
\`\`\`

**Step 2: IDE Integration**

**VS Code tasks.json:**
\`\`\`json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Compile & Run",
            "type": "shell",
            "command": "g++ -std=c++17 -O2 -Wall -Wextra \${file} -o \${fileDirname}/solution && \${fileDirname}/solution < \${fileDirname}/input.txt",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ]
}
\`\`\`

**Keybinding:** Ctrl+Shift+B

**Step 3: Automatic Testing**

Create \`test.sh\`:
\`\`\`bash
#!/bin/bash

# Compile
g++ -std=c++17 -O2 -Wall -Wextra solution.cpp -o solution || exit 1

# Test on all inputs
for input in test*.txt; do
    echo "Testing $input..."
    ./solution < "$input"
    echo "---"
done
\`\`\`

**Step 4: Error Handling Workflow**

**When you see errors:**

\`\`\`bash
# Step 1: Compile and capture errors
g++ solution.cpp 2> errors.txt

# Step 2: View first error only
head -1 errors.txt

# Step 3: Fix it

# Step 4: Recompile
g++ solution.cpp
\`\`\`

**Or use this alias:**
\`\`\`bash
alias compile='g++ -std=c++17 -O2 -Wall -Wextra'
alias testcp='compile solution.cpp && ./a.out < input.txt'
\`\`\`

**Step 5: Quick Fix Checklist**

Before compiling, check:
\`\`\`
✓ All semicolons present?
✓ All braces matched?
✓ Includes at top?
✓ using namespace std?
✓ Template arguments complete?
\`\`\`

**Auto-format (Ctrl+Shift+I in VS Code) helps catch these!**

**Step 6: Fast Debugging During Errors**

**Use preprocessor flags:**
\`\`\`cpp
#define LOCAL

#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif

int main() {
    int x = 42;
    debug(x);  // Only shows locally
    // ...
}
\`\`\`

Compile with:
\`\`\`bash
g++ -DLOCAL solution.cpp
\`\`\`

**Step 7: Makefile for One-Command Everything**

\`\`\`makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wshadow

.PHONY: test clean

# Compile and run
test: solution.cpp
	$(CXX) $(CXXFLAGS) solution.cpp -o solution
	./solution < input.txt

# Just compile
build: solution.cpp
	$(CXX) $(CXXFLAGS) solution.cpp -o solution

# Clean
clean:
	rm -f solution a.out
\`\`\`

Usage: \`make test\`

**Step 8: Error Pattern Recognition**

**Train your brain to recognize errors instantly:**

See "expected ';'" → Add semicolon to previous line (5 seconds)
See "was not declared" → Add include (5 seconds)
See "missing template" → Add <type> (5 seconds)

**Don't read the whole message, just the key phrase!**

**Step 9: Multiple Test Cases**

\`\`\`bash
#!/bin/bash

g++ solution.cpp -o solution || exit 1

passed=0
failed=0

for i in {1..10}; do
    if ./solution < test$i.txt | diff - output$i.txt > /dev/null; then
        echo "✓ Test $i passed"
        ((passed++))
    else
        echo "✗ Test $i failed"
        echo "Input:"
        cat test$i.txt
        echo "Expected:"
        cat output$i.txt
        echo "Got:"
        ./solution < test$i.txt
        ((failed++))
    fi
done

echo "$passed passed, $failed failed"
\`\`\`

**Step 10: Contest Day Workflow**

\`\`\`
Problem opened
    ↓
Copy template → solution.cpp
    ↓
Write solution
    ↓
Ctrl+Shift+B (compile & run on samples)
    ↓
Fix errors (if any) → repeat
    ↓
All samples pass?
    ↓
Test edge cases
    ↓
Submit!
\`\`\`

**Time Budget:**

For a 20-minute problem:
- Writing: 12 minutes
- Compiling/fixing errors: 2 minutes  
- Testing: 4 minutes
- Submitting: 2 minutes

**If spending >2 minutes on compilation errors, something is wrong with workflow!**

**Optimization Tips:**

**1. Precompiled Headers (Advanced)**
\`\`\`bash
# Precompile bits/stdc++.h
g++ -std=c++17 -O2 /usr/include/x86_64-linux-gnu/c++/9/bits/stdc++.h

# Now compiles faster
\`\`\`

**2. Compiler Cache**
\`\`\`bash
# Install ccache
sudo apt install ccache

# Use with g++
ccache g++ solution.cpp
\`\`\`

**3. Split Screen Setup**
\`\`\`
+----------------+----------------+
|                |                |
|   Code Editor  |  Terminal      |
|                |  (running      |
|                |   solution)    |
|                |                |
+----------------+----------------+
\`\`\`

No context switching!

**Common Workflow Mistakes:**

**Mistake 1: Not using fast I/O setup**
Always include:
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**Mistake 2: Manually typing compile command**
Use scripts/makefiles/IDE shortcuts!

**Mistake 3: Testing one input at a time**
Test all samples at once!

**Mistake 4: Not fixing warnings**
Warnings today = bugs tomorrow

**Mistake 5: No input file**
Keep input.txt with samples ready

**My Personal Workflow:**

\`\`\`bash
# Setup
alias c='g++ -std=c++17 -O2 -Wall -Wextra'
alias r='./a.out < input.txt'
alias t='c solution.cpp && r'

# In contest:
vim solution.cpp  # Write solution
:!t               # Compile and test (from vim)
# If errors, fix and repeat
# If works, submit
\`\`\`

**Time saved per problem: 2-3 minutes**
**Time saved per contest: 10-15 minutes**

**That's enough time for an extra problem!**

**Pre-Contest Checklist:**

Before contest starts:
\`\`\`
✓ Template file ready
✓ Compile script works
✓ Test inputs/outputs setup
✓ IDE shortcuts configured
✓ Fast I/O in template
✓ Debug macros ready
✓ Internet stable
✓ Backup editor ready
\`\`\`

**Bottom Line:**

Efficient workflow:
- ✅ One-key compilation
- ✅ Automatic testing  
- ✅ Fast error fixing
- ✅ No context switching
- ✅ Practiced and automatic

Set it up once, benefit in every contest!`,
        },
        {
            question: 'Some compiler errors are particularly misleading or confusing for beginners. Discuss the most commonly misunderstood error messages in C++ and explain what they actually mean with examples.',
            answer: `Some C++ errors are notorious for being confusing. Here's the truth behind the most misleading ones:

**Misleading Error #1: "Expected Primary Expression"**

**What you see:**
\`\`\`
error: expected primary-expression before 'int'
\`\`\`

**What it actually means:**
You used a type name where a value was expected

**Example:**
\`\`\`cpp
// Wrong:
vector<int> v;
v.push_back(int);  // Error here!

// You meant:
v.push_back(5);  // A value, not a type
\`\`\`

**Another common cause:**
\`\`\`cpp
// Wrong:
if(x > 5) {
    int y = 10;
}
cout << y;  // Error: y not in scope

// This gives "expected primary-expression" because y doesn't exist here
\`\`\`

**Misleading Error #2: "No Match for Operator<<"**

**What you see:**
\`\`\`
error: no match for 'operator<<' (operand types are 'std::ostream' and 'Point')
\`\`\`

**What beginners think:** "cout is broken!"

**What it actually means:**
Your custom type doesn't have a << operator defined

**Example:**
\`\`\`cpp
struct Point {
    int x, y;
};

Point p = {1, 2};
cout << p;  // Error! How to print a Point?

// Fix: Define operator<<
ostream& operator<<(ostream& os, const Point& p) {
    return os << "(" << p.x << ", " << p.y << ")";
}

// Now works:
cout << p;  // Prints: (1, 2)
\`\`\`

**Misleading Error #3: "Invalid Use of Template-Name"**

**What you see:**
\`\`\`
error: invalid use of template-name 'vector' without an argument list
\`\`\`

**What beginners think:** "I can't use vector?"

**What it actually means:**
Forgot template arguments

**Example:**
\`\`\`cpp
// Wrong:
vector v;  // What type of elements?

// Right:
vector<int> v;  // vector OF int
\`\`\`

**Misleading Error #4: "Cannot Convert"**

**What you see:**
\`\`\`
error: cannot convert 'std::__cxx11::basic_string<char>' to 'int'
\`\`\`

**What beginners think:** *confused by std::__cxx11::basic_string*

**What it actually means:**
\`\`\`
cannot convert 'string' to 'int'
\`\`\`
(Ignore the std::__cxx11 noise!)

**Example:**
\`\`\`cpp
int x = "hello";  // Can't assign string to int!
\`\`\`

**Misleading Error #5: "Template Argument Deduction/Substitution Failed"**

**What you see:**
\`\`\`
error: template argument deduction/substitution failed
note: deduced conflicting types for parameter 'T' ('int' and 'double')
\`\`\`

**What beginners think:** *runs away screaming*

**What it actually means:**
Function template can't figure out which type to use

**Example:**
\`\`\`cpp
template<typename T>
T max(T a, T b) { return (a > b) ? a : b; }

max(5, 3.14);  // Error! T is int or double?

// Fix 1: Be explicit
max<double>(5, 3.14);

// Fix 2: Make both same type
max(5.0, 3.14);
\`\`\`

**Misleading Error #6: "Does Not Name a Type"**

**What you see:**
\`\`\`
error: 'string' does not name a type
\`\`\`

**What beginners think:** "String is not a type??"

**What it actually means:**
One of these:
1. Missing include
2. Forgot std:: or using namespace std
3. Typo

**Example:**
\`\`\`cpp
// Missing include:
string s = "hello";  // Error!

// Fix:
#include <string>
using namespace std;
string s = "hello";  // OK
\`\`\`

**Misleading Error #7: "No Matching Function for Call"**

**What you see:**
\`\`\`
error: no matching function for call to 'solve(int&)'
note: candidate: void solve()
note:   candidate expects 0 arguments, 1 provided
\`\`\`

**What beginners think:** "But I have a solve function!"

**What it actually means:**
Function exists but with wrong signature

**Example:**
\`\`\`cpp
void solve() {  // Takes 0 arguments
    // ...
}

int main() {
    int n = 5;
    solve(n);  // Calling with 1 argument - mismatch!
}

// Fix: Either change definition or remove argument
\`\`\`

**Misleading Error #8: "Expected Unqualified-Id Before..."**

**What you see:**
\`\`\`
error: expected unqualified-id before 'for'
\`\`\`

**What it actually means:**
Missing semicolon on previous line

**Example:**
\`\`\`cpp
int x = 5  // Missing semicolon
for(int i = 0; i < n; i++) {  // Error reported here
    // ...
}

// Fix: Add semicolon to line above
int x = 5;
\`\`\`

**Misleading Error #9: "Undefined Reference to..."**

**What you see:**
\`\`\`
undefined reference to \`solve()'
\`\`\`

**What beginners think:** "But I have solve()!"

**What it actually means:**
You DECLARED solve() but never DEFINED it

**Example:**
\`\`\`cpp
void solve();  // Declaration only

int main() {
    solve();  // Call
}
// Error: Where's the implementation?

// Fix: Add definition
void solve() {  // Definition
    // Implementation here
}
\`\`\`

**Misleading Error #10: "Lvalue Required as Left Operand"**

**What you see:**
\`\`\`
error: lvalue required as left operand of assignment
\`\`\`

**What beginners think:** *confused by "lvalue"*

**What it actually means:**
Trying to assign to something that can't be assigned to

**Example:**
\`\`\`cpp
5 = x;  // Can't assign to literal 5!
(x + y) = 10;  // Can't assign to expression!

// Often caused by:
if(x = 5) { }  // Assignment in condition (probably meant ==)
\`\`\`

**Misleading Error #11: "Use of Deleted Function"**

**What you see:**
\`\`\`
error: use of deleted function 'std::pair<int, int>::pair()'
\`\`\`

**What it actually means:**
Trying to use a function that doesn't exist (was deleted)

**Example:**
\`\`\`cpp
pair<int, int> p;  // Error if no default constructor
// Some types don't have default constructors

// Fix:
pair<int, int> p = {1, 2};  // Provide values
// Or:
pair<int, int> p(1, 2);
\`\`\`

**Misleading Error #12: "Incomplete Type"**

**What you see:**
\`\`\`
error: invalid use of incomplete type 'struct Node'
\`\`\`

**What it actually means:**
Forward declared but not defined yet

**Example:**
\`\`\`cpp
struct Node;  // Forward declaration

Node* ptr;  // OK - pointer to incomplete type
Node n;  // Error - need full definition!

// Fix: Define the struct
struct Node {
    int value;
    Node* next;
};
\`\`\`

**How to Decode Any Error:**

**Step 1: Ignore the noise**
\`\`\`
error: cannot convert 'std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>' to 'int'
\`\`\`
becomes:
\`\`\`
cannot convert 'string' to 'int'
\`\`\`

**Step 2: Find the actual error**
Skip "note:", "candidate:", "in instantiation of"
Focus on the line starting with "error:"

**Step 3: Check common causes**
- Semicolon missing?
- Include missing?
- Type wrong?
- Arguments wrong?

**Error Message Translator:**

| Cryptic Message | Plain English |
|-----------------|---------------|
| expected primary-expression | Used type instead of value |
| does not name a type | Missing include or typo |
| no matching function | Wrong arguments |
| cannot convert | Wrong type |
| undefined reference | Declared but not defined |
| template deduction failed | Types don't match |
| lvalue required | Can't assign to this |
| incomplete type | Need full definition |

**Pro Tip: Google Smart**

Don't copy entire error message!

**Bad search:**
\`std::__cxx11::basic_string<char> cannot convert int line 42\`

**Good search:**
\`cpp cannot convert string to int\`

**Bottom Line:**

Most "confusing" errors are actually simple:
- ✅ Ignore template noise
- ✅ Read first error only
- ✅ Check obvious causes
- ✅ Google the cleaned-up error

With experience, these errors become instantly recognizable!`,
        },
    ],
} as const;

