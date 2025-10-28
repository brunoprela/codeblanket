export default {
  id: 'cp-m1-s14-discussion',
  title: "Reading Other People's C++ Code - Discussion Questions",
  questions: [
    {
      question:
        "Top competitive programmers often have unique coding styles with heavy macro usage and terse notation. Describe strategies for quickly understanding unfamiliar competitive programming code, especially when learning from others' solutions.",
      answer: `Reading others' CP code is essential for learning, but it can be cryptic. Here's how to decode it efficiently:

**The Reality of CP Code:**

Tourist's code:
\`\`\`cpp
#define pb push_back
#define eb emplace_back
#define sz(x) (int)(x).size()
#define all(x) (x).begin(),(x).end()
#define F first
#define S second
typedef long long ll;
typedef pair<int,int> pii;
\`\`\`

**Beginners see:** Gibberish
**Experts see:** Standard abbreviations

**Strategy 1: Learn Common Macro Patterns**

**Essential Macros Dictionary:**

| Macro | Meaning | Example Usage |
|-------|---------|---------------|
| \`ll\` | \`long long\` | \`ll x = 1e18;\` |
| \`pii\` | \`pair<int,int>\` | \`pii p = {1, 2};\` |
| \`vi\` | \`vector<int>\` | \`vi arr(n);\` |
| \`pb\` | \`push_back\` | \`v.pb(5);\` |
| \`eb\` | \`emplace_back\` | \`v.eb(5);\` |
| \`all(x)\` | \`x.begin(),x.end()\` | \`sort(all(v));\` |
| \`sz(x)\` | \`(int)x.size()\` | \`for(int i=0;i<sz(v);i++)\` |
| \`F\` | \`first\` | \`p.F = 5;\` |
| \`S\` | \`second\` | \`p.S = 10;\` |
| \`rep(i,n)\` | \`for(int i=0;i<n;i++)\` | \`rep(i,n) {...}\` |
| \`FOR(i,a,b)\` | \`for(int i=a;i<=b;i++)\` | \`FOR(i,1,n) {...}\` |

**Decode this:**
\`\`\`cpp
ll solve(vi& arr) {
    ll sum = 0;
    rep(i, sz(arr)) sum += arr[i];
    return sum;
}
\`\`\`

**Expanded:**
\`\`\`cpp
long long solve(vector<int>& arr) {
    long long sum = 0;
    for(int i = 0; i < (int)arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}
\`\`\`

**Strategy 2: Start from Main, Work Backwards**

Typical CP structure:
\`\`\`cpp
void solve() {
    // Solution logic
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t; cin >> t;
    while(t--) {
        solve();
    }
    return 0;
}
\`\`\`

**Reading order:**
1. Start at \`main()\` - understand structure
2. Look at \`solve()\` - main logic
3. Check helper functions as needed
4. Understand macros last

**Strategy 3: Identify Code Sections**

Well-structured CP code has sections:
\`\`\`cpp
// ============ MACROS & TYPES ============
#define ll long long
#define pii pair<int,int>
// ... more macros ...

// ============ CONSTANTS ============
const int MOD = 1e9 + 7;
const int MAXN = 1e5 + 5;

// ============ GLOBAL VARIABLES ============
int n, m;
vector<int> arr;

// ============ HELPER FUNCTIONS ============
int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}

// ============ MAIN SOLUTION ============
void solve() {
    // Core logic
}

// ============ MAIN ============
int main() {
    solve();
    return 0;
}
\`\`\`

**Read in this order:**
1. Main solution (\`solve()\`)
2. Helper functions (when referenced)
3. Global variables (when used)
4. Constants (when used)
5. Macros (when confused)

**Strategy 4: Pattern Recognition**

**Pattern: Segment Tree**
\`\`\`cpp
int tree[4*MAXN];

void build(int v, int tl, int tr) {
    if(tl == tr) {
        tree[v] = arr[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(2*v, tl, tm);
        build(2*v+1, tm+1, tr);
        tree[v] = tree[2*v] + tree[2*v+1];
    }
}
\`\`\`

**If you see:** \`tree[4*MAXN]\`, \`2*v\`, \`2*v+1\`, recursive structure
**It's probably:** Segment tree

**Pattern: DFS**
\`\`\`cpp
bool vis[MAXN];
vector<int> g[MAXN];

void dfs(int u) {
    vis[u] = true;
    for(int v : g[u]) {
        if(!vis[v]) dfs(v);
    }
}
\`\`\`

**If you see:** \`vis[]\` array, \`g[]\` adjacency list, recursive calls
**It's:** DFS

**Pattern: BFS**
\`\`\`cpp
queue<int> q;
bool vis[MAXN];

void bfs(int start) {
    q.push(start);
    vis[start] = true;
    while(!q.empty()) {
        int u = q.front(); q.pop();
        for(int v : g[u]) {
            if(!vis[v]) {
                vis[v] = true;
                q.push(v);
            }
        }
    }
}
\`\`\`

**If you see:** \`queue<>\`, \`while(!q.empty())\`, \`q.front()\`, \`q.pop()\`
**It's:** BFS

**Pattern: Dynamic Programming**
\`\`\`cpp
int dp[MAXN][MAXN];

void solve() {
    // Base cases
    for(int i = 0; i < n; i++) dp[i][0] = base[i];
    
    // Transitions
    for(int i = 0; i < n; i++) {
        for(int j = 1; j < m; j++) {
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + arr[i][j];
        }
    }
}
\`\`\`

**If you see:** \`dp[][]\` array, nested loops, building from smaller subproblems
**It's:** DP

**Strategy 5: Trace with Example**

**Given code:**
\`\`\`cpp
ll solve(vi& arr) {
    ll ans = 0;
    rep(i, sz(arr)) {
        if(arr[i] > 0) ans += arr[i];
    }
    return ans;
}
\`\`\`

**Trace with example:** \`arr = [-1, 2, -3, 4]\`

| i | arr[i] | arr[i] > 0? | ans |
|---|--------|-------------|-----|
| 0 | -1 | No | 0 |
| 1 | 2 | Yes | 2 |
| 2 | -3 | No | 2 |
| 3 | 4 | Yes | 6 |

**Conclusion:** Sums positive elements

**Strategy 6: Comment Complex Parts**

When reading:
\`\`\`cpp
// Original:
rep(i, n) {
    rep(j, m) {
        dp[i][j] = (i > 0 ? dp[i-1][j] : 0) + (j > 0 ? dp[i][j-1] : 0);
    }
}

// Add comments while reading:
rep(i, n) {  // For each row
    rep(j, m) {  // For each column
        // dp[i][j] = value from above + value from left
        dp[i][j] = (i > 0 ? dp[i-1][j] : 0) + (j > 0 ? dp[i][j-1] : 0);
    }
}
\`\`\`

**Strategy 7: Expand Macros Mentally**

**Original:**
\`\`\`cpp
vi arr(n);
rep(i, n) cin >> arr[i];
sort(all(arr));
\`\`\`

**Expanded (in your mind):**
\`\`\`cpp
vector<int> arr(n);
for(int i = 0; i < n; i++) cin >> arr[i];
sort(arr.begin(), arr.end());
\`\`\`

**Strategy 8: Recognize Idioms**

**Idiom: Fast I/O**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`
**Meaning:** Speed up cin/cout

**Idiom: Multiple Test Cases**
\`\`\`cpp
int t; cin >> t;
while(t--) solve();
\`\`\`
**Meaning:** Run solve() t times

**Idiom: 1-indexed Arrays**
\`\`\`cpp
int arr[MAXN + 1];
for(int i = 1; i <= n; i++) cin >> arr[i];
\`\`\`
**Meaning:** Array starts at index 1

**Idiom: Pair Sorting**
\`\`\`cpp
vector<pii> v;
sort(all(v));  // Sorts by first, then second
\`\`\`

**Idiom: Custom Comparator**
\`\`\`cpp
sort(all(v), greater<int>());  // Descending order
sort(all(v), [](pii a, pii b) { return a.S < b.S; });  // By second element
\`\`\`

**Strategy 9: Look for Edge Cases**

Experienced coders handle edge cases implicitly:
\`\`\`cpp
// Handles n=0 implicitly
rep(i, n) {  // Loop doesn't run if n=0
    // ...
}

// Handles empty container
if(!v.empty()) {  // Check before accessing
    int x = v.back();
}

// Handles single element
sort(all(v));  // Works even if v.size() == 1
\`\`\`

**Strategy 10: Understand Coding Style**

Different coders have different styles:

**Tourist's style:**
- Heavy macro usage
- Very short variable names
- Dense code

**Benq's style:**
- Templates and classes
- Modular structure
- Reusable components

**Um_nik's style:**
- Clear logic flow
- Descriptive names
- Less macro-heavy

**Practice Exercise:**

Decode this:
\`\`\`cpp
#define F first
#define S second
typedef pair<int,int> pii;

ll solve(vector<pii>& v) {
    sort(all(v), [](pii a, pii b) { return a.S < b.S; });
    ll ans = 0, last = -1e18;
    for(auto& p : v) {
        if(p.F >= last) {
            ans++;
            last = p.S;
        }
    }
    return ans;
}
\`\`\`

**Expanded:**
\`\`\`cpp
long long solve(vector<pair<int,int>>& v) {
    // Sort pairs by second element
    sort(v.begin(), v.end(), [](pair<int,int> a, pair<int,int> b) {
        return a.second < b.second;
    });
    
    long long ans = 0;
    long long last = -1000000000000000000LL;
    
    for(auto& p : v) {
        // If pair.first >= last endpoint
        if(p.first >= last) {
            ans++;
            last = p.second;
        }
    }
    return ans;
}
\`\`\`

**Algorithm:** Activity selection / interval scheduling

**Quick Tips:**

1. **Don't get intimidated** by macros - they're just abbreviations
2. **Focus on logic first**, syntax second
3. **Trace with examples** to understand behavior
4. **Recognize patterns** - most problems use standard techniques
5. **Expand macros mentally** if confused
6. **Read top coders' code** to learn new tricks
7. **Keep a macro reference** handy

**Resources for Learning:**

- **Codeforces submissions:** Filter by rating, read AC solutions
- **GitHub:** Search for "competitive programming template"
- **AtCoder:** Solutions are often clean and readable
- **YouTube:** Errichto, SecondThread explain code

**Common Pitfalls:**

**Pitfall 1: Assuming macros are standard**
- Every coder has their own macros
- Check definitions at top

**Pitfall 2: Missing global state**
- CP code often uses global variables
- Check what's defined outside functions

**Pitfall 3: Not understanding modular arithmetic**
\`\`\`cpp
ans = (ans + x) % MOD;
\`\`\`
This is everywhere in CP!

**Bottom Line:**

Reading CP code:
- ✅ Learn common macro patterns
- ✅ Start from main, work backwards
- ✅ Recognize standard algorithms
- ✅ Trace with examples
- ✅ Expand macros mentally
- ✅ Practice regularly

**The more code you read, the faster you'll get!**`,
    },
    {
      question:
        'Converting terse competitive code to readable format is a useful skill for learning. Walk through the process of taking a heavily macro-ed solution and expanding it into more readable code while maintaining functionality.',
      answer: `Converting terse CP code to readable format helps understand algorithms. Here's the complete process:

**Original Terse Code:**

\`\`\`cpp
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int,int>
#define vi vector<int>
#define pb push_back
#define all(x) (x).begin(),(x).end()
#define rep(i,n) for(int i=0;i<(n);i++)
#define F first
#define S second

ll solve(vi& a) {
    int n=sz(a);
    ll ans=0;
    rep(i,n) if(a[i]>0) ans+=a[i];
    return ans;
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;cin>>t;
    while(t--) {
        int n;cin>>n;
        vi a(n);
        rep(i,n) cin>>a[i];
        cout<<solve(a)<<'\\n';
    }
}
\`\`\`

**Step-by-Step Conversion:**

**Step 1: Replace Type Aliases**

\`\`\`cpp
// Before:
ll ans = 0;
vi a(n);
pii p = {1, 2};

// After:
long long ans = 0;
vector<int> a(n);
pair<int, int> p = {1, 2};
\`\`\`

**Step 2: Expand Macros**

\`\`\`cpp
// Before:
rep(i, n) { ... }

// After:
for(int i = 0; i < n; i++) { ... }

// Before:
sort(all(a));

// After:
sort(a.begin(), a.end());

// Before:
v.pb(5);

// After:
v.push_back(5);

// Before:
p.F = 5;
p.S = 10;

// After:
p.first = 5;
p.second = 10;
\`\`\`

**Step 3: Add Whitespace**

\`\`\`cpp
// Before:
int t;cin>>t;
while(t--){
    int n;cin>>n;
    vi a(n);
}

// After:
int t;
cin >> t;

while(t--) {
    int n;
    cin >> n;
    vector<int> a(n);
}
\`\`\`

**Step 4: Expand bits/stdc++.h**

\`\`\`cpp
// Before:
#include <bits/stdc++.h>

// After:
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
// ... (only what's actually used)
\`\`\`

**Step 5: Add Comments**

\`\`\`cpp
// Before:
ll solve(vi& a) {
    ll ans=0;
    rep(i,sz(a)) if(a[i]>0) ans+=a[i];
    return ans;
}

// After:
// Calculates sum of positive elements in array
long long solve(vector<int>& a) {
    long long ans = 0;
    
    // Iterate through all elements
    for(int i = 0; i < (int)a.size(); i++) {
        // Add only positive elements to sum
        if(a[i] > 0) {
            ans += a[i];
        }
    }
    
    return ans;
}
\`\`\`

**Step 6: Improve Variable Names**

\`\`\`cpp
// Before:
ll solve(vi& a) {
    int n = sz(a);
    ll ans = 0;
    // ...
}

// After:
long long solve(vector<int>& arr) {
    int array_size = arr.size();
    long long sum_positive = 0;
    // ...
}
\`\`\`

**Complete Conversion:**

**Before (Terse):**
\`\`\`cpp
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int,int>
#define vi vector<int>
#define pb push_back
#define all(x) (x).begin(),(x).end()
#define rep(i,n) for(int i=0;i<(n);i++)
#define sz(x) (int)(x).size()

ll solve(vi& a) {
    int n=sz(a);
    ll ans=0;
    rep(i,n) if(a[i]>0) ans+=a[i];
    return ans;
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;cin>>t;
    while(t--) {
        int n;cin>>n;
        vi a(n);
        rep(i,n) cin>>a[i];
        cout<<solve(a)<<'\\n';
    }
}
\`\`\`

**After (Readable):**
\`\`\`cpp
#include <iostream>
#include <vector>
using namespace std;

/**
 * Calculate the sum of all positive elements in an array
 * @param arr: Input array of integers
 * @return: Sum of positive elements
 */
long long calculatePositiveSum(vector<int>& arr) {
    int n = arr.size();
    long long positive_sum = 0;
    
    // Iterate through all elements
    for(int i = 0; i < n; i++) {
        // Only add positive numbers to sum
        if(arr[i] > 0) {
            positive_sum += arr[i];
        }
    }
    
    return positive_sum;
}

int main() {
    // Fast I/O optimization
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Read number of test cases
    int num_tests;
    cin >> num_tests;
    
    // Process each test case
    while(num_tests--) {
        // Read array size
        int n;
        cin >> n;
        
        // Read array elements
        vector<int> arr(n);
        for(int i = 0; i < n; i++) {
            cin >> arr[i];
        }
        
        // Calculate and output result
        long long result = calculatePositiveSum(arr);
        cout << result << '\\n';
    }
    
    return 0;
}
\`\`\`

**Complex Example:**

**Before (Very Terse):**
\`\`\`cpp
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<ll,ll>
#define F first
#define S second
#define all(x) (x).begin(),(x).end()

ll solve(vector<pii>& v) {
    sort(all(v),[](pii a,pii b){return a.S<b.S;});
    ll ans=0,last=-1e18;
    for(auto& p:v){
        if(p.F>=last){
            ans++;
            last=p.S;
        }
    }
    return ans;
}

int main(){
    ios_base::sync_with_stdio(0);cin.tie(0);
    int n;cin>>n;
    vector<pii> v(n);
    for(auto& p:v) cin>>p.F>>p.S;
    cout<<solve(v);
}
\`\`\`

**After (Readable):**
\`\`\`cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/**
 * Activity Selection Problem / Interval Scheduling
 * Select maximum number of non-overlapping intervals
 * 
 * @param intervals: Vector of pairs (start_time, end_time)
 * @return: Maximum number of non-overlapping intervals
 */
long long selectActivities(vector<pair<long long, long long>>& intervals) {
    int n = intervals.size();
    
    // Sort intervals by end time (greedy approach)
    sort(intervals.begin(), intervals.end(), 
         [](pair<long long, long long> a, pair<long long, long long> b) {
             return a.second < b.second;
         });
    
    long long selected_count = 0;
    long long last_end_time = -1000000000000000000LL;  // -1e18
    
    // Greedy selection
    for(auto& interval : intervals) {
        long long start_time = interval.first;
        long long end_time = interval.second;
        
        // If current interval doesn't overlap with last selected
        if(start_time >= last_end_time) {
            selected_count++;
            last_end_time = end_time;
        }
    }
    
    return selected_count;
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Read number of intervals
    int n;
    cin >> n;
    
    // Read intervals (start_time, end_time)
    vector<pair<long long, long long>> intervals(n);
    for(int i = 0; i < n; i++) {
        cin >> intervals[i].first >> intervals[i].second;
    }
    
    // Calculate and output result
    long long result = selectActivities(intervals);
    cout << result << endl;
    
    return 0;
}
\`\`\`

**Conversion Checklist:**

\`\`\`
✓ Replace ll → long long
✓ Replace pii → pair<int,int>
✓ Replace vi → vector<int>
✓ Replace pb → push_back
✓ Replace all(x) → x.begin(), x.end()
✓ Replace rep(i,n) → for(int i=0; i<n; i++)
✓ Replace F → first
✓ Replace S → second
✓ Replace sz(x) → (int)x.size()
✓ Expand bits/stdc++.h (optional but cleaner)
✓ Add whitespace around operators
✓ Add blank lines between sections
✓ Add comments explaining logic
✓ Rename variables to be descriptive
✓ Add function documentation
\`\`\`

**Tools for Automatic Conversion:**

**1. C++ Formatter (clang-format)**
\`\`\`bash
clang-format -i solution.cpp
# Fixes spacing and formatting
\`\`\`

**2. Macro Expander Script**
\`\`\`python
# expand_macros.py
import re

macros = {
    r'\\bll\\b': 'long long',
    r'\\bpii\\b': 'pair<int,int>',
    r'\\bvi\\b': 'vector<int>',
    r'\\.pb\\(': '.push_back(',
    r'all\\(([^)]*)\\)': r'\\1.begin(), \\1.end()',
    r'rep\\((\w+),\s*(\w+)\\)': r'for(int \\1 = 0; \\1 < \\2; \\1++)',
    r'\\.F\\b': '.first',
    r'\\.S\\b': '.second',
}

def expand_macros(code):
    for pattern, replacement in macros.items():
        code = re.sub(pattern, replacement, code)
    return code

# Usage:
with open('solution.cpp') as f:
    code = f.read()

expanded = expand_macros(code)
print(expanded)
\`\`\`

**When to Expand vs Keep Terse:**

**Keep terse for:**
- Contest submissions (speed matters)
- Your own practice (if you know macros)
- Quick prototyping

**Expand for:**
- Learning algorithms
- Sharing code with others
- Teaching
- Long-term maintenance
- Team competitions

**Best of Both Worlds:**

Keep two versions:
\`\`\`
solution_terse.cpp    # For contests
solution_readable.cpp # For learning/sharing
\`\`\`

**Bottom Line:**

Converting terse code:
- ✅ Replace type aliases first
- ✅ Expand macros
- ✅ Add whitespace and formatting
- ✅ Add comments and documentation
- ✅ Rename variables descriptively
- ✅ Use tools to automate where possible

**Result: Code that's easy to understand and learn from!**`,
    },
    {
      question:
        'Different competitive programming platforms and communities have different coding conventions. Compare coding styles across platforms (Codeforces, AtCoder, TopCoder) and explain what to look for when reading solutions from each.',
      answer: `Each platform has distinct coding cultures. Understanding these helps you learn effectively from each community:

**Codeforces Style:**

**Characteristics:**
- Heavy macro usage
- Very terse variable names
- Global variables common
- Fast I/O always included
- Competitive, speed-focused

**Typical Code:**
\`\`\`cpp
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int,int>
#define pb push_back
#define all(x) (x).begin(),(x).end()
#define F first
#define S second

const int N=2e5+5;
int n,m;
ll a[N];

void solve(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
    ll ans=0;
    for(int i=0;i<n;i++) ans+=a[i];
    cout<<ans<<'\\n';
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;cin>>t;
    while(t--) solve();
}
\`\`\`

**What to look for:**
- ✅ Check macro definitions at top
- ✅ Global arrays (N, MAXN constants)
- ✅ Multiple test cases (while(t--))
- ✅ Fast I/O optimization
- ✅ Very compact code

**Top Codeforces coders:**
- **Tourist:** Extremely terse, many custom macros
- **jiangly:** Modular, template-heavy
- **Petr:** Clear logic, less macro abuse

**AtCoder Style:**

**Characteristics:**
- More readable than Codeforces
- Moderate macro usage
- Better variable names
- Clean code encouraged
- Often uses Japanese comments (can ignore)

**Typical Code:**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for(int i = 0; i < n; i++) {
        cin >> a[i];
    }
    
    ll sum = 0;
    for(int i = 0; i < n; i++) {
        sum += a[i];
    }
    
    cout << sum << endl;
    return 0;
}
\`\`\`

**What to look for:**
- ✅ \`using ll = long long\` (modern C++ typedef)
- ✅ More whitespace and formatting
- ✅ Descriptive variable names
- ✅ Less macro abuse
- ✅ Often single test case per run

**Top AtCoder coders:**
- **tourist (also competes here):** Cleaner than on CF
- **Petr:** Very readable
- **chokudai:** Japanese legend, clean code

**TopCoder Style:**

**Characteristics:**
- Class-based structure (required by platform)
- Very clean, readable code
- Almost no macros
- Professional coding style
- Long method names

**Typical Code:**
\`\`\`cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    long long calculateSum(vector<int> arr) {
        long long sum = 0;
        for(int i = 0; i < arr.size(); i++) {
            sum += arr[i];
        }
        return sum;
    }
};
\`\`\`

**What to look for:**
- ✅ Class structure (required)
- ✅ Public method with exact signature
- ✅ Very readable code
- ✅ No macros usually
- ✅ Specific includes (not bits/stdc++.h)

**Google Code Jam / Kickstart Style:**

**Characteristics:**
- Multiple test cases always
- Output format matters (Case #X:)
- Local testing important
- Mix of terse and readable

**Typical Code:**
\`\`\`cpp
#include <iostream>
using namespace std;

void solve(int case_num) {
    int n;
    cin >> n;
    
    long long ans = 0;
    for(int i = 0; i < n; i++) {
        int x;
        cin >> x;
        ans += x;
    }
    
    cout << "Case #" << case_num << ": " << ans << endl;
}

int main() {
    int t;
    cin >> t;
    for(int i = 1; i <= t; i++) {
        solve(i);
    }
    return 0;
}
\`\`\`

**What to look for:**
- ✅ Case numbering (Case #X:)
- ✅ 1-indexed test cases
- ✅ Often more readable than CF
- ✅ Structured output

**ICPC Style:**

**Characteristics:**
- Team competitions
- Very readable code (team needs to understand)
- Often modular
- Libraries and templates common
- Professional style

**Typical Code:**
\`\`\`cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Segment Tree for range sum queries
class SegmentTree {
private:
    vector<long long> tree;
    int n;
    
public:
    SegmentTree(int size) : n(size) {
        tree.resize(4 * n);
    }
    
    void update(int pos, long long value) {
        // Implementation
    }
    
    long long query(int l, int r) {
        // Implementation
    }
};

int main() {
    int n;
    cin >> n;
    
    SegmentTree st(n);
    // Use segment tree...
    
    return 0;
}
\`\`\`

**What to look for:**
- ✅ Classes and OOP structure
- ✅ Very readable
- ✅ Modular design
- ✅ Comments for teammates
- ✅ Reusable components

**Platform Comparison Table:**

| Platform | Macro Usage | Readability | Style |
|----------|-------------|-------------|-------|
| Codeforces | Heavy | Low | Terse |
| AtCoder | Moderate | Medium | Balanced |
| TopCoder | Minimal | High | Clean |
| Code Jam | Moderate | Medium | Structured |
| ICPC | Minimal | High | Professional |

**Cultural Differences:**

**Codeforces:**
- Speed is king
- Shortest code wins (mentally)
- Community loves clever hacks
- Russian/Eastern European influence

**AtCoder:**
- Balance speed and clarity
- Japanese influence (cleaner code)
- Educational focus
- Beginner-friendly

**TopCoder:**
- Professional style
- Platform enforces structure
- American corporate influence
- Focus on correctness

**ICPC:**
- Team dynamics matter
- Code must be maintainable
- Everyone needs to understand
- Academic focus

**Reading Strategy by Platform:**

**For Codeforces:**
1. Expand macros mentally
2. Trace with examples
3. Focus on algorithm, not syntax
4. Look for clever tricks

**For AtCoder:**
1. Usually straightforward to read
2. Good starting point for learning
3. Check editorial explanations
4. Often uses standard library well

**For TopCoder:**
1. Class structure is boilerplate
2. Focus on the main method
3. Very readable, easy to understand
4. Good for learning clean code

**For ICPC:**
1. Look for modular components
2. Classes are actual data structures
3. Very educational
4. Good templates to copy

**Common Patterns Across All:**

**1. Fast I/O (everywhere except TopCoder):**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**2. Multiple test cases (CF, AtCoder, GCJ):**
\`\`\`cpp
int t; cin >> t;
while(t--) solve();
\`\`\`

**3. Global arrays (CF, AtCoder):**
\`\`\`cpp
const int MAXN = 1e5 + 5;
int arr[MAXN];
\`\`\`

**4. Using namespace std (almost everywhere):**
\`\`\`cpp
using namespace std;
\`\`\`

**Learning Resources by Platform:**

**Codeforces:**
- View submissions after solving
- Read tourist, jiangly solutions
- Blogs/tutorials by community
- Very active forums

**AtCoder:**
- Official editorials (great quality)
- English translations available
- Clean reference solutions
- Educational problems

**TopCoder:**
- Match editorials
- Tutorial section
- Algorithm tutorials
- Historical value

**Code Jam:**
- Analysis after rounds
- Community solutions
- Focus on problem-solving

**My Recommendation:**

**For learning algorithms:**
1. Start with **AtCoder** (most readable)
2. Read **ICPC** solutions (well-structured)
3. Progress to **Codeforces** (once comfortable with macros)

**For competitive speed:**
1. Study **Codeforces** top coders
2. Learn their macro patterns
3. Develop your own template
4. Practice, practice, practice

**Bottom Line:**

Platform coding styles:
- **Codeforces:** Terse, macro-heavy, speed-focused
- **AtCoder:** Balanced, readable, educational
- **TopCoder:** Clean, class-based, professional
- **ICPC:** Modular, team-friendly, structured

**Learn from all platforms to become well-rounded!**`,
    },
  ],
} as const;
