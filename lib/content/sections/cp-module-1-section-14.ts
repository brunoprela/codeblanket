export const readingOthersCppCodeSection = {
  id: 'cp-m1-s14',
  title: "Reading Other People's C++ Code",
  content: `

# Reading Other People's C++ Code

## Introduction

You've just spent an hour on a problem. Still stuck. You check the editorial, and there it is—a beautiful, elegant 20-line solution. But when you look at the code... it might as well be written in ancient hieroglyphics.

\`\`\`cpp
#define ll long long
#define pb push_back
#define F first
#define S second
vi a(n); rep(i,n) cin>>a[i]; sort(all(a));
ll ans=0; rep(i,n-1) ans+=a[i]*a[i+1]; cout<<ans;
\`\`\`

What is this?! Where are the spaces? What's \`vi\`? What's \`rep\`? Why is everything so compressed?

**Welcome to competitive programming code style.** It's dense, it uses heavy macros, and it's optimized for speed of writing—not readability. But learning to read it is **essential** for learning from top coders, understanding editorials, and improving your skills.

In this comprehensive section, we'll decode common CP coding patterns, understand macro-heavy code, learn to read different coding styles, extract algorithmic insights, and master the art of learning from others' solutions.

**Goal**: Decode and understand other people's competitive programming code efficiently to learn techniques and improve your skills.

---

## Why Reading Others' Code Matters

### Learning Benefits

**1. Discover new techniques**
- See elegant solutions you wouldn't think of
- Learn standard patterns and idioms
- Understand optimal approaches

**2. Improve code quality**
- See how experts structure code
- Learn efficient implementations
- Pick up useful tricks

**3. Expand your toolkit**
- Find reusable code snippets
- Build your template library
- Learn new C++ features

**4. Understand editorials**
- Most editorials include code
- Faster to read code than long explanations
- See implementation details clearly

**5. Competitive advantage**
- Learn faster than just solving problems alone
- Understand multiple approaches
- Build intuition for good solutions

---

## Common CP Macros and Abbreviations

### The Essential Macro Dictionary

\`\`\`cpp
// Type shortcuts
#define ll long long
#define ull unsigned long long
#define ld long double
#define vi vector<int>
#define vll vector<long long>
#define pii pair<int, int>
#define pll pair<long long, long long>

// Container operations
#define pb push_back
#define mp make_pair
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define sz(v) (int)v.size()

// Loops
#define rep(i, n) for(int i = 0; i < (n); i++)
#define rep1(i, n) for(int i = 1; i <= (n); i++)
#define rrep(i, n) for(int i = (n) - 1; i >= 0; i--)
#define FOR(i, a, b) for(int i = (a); i < (b); i++)

// Pair access
#define F first
#define S second

// Utility
#define endl '\\n'
#define inf 1e9
#define INF 1e18
#define MOD 1000000007
#define PI 3.14159265358979323846
\`\`\`

### Translating Macro-Heavy Code

**Before (macro-heavy):**
\`\`\`cpp
vi a(n); 
rep(i,n) cin>>a[i]; 
sort(all(a));
ll ans=0; 
rep(i,n-1) ans+=a[i]*a[i+1]; 
cout<<ans<<endl;
\`\`\`

**After (expanded):**
\`\`\`cpp
vector<int> a(n);
for(int i = 0; i < n; i++) cin >> a[i];
sort(a.begin(), a.end());
long long ans = 0;
for(int i = 0; i < n-1; i++) ans += a[i] * a[i+1];
cout << ans << '\\n';
\`\`\`

**Much clearer!** When reading macro-heavy code, mentally expand the macros.

---

## Decoding Dense Code

### Pattern 1: Compressed Variable Names

\`\`\`cpp
int n,m,k,x,y,z,a,b,c,d,i,j;
\`\`\`

**What it means:**
- \`n, m\`: Usually array/matrix sizes
- \`k\`: Often a parameter or constraint
- \`x, y, z\`: Coordinates or generic variables
- \`a, b, c\`: Generic variables
- \`i, j\`: Loop indices

**Reading tip:** Infer meaning from usage, not names!

### Pattern 2: No Spacing

\`\`\`cpp
for(int i=0;i<n;i++)for(int j=0;j<m;j++)dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
\`\`\`

**Reading strategy:**
1. Add mental spaces at keywords: \`for\`, \`if\`, \`while\`
2. Identify operators: \`= \`, \` == \`, \`<\`, \`>\`
3. Break into logical chunks

**Reformatted:**
\`\`\`cpp
for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
        dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
    }
}
\`\`\`

### Pattern 3: Chained Operations

\`\`\`cpp
cin>>n>>m>>k;
a.resize(n); rep(i,n) cin>>a[i];
sort(all(a)); reverse(all(a));
\`\`\`

**Reading strategy:**
1. Each \`; \` is a statement
2. \` >>\`
    chains input operations
3. Parse line by line

**What it does:**
1. Read three integers
2. Resize vector and read n elements
3. Sort then reverse (descending order)

### Pattern 4: Ternary Operators

\`\`\`cpp
int result = (x > y) ? (x > z ? x : z) : (y > z ? y : z);
\`\`\`

**Reading tip:** Convert to if-else mentally:
\`\`\`cpp
int result;
if (x > y) {
    if (x > z) result = x;
    else result = z;
} else {
    if (y > z) result = y;
    else result = z;
}
// This finds max(x, y, z)!
\`\`\`

### Pattern 5: Lambda Functions

\`\`\`cpp
sort(all(v), [](int a, int b){ return a > b; });
\`\`\`

**What it is:** Anonymous function (lambda)
**What it does:** Sort in descending order

**Reading tip:** 
- \`[]\` = capture clause (empty = capture nothing)
- \`(int a, int b)\` = parameters
- \`{ return a > b; } \` = function body

---

## Common Code Patterns

### Pattern: Fast I/O

\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**What it does:** Speeds up cin/cout (don't mix with scanf/printf!)

### Pattern: Reading N Elements

\`\`\`cpp
// Pattern 1: Loop
vector<int> a(n);
for(int i = 0; i < n; i++) cin >> a[i];

// Pattern 2: Macro
vi a(n); rep(i,n) cin>>a[i];

// Pattern 3: Function
vector<int> read(int n) {
    vector<int> v(n);
    for(auto& x : v) cin >> x;
    return v;
}
auto a = read(n);
\`\`\`

### Pattern: Pair Sorting

\`\`\`cpp
vector<pii> v;
// Read pairs...
sort(all(v));  // Sorts by first, then second

// Custom sort by second:
sort(all(v), [](pii a, pii b){ return a.S < b.S; });

// Or without macro:
sort(v.begin(), v.end(), [](pair<int,int> a, pair<int,int> b){ 
    return a.second < b.second; 
});
\`\`\`

### Pattern: Map Default Values

\`\`\`cpp
map<int, int> cnt;
for(auto x : a) cnt[x]++;  // Auto-creates with 0

// Checking existence:
if(cnt.count(x)) { ... }  // Pattern 1
if(cnt.find(x) != cnt.end()) { ... }  // Pattern 2
\`\`\`

### Pattern: Set Operations

\`\`\`cpp
set<int> s;
s.insert(x);  // Add element
s.erase(x);   // Remove element
s.count(x);   // Check if exists (returns 0 or 1)
s.size();     // Number of elements

// Iterate:
for(auto x : s) cout << x << " ";  // Sorted order!
\`\`\`

### Pattern: Priority Queue (Heap)

\`\`\`cpp
// Max heap (default):
priority_queue<int> pq;
pq.push(5);
int top = pq.top();  // Maximum element
pq.pop();

// Min heap:
priority_queue<int, vector<int>, greater<int>> pq;
// Same operations, but top() gives minimum
\`\`\`

### Pattern: memset for Arrays

\`\`\`cpp
int arr[100];
memset(arr, 0, sizeof(arr));  // Set all to 0
memset(arr, -1, sizeof(arr)); // Set all to -1

// WARNING: Only works for 0 and -1!
// For other values, use loop or fill:
fill(arr, arr + 100, 42);  // Set all to 42
\`\`\`

---

## Understanding Algorithm Flow

### Step-by-Step Code Reading

**Example problem:** Find maximum subarray sum

\`\`\`cpp
ll ans=-INF,curr=0;
rep(i,n){
    curr+=a[i];
    ans=max(ans,curr);
    if(curr<0)curr=0;
}
cout<<ans;
\`\`\`

**Step 1: Expand macros**
\`\`\`cpp
long long ans = -INF, curr = 0;
for(int i = 0; i < n; i++) {
    curr += a[i];
    ans = max(ans, curr);
    if(curr < 0) curr = 0;
}
cout << ans;
\`\`\`

**Step 2: Understand variables**
- \`ans\`: Current maximum sum found
- \`curr\`: Current subarray sum being considered

**Step 3: Trace algorithm**
- Start with \`curr = 0\`
- Add each element to \`curr\`
- Update maximum if \`curr\` is better
- If \`curr\` becomes negative, reset to 0

**Recognized:** Kadane's Algorithm for maximum subarray sum!

---

## Different Coding Styles

### Style 1: Macro-Heavy

\`\`\`cpp
#define ll long long
#define rep(i,n) for(int i=0;i<(n);i++)
#define all(v) v.begin(),v.end()

void solve(){
    int n;cin>>n;
    vi a(n);rep(i,n)cin>>a[i];
    sort(all(a));
    cout<<a[n-1]-a[0]<<endl;
}
\`\`\`

**Characteristics:**
- Heavy use of macros
- Very compact
- Fast to write
- Hard to read for beginners

### Style 2: Clean and Readable

\`\`\`cpp
void solve() {
    int n;
    cin >> n;
    
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    
    sort(a.begin(), a.end());
    
    int result = a[n-1] - a[0];
    cout << result << endl;
}
\`\`\`

**Characteristics:**
- Minimal macros
- Proper spacing
- Clear variable names
- Easier to read

### Style 3: Functional

\`\`\`cpp
void solve() {
    int n; cin >> n;
    vector<int> a(n);
    for (auto& x : a) cin >> x;
    
    auto [minElem, maxElem] = minmax_element(a.begin(), a.end());
    cout << *maxElem - *minElem << endl;
}
\`\`\`

**Characteristics:**
- Modern C++ features
- STL algorithms
- Concise but readable

### Style 4: Global Variables

\`\`\`cpp
int n, a[100005];

void solve() {
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a, a + n);
    cout << a[n-1] - a[0] << endl;
}
\`\`\`

**Characteristics:**
- Global arrays/variables
- Common in CP
- Avoids passing parameters

**All four solve the same problem!** Practice reading all styles.

---

## Learning from Solutions

### Effective Code Reading Strategy

**1. Understand the problem first**
- Read problem statement
- Understand input/output
- Know the constraints

**2. Skim the code structure**
- How many functions?
- Main algorithm location?
- Data structures used?

**3. Identify key variables**
- What do they represent?
- Initial values?
- How do they change?

**4. Trace the algorithm**
- Follow the main logic
- Understand key steps
- See how it handles edge cases

**5. Run mentally on example**
- Use sample input
- Trace step by step
- Verify output matches

**6. Extract insights**
- What technique is used?
- Why does it work?
- Can I apply this elsewhere?

### Example: Reading a DP Solution

\`\`\`cpp
ll dp[1005][1005];
void solve(){
    int n,m;cin>>n>>m;
    vi a(n);rep(i,n)cin>>a[i];
    memset(dp,-1,sizeof dp);
    dp[0][0]=0;
    rep(i,n)rep(j,m+1){
        if(dp[i][j]==-1)continue;
        dp[i+1][j]=max(dp[i+1][j],dp[i][j]);
        if(j+a[i]<=m)dp[i+1][j+a[i]]=max(dp[i+1][j+a[i]],dp[i][j]+a[i]);
    }
    ll ans=0;rep(j,m+1)ans=max(ans,dp[n][j]);
    cout<<ans<<endl;
}
\`\`\`

**Reading process:**

**Step 1: Expand and format**
\`\`\`cpp
long long dp[1005][1005];

void solve() {
    int n, m;
    cin >> n >> m;
    
    vector<int> a(n);
    for(int i = 0; i < n; i++) cin >> a[i];
    
    memset(dp, -1, sizeof dp);
    dp[0][0] = 0;
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j <= m; j++) {
            if(dp[i][j] == -1) continue;
            
            // Option 1: Don't take item i
            dp[i+1][j] = max(dp[i+1][j], dp[i][j]);
            
            // Option 2: Take item i (if possible)
            if(j + a[i] <= m) {
                dp[i+1][j+a[i]] = max(dp[i+1][j+a[i]], dp[i][j] + a[i]);
            }
        }
    }
    
    long long ans = 0;
    for(int j = 0; j <= m; j++) {
        ans = max(ans, dp[n][j]);
    }
    
    cout << ans << endl;
}
\`\`\`

**Step 2: Identify pattern**
- \`dp[i][j]\` = maximum value using first i items with weight ≤ j
- This is **0/1 Knapsack DP**!

**Step 3: Understand transitions**
- Either take item i or don't
- Track maximum value for each weight

**Insight:** Standard knapsack pattern, can reuse in future problems!

---

## Common Pitfalls When Reading Code

### Pitfall 1: Assuming Variables Are Initialized

\`\`\`cpp
int sum;  // Might not be 0!
for(int i = 0; i < n; i++) sum += a[i];
\`\`\`

**Check:** Look for initialization! Global variables are 0, local ones aren't.

### Pitfall 2: Ignoring Edge Cases

\`\`\`cpp
int result = a[0];  // What if n = 0?
\`\`\`

**Check:** How does code handle n=0, n=1, empty input?

### Pitfall 3: Missing Modular Arithmetic

\`\`\`cpp
ans = (ans + x) % MOD;  // vs
ans = ans + x;  // Overflow!
\`\`\`

**Check:** Does problem require modulo? Is it applied correctly?

### Pitfall 4: Off-by-One in Loops

\`\`\`cpp
for(int i = 0; i <= n; i++)  // vs
for(int i = 0; i < n; i++)
\`\`\`

**Check:** Loop bounds carefully!

---

## Building Your Reading Skills

### Practice Exercises

**1. Read accepted solutions on Codeforces**
- After solving a problem, read others' solutions
- Look for faster/cleaner approaches
- Learn new techniques

**2. Read editorials with code**
- Understand the algorithm
- See implementation details
- Compare with your approach

**3. Read top coders' submissions**
- Visit profiles of red coders
- See how they structure code
- Notice patterns they use

**4. Participate in virtual contests**
- After contest, read fastest solutions
- Learn speed-coding techniques
- See common optimizations

### Tools for Reading Code

**1. Codeforces Standings**
- Click on problem letter → see all submissions
- Sort by fastest time
- Read top solutions

**2. CodeForces Edu Section**
- Detailed tutorials with code
- Well-commented implementations
- Step-by-step explanations

**3. GitHub Repositories**
- Search "competitive programming templates"
- See organized code libraries
- Learn common patterns

**4. Online IDEs**
- Copy code to online IDE
- Add prints to understand flow
- Modify and experiment

---

## Creating Your Own Template Library

### Building Reusable Components

**From reading others' code, extract:**

**1. Useful functions**
\`\`\`cpp
// GCD
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

// Power
ll power(ll a, ll b, ll mod) {
    ll res = 1;
    while(b > 0) {
        if(b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
\`\`\`

**2. Useful macros (your choice)**
\`\`\`cpp
#define ll long long
#define all(v) v.begin(), v.end()
#define rep(i,n) for(int i=0; i<(n); i++)
\`\`\`

**3. Common patterns**
\`\`\`cpp
// Read array
vector<int> read(int n) {
    vector<int> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

// 2D vector
auto create2D = [](int n, int m, auto val) {
    return vector(n, vector(m, val));
};
\`\`\`

**4. Data structures**
\`\`\`cpp
// Union-Find, Segment Tree, etc.
// Copy implementations you understand well
\`\`\`

---

## Summary

**Key Skills:**

✅ **Expand macros mentally** (understand what they do)
✅ **Reformat dense code** (add spacing to understand)
✅ **Trace algorithms** (run mentally on examples)
✅ **Recognize patterns** (common techniques)
✅ **Extract insights** (learn techniques for reuse)

**Common CP Macros:**

✅ \`ll\` = long long
✅ \`vi\` = vector<int>
✅ \`pii\` = pair<int, int>
✅ \`pb\` = push_back
✅ \`all(v)\` = v.begin(), v.end()
✅ \`rep(i, n)\` = for(int i=0; i<n; i++)
✅ \`F / S\` = first/second (for pairs)

**Reading Strategy:**

1. Understand problem first
2. Skim code structure
3. Expand macros
4. Reformat for readability
5. Trace algorithm
6. Run on example
7. Extract insights

**Learning Sources:**

- Codeforces accepted solutions
- Editorial code
- Top coders' profiles
- GitHub CP repositories
- Educational platforms

---

## Next Steps

Now let's learn **Contest-Day C++ Tips** - practical advice for writing fast, correct code under time pressure!

**Key Takeaway**: Reading others' code is essential for growth in competitive programming. Don't be intimidated by macro-heavy, dense code—with practice, you'll read it as easily as clean code. Always extract insights and techniques to add to your own toolkit. Every solution you read makes you a better programmer!
`,
  quizId: 'cp-m1-s14-quiz',
  discussionId: 'cp-m1-s14-discussion',
} as const;
