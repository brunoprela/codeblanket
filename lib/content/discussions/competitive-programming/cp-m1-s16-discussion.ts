export default {
  id: 'cp-m1-s16-discussion',
  title: 'Building a Robust CP Starter Template - Discussion Questions',
  questions: [
    {
      question:
        "A well-designed starter template can save significant time in contests. Design a comprehensive C++ template that balances functionality, simplicity, and flexibility. Explain each component and why it's included.",
      answer: `A great template is your competitive programming foundation. Here's the complete, battle-tested design:

**The Complete Template:**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// ==================== TYPE ALIASES ====================
#define ll long long
#define ull unsigned long long
#define ld long double

#define pii pair<int, int>
#define pll pair<ll, ll>
#define pdd pair<double, double>

#define vi vector<int>
#define vll vector<ll>
#define vvi vector<vi>
#define vpii vector<pii>

// ==================== SHORTCUTS ====================
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define mt make_tuple
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define sz(x) (int)(x).size()
#define F first
#define S second

// ==================== LOOPS ====================
#define rep(i, n) for(int i = 0; i < (int)(n); i++)
#define rep1(i, n) for(int i = 1; i <= (int)(n); i++)
#define rrep(i, n) for(int i = (int)(n) - 1; i >= 0; i--)
#define FOR(i, a, b) for(int i = (int)(a); i < (int)(b); i++)
#define RFOR(i, a, b) for(int i = (int)(a) - 1; i >= (int)(b); i--)

// ==================== CONSTANTS ====================
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;
const double EPS = 1e-9;
const double PI = acos(-1.0);

// ==================== MATH UTILITIES ====================
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a % b) : a; }

template<typename T>
T lcm(T a, T b) { return a / gcd(a, b) * b; }

template<typename T>
T power(T base, ll exp) {
    T result = 1;
    while(exp > 0) {
        if(exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

ll modpow(ll base, ll exp, ll mod = MOD) {
    ll result = 1;
    base %= mod;
    while(exp > 0) {
        if(exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

ll modinv(ll a, ll mod = MOD) {
    return modpow(a, mod - 2, mod);
}

// ==================== DEBUG (LOCAL ONLY) ====================
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << endl
#define debug2(x, y) cerr << #x << " = " << (x) << ", " << #y << " = " << (y) << endl
#define debug3(x, y, z) cerr << #x << " = " << (x) << ", " << #y << " = " << (y) << ", " << #z << " = " << (z) << endl

template<typename T>
void debugv(const vector<T>& v, const string& name = "vector") {
    cerr << name << " = [";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}

#define TRACE cerr << "→ " << __FUNCTION__ << ":" << __LINE__ << endl
#else
#define debug(x)
#define debug2(x, y)
#define debug3(x, y, z)
#define debugv(v, name)
#define TRACE
#endif

// ==================== SOLUTION ====================

void solve() {
    // Your solution goes here
    
}

// ==================== MAIN ====================
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    #ifdef LOCAL
    freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    #endif
    
    int t = 1;
    // cin >> t;  // Uncomment for multiple test cases
    
    while(t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

**Component-by-Component Explanation:**

**1. Header: bits/stdc++.h**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;
\`\`\`

**Why:**
- ✅ Includes everything (iostream, vector, algorithm, etc.)
- ✅ Saves time (no need to remember specific headers)
- ✅ Standard in competitive programming

**Drawback:** Not portable to production code (only works with GCC)
**Verdict:** Perfect for CP, use it!

**2. Type Aliases**

\`\`\`cpp
#define ll long long
#define pii pair<int, int>
#define vi vector<int>
\`\`\`

**Why:**
- ✅ Type less: \`ll\` vs \`long long\`
- ✅ More readable: \`vi\` vs \`vector<int>\`
- ✅ Standard abbreviations everyone knows

**Usage:**
\`\`\`cpp
ll sum = 0;  // Instead of long long sum = 0;
vi arr(n);   // Instead of vector<int> arr(n);
pii p = {1, 2};  // Instead of pair<int,int> p = {1, 2};
\`\`\`

**3. Container Shortcuts**

\`\`\`cpp
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define sz(x) (int)(x).size()
\`\`\`

**Why:**
- ✅ Faster typing
- ✅ Less verbose

**Usage:**
\`\`\`cpp
v.pb(5);         // v.push_back(5);
sort(all(v));    // sort(v.begin(), v.end());
rep(i, sz(v))    // for(int i = 0; i < (int)v.size(); i++)
\`\`\`

**Note:** \`sz(x)\` casts to int to avoid signed/unsigned comparison warnings

**4. Loop Macros**

\`\`\`cpp
#define rep(i, n) for(int i = 0; i < (int)(n); i++)
#define rep1(i, n) for(int i = 1; i <= (int)(n); i++)
\`\`\`

**Why:**
- ✅ Faster loop writing
- ✅ Less typing

**Usage:**
\`\`\`cpp
rep(i, n) {  // 0-indexed loop
    cout << i << " ";
}

rep1(i, n) {  // 1-indexed loop
    cout << i << " ";
}
\`\`\`

**5. Constants**

\`\`\`cpp
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;
\`\`\`

**Why:**
- ✅ Commonly used values
- ✅ Consistent across problems
- ✅ Easy to change if needed

**Usage:**
\`\`\`cpp
ans = (ans + x) % MOD;  // Modular arithmetic
int dist[MAXN];
fill(dist, dist + MAXN, INF);  // Initialize to infinity
\`\`\`

**6. Math Utilities**

\`\`\`cpp
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a % b) : a; }

ll modpow(ll base, ll exp, ll mod = MOD) { /* ... */ }
\`\`\`

**Why:**
- ✅ Commonly needed functions
- ✅ Avoid reimplementing every time
- ✅ Templates work with different types

**Usage:**
\`\`\`cpp
int g = gcd(12, 18);  // 6
ll result = modpow(2, 10, MOD);  // 2^10 mod MOD
\`\`\`

**7. Debug Macros**

\`\`\`cpp
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << endl
#else
#define debug(x)
#endif
\`\`\`

**Why:**
- ✅ Debug locally without affecting submission
- ✅ Automatically disabled when \`LOCAL\` not defined
- ✅ Clean code (no commented debug statements)

**Usage:**
\`\`\`cpp
int x = 42;
debug(x);  // Prints: x = 42 (only locally)

// Submit without changing code!
\`\`\`

**8. Fast I/O**

\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
cout.tie(nullptr);
\`\`\`

**Why:**
- ✅ Makes cin/cout as fast as scanf/printf
- ✅ Essential for large input/output
- ✅ Can prevent TLE

**Warning:** Don't mix with scanf/printf after this!

**9. File I/O (Local Testing)**

\`\`\`cpp
#ifdef LOCAL
freopen("input.txt", "r", stdin);
#endif
\`\`\`

**Why:**
- ✅ Test with file input locally
- ✅ Automatically disabled on submission
- ✅ No need to paste input every time

**10. Multiple Test Cases Structure**

\`\`\`cpp
int t = 1;
// cin >> t;  // Uncomment for multiple test cases

while(t--) {
    solve();
}
\`\`\`

**Why:**
- ✅ Easy to toggle multiple test cases
- ✅ Clean separation of solution logic
- ✅ One uncomment to enable

**Variations and Customizations:**

**Minimal Template (If You Prefer Less Macros):**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define all(x) (x).begin(), (x).end()

void solve() {
    // Solution
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;
    while(t--) solve();
}
\`\`\`

**Extended Template (More Utilities):**

\`\`\`cpp
// Add to template:

// Range min/max
template<typename T>
T minimum(T a, T b) { return min(a, b); }
template<typename T, typename... Args>
T minimum(T first, Args... args) {
    return min(first, minimum(args...));
}

// Coordinate compression
template<typename T>
vector<T> compress(vector<T>& v) {
    vector<T> ret = v;
    sort(all(ret));
    ret.erase(unique(all(ret)), ret.end());
    for(auto& x : v) {
        x = lower_bound(all(ret), x) - ret.begin();
    }
    return ret;
}

// 2D prefix sum
vector<vll> prefix2D(const vector<vll>& grid) {
    int n = sz(grid), m = sz(grid[0]);
    vector<vll> pre(n + 1, vll(m + 1, 0));
    rep(i, n) rep(j, m) {
        pre[i+1][j+1] = grid[i][j] + pre[i][j+1] + pre[i+1][j] - pre[i][j];
    }
    return pre;
}
\`\`\`

**Template Best Practices:**

**1. Personalize It**
- Add utilities YOU use often
- Remove what you never use
- Make it YOUR template

**2. Test It Regularly**
- Use in every contest
- Fix bugs as you find them
- Evolve over time

**3. Keep It Updated**
- Add new tricks you learn
- Remove outdated patterns
- Stay current

**4. Know It By Heart**
- What every macro does
- When to use each utility
- Where everything is

**5. Have Variations**
- Minimal template for simple problems
- Full template for complex problems
- Specialized for specific domains (geometry, graphs)

**Template Organization Tips:**

**Section Headers:**
\`\`\`cpp
// ==================== SECTION ====================
\`\`\`
- Makes template scannable
- Easy to find what you need
- Professional look

**Ordering:**
1. Includes
2. Type aliases
3. Shortcuts
4. Constants
5. Utilities
6. Debug
7. Solution
8. Main

**Logical flow from general to specific**

**Common Template Mistakes:**

**Mistake 1: Too Many Macros**
- Becomes unreadable
- Hard to debug
- Use moderation

**Mistake 2: Untested Utilities**
- Add function, never test
- Breaks in contest
- Test everything!

**Mistake 3: Copy-Paste Random Code**
- Don't understand it
- Causes bugs
- Only add what you know

**Mistake 4: Never Updating**
- Template gets stale
- Miss new tricks
- Update after each contest

**My Template Evolution:**

**Beginner (Year 1):**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    // Solution
}
\`\`\`

**Intermediate (Year 2):**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pb push_back
#define all(x) (x).begin(), (x).end()

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    // Solution
}
\`\`\`

**Advanced (Year 3+):**
\`\`\`cpp
// Complete template shown above
\`\`\`

**Template grows with your skills!**

**Platform-Specific Considerations:**

**Codeforces:**
- Fast I/O essential
- Multiple test cases common
- Heavy macros accepted

**AtCoder:**
- Fast I/O helpful
- Clean code preferred
- Moderate macros

**TopCoder:**
- Class-based required
- Different template needed
- No bits/stdc++.h

**Google Code Jam:**
- File I/O sometimes
- Case numbering needed
- Careful output format

**Template Storage:**

**Location:**
\`\`\`
~/cp/template.cpp
\`\`\`

**Quick Copy:**
\`\`\`bash
# Add alias to shell
alias cptemplate='cp ~/cp/template.cpp solution.cpp'

# Or function
newcp() {
    cp ~/cp/template.cpp "$1.cpp"
    vim "$1.cpp"
}

# Usage:
newcp A  # Creates A.cpp from template
\`\`\`

**Backup:**
- GitHub repository
- Google Drive
- Multiple locations

**Don't lose your template!**

**Bottom Line:**

Good template:
- ✅ Saves time (5-10 minutes per contest)
- ✅ Reduces errors (tested utilities)
- ✅ Consistent (same structure every time)
- ✅ Personal (fits YOUR style)
- ✅ Evolving (gets better over time)

**Invest time building it, save time using it!**`,
    },
    {
      question:
        'Different problem types may benefit from specialized templates. Describe how to organize multiple template variants (basic, graph, geometry, string, etc.) and when to use each.',
      answer: `Multiple specialized templates can accelerate solving specific problem types. Here's the complete organizational system:

**The Template System:**

\`\`\`
~/cp/templates/
├── basic.cpp           # Default template
├── graph.cpp           # Graph problems
├── geometry.cpp        # Geometry problems
├── string.cpp          # String algorithms
├── number_theory.cpp   # Math problems
├── data_structures.cpp # Advanced DS
└── README.md          # Quick reference
\`\`\`

**1. Basic Template (Default)**

**When to use:**
- Simple problems
- Array/vector manipulation
- Basic algorithms
- Unsure what's needed

**Contents:**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define vi vector<int>
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define rep(i,n) for(int i=0;i<(n);i++)

void solve() {
    // Solution
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;
    while(t--) solve();
}
\`\`\`

**Size:** ~30 lines
**Philosophy:** Minimal, fast to modify

**2. Graph Template**

**When to use:**
- DFS/BFS problems
- Shortest path
- Minimum spanning tree
- Topological sort
- Connected components

**Additional contents:**
\`\`\`cpp
// ==================== GRAPH UTILITIES ====================

const int MAXN = 1e5 + 5;

// Adjacency list representation
vector<int> graph[MAXN];
vector<pii> weighted_graph[MAXN];  // {neighbor, weight}
bool visited[MAXN];
int dist[MAXN];
int parent[MAXN];

// Clear graph (for multiple test cases)
void clear_graph(int n) {
    for(int i = 0; i <= n; i++) {
        graph[i].clear();
        weighted_graph[i].clear();
        visited[i] = false;
        dist[i] = INF;
        parent[i] = -1;
    }
}

// DFS
void dfs(int u) {
    visited[u] = true;
    for(int v : graph[u]) {
        if(!visited[v]) {
            parent[v] = u;
            dfs(v);
        }
    }
}

// BFS
void bfs(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;
    dist[start] = 0;
    
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        
        for(int v : graph[u]) {
            if(!visited[v]) {
                visited[v] = true;
                dist[v] = dist[u] + 1;
                parent[v] = u;
                q.push(v);
            }
        }
    }
}

// Dijkstra
void dijkstra(int start, int n) {
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    fill(dist, dist + n + 1, INF);
    dist[start] = 0;
    pq.push({0, start});
    
    while(!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        if(d > dist[u]) continue;
        
        for(auto [v, w] : weighted_graph[u]) {
            if(dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}

// Union-Find (Disjoint Set Union)
int dsu_parent[MAXN];
int dsu_rank[MAXN];

void dsu_init(int n) {
    for(int i = 0; i <= n; i++) {
        dsu_parent[i] = i;
        dsu_rank[i] = 0;
    }
}

int dsu_find(int x) {
    if(dsu_parent[x] != x) {
        dsu_parent[x] = dsu_find(dsu_parent[x]);  // Path compression
    }
    return dsu_parent[x];
}

void dsu_union(int x, int y) {
    int px = dsu_find(x);
    int py = dsu_find(y);
    
    if(px == py) return;
    
    // Union by rank
    if(dsu_rank[px] < dsu_rank[py]) {
        dsu_parent[px] = py;
    } else if(dsu_rank[px] > dsu_rank[py]) {
        dsu_parent[py] = px;
    } else {
        dsu_parent[py] = px;
        dsu_rank[px]++;
    }
}
\`\`\`

**Size:** ~150 lines
**Use case:** Any graph problem

**3. Geometry Template**

**When to use:**
- Points, lines, polygons
- Convex hull
- Closest pair
- Geometric algorithms

**Additional contents:**
\`\`\`cpp
// ==================== GEOMETRY ====================

const double EPS = 1e-9;
const double PI = acos(-1.0);

// Point structure
struct Point {
    double x, y;
    
    Point() : x(0), y(0) {}
    Point(double x, double y) : x(x), y(y) {}
    
    // Vector operations
    Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
    Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
    Point operator*(double t) const { return Point(x * t, y * t); }
    Point operator/(double t) const { return Point(x / t, y / t); }
    
    // Dot product
    double dot(const Point& p) const { return x * p.x + y * p.y; }
    
    // Cross product
    double cross(const Point& p) const { return x * p.y - y * p.x; }
    
    // Length
    double length() const { return sqrt(x * x + y * y); }
    
    // Normalize
    Point normalize() const { return *this / length(); }
    
    // Angle
    double angle() const { return atan2(y, x); }
};

// Line structure
struct Line {
    Point p1, p2;
    
    Line() {}
    Line(Point p1, Point p2) : p1(p1), p2(p2) {}
    
    // Check if point is on line
    bool on_line(const Point& p) const {
        return abs((p - p1).cross(p2 - p1)) < EPS;
    }
    
    // Distance from point to line
    double distance(const Point& p) const {
        return abs((p - p1).cross(p2 - p1)) / (p2 - p1).length();
    }
};

// Distance between two points
double dist(const Point& a, const Point& b) {
    return (a - b).length();
}

// Orientation test (counter-clockwise, clockwise, collinear)
int orientation(const Point& a, const Point& b, const Point& c) {
    double cross = (b - a).cross(c - a);
    if(abs(cross) < EPS) return 0;  // Collinear
    return cross > 0 ? 1 : -1;  // CCW : CW
}

// Convex hull (Graham scan)
vector<Point> convex_hull(vector<Point> points) {
    int n = points.size();
    if(n < 3) return points;
    
    // Find bottom-most point
    swap(points[0], *min_element(all(points), [](const Point& a, const Point& b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    }));
    
    Point pivot = points[0];
    
    // Sort by polar angle
    sort(points.begin() + 1, points.end(), [&](const Point& a, const Point& b) {
        int o = orientation(pivot, a, b);
        if(o == 0) return dist(pivot, a) < dist(pivot, b);
        return o > 0;
    });
    
    vector<Point> hull;
    for(const Point& p : points) {
        while(hull.size() > 1 && orientation(hull[hull.size()-2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.pb(p);
    }
    
    return hull;
}
\`\`\`

**Size:** ~120 lines
**Use case:** Computational geometry

**4. String Template**

**When to use:**
- String matching
- Palindromes
- Suffix arrays
- String DP

**Additional contents:**
\`\`\`cpp
// ==================== STRING ALGORITHMS ====================

// KMP (Knuth-Morris-Pratt) pattern matching
vector<int> compute_lps(const string& pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0;
    int i = 1;
    
    while(i < m) {
        if(pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if(len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

vector<int> kmp_search(const string& text, const string& pattern) {
    vector<int> lps = compute_lps(pattern);
    vector<int> matches;
    
    int n = text.length();
    int m = pattern.length();
    int i = 0, j = 0;
    
    while(i < n) {
        if(text[i] == pattern[j]) {
            i++;
            j++;
        }
        
        if(j == m) {
            matches.pb(i - j);
            j = lps[j - 1];
        } else if(i < n && text[i] != pattern[j]) {
            if(j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    return matches;
}

// Z-algorithm
vector<int> z_algorithm(const string& s) {
    int n = s.length();
    vector<int> z(n);
    int l = 0, r = 0;
    
    for(int i = 1; i < n; i++) {
        if(i <= r) {
            z[i] = min(r - i + 1, z[i - l]);
        }
        while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if(i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
    
    return z;
}

// Rabin-Karp hashing
struct StringHash {
    static const ll MOD = 1e9 + 7;
    static const ll BASE = 31;
    
    vector<ll> hash, pow;
    
    StringHash(const string& s) {
        int n = s.length();
        hash.resize(n + 1);
        pow.resize(n + 1);
        
        pow[0] = 1;
        for(int i = 0; i < n; i++) {
            hash[i + 1] = (hash[i] * BASE + (s[i] - 'a' + 1)) % MOD;
            pow[i + 1] = (pow[i] * BASE) % MOD;
        }
    }
    
    ll get_hash(int l, int r) {  // [l, r)
        ll h = (hash[r] - hash[l] * pow[r - l]) % MOD;
        return (h + MOD) % MOD;
    }
};
\`\`\`

**Size:** ~100 lines
**Use case:** String problems

**5. Number Theory Template**

**When to use:**
- Modular arithmetic
- Prime numbers
- GCD/LCM
- Combinatorics

**Additional contents:**
\`\`\`cpp
// ==================== NUMBER THEORY ====================

// Sieve of Eratosthenes
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    for(int i = 2; i * i <= n; i++) {
        if(is_prime[i]) {
            for(int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    
    return is_prime;
}

// Prime factorization
map<int, int> prime_factors(int n) {
    map<int, int> factors;
    for(int i = 2; i * i <= n; i++) {
        while(n % i == 0) {
            factors[i]++;
            n /= i;
        }
    }
    if(n > 1) factors[n]++;
    return factors;
}

// Modular arithmetic utilities
ll mod_add(ll a, ll b, ll mod = MOD) {
    return ((a % mod) + (b % mod)) % mod;
}

ll mod_sub(ll a, ll b, ll mod = MOD) {
    return ((a % mod) - (b % mod) + mod) % mod;
}

ll mod_mul(ll a, ll b, ll mod = MOD) {
    return ((a % mod) * (b % mod)) % mod;
}

// Factorial and combinatorics
const int MAXF = 1e6 + 5;
ll fact[MAXF], inv_fact[MAXF];

void precompute_factorial() {
    fact[0] = 1;
    for(int i = 1; i < MAXF; i++) {
        fact[i] = (fact[i-1] * i) % MOD;
    }
    
    inv_fact[MAXF-1] = modinv(fact[MAXF-1]);
    for(int i = MAXF-2; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD;
    }
}

ll nCr(int n, int r) {
    if(r > n || r < 0) return 0;
    return (fact[n] * inv_fact[r] % MOD) * inv_fact[n-r] % MOD;
}

ll nPr(int n, int r) {
    if(r > n || r < 0) return 0;
    return (fact[n] * inv_fact[n-r]) % MOD;
}
\`\`\`

**Size:** ~100 lines
**Use case:** Math/number theory problems

**Template Selection Guide:**

| Problem Type | Template | Key Indicators |
|--------------|----------|----------------|
| Sorting, searching | Basic | Arrays, simple logic |
| Shortest path, connectivity | Graph | "Graph", "tree", "path" |
| Points, lines, polygons | Geometry | Coordinates, distances |
| Pattern matching, palindromes | String | Text processing |
| Primes, modulo, GCD | Number Theory | Divisibility, factors |
| Segment trees, fenwick | Data Structures | Range queries |

**Workflow:**

1. Read problem
2. Identify type
3. Copy appropriate template
4. Modify as needed
5. Solve!

**Template Management:**

**Quick Copy Script:**
\`\`\`bash
#!/bin/bash
# cp_template.sh

TEMPLATE_DIR=~/cp/templates

case $1 in
    basic|b)
        cp $TEMPLATE_DIR/basic.cpp $2.cpp
        ;;
    graph|g)
        cp $TEMPLATE_DIR/graph.cpp $2.cpp
        ;;
    geometry|geo)
        cp $TEMPLATE_DIR/geometry.cpp $2.cpp
        ;;
    string|s)
        cp $TEMPLATE_DIR/string.cpp $2.cpp
        ;;
    number|n)
        cp $TEMPLATE_DIR/number_theory.cpp $2.cpp
        ;;
    *)
        echo "Usage: cp_template.sh [basic|graph|geometry|string|number] <filename>"
        ;;
esac

echo "Created $2.cpp"
\`\`\`

**Usage:**
\`\`\`bash
./cp_template.sh graph A  # Creates A.cpp with graph template
./cp_template.sh basic B  # Creates B.cpp with basic template
\`\`\`

**Bottom Line:**

Specialized templates:
- ✅ Save time on specific problems
- ✅ Reduce implementation errors
- ✅ Standard algorithms ready
- ✅ Focus on problem logic, not boilerplate

**Build your template library over time, one problem type at a time!**`,
    },
    {
      question:
        'A robust template should evolve based on experience and lessons learned. Describe a process for continuously improving your template, including testing, documentation, and version control.',
      answer: `Your template is a living document that should grow with your competitive programming journey. Here's the complete improvement system:

**Template Lifecycle:**

\`\`\`
1. CREATE initial template
   ↓
2. USE in contests
   ↓
3. IDENTIFY issues/missing features
   ↓
4. TEST improvements
   ↓
5. UPDATE template
   ↓
6. DOCUMENT changes
   ↓
Back to step 2
\`\`\`

**Version Control System:**

**Git Setup:**
\`\`\`bash
# Initialize repository
cd ~/cp
git init
git add templates/
git commit -m "Initial template version"

# Tag versions
git tag v1.0
\`\`\`

**Structure:**
\`\`\`
cp/
├── .git/
├── templates/
│   ├── basic.cpp
│   ├── graph.cpp
│   ├── geometry.cpp
│   └── ...
├── CHANGELOG.md     # Track changes
├── README.md        # Documentation
├── tests/           # Test cases for template components
│   ├── test_gcd.cpp
│   ├── test_dijkstra.cpp
│   └── ...
└── archive/         # Old versions
    ├── v1.0/
    ├── v2.0/
    └── ...
\`\`\`

**CHANGELOG.md Example:**
\`\`\`markdown
# Template Changelog

## v3.0 - 2024-03-15
### Added
- Dijkstra's algorithm to graph template
- String hashing function
- Geometry convex hull

### Fixed
- GCD template now handles negative numbers
- Fast I/O bug with interactive problems

### Removed
- Unused matrix multiplication (never used)

## v2.0 - 2024-02-01
### Added
- Debug macros with LOCAL flag
- Multiple test case handling

### Changed
- Renamed ll to long long in some places for clarity

## v1.0 - 2024-01-01
- Initial template version
\`\`\`

**Testing System:**

**Test Framework:**
\`\`\`cpp
// tests/test_template.cpp
#include "../templates/basic.cpp"
#include <cassert>

void test_gcd() {
    assert(gcd(12, 18) == 6);
    assert(gcd(7, 11) == 1);
    assert(gcd(0, 5) == 5);
    assert(gcd(-12, 18) == 6);  // Should handle negatives
    cout << "✓ GCD tests passed" << endl;
}

void test_modpow() {
    assert(modpow(2, 10, 1000) == 24);
    assert(modpow(3, 4, 10) == 1);
    assert(modpow(5, 0, 10) == 1);
    cout << "✓ Modpow tests passed" << endl;
}

void test_lcm() {
    assert(lcm(12, 18) == 36);
    assert(lcm(7, 11) == 77);
    cout << "✓ LCM tests passed" << endl;
}

int main() {
    test_gcd();
    test_modpow();
    test_lcm();
    
    cout << "All tests passed!" << endl;
    return 0;
}
\`\`\`

**Run tests:**
\`\`\`bash
g++ tests/test_template.cpp -o test && ./test
\`\`\`

**Continuous Improvement Process:**

**Phase 1: Post-Contest Review (After Every Contest)**

\`\`\`markdown
Review Questions:
1. What did I reimplement that should be in template?
2. What template component had bugs?
3. What was missing that I needed?
4. What did I use that I've never used before?
5. What slowed me down?
\`\`\`

**Example:**
\`\`\`
Contest: Codeforces Round #800
Date: 2024-03-15

Issues:
- Dijkstra had bug with visited array
- Missing prefix sum utility
- Debug macro didn't work with vectors

Action Items:
- Fix Dijkstra in graph template
- Add prefix sum to basic template
- Enhance debug macro

Changes Made:
- ✅ Fixed Dijkstra
- ✅ Added prefix_sum function
- ✅ Added debugv macro for vectors
\`\`\`

**Phase 2: Testing New Features**

**Before adding to template:**
\`\`\`cpp
// test_new_feature.cpp
// Test the new feature in isolation

#include <bits/stdc++.h>
using namespace std;

// New feature to test
template<typename T>
vector<T> prefix_sum(const vector<T>& arr) {
    int n = arr.size();
    vector<T> prefix(n + 1, 0);
    for(int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + arr[i];
    }
    return prefix;
}

int main() {
    // Test case 1: Basic
    vector<int> v1 = {1, 2, 3, 4, 5};
    auto p1 = prefix_sum(v1);
    assert(p1[5] == 15);  // Sum of 1-5
    
    // Test case 2: Empty
    vector<int> v2 = {};
    auto p2 = prefix_sum(v2);
    assert(p2[0] == 0);
    
    // Test case 3: Negative
    vector<int> v3 = {-1, 2, -3, 4};
    auto p3 = prefix_sum(v3);
    assert(p3[4] == 2);
    
    cout << "All tests passed!" << endl;
    return 0;
}
\`\`\`

**Only add after thorough testing!**

**Phase 3: Documentation**

**README.md Structure:**
\`\`\`markdown
# CP Template Library

## Quick Start
\`\`\`bash
# Copy basic template
cp ~/cp/templates/basic.cpp solution.cpp
\`\`\`

## Templates

### Basic Template
**Use for:** Simple problems, arrays, basic algorithms

**Contents:**
- Fast I/O
- Common type aliases
- Loop macros
- Debug macros

### Graph Template
**Use for:** Graph problems, trees, shortest paths

**Contents:**
- DFS/BFS
- Dijkstra
- Union-Find
- Topological sort

**Example:**
\`\`\`cpp
// Reading graph
int n, m;
cin >> n >> m;
rep(i, m) {
    int u, v;
    cin >> u >> v;
    graph[u].pb(v);
    graph[v].pb(u);
}

// Run BFS
bfs(1);
\`\`\`

## Utility Reference

### gcd(a, b)
**Purpose:** Calculate greatest common divisor

**Usage:**
\`\`\`cpp
int g = gcd(12, 18);  // Returns 6
\`\`\`

**Time:** O(log(min(a, b)))

### modpow(base, exp, mod)
**Purpose:** Modular exponentiation

**Usage:**
\`\`\`cpp
ll result = modpow(2, 10, MOD);  // 2^10 mod MOD
\`\`\`

**Time:** O(log exp)

[... document each function ...]

## Common Patterns

### Reading Array
\`\`\`cpp
int n; cin >> n;
vi arr(n);
rep(i, n) cin >> arr[i];
\`\`\`

### Multiple Test Cases
\`\`\`cpp
int t; cin >> t;
while(t--) solve();
\`\`\`

### Sorting Pairs by Second Element
\`\`\`cpp
sort(all(v), [](pii a, pii b) {
    return a.S < b.S;
});
\`\`\`

## Debugging

### Local Testing
\`\`\`bash
g++ -DLOCAL -std=c++17 -O2 solution.cpp
./a.out < input.txt
\`\`\`

### Debug Output
\`\`\`cpp
debug(x);        // Print single variable
debug2(x, y);    // Print two variables
debugv(arr);     // Print vector
\`\`\`

## Changelog
See [CHANGELOG.md](CHANGELOG.md)
\`\`\`

**Phase 4: Periodic Review**

**Monthly Review:**
\`\`\`
□ Check for unused components (remove?)
□ Review recent contests for patterns
□ Test all utilities still work
□ Update documentation
□ Clean up code formatting
□ Tag new version if significant changes
\`\`\`

**Improvement Tracking:**

**Keep a journal:**
\`\`\`markdown
## Template Improvement Log

### 2024-03-15
**Problem:** Dijkstra had wrong initialization
**Solution:** Changed INF handling
**Test:** Verified on CF #800 Problem C
**Status:** ✅ Fixed

### 2024-03-10
**Problem:** No way to debug pairs
**Solution:** Added operator<< overload for pairs
**Test:** Used in practice problems
**Status:** ✅ Added

### 2024-03-05
**Idea:** Add segment tree template
**Status:** ⏳ In progress, testing different implementations
\`\`\`

**Quality Standards:**

**Before adding anything:**
1. ✅ Tested on at least 3 problems
2. ✅ Handles edge cases
3. ✅ Documented
4. ✅ Clear variable names
5. ✅ Consistent style

**Code Review Checklist:**
\`\`\`
□ No bugs found in testing?
□ Handles n=0, n=1?
□ Handles negative numbers (if applicable)?
□ Handles overflow (uses ll where needed)?
□ Clear and readable?
□ Consistent with rest of template?
□ Documented in README?
\`\`\`

**Backup Strategy:**

**Multiple Backups:**
1. **Git** (local + remote)
2. **GitHub** (public or private repo)
3. **Google Drive** (backup folder)
4. **USB Drive** (physical backup)

**Never lose your template!**

**Git Workflow:**
\`\`\`bash
# After improvements
git add templates/
git commit -m "Add Dijkstra to graph template"

# Tag major versions
git tag v3.0
git push origin v3.0

# Push to GitHub
git push origin main
\`\`\`

**Template Evolution Over Time:**

**Beginner (Months 1-6):**
- Basic template only
- Few macros
- Standard utilities

**Intermediate (Months 6-12):**
- Multiple templates
- More utilities
- Personal customizations

**Advanced (Year 2+):**
- Specialized templates
- Tested components
- Well-documented
- Version controlled

**Anti-Patterns to Avoid:**

**1. Kitchen Sink Template**
- Adding everything you see online
- 1000+ line template
- Most code never used
- Hard to navigate

**Solution:** Only add what YOU use

**2. Never Testing**
- Add code without testing
- Discovers bugs in contest
- Wastes time

**Solution:** Test everything before adding

**3. No Documentation**
- Forget what functions do
- Waste time reading code

**Solution:** Document as you go

**4. Never Updating**
- Template gets stale
- Missing useful features

**Solution:** Review after every contest

**Sharing and Learning:**

**Learn from others:**
\`\`\`bash
# Check out top coders' templates on GitHub
# Search: "competitive programming template"
\`\`\`

**Share your template:**
- GitHub repository
- Blog post
- Help beginners

**But make it YOUR OWN:**
- Understand every line
- Customize to your style
- Test thoroughly

**Automation:**

**Automated Testing:**
\`\`\`bash
#!/bin/bash
# test_all.sh

echo "Running template tests..."

for test in tests/*.cpp; do
    echo "Testing $test..."
    g++ -std=c++17 $test -o test_out || exit 1
    ./test_out || exit 1
    rm test_out
done

echo "✓ All tests passed!"
\`\`\`

**Run before every commit:**
\`\`\`bash
./test_all.sh && git commit
\`\`\`

**Bottom Line:**

Template improvement:
- ✅ Version control (Git)
- ✅ Test new features
- ✅ Document everything
- ✅ Review after contests
- ✅ Backup regularly
- ✅ Evolve continuously

**Your template should be:**
- Tested: No bugs
- Documented: Easy to use
- Versioned: Track changes
- Personal: Fits your style
- Growing: Always improving

**Invest in your template, it's an investment in your competitive programming career!** 🚀`,
    },
  ],
} as const;
