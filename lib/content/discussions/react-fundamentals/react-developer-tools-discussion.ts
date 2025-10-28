export const reactDeveloperToolsDiscussion = {
  title: 'React Developer Tools Discussion Questions',
  id: 'react-developer-tools-discussion',
  sectionId: 'react-developer-tools',
  questions: [
    {
      id: 'q1',
      question:
        'Explain how you would use React DevTools to debug a component that is re-rendering too frequently. Walk through your debugging process step-by-step, including which DevTools features you would use and what information you would look for.',
      answer: `Debugging unnecessary re-renders is a common performance issue. Here's a systematic approach using React DevTools:

**Step 1: Confirm the Problem Exists**

First, verify the component is actually re-rendering more than expected.

\`\`\`tsx
// Add console.log to suspect component
function ExpensiveComponent({ userId, onUpdate }) {
  console.log('ExpensiveComponent rendered');
  
  // ... expensive rendering logic
  
  return <div>Content</div>;
}

// User types in search box → console shows:
// "ExpensiveComponent rendered" (x50 times)
// Problem confirmed: Re-rendering on every keystroke!
\`\`\`

**Step 2: Open React DevTools Profiler**

1. Open browser DevTools (F12)
2. Navigate to **"Profiler"** tab
3. Click **"Record"** button (red circle)
4. Reproduce the issue (e.g., type in search box)
5. Click **"Stop"**

**Step 3: Analyze the Commit Timeline**

View all re-renders that occurred:

\`\`\`
Commits:
1. Initial Mount (12ms)
2. Keystroke "h" (8ms)
3. Keystroke "he" (7ms)
4. Keystroke "hel" (8ms)
5. Keystroke "hell" (7ms)
6. Keystroke "hello" (8ms)
\`\`\`

**Observation:** Component renders on every keystroke (expected), but let's verify if expensive components are re-rendering unnecessarily.

**Step 4: Check Ranked Chart**

Click any commit → Select **"Ranked"** view:

\`\`\`
Components sorted by render time:
1. ExpensiveComponent  7.2ms  ← Problem!
2. SearchBox           0.5ms
3. App                 0.3ms
\`\`\`

**Finding:** ExpensiveComponent is slow AND re-rendering frequently.

**Step 5: Investigate Why It Rendered**

Select ExpensiveComponent in the Profiler → Check "Why did this render?":

\`\`\`
ExpensiveComponent
Why did this render?
  • Props changed: onUpdate
  • Parent <App> rendered
\`\`\`

**Key insight:** \`onUpdate\` prop is changing on every render!

**Step 6: Switch to Components Tab**

1. Go to **"Components"** tab
2. Select ExpensiveComponent
3. View props in right panel:

\`\`\`
ExpensiveComponent
├─ props
│  ├─ userId: 123
│  └─ onUpdate: ƒ onUpdate()  ← Function
\`\`\`

**Step 7: Investigate Parent Component**

1. Look at component stack: ExpensiveComponent → App
2. Select App in tree
3. View App's code:

\`\`\`tsx
// Problem code
function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [userId] = useState(123);
  
  // ❌ New function created every render!
  function handleUpdate() {
    updateUser(userId);
  }
  
  return (
    <div>
      <SearchBox value={searchQuery} onChange={setSearchQuery} />
      <ExpensiveComponent userId={userId} onUpdate={handleUpdate} />
    </div>
  );
}

// Every time searchQuery changes:
// 1. App re-renders
// 2. handleUpdate is recreated (new function reference)
// 3. ExpensiveComponent receives new onUpdate prop
// 4. ExpensiveComponent re-renders (even though logic is same)
\`\`\`

**Step 8: Verify the Issue**

In Components tab, select ExpensiveComponent multiple times while interacting:
- Note the function reference changes: \`ƒ onUpdate()\` has different memory address
- This confirms new function created every render

**Step 9: Fix the Issue**

\`\`\`tsx
import { useCallback, memo } from 'react';

// Wrap component in memo to prevent re-renders when props unchanged
const ExpensiveComponent = memo(function ExpensiveComponent({ userId, onUpdate }) {
  console.log('ExpensiveComponent rendered');
  return <div>Content</div>;
});

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [userId] = useState(123);
  
  // ✅ Wrap in useCallback to maintain stable function reference
  const handleUpdate = useCallback(() => {
    updateUser(userId);
  }, [userId]);  // Only recreate if userId changes
  
  return (
    <div>
      <SearchBox value={searchQuery} onChange={setSearchQuery} />
      <ExpensiveComponent userId={userId} onUpdate={handleUpdate} />
    </div>
  );
}
\`\`\`

**Step 10: Verify the Fix**

1. Profiler → Record
2. Type in search box
3. Stop recording
4. Check Ranked chart:

\`\`\`
Before fix:
1. ExpensiveComponent  7.2ms (rendered 5 times)

After fix:
1. SearchBox  0.5ms (rendered 5 times)
// ExpensiveComponent not in list! (didn't render)
\`\`\`

**Success:** ExpensiveComponent no longer re-renders when search query changes.

**Alternative Scenarios and Solutions:**

**Scenario 2: Props Appear Unchanged**

\`\`\`
ExpensiveComponent
Why did this render?
  • Props changed: data
\`\`\`

But \`data\` looks the same! Investigate:

1. Select component
2. Right-click props.data → "Copy value"
3. Paste into console
4. Record again
5. Copy new props.data
6. Compare with \`===\`:

\`\`\`javascript
const before = { user: { name: 'Alice' } };
const after = { user: { name: 'Alice' } };
console.log(before === after);  // false! Different object reference
\`\`\`

**Problem:** New object created every render with same content.

**Solution:**
\`\`\`tsx
// ❌ Creates new object every render
function Parent() {
  return <Child data={{ user: { name: 'Alice' } }} />;
}

// ✅ Stable reference
function Parent() {
  const data = useMemo(() => ({ user: { name: 'Alice' } }), []);
  return <Child data={data} />;
}

// ✅ Or extract to constant (if never changes)
const STATIC_DATA = { user: { name: 'Alice' } };

function Parent() {
  return <Child data={STATIC_DATA} />;
}
\`\`\`

**Scenario 3: Context Causing Re-renders**

\`\`\`
ExpensiveComponent
Why did this render?
  • Context changed: ThemeContext
\`\`\`

**Problem:** Component consumes context, context provider value changes frequently.

**Solution 1: Split context**
\`\`\`tsx
// ❌ Single context with unrelated values
<AppContext.Provider value={{ theme, user, cart, notifications }}>
  <App />
</AppContext.Provider>

// ✅ Split into separate contexts
<ThemeContext.Provider value={theme}>
  <UserContext.Provider value={user}>
    <CartContext.Provider value={cart}>
      <App />
    </CartContext.Provider>
  </UserContext.Provider>
</ThemeContext.Provider>
\`\`\`

**Solution 2: Memoize context value**
\`\`\`tsx
// ❌ New object every render
function App() {
  return (
    <UserContext.Provider value={{ user, updateUser }}>
      <Content />
    </UserContext.Provider>
  );
}

// ✅ Memoized value
function App() {
  const contextValue = useMemo(
    () => ({ user, updateUser }),
    [user, updateUser]
  );
  
  return (
    <UserContext.Provider value={contextValue}>
      <Content />
    </UserContext.Provider>
  );
}
\`\`\`

**Scenario 4: Parent Always Renders**

\`\`\`
ExpensiveComponent
Why did this render?
  • Parent <Parent> rendered
\`\`\`

No props changed, but parent rendered → child rendered.

**Solution:** Wrap in \`memo\`
\`\`\`tsx
const ExpensiveComponent = memo(function ExpensiveComponent({ userId }) {
  return <div>Content</div>;
});

// Now only re-renders if userId changes, not when parent renders
\`\`\`

**Complete Debugging Checklist:**

\`\`\`
□ Confirm frequent re-renders (console.log or visual observation)
□ Open Profiler tab
□ Record interaction that causes issue
□ Check Commit timeline (how many renders?)
□ Check Ranked chart (which components slow?)
□ Select slow component
□ Read "Why did this render?"
□ Check which props/state/context changed
□ Switch to Components tab
□ Inspect actual prop/state values
□ Compare values across renders
□ Identify unstable references (functions, objects, arrays)
□ Apply fixes:
  □ useCallback for functions
  □ useMemo for objects/arrays
  □ memo for components
  □ Split context
  □ Lift state closer to where it's used
□ Record again to verify fix
□ Compare before/after performance
\`\`\`

**Interview Insight:**
Walking through a methodical debugging process demonstrates problem-solving skills. Explaining "Why did this render?" feature shows DevTools proficiency. Discussing multiple scenarios (functions, objects, context) shows breadth of knowledge.`,
    },
    {
      id: 'q2',
      question:
        'How would you use React DevTools to debug a component displaying stale data? Describe your investigation process and explain how DevTools helps you trace the data flow through the component tree.',
      answer: `Debugging stale data is a common challenge in React applications. Here's a systematic approach using DevTools:

**Scenario:** User updates their profile, but the UI still shows old data.

**Step 1: Identify the Stale Component**

1. Look at the page and identify which component shows stale data
2. Right-click the stale element
3. Select "Inspect"
4. Switch to "Components" tab
5. DevTools automatically selects the component

**Example:**
\`\`\`tsx
// UI shows: "Name: John Doe"
// But user just updated to: "Name: Jane Smith"
// ProfileDisplay component is showing stale data
\`\`\`

**Step 2: Check Current Props**

In Components tab, check ProfileDisplay's props:

\`\`\`
ProfileDisplay
├─ props
│  ├─ user: Object
│  │  ├─ id: 123
│  │  ├─ name: "John Doe"  ← Stale!
│  │  └─ email: "john@example.com"
\`\`\`

**Finding:** Component is receiving stale data via props.

**Step 3: Trace Up the Component Tree**

Who's passing this prop? Check the parent:

\`\`\`
ProfileDisplay
  ↑ rendered by Profile
    ↑ rendered by UserDashboard
      ↑ rendered by App
\`\`\`

Select Profile (parent):

\`\`\`
Profile
├─ props
│  ├─ userId: 123
├─ state
│  ├─ user: Object
│  │  ├─ id: 123
│  │  ├─ name: "John Doe"  ← Also stale!
│  │  └─ email: "john@example.com"
│  └─ isLoading: false
\`\`\`

**Finding:** Profile component has stale data in state. It's not fetching updated data.

**Step 4: Inspect the Source Code**

Click the \`< >\` icon next to Profile → Jumps to source:

\`\`\`tsx
function Profile({ userId }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(data => {
        setUser(data);
        setIsLoading(false);
      });
  }, []);  // ← Problem! Empty dependency array
  
  return <ProfileDisplay user={user} />;
}

// The effect only runs once on mount
// When userId changes, it doesn't refetch
// So data becomes stale
\`\`\`

**Problem identified:** useEffect missing \`userId\` in dependencies!

**Step 5: Verify the Issue**

Test the hypothesis using DevTools:

1. Note current userId in props: \`123\`
2. Navigate to different user (triggers userId change to \`456\`)
3. Check Profile's props:

\`\`\`
Profile
├─ props
│  ├─ userId: 456  ← Changed!
├─ state
│  ├─ user: Object
│  │  ├─ id: 123  ← Still old user!
\`\`\`

**Confirmed:** Props changed but state didn't update.

**Step 6: Fix the Issue**

\`\`\`tsx
function Profile({ userId }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    setIsLoading(true);
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(data => {
        setUser(data);
        setIsLoading(false);
      });
  }, [userId]);  // ✅ Added userId to dependencies
  
  return <ProfileDisplay user={user} />;
}
\`\`\`

**Step 7: Verify the Fix**

1. Refresh page
2. Open DevTools Components tab
3. Navigate to different user
4. Watch Profile's state update in real-time:

\`\`\`
Profile
├─ props
│  ├─ userId: 456  ← Changed
├─ state
│  ├─ user: Object
│  │  ├─ id: 456  ← Updated! ✓
│  │  ├─ name: "Jane Smith"
\`\`\`

**Success:** Data now updates correctly.

**Alternative Scenarios:**

**Scenario 2: Props Not Passed Down**

\`\`\`tsx
// Parent has fresh data
<Parent>
├─ state
│  └─ user: { name: "Jane Smith" }  ← Fresh

// But child has stale data
<Child>
├─ props
│  └─ userName: "John Doe"  ← Stale
\`\`\`

**Investigation:**
1. Check Parent's render method
2. Find: \`<Child userName={oldUser.name} />\`
3. Problem: Passing old variable

**Fix:**
\`\`\`tsx
// ❌ Wrong variable
<Child userName={oldUser.name} />

// ✅ Correct variable
<Child userName={user.name} />
\`\`\`

**Scenario 3: Closure Capturing Old Value**

\`\`\`tsx
function Component() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      console.log('Count:', count);  // Always logs 0!
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);  // ← Empty deps captures initial count (0)
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>Increment</button>
    </div>
  );
}
\`\`\`

**DevTools Investigation:**
1. Select Component
2. Check state: \`count: 5\`
3. But console still logs: \`"Count: 0"\`
4. Problem: Closure captured initial value

**Fix:**
\`\`\`tsx
useEffect(() => {
  const interval = setInterval(() => {
    console.log('Count:', count);
  }, 1000);
  
  return () => clearInterval(interval);
}, [count]);  // ✅ Add count to deps
\`\`\`

**Scenario 4: State Not Updating After Async Operation**

\`\`\`tsx
function Component({ userId }) {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetchData(userId).then(result => {
      setData(result);
    });
  }, [userId]);
  
  // User quickly switches from userId=1 to userId=2
  // Both requests fire, but userId=1 response comes back LAST
  // State shows data for userId=1 even though userId=2 is current
  
  return <div>{data?.name}</div>;
}
\`\`\`

**DevTools Investigation:**
1. Select Component
2. Props show: \`userId: 2\`
3. State shows: \`data: { id: 1, name: "User 1" }\` ← Wrong user!
4. Problem: Race condition

**Fix:**
\`\`\`tsx
useEffect(() => {
  let cancelled = false;
  
  fetchData(userId).then(result => {
    if (!cancelled) {  // Only update if still relevant
      setData(result);
    }
  });
  
  return () => {
    cancelled = true;  // Ignore stale responses
  };
}, [userId]);
\`\`\`

**Advanced: Using DevTools to Edit State for Testing**

Test how component behaves with different data WITHOUT changing code:

1. Select component
2. Find state value
3. Double-click to edit
4. Enter new value
5. Press Enter
6. Component re-renders with new state

**Example:**
\`\`\`
// Current state:
user: { name: "Alice", role: "user" }

// Edit to:
user: { name: "Alice", role: "admin" }

// Instantly see how component handles admin role
// Without deploying code changes or creating test data
\`\`\`

**Debugging Checklist for Stale Data:**

\`\`\`
□ Identify which component shows stale data
□ Open Components tab
□ Select component (use element picker)
□ Check props: Are they stale?
  ├─ Yes → Problem is parent not passing fresh data
  └─ No → Problem is within this component
□ Check state: Is it stale?
  ├─ Yes → Component not updating state
  └─ No → Problem is how props/state are used in render
□ Check hooks: Which hooks are used?
  ├─ useEffect with [] deps → Might be missing dependencies
  ├─ useMemo/useCallback → Might have stale dependencies
  └─ Custom hooks → Check their dependencies
□ Trace up component tree:
  □ Check each parent's props/state
  □ Find where fresh data exists
  □ Find where it becomes stale
□ Check source code (click < > icon)
□ Identify root cause:
  □ Missing dependencies
  □ Closure capturing old value
  □ Race condition
  □ Not passing updated props
  □ Comparing with stale reference
□ Apply fix
□ Verify in DevTools
□ Test edge cases by editing state in DevTools
\`\`\`

**Interview Insight:**
Demonstrating a systematic approach to tracing data flow through the component tree shows debugging expertise. Explaining how to use DevTools to inspect props at each level and identify where data becomes stale demonstrates practical problem-solving skills.`,
    },
    {
      id: 'q3',
      question:
        "Explain how the React DevTools Profiler works and how you would use it to identify and fix performance bottlenecks in a React application. What metrics should you focus on and what do they tell you about your application's performance?",
      answer: `The React DevTools Profiler is a powerful performance analysis tool. Here's a comprehensive guide to using it effectively:

**What the Profiler Measures:**

The Profiler captures:
1. **Which components rendered**
2. **How long each render took**
3. **Why each component rendered**
4. **Number of times each component rendered**
5. **Timeline of all renders (commits)**

**Key Concepts:**

**Commit:** A single render cycle where React updates the DOM
- Can involve multiple component renders
- Measured in milliseconds (ms)

**Render:** One component function executing
- Part of a commit
- Can be expensive if component does heavy computation

**Step-by-Step Performance Optimization Workflow:**

**Step 1: Record a Profile**

1. Open DevTools → Profiler tab
2. Click **"Record"** button
3. Perform the slow interaction (e.g., type in search, scroll list, click button)
4. Click **"Stop"**
5. Profiler shows all commits that occurred

**Step 2: Analyze the Commit Timeline**

View the timeline graph:

\`\`\`
Commits over time:
────────────────────────────────────────────
│    ▂▄█▆▅▃▂  ← Renders
│   
0ms                                     500ms
\`\`\`

Each bar represents one commit. Height = render duration.

**What to look for:**
- **Tall bars**: Slow renders (> 16ms can cause dropped frames)
- **Many bars**: Too many renders
- **Regular pattern**: Might be polling/interval
- **Spikes**: Sudden slowness (investigate these first)

**Step 3: Select Slowest Commit**

Click the tallest bar. See details:

\`\`\`
Commit #5
Duration: 45ms
Rendered at: 2.3s
Components: 24 rendered
\`\`\`

**45ms is slow!** (At 60fps, each frame must render in 16ms)

**Step 4: Check Ranked Chart**

Switch to "Ranked" view to see slowest components:

\`\`\`
Ranked by render time:
1. ProductList        32.4ms  ← Culprit!
2. ProductCard (×20)   1.2ms each (24ms total)
3. App                 2.1ms
4. Header              0.8ms
5. Footer              0.3ms
\`\`\`

**Finding:** ProductList takes 32ms out of 45ms total!

**Step 5: Check Flamegraph**

Shows call stack of renders:

\`\`\`
App (45ms)
├─ Header (0.8ms)
├─ ProductList (32.4ms)  ← Most time here!
│  └─ ProductCard (×20, 1.2ms each)
└─ Footer (0.3ms)
\`\`\`

**Colors:**
- **Green**: Fast (< 1ms)
- **Yellow**: Moderate (1-5ms)
- **Orange**: Slow (5-10ms)
- **Red**: Very slow (> 10ms)

ProductList is red → Very slow!

**Step 6: Investigate Why It Rendered**

Select ProductList in Profiler. Check "Why did this render?":

\`\`\`
ProductList
Why did this render?
  • Props changed: onFilterChange
  • State changed: sortBy
  • Parent <App> rendered
\`\`\`

**Step 7: Check Component Implementation**

Click \`< >\` icon → Jump to source:

\`\`\`tsx
// Problem code
function ProductList({ products, onFilterChange }) {
  const [sortBy, setSortBy] = useState('name');
  
  // ❌ Expensive: Sorts on every render!
  const sortedProducts = products.sort((a, b) => {
    return a[sortBy].localeCompare(b[sortBy]);
  });
  
  return (
    <div>
      {sortedProducts.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}

// Problems:
// 1. Sorting happens every render (not memoized)
// 2. Array.sort() mutates original array
// 3. No memoization of ProductCard list
\`\`\`

**Step 8: Apply Fixes**

\`\`\`tsx
import { useMemo, memo } from 'react';

// Fix 1: Memoize expensive sorting
function ProductList({ products, onFilterChange }) {
  const [sortBy, setSortBy] = useState('name');
  
  // ✅ Only re-sort when products or sortBy changes
  const sortedProducts = useMemo(() => {
    return [...products].sort((a, b) => {  // Copy before sorting
      return a[sortBy].localeCompare(b[sortBy]);
    });
  }, [products, sortBy]);
  
  return (
    <div>
      {sortedProducts.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}

// Fix 2: Memoize ProductCard to prevent unnecessary re-renders
const ProductCard = memo(function ProductCard({ product }) {
  return (
    <div>
      <h3>{product.name}</h3>
      <p>{\`\$\${product.price}\`}</p>
    </div>
  );
});
\`\`\`

**Step 9: Record Again and Compare**

1. Profiler → Record
2. Perform same interaction
3. Stop
4. Compare results:

\`\`\`
Before:
Commit #5: 45ms
  ProductList: 32.4ms

After:
Commit #5: 3.2ms  ← 14× faster!
  ProductList: 1.8ms
\`\`\`

**Huge improvement!**

**Key Metrics to Focus On:**

**1. Commit Duration**
- **Target**: < 16ms (60fps) or < 33ms (30fps)
- **If higher**: User will perceive lag

**2. Component Render Count**
- **What it shows**: How many times component rendered
- **Red flag**: Same component rendered many times in one commit
- **Fix**: Add React.memo or optimize parent

**3. Self Time vs Total Time**
- **Self time**: Time spent in this component's code
- **Total time**: Self time + all children
- **If Total >> Self**: Problem is in children
- **If Self is high**: Problem is in this component's logic

**4. Number of Renders in Timeline**
- **What it shows**: How often components render
- **Red flag**: Hundreds of renders for simple interaction
- **Fix**: Reduce unnecessary renders with memo/useCallback/useMemo

**5. Why Component Rendered**
- **Most important metric!** Tells you root cause
- **Common causes**:
  - Props changed (check which props)
  - State changed (check which state)
  - Parent rendered (add memo to child)
  - Context changed (split context or memoize value)

**Common Performance Patterns:**

**Pattern 1: Function Prop Causing Re-renders**

\`\`\`tsx
// ❌ Bad: New function every render
function Parent() {
  return <Child onUpdate={() => update()} />;
}

const Child = memo(function Child({ onUpdate }) {
  // Re-renders even though memo'd because onUpdate changes
  return <button onClick={onUpdate}>Update</button>;
});

// Profiler shows:
// Child: Why rendered? → Props changed: onUpdate
\`\`\`

**Fix:**
\`\`\`tsx
function Parent() {
  const handleUpdate = useCallback(() => update(), []);
  return <Child onUpdate={handleUpdate} />;
}

// Profiler after fix:
// Child doesn't appear in commits (not rendering!)
\`\`\`

**Pattern 2: Large List Re-rendering**

\`\`\`tsx
// ❌ Bad: All items re-render
function List({ items, onItemClick }) {
  return (
    <div>
      {items.map(item => (
        <Item 
          key={item.id}
          item={item}
          onClick={() => onItemClick(item.id)}  // New function!
        />
      ))}
    </div>
  );
}

// Profiler shows:
// Item (×100) each rendered: 2ms = 200ms total!
\`\`\`

**Fix:**
\`\`\`tsx
const Item = memo(function Item({ item, onClick }) {
  return <div onClick={onClick}>{item.name}</div>;
});

function List({ items, onItemClick }) {
  return (
    <div>
      {items.map(item => {
        // Stable callback per item
        const handleClick = () => onItemClick(item.id);
        return (
          <Item 
            key={item.id}
            item={item}
            onClick={handleClick}
          />
        );
      })}
    </div>
  );
}

// Or better: Pass ID and handler separately
function List({ items, onItemClick }) {
  return (
    <div>
      {items.map(item => (
        <Item 
          key={item.id}
          item={item}
          onItemClick={onItemClick}  // Same function reference
        />
      ))}
    </div>
  );
}

const Item = memo(function Item({ item, onItemClick }) {
  return (
    <div onClick={() => onItemClick(item.id)}>
      {item.name}
    </div>
  );
});

// Profiler after fix:
// Only changed items render (not entire list)
\`\`\`

**Pattern 3: Context Causing Widespread Re-renders**

\`\`\`tsx
// ❌ Bad: All consumers re-render when any value changes
const AppContext = createContext();

function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');
  const [cart, setCart] = useState([]);
  
  return (
    <AppContext.Provider value={{ user, theme, cart }}>
      <Components />
    </AppContext.Provider>
  );
}

// When cart updates:
// All components using context re-render (even if they only need user)
\`\`\`

**Profiler shows:**
\`\`\`
50 components rendered
Why? → Context changed: AppContext
\`\`\`

**Fix:**
\`\`\`tsx
// ✅ Split into separate contexts
<UserContext.Provider value={user}>
  <ThemeContext.Provider value={theme}>
    <CartContext.Provider value={cart}>
      <Components />
    </CartContext.Provider>
  </ThemeContext.Provider>
</UserContext.Provider>

// Now only components using CartContext re-render when cart updates
\`\`\`

**Performance Optimization Checklist:**

\`\`\`
□ Record profile of slow interaction
□ Find slowest commit (tallest bar)
□ Switch to Ranked view
□ Identify slowest components (top of list)
□ Check Flamegraph for visual hierarchy
□ Select slow component
□ Check "Why did this render?"
□ Identify root cause:
  □ Props changing unnecessarily?
  □ State updating too often?
  □ Parent causing child re-renders?
  □ Context value changing?
  □ Expensive computation not memoized?
□ Apply appropriate fix:
  □ useCallback for functions
  □ useMemo for computed values
  □ React.memo for components
  □ Split context
  □ Optimize state structure
  □ Virtualize long lists
□ Record again
□ Compare before/after metrics
□ Repeat for next slowest component
\`\`\`

**Interview Insight:**
Explaining the Profiler's metrics (commit duration, why rendered, self vs total time) shows deep understanding of performance. Walking through a systematic optimization process demonstrates production experience. Mentioning specific patterns (function props, context, lists) shows breadth of knowledge.`,
    },
  ],
};
