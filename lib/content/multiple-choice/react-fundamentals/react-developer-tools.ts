export const reactDeveloperToolsQuiz = {
  title: 'React Developer Tools Quiz',
  id: 'react-developer-tools-quiz',
  sectionId: 'react-developer-tools',
  questions: [
    {
      id: 'q1',
      question:
        'What information does React DevTools NOT show you about a component?',
      options: [
        'Current props and state',
        'Why the component re-rendered',
        'The compiled JavaScript output',
        'Which hooks the component uses',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"The compiled JavaScript output"**.

React DevTools shows you React-specific information about your components, not the compiled/transpiled JavaScript code. For compiled output, you would use the browser's Sources tab.

**What React DevTools DOES show:**

**1. Props and State:**
\`\`\`
UserProfile
├─ props
│  ├─ userId: 123
│  ├─ name: "Alice"
│  └─ onUpdate: ƒ handleUpdate()
├─ state
│  ├─ isEditing: false
│  └─ editedName: ""
\`\`\`

You can:
- View current values
- Edit them live
- See nested objects/arrays
- Track changes over time

**2. Why Component Re-rendered:**
\`\`\`
Post
Why did this render?
  • Props changed: likes (42 → 43)
  • Parent <PostList> rendered
\`\`\`

Shows exactly what triggered the re-render:
- Which props changed
- Which state changed
- Which context changed
- Parent re-rendered
- Hook dependencies changed

**3. Hooks:**
\`\`\`
hooks
├─ State: "Alice"           (useState)
├─ State: false             (useState)
├─ Effect: ƒ ()             (useEffect)
├─ Callback: ƒ handleClick() (useCallback)
└─ Memo: { computed: true } (useMemo)
\`\`\`

Shows:
- Hook types
- Current values
- Hook execution order
- Custom hook composition

**What React DevTools DOESN'T Show:**

**Compiled JavaScript:**

\`\`\`tsx
// Your JSX code
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Compiled to (via Babel):
function Greeting(_ref) {
  var name = _ref.name;
  return React.createElement("h1", null, "Hello, ", name, "!");
}

// DevTools shows: Original JSX component structure
// DevTools does NOT show: Compiled JavaScript
\`\`\`

To see compiled output:
1. Use browser DevTools → Sources tab
2. Or use Babel REPL
3. Or webpack bundle analyzer

**Other React DevTools Features:**

**Component Tree:**
\`\`\`
App
├─ Header
│  └─ Navigation
├─ Main
│  ├─ Sidebar
│  └─ Content
└─ Footer
\`\`\`

**Context Values:**
\`\`\`
Button
├─ hooks
│  └─ Context: "dark"  (ThemeContext)
\`\`\`

**Performance Profiling:**
- Commit timeline
- Flamegraph
- Ranked chart
- Render durations

**Source Mapping:**
- Click \`< >\` button → Jump to source code
- Only works if source maps enabled
- Shows your original code (not compiled)

**Component Highlighting:**
- Hover component in DevTools → Highlights on page
- Shows component bounds, name, dimensions

**Component Filtering:**
- Filter by name
- Filter by props
- Hide library components

**Why You'd Want to See Compiled Output (use Sources tab instead):**

1. **Debugging transpilation issues:**
\`\`\`tsx
// Your code
const App = () => <div>Hello</div>;

// If you see errors about "unexpected token <"
// Check compiled output in Sources tab
// Might reveal Babel not running
\`\`\`

2. **Understanding bundle size:**
- Use webpack-bundle-analyzer
- Not React DevTools

3. **Debugging minified production code:**
- Use Sources tab + source maps
- React DevTools shows component structure regardless

**Interview Tip:**
Understanding that DevTools is for React-specific debugging (components, props, state, hooks) while Sources tab is for JavaScript debugging (breakpoints, compiled code, network) shows clear separation of concerns.`,
    },
    {
      id: 'q2',
      question:
        'In the React DevTools Profiler, what does a "commit" represent?',
      options: [
        'A single component function execution',
        'A complete render cycle where React updates the DOM',
        'A state update in a component',
        'A prop change passed to a component',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"A complete render cycle where React updates the DOM"**.

A **commit** is React's term for a complete update cycle—from when state/props change to when the DOM is updated and browser paints the changes.

**Understanding Commits:**

**Commit = One Render Cycle**

\`\`\`
User clicks button
  ↓
setState called
  ↓
React starts commit
  ├─ App renders
  ├─ Header renders
  ├─ Content renders
  │  ├─ PostList renders
  │  └─ Sidebar renders
  └─ Footer renders
  ↓
React updates DOM
  ↓
Browser paints
  ↓
Commit complete ✓
\`\`\`

**One Commit, Multiple Component Renders:**

\`\`\`tsx
function App() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <Header />
      <Counter count={count} />
      <Footer />
      <button onClick={() => setCount(c => c + 1)}>
        Increment
      </button>
    </div>
  );
}

// When button clicked:
// - One commit occurs
// - Multiple components render:
//   - App (state changed)
//   - Counter (props changed)
//   - Possibly Header/Footer (if no memo)
\`\`\`

**In DevTools Profiler:**

**Commit Timeline:**
\`\`\`
Commits over time:
1. Initial Mount (15ms)
2. Button click (3ms)
3. Input change (1ms)
4. Form submit (8ms)
\`\`\`

Each number is one commit.

**Commit Details:**
\`\`\`
Commit #2
Duration: 3ms
Rendered at: 1.2s
Components: 5 rendered
  - App (0.5ms)
  - Header (0.2ms)
  - Counter (1.8ms)
  - Button (0.3ms)
  - Footer (0.2ms)
\`\`\`

**Commit Phases:**

React commits happen in two phases:

\`\`\`
Render Phase (can be paused/restarted):
  - Call component functions
  - Calculate what changed
  - Build virtual DOM tree
  - Prepare updates

↓

Commit Phase (synchronous, can't be interrupted):
  - Apply updates to DOM
  - Run layout effects (useLayoutEffect)
  - Browser paints
  - Run effects (useEffect)
\`\`\`

The Profiler measures the entire commit (both phases).

**Why Other Options Are Wrong:**

**"A single component function execution":**
- That's called a **render**, not a commit
- One commit includes many renders

\`\`\`
Commit #1:
  ├─ App rendered
  ├─ Header rendered
  ├─ Content rendered
  └─ Footer rendered
\`\`\`

**"A state update in a component":**
- State updates TRIGGER commits
- They're not commits themselves

\`\`\`tsx
// This causes a commit:
setState(newValue);

// The commit includes:
// - Re-rendering components
// - Updating DOM
// - Running effects
\`\`\`

**"A prop change passed to a component":**
- Prop changes happen DURING commits
- They're not commits themselves

\`\`\`tsx
// Parent renders, passes new prop to child
<Child value={newValue} />

// This happens inside a commit
// It's not a separate commit
\`\`\`

**Real-World Example:**

\`\`\`tsx
function TodoApp() {
  const [todos, setTodos] = useState([]);
  const [filter, setFilter] = useState('all');
  
  function addTodo(text) {
    setTodos([...todos, { id: Date.now(), text }]);
    // ↑ This triggers ONE commit
  }
  
  function changeFilter(newFilter) {
    setFilter(newFilter);
    // ↑ This triggers ONE commit
  }
  
  return (
    <div>
      <AddTodoForm onAdd={addTodo} />
      <FilterButtons filter={filter} onChangeFilter={changeFilter} />
      <TodoList todos={todos} filter={filter} />
    </div>
  );
}

// User types "Buy milk" and clicks Add:
// Commit #1: addTodo called
//   - TodoApp re-renders
//   - AddTodoForm re-renders (props change)
//   - TodoList re-renders (props change)
//     - New TodoItem renders (for "Buy milk")

// User clicks "Completed" filter:
// Commit #2: changeFilter called
//   - TodoApp re-renders
//   - FilterButtons re-renders (props change)
//   - TodoList re-renders (props change)
//     - All TodoItems re-render (filtered list)

// Two separate commits (one per state update)
\`\`\`

**Profiler Shows:**
\`\`\`
Commit #1 (addTodo): 5.2ms
  - TodoApp: 0.5ms
  - TodoList: 3.8ms
    - TodoItem (×5): 0.6ms each

Commit #2 (changeFilter): 4.1ms
  - TodoApp: 0.4ms
  - FilterButtons: 0.3ms
  - TodoList: 2.9ms
    - TodoItem (×2): 0.7ms each (filtered to 2 items)
\`\`\`

**Batching Multiple State Updates:**

React batches multiple state updates into one commit:

\`\`\`tsx
function handleClick() {
  setCount(c => c + 1);
  setName('Alice');
  setEmail('alice@example.com');
  // All three updates happen in ONE commit!
}

// Not three commits, just one!
// More efficient
\`\`\`

**Commit Performance Goals:**

- **16ms per commit** = 60 FPS (smooth)
- **33ms per commit** = 30 FPS (acceptable)
- **> 50ms** = Noticeable lag

\`\`\`
Commit Timeline:
│     16ms
│  ┌────┐
│  │    │ ← Under 16ms = smooth
│  └────┘
│
│        50ms
│  ┌──────────┐
│  │          │ ← Over 50ms = laggy
│  └──────────┘
\`\`\`

**Using Profiler to Optimize:**

1. Record profile
2. Find slow commits (> 16ms)
3. Check which components took longest
4. Optimize those components
5. Re-record and compare

\`\`\`
Before optimization:
Commit #3: 45ms (slow!)
  - ProductList: 38ms

After optimization:
Commit #3: 3ms (fast!)
  - ProductList: 1.2ms
\`\`\`

**Interview Tip:**
Explaining that a commit is a complete render cycle (multiple components rendering → DOM update → paint) shows understanding of React's reconciliation process. Mentioning the 16ms target for 60fps demonstrates performance awareness.`,
    },
    {
      id: 'q3',
      question: 'What does it mean when the React DevTools icon is gray?',
      options: [
        'The site is using an old version of React',
        'React is detected but the page is currently loading',
        'No React application is detected on the page',
        'React DevTools needs to be updated',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"No React application is detected on the page"**.

The React DevTools icon color indicates whether React is present on the current page:

**Icon Colors:**

**🔴 Red Icon:**
- **Meaning:** React detected (production build)
- **Site is using:** Minified React build
- **Characteristics:**
  - Smaller bundle size
  - Faster performance
  - No development warnings
  - Harder to debug (minified names)

\`\`\`tsx
// Production build
// Component names might show as "r", "t", "n"
// PropTypes checks disabled
// Development warnings removed
\`\`\`

**🔵 Blue/Teal Icon:**
- **Meaning:** React detected (development build)
- **Site is using:** Development React build
- **Characteristics:**
  - Larger bundle size
  - Helpful warnings
  - Better error messages
  - Full component names
  - PropTypes validation

\`\`\`tsx
// Development build
// Component names: "UserProfile", "Header", etc.
// PropTypes warnings
// useEffect dependency warnings
// Key prop warnings
\`\`\`

**⚫ Gray Icon:**
- **Meaning:** No React detected
- **Possible reasons:**
  1. Site doesn't use React
  2. React hasn't loaded yet (page still loading)
  3. React loaded via unusual method (iframe, shadow DOM)
  4. React version too old (< 0.14)

**How React DevTools Detects React:**

When page loads, DevTools looks for:
\`\`\`javascript
// On window object:
window.__REACT_DEVTOOLS_GLOBAL_HOOK__

// React sets this when it initializes
// DevTools reads it to detect React
\`\`\`

**Testing the Icons:**

**Gray Icon Example:**
\`\`\`html
<!-- Plain HTML site, no React -->
<!DOCTYPE html>
<html>
  <head><title>No React Here</title></head>
  <body>
    <h1>Just HTML</h1>
    <script>
      document.querySelector('h1').textContent = 'Vanilla JS';
    </script>
  </body>
</html>

<!-- DevTools icon: Gray -->
\`\`\`

**Blue Icon Example (Development):**
\`\`\`html
<!DOCTYPE html>
<html>
  <body>
    <div id="root"></div>
    
    <!-- React development build -->
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    
    <script>
      const root = ReactDOM.createRoot(document.getElementById('root'));
      root.render(React.createElement('h1', null, 'Hello React!'));
    </script>
  </body>
</html>

<!-- DevTools icon: Blue -->
\`\`\`

**Red Icon Example (Production):**
\`\`\`html
<!DOCTYPE html>
<html>
  <body>
    <div id="root"></div>
    
    <!-- React production build -->
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    
    <script>
      const root = ReactDOM.createRoot(document.getElementById('root'));
      root.render(React.createElement('h1', null, 'Hello React!'));
    </script>
  </body>
</html>

<!-- DevTools icon: Red -->
\`\`\`

**Why Other Options Are Wrong:**

**"The site is using an old version of React":**
- Old React versions (< 16) show blue/red icon (just older DevTools UI)
- Very old versions (< 0.14) might show gray
- But this isn't the primary meaning

**"React is detected but the page is currently loading":**
- While loading, icon stays gray until React initializes
- Once React loads, icon immediately changes to blue/red
- Gray doesn't mean "loading React", it means "no React found"

**"React DevTools needs to be updated":**
- Outdated DevTools still show blue/red for React
- They just show limited features
- Gray specifically means no React detected

**Troubleshooting Gray Icon:**

**Problem:** "I know my site uses React, but icon is gray"

**Solutions:**

**1. Wait for page to fully load:**
\`\`\`
Page loading... (gray)
  ↓
React bundle loads
  ↓
React initializes
  ↓
Icon changes to blue/red
\`\`\`

**2. Check if React is actually loaded:**
\`\`\`javascript
// Open console and type:
console.log(React);

// If React is loaded: Object { ... }
// If not loaded: ReferenceError: React is not defined
\`\`\`

**3. Check for errors:**
\`\`\`javascript
// Open Console tab
// Look for errors that might have prevented React from loading:

// Example errors:
// "Uncaught SyntaxError: Unexpected token '<'"
// "Failed to load resource: net::ERR_FILE_NOT_FOUND"
// "React is not defined"
\`\`\`

**4. Check if it's a React app:**
\`\`\`javascript
// In console:
document.querySelector('[data-reactroot]');
// or
document.querySelector('#root');

// If React app, should find elements
\`\`\`

**5. Refresh the page:**
- Sometimes DevTools doesn't detect React if it loaded before extension
- Refresh page after opening DevTools

**Real-World Scenarios:**

**Scenario 1: Checking competitor's stack:**
\`\`\`
Visit competitor.com
Check DevTools icon:
  Red → Using React in production
  Blue → Using React in development (rare for production sites)
  Gray → Not using React (might be Vue, Angular, vanilla JS)
\`\`\`

**Scenario 2: Debugging "React not detected":**
\`\`\`
Your app shows gray icon but should use React:

1. Check console for errors
2. Verify React script tags loaded
3. Check network tab: react.js loaded?
4. Try: window.React (should not be undefined)
5. Check if using server-side only (no client React)
\`\`\`

**Scenario 3: Client vs Server Rendering:**
\`\`\`tsx
// Server-side rendered HTML only (no client React):
// - Gray icon (no React on client)
// - Fast initial load
// - No interactivity

// Server-side rendered + hydrated:
// - Blue/Red icon (React on client)
// - Fast initial load
// - Full interactivity
\`\`\`

**Icon Reference Chart:**

| Icon | Meaning | When You See It |
|------|---------|-----------------|
| 🔴 Red | Production React | Live websites |
| 🔵 Blue | Development React | Local development, staging |
| ⚫ Gray | No React | Non-React sites, before load |

**Pro Tip:**

Always use development build (blue icon) when developing:
- Better error messages
- Component names preserved
- Helpful warnings
- Easier debugging

Switch to production build (red icon) only for deployment:
- Smaller bundle
- Faster performance
- No development overhead

**Interview Tip:**
Explaining the icon colors shows familiarity with development workflows. Mentioning that gray means "no React detected" and discussing troubleshooting steps demonstrates practical debugging experience.`,
    },
    {
      id: 'q4',
      question:
        'In React DevTools, what can you do by double-clicking a prop or state value?',
      options: [
        "View the component's source code",
        'See the history of how the value changed',
        'Edit the value and see the component re-render with the new value',
        'Copy the value to clipboard',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"Edit the value and see the component re-render with the new value"**.

This is one of the most powerful features of React DevTools—live editing props and state for instant testing without changing code.

**How Live Editing Works:**

**Step-by-Step:**

1. Open DevTools → Components tab
2. Select any component
3. Find a prop or state value in right panel
4. **Double-click** the value
5. Type new value
6. Press **Enter**
7. Component **instantly re-renders** with new value!

**Example:**

\`\`\`tsx
// Your component
function UserProfile({ name, age, isAdmin }) {
  return (
    <div>
      <h1>{name}</h1>
      <p>Age: {age}</p>
      {isAdmin && <button>Admin Panel</button>}
    </div>
  );
}

// Current props in DevTools:
// name: "Alice"
// age: 25
// isAdmin: false

// Double-click "Alice" → Edit to "Bob"
// → Component instantly shows "Bob"

// Double-click 25 → Edit to 99
// → Component instantly shows "Age: 99"

// Double-click false → Edit to true
// → Admin Panel button appears!
\`\`\`

**What You Can Edit:**

**Strings:**
\`\`\`
props.username: "alice"
Double-click → Edit to: "bob123"
Component shows: "bob123"
\`\`\`

**Numbers:**
\`\`\`
props.count: 42
Double-click → Edit to: 999999
Component shows: 999999
\`\`\`

**Booleans:**
\`\`\`
state.isOpen: false
Double-click → Edit to: true
Component shows open state
\`\`\`

**Null/Undefined:**
\`\`\`
props.user: { name: "Alice" }
Double-click → Edit to: null
Component shows null user state (tests error handling!)
\`\`\`

**Arrays:**
\`\`\`
state.items: [1, 2, 3]
Click to expand, edit individual items
Or edit entire array: [1, 2, 3, 4, 5]
\`\`\`

**Objects:**
\`\`\`
props.user: { name: "Alice", age: 25 }
Click to expand, edit individual properties
user.age: 25 → 26
\`\`\`

**Why This is Powerful:**

**1. Test Edge Cases Without Code Changes:**

\`\`\`tsx
function UserList({ users }) {
  return (
    <ul>
      {users.map(user => <li key={user.id}>{user.name}</li>)}
    </ul>
  );
}

// In DevTools:
// users: [{ id: 1, name: "Alice" }, { id: 2, name: "Bob" }]

// Test empty array:
// Edit to: []
// See: Empty state handling (or lack thereof!)

// Test large array:
// Edit to: [1, 2, 3, ..., 1000]
// See: Performance issues?

// Test invalid data:
// Edit to: [{ id: 1 }]  // Missing name
// See: Does component crash?
\`\`\`

**2. Debug Conditional Rendering:**

\`\`\`tsx
function Dashboard({ user }) {
  if (!user) return <p>Loading...</p>;
  if (!user.isActive) return <p>Account inactive</p>;
  if (user.isAdmin) return <AdminDashboard />;
  return <UserDashboard />;
}

// In DevTools:
// user: { isActive: true, isAdmin: false }

// Test admin view:
// Edit isAdmin: false → true
// Instantly see AdminDashboard!

// Test inactive account:
// Edit isActive: true → false
// Instantly see "Account inactive"

// Test loading state:
// Edit user: {...} → null
// Instantly see "Loading..."
\`\`\`

**3. Test Form Validation:**

\`\`\`tsx
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  const isValid = email.includes('@') && password.length >= 8;
  
  return (
    <form>
      <input value={email} onChange={e => setEmail(e.target.value)} />
      <input value={password} onChange={e => setPassword(e.target.value)} />
      <button disabled={!isValid}>Login</button>
    </form>
  );
}

// In DevTools:
// email: ""
// password: ""

// Test valid state:
// Edit email → "test@example.com"
// Edit password → "password123"
// Button becomes enabled!

// Test invalid email:
// Edit email → "notanemail"
// Button stays disabled
\`\`\`

**4. Test Loading and Error States:**

\`\`\`tsx
function DataComponent() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  if (isLoading) return <Spinner />;
  if (error) return <Error message={error} />;
  if (!data) return <Empty />;
  return <Display data={data} />;
}

// In DevTools:
// isLoading: true
// error: null
// data: null

// Test error state:
// Edit error → "Network error"
// See: Error component

// Test loaded state:
// Edit isLoading → false
// Edit data → { name: "Test" }
// See: Display component

// Test empty state:
// Edit data → null
// See: Empty component
\`\`\`

**Why Other Options Are Wrong:**

**"View the component's source code":**
- That's done by clicking the \`< >\` button next to component name
- Not by double-clicking values

**"See the history of how the value changed":**
- DevTools doesn't track value history
- You can use the Profiler to see what changed between commits
- But not a history timeline of one value

**"Copy the value to clipboard":**
- That's done by right-clicking → "Copy value to clipboard"
- Not by double-clicking
- Double-clicking opens edit mode

**Limitations:**

**1. Functions can't be edited:**
\`\`\`
props.onClick: ƒ handleClick()
// Can't double-click to edit function body
// Only see function reference
\`\`\`

**2. Changes are temporary:**
\`\`\`
// Edited values lost on page refresh
// Edited values lost when parent re-renders and passes original prop
// Only lasts for current session
\`\`\`

**3. Can't edit computed values:**
\`\`\`tsx
function Component({ items }) {
  const count = items.length;  // Derived value
  // Can't edit count in DevTools
  // Can only edit items (source of truth)
}
\`\`\`

**Pro Workflow:**

\`\`\`
Development workflow:

1. Write component
2. Run app
3. Open DevTools
4. Test edge cases by editing props/state:
   - Empty values
   - Null/undefined
   - Very long strings
   - Large numbers
   - Negative numbers
   - Empty arrays
   - Large arrays
5. Find bugs
6. Fix in code
7. Repeat
\`\`\`

**Real-World Example:**

\`\`\`tsx
// Building a shopping cart
function Cart({ items, total }) {
  return (
    <div>
      <h2>Cart ({items.length} items)</h2>
      <p>Total: ${total}</p>
      {items.length === 0 && <p>Your cart is empty</p>}
      {items.length > 10 && <p>Wow, that's a lot of items!</p>}
    </div>
  );
}

// In DevTools, test:
// items: [] → See empty message
// items: [{...}] (11 items) → See "lot of items" message
// total: -100 → Does component handle negative totals?
// total: 999999999 → How does UI look with huge numbers?

// Find issues without:
// - Writing tests
// - Modifying code
// - Restarting server
// - Clicking through UI
\`\`\`

**Interview Tip:**
Explaining that you can live-edit props/state in DevTools shows hands-on debugging experience. Describing use cases (testing edge cases, conditional rendering, validation) demonstrates practical problem-solving skills.`,
    },
    {
      id: 'q5',
      question:
        'What should you do if React DevTools Components tab is not showing up even though the site uses React?',
      options: [
        'Reinstall React DevTools',
        'Clear browser cache',
        'Refresh the page (React DevTools needs to be open before page loads)',
        'Update to the latest version of React',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"Refresh the page (React DevTools needs to be open before page loads)"**.

React DevTools needs to inject its hook into the page **before React initializes**. If you open DevTools after React has already loaded, you need to refresh.

**Why This Happens:**

**Normal Flow (DevTools appear):**
\`\`\`
1. Open DevTools (F12)
2. DevTools injects hook into page
3. Navigate to page or refresh
4. Page loads
5. React loads
6. React detects DevTools hook
7. React connects to DevTools
8. Components tab appears ✓
\`\`\`

**Problem Flow (DevTools don't appear):**
\`\`\`
1. Navigate to page
2. Page loads
3. React loads
4. React looks for DevTools hook (not found)
5. React initializes without DevTools
6. Open DevTools (F12)
7. DevTools injects hook
8. But React already initialized!
9. Components tab missing ✗
\`\`\`

**Solution:**

\`\`\`
1. Open DevTools (F12)
2. Refresh page (Cmd/Ctrl + R)
3. Components tab appears! ✓
\`\`\`

**How React DevTools Injection Works:**

When the React DevTools extension loads, it injects a script:

\`\`\`javascript
// DevTools injects this into every page:
window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
  inject: function() { /* ... */ },
  // ... other methods
};

// When React loads, it checks:
if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
  // Connect to DevTools
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__.inject(/* React internals */);
}

// If hook exists: Components tab appears
// If hook doesn't exist: No Components tab
\`\`\`

**Testing This:**

**Experiment 1: Open DevTools First**
\`\`\`
1. Open new tab
2. Open DevTools (F12) FIRST
3. Navigate to React app
4. Components tab appears immediately ✓
\`\`\`

**Experiment 2: Open DevTools After**
\`\`\`
1. Open new tab
2. Navigate to React app
3. Let page fully load
4. Open DevTools (F12)
5. Components tab missing ✗
6. Refresh page
7. Components tab appears ✓
\`\`\`

**When This Commonly Happens:**

**Scenario 1: Bookmarking a React app**
\`\`\`
1. Click bookmark → Page loads
2. Open DevTools to debug
3. No Components tab
4. Solution: Refresh page
\`\`\`

**Scenario 2: Clicking links within app**
\`\`\`
1. On a React app with DevTools closed
2. Click link (client-side navigation)
3. Open DevTools
4. Components tab there (React still connected) ✓

// But if you:
1. Close tab
2. Open new tab
3. Navigate to app (fresh page load)
4. Open DevTools
5. No Components tab
6. Solution: Refresh
\`\`\`

**Scenario 3: Clearing React state during development**
\`\`\`
1. Make code changes
2. Page hot-reloads
3. Sometimes React reinitializes
4. Components tab might disappear
5. Solution: Refresh page
\`\`\`

**Why Other Options Are Wrong (or less effective):**

**"Reinstall React DevTools":**
- This rarely fixes the issue
- The problem is timing, not installation
- Only reinstall if DevTools icon never appears at all

**"Clear browser cache":**
- Cache doesn't affect DevTools injection
- This solves different problems (stale files, old JavaScript)
- Won't help if DevTools just need a refresh

**"Update to the latest version of React":**
- React version doesn't affect whether Components tab shows
- Even React 16, 17, 18 all work with modern DevTools
- Only very old React (< 0.14) has compatibility issues

**Troubleshooting Checklist:**

If Components tab is missing:

\`\`\`
□ Check icon color:
  ├─ Gray → No React detected (wrong site or React didn't load)
  ├─ Red/Blue → React detected
  └─ If Red/Blue but no tab → Try steps below

□ Step 1: Refresh page (fixes 90% of cases)
  Cmd/Ctrl + R

□ Step 2: Hard refresh (clears cache too)
  Cmd/Ctrl + Shift + R

□ Step 3: Check DevTools was installed
  - Chrome: chrome://extensions
  - Look for "React Developer Tools"
  - Ensure "Enabled" is checked

□ Step 4: Check React actually loaded
  - Open Console tab
  - Type: React
  - Should see: Object { ... }
  - If ReferenceError → React didn't load

□ Step 5: Try in incognito mode
  - Rules out extension conflicts
  - If works in incognito → Disable other extensions

□ Step 6: Reinstall DevTools (last resort)
  - Remove extension
  - Restart browser
  - Reinstall from store
\`\`\`

**Best Practice:**

**Always open DevTools before navigating:**
\`\`\`
1. New tab → Open DevTools FIRST (F12)
2. Then navigate to site
3. DevTools always work ✓

// Or keep DevTools open:
1. DevTools→ Settings (F1)
2. Check "Auto-open DevTools for popups"
3. DevTools open automatically ✓
\`\`\`

**Alternative: Standalone React DevTools**

If browser extension has issues, use standalone app:

\`\`\`bash
# Install globally
npm install -g react-devtools

# Run
react-devtools

# Add to your app:
<script src="http://localhost:8097"></script>

# Standalone app connects to your React app
# No browser extension needed
\`\`\`

**When Components Tab Appears But Is Empty:**

Different issue:
\`\`\`
□ Check: Is React actually rendering?
  - View page source
  - Look for <div id="root"> or similar
  - Should have content inside

□ Check: Server-side rendering only?
  - If only server HTML (no client React)
  - No components to show

□ Check: React inside iframe?
  - DevTools might not detect
  - Open iframe's DevTools separately
\`\`\`

**Interview Tip:**
Explaining that DevTools need to be open before React loads shows understanding of the injection mechanism. Mentioning that refreshing usually fixes it demonstrates practical troubleshooting experience. Discussing the timing issue shows attention to technical details.`,
    },
  ],
};
