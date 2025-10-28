export const legacyPatternsAndMigrationQuiz = {
  title: 'Legacy Patterns & Migration Quiz',
  id: 'legacy-patterns-and-migration-quiz',
  sectionId: 'legacy-patterns-and-migration',
  questions: [
    {
      id: 'q1',
      question: 'What is the hook equivalent of componentDidMount?',
      options: [
        'useEffect(() => { }, [])',
        'useEffect(() => { })',
        'useLayoutEffect(() => { })',
        'useMemo(() => { }, [])',
      ],
      correctAnswer: 0,
      explanation: `The correct answer is **"useEffect(() => { }, [])"**.

An empty dependency array tells useEffect to run only once after the initial render—equivalent to componentDidMount.

**componentDidMount (Class):**
\`\`\`tsx
class Component extends React.Component {
  componentDidMount() {
    // Runs once after component mounts
    console.log('Component mounted');
    fetchData();
  }
  
  render() {
    return <div>Content</div>;
  }
}
\`\`\`

**useEffect with Empty Deps (Function):**
\`\`\`tsx
function Component() {
  useEffect(() => {
    // Runs once after component mounts
    console.log('Component mounted');
    fetchData();
  }, []);  // ← Empty array = run once
  
  return <div>Content</div>;
}
\`\`\`

**How the Empty Dependency Array Works:**

\`\`\`tsx
// No deps: Runs after EVERY render
useEffect(() => {
  console.log('Runs every render');
});

// Empty deps []: Runs ONCE after first render
useEffect(() => {
  console.log('Runs once (mount)');
}, []);

// Deps [value]: Runs when value changes
useEffect(() => {
  console.log('Runs when value changes');
}, [value]);
\`\`\`

**Complete Lifecycle Mapping:**

\`\`\`tsx
// Class component
class Component extends React.Component {
  componentDidMount() {
    // Setup: Runs once
    const timer = setInterval(() => {
      console.log('Tick');
    }, 1000);
    
    this.timer = timer;
  }
  
  componentWillUnmount() {
    // Cleanup: Runs once before unmount
    clearInterval(this.timer);
  }
  
  render() {
    return <div>Content</div>;
  }
}

// Function component
function Component() {
  useEffect(() => {
    // Setup: Runs once (componentDidMount)
    const timer = setInterval(() => {
      console.log('Tick');
    }, 1000);
    
    // Cleanup: Runs before unmount (componentWillUnmount)
    return () => {
      clearInterval(timer);
    };
  }, []);  // Empty deps = mount + unmount only
  
  return <div>Content</div>;
}
\`\`\`

**Real-World Example:**

\`\`\`tsx
// Fetch data on mount
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Runs once on mount
    setLoading(true);
    
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      });
  }, []);  // Empty deps = fetch once
  
  if (loading) return <div>Loading...</div>;
  return <div>{user?.name}</div>;
}
\`\`\`

**Why Other Options Are Wrong:**

**"useEffect(() => { })" (no deps array):**
\`\`\`tsx
// This runs after EVERY render, not just mount
useEffect(() => {
  console.log('Runs every render');  // Runs constantly!
});

// Equivalent to:
componentDidMount() { console.log('Mount'); }
componentDidUpdate() { console.log('Update'); }

// Not the same as just componentDidMount
\`\`\`

**"useLayoutEffect(() => { })" (no deps array):**
\`\`\`tsx
// useLayoutEffect fires synchronously before browser paint
// Empty deps would make it run once, but:
// - It's for layout measurements
// - Blocks visual updates
// - Not the standard equivalent

// Use useLayoutEffect only when you need to measure/modify DOM
// before browser paints
\`\`\`

**"useMemo(() => { }, [])":**
\`\`\`tsx
// useMemo is for memoizing values, not running effects
const cachedValue = useMemo(() => expensiveComputation(), []);

// Wrong usage:
useMemo(() => {
  fetchData();  // ❌ Side effect in useMemo!
}, []);

// useMemo returns a value, useEffect runs side effects
\`\`\`

**Key Differences:**

| Pattern | When It Runs | Purpose |
|---------|--------------|---------|
| \`useEffect(() => {}, [])\` | Once after mount | Side effects on mount (componentDidMount) |
| \`useEffect(() => {})\` | After every render | Side effects on mount + updates |
| \`useLayoutEffect(() => {}, [])\` | Once after mount (synchronous) | DOM measurements/mutations |
| \`useMemo(() => value, [])\` | Once on mount | Memoize expensive computation |

**Common Mistake:**

\`\`\`tsx
// ❌ WRONG: Missing empty array
function Component() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('/api/data')
      .then(r => r.json())
      .then(setData);
  });  // ← Missing []!
  
  return <div>{data?.name}</div>;
}

// Problem: Infinite loop!
// 1. Component renders
// 2. useEffect runs (no deps = runs every render)
// 3. setData() called
// 4. Component re-renders
// 5. useEffect runs again...
// → Infinite loop! 💥
\`\`\`

**Correct:**
\`\`\`tsx
// ✅ CORRECT: With empty array
useEffect(() => {
  fetch('/api/data')
    .then(r => r.json())
    .then(setData);
}, []);  // ← Runs once!
\`\`\`

**Interview Tip:** Explaining that the empty dependency array \`[]\` is what makes useEffect equivalent to componentDidMount shows understanding of how hooks dependencies work. Mentioning the cleanup function for componentWillUnmount demonstrates complete knowledge of the lifecycle mapping.`,
    },
    {
      id: 'q2',
      question:
        'In a class component, why do you need to bind event handlers in the constructor (or use arrow functions)?',
      options: [
        'To improve performance',
        'To allow the handler to access component props',
        'To ensure "this" refers to the component instance',
        'To prevent memory leaks',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"To ensure 'this' refers to the component instance"**.

In JavaScript classes, methods lose their \`this\` context when passed as callbacks unless explicitly bound.

**The Problem:**

\`\`\`tsx
class Counter extends React.Component {
  state = { count: 0 };
  
  increment() {
    // ❌ this is undefined here!
    this.setState({ count: this.state.count + 1 });
    // TypeError: Cannot read property 'setState' of undefined
  }
  
  render() {
    return <button onClick={this.increment}>Count: {this.state.count}</button>;
  }
}
\`\`\`

**Why \`this\` is undefined:**

When you pass a method as a callback, JavaScript doesn't automatically bind \`this\`:

\`\`\`javascript
class MyClass {
  name = 'MyClass';
  
  sayName() {
    console.log(this.name);
  }
}

const instance = new MyClass();

// Direct call: Works
instance.sayName();  // "MyClass"

// Callback: Loses this
const callback = instance.sayName;
callback();  // undefined (or error in strict mode)

// Why? When called as callback, there's no object before the dot
// So JavaScript doesn't know what \`this\` should be
\`\`\`

**Solution 1: Bind in Constructor**

\`\`\`tsx
class Counter extends React.Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
    
    // Bind in constructor: Creates new bound function once
    this.increment = this.increment.bind(this);
  }
  
  increment() {
    // ✅ Now \`this\` is bound!
    this.setState({ count: this.state.count + 1 });
  }
  
  render() {
    // Pass bound function
    return <button onClick={this.increment}>Count: {this.state.count}</button>;
  }
}
\`\`\`

**What \`.bind()\` does:**
\`\`\`javascript
// Creates a new function with \`this\` permanently bound
this.increment = this.increment.bind(this);

// Equivalent to:
this.increment = () => {
  // Original increment code with \`this\` fixed
};
\`\`\`

**Solution 2: Arrow Function (Class Field)**

\`\`\`tsx
class Counter extends React.Component {
  state = { count: 0 };
  
  // Arrow function: Automatically binds \`this\`
  increment = () => {
    // ✅ \`this\` is bound!
    this.setState({ count: this.state.count + 1 });
  }
  
  render() {
    return <button onClick={this.increment}>Count: {this.state.count}</button>;
  }
}
\`\`\`

**Why arrow functions work:**

Arrow functions don't have their own \`this\`—they inherit it from the surrounding scope:

\`\`\`javascript
class MyClass {
  name = 'MyClass';
  
  // Regular method: Has its own \`this\`
  regularMethod() {
    console.log(this.name);  // Depends on how it's called
  }
  
  // Arrow function: Inherits \`this\` from class
  arrowMethod = () => {
    console.log(this.name);  // Always refers to instance
  }
}

const instance = new MyClass();

// Both work when called directly
instance.regularMethod();  // "MyClass"
instance.arrowMethod();    // "MyClass"

// Only arrow works as callback
const callback1 = instance.regularMethod;
callback1();  // undefined ❌

const callback2 = instance.arrowMethod;
callback2();  // "MyClass" ✅
\`\`\`

**Solution 3: Arrow Function in JSX (Not Recommended)**

\`\`\`tsx
class Counter extends React.Component {
  state = { count: 0 };
  
  increment() {
    this.setState({ count: this.state.count + 1 });
  }
  
  render() {
    // ⚠️ Creates new function every render
    return <button onClick={() => this.increment()}>Count: {this.state.count}</button>;
  }
}
\`\`\`

**Problem:** Creates a new function on every render—bad for performance and child re-renders.

**Why Other Options Are Wrong:**

**"To improve performance":**
- Binding is about correctness, not performance
- Actually, binding in constructor is slightly better for performance (one function vs creating new ones)
- But that's a side effect, not the main reason

**"To allow the handler to access component props":**
- Handlers can access props via \`this.props\` regardless of binding
- The issue is accessing \`this\` itself, not props specifically

\`\`\`tsx
class Component extends React.Component {
  handleClick() {
    // The problem isn't accessing props
    // The problem is \`this\` is undefined
    console.log(this);       // undefined ❌
    console.log(this.props); // TypeError: Cannot read property 'props' of undefined
  }
  
  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}
\`\`\`

**"To prevent memory leaks":**
- Binding doesn't prevent memory leaks
- Memory leaks in React usually come from:
  - Not cleaning up timers/intervals
  - Not removing event listeners
  - Not canceling async operations
- Binding is unrelated to these

**Function Components Don't Have This Problem:**

\`\`\`tsx
// Function components: No \`this\`, no binding needed!
function Counter() {
  const [count, setCount] = useState(0);
  
  function increment() {
    setCount(count + 1);  // Just works!
  }
  
  return <button onClick={increment}>Count: {count}</button>;
}

// Or inline:
function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}
\`\`\`

**Comparison of Binding Methods:**

| Method | Pros | Cons |
|--------|------|------|
| Bind in constructor | ✅ One function created<br>✅ Good performance | ❌ Verbose<br>❌ Easy to forget |
| Arrow class field | ✅ Automatic binding<br>✅ Clean syntax | ❌ Slightly larger bundle (not in prototype) |
| Arrow in JSX | ✅ Simple | ❌ New function every render<br>❌ Breaks PureComponent<br>❌ Performance issues |

**Real-World Example:**

\`\`\`tsx
// Legacy codebase: Class with multiple handlers
class TodoList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      todos: [],
      newTodoText: ''
    };
    
    // Must bind each handler
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleDelete = this.handleDelete.bind(this);
    this.handleToggle = this.handleToggle.bind(this);
  }
  
  handleSubmit(e) {
    e.preventDefault();
    // \`this\` works because bound
    this.setState({ /* ... */ });
  }
  
  handleChange(e) {
    this.setState({ newTodoText: e.target.value });
  }
  
  handleDelete(id) {
    this.setState({ /* ... */ });
  }
  
  handleToggle(id) {
    this.setState({ /* ... */ });
  }
  
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <input value={this.state.newTodoText} onChange={this.handleChange} />
        <button type="submit">Add</button>
        {/* ... */}
      </form>
    );
  }
}

// Modern: Function component (no binding needed!)
function TodoList() {
  const [todos, setTodos] = useState([]);
  const [newTodoText, setNewTodoText] = useState('');
  
  function handleSubmit(e) {
    e.preventDefault();
    // Just works! No \`this\`
    setTodos([...]);
  }
  
  function handleChange(e) {
    setNewTodoText(e.target.value);
  }
  
  function handleDelete(id) {
    setTodos([...]);
  }
  
  function handleToggle(id) {
    setTodos([...]);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input value={newTodoText} onChange={handleChange} />
      <button type="submit">Add</button>
      {/* ... */}
    </form>
  );
}
\`\`\`

**Interview Tip:** Explaining that binding fixes the \`this\` context shows understanding of JavaScript fundamentals. Mentioning that function components don't have this issue demonstrates knowledge of why hooks are simpler than class components.`,
    },
    {
      id: 'q3',
      question:
        'What is the main advantage of custom hooks over Higher-Order Components (HOCs)?',
      options: [
        'Custom hooks are faster to execute',
        "Custom hooks don't add extra components to the React tree",
        'Custom hooks can only be used with function components',
        'Custom hooks automatically memoize their return values',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"Custom hooks don't add extra components to the React tree"**.

This is the primary advantage—hooks provide code reuse without adding wrapper components, solving the "wrapper hell" problem of HOCs.

**Higher-Order Components (HOCs): Add Wrappers**

\`\`\`tsx
// HOC: Returns a new component
function withAuth(Component) {
  return function AuthWrapper(props) {
    const user = useAuth();
    if (!user) return <LoginPrompt />;
    return <Component {...props} user={user} />;
  };
}

function withTheme(Component) {
  return function ThemeWrapper(props) {
    const theme = useTheme();
    return <Component {...props} theme={theme} />;
  };
}

// Usage
const Dashboard = withAuth(withTheme(DashboardComponent));

// React DevTools shows:
<AuthWrapper>
  <ThemeWrapper>
    <DashboardComponent />
  </ThemeWrapper>
</AuthWrapper>

// Three components in the tree!
\`\`\`

**Custom Hooks: No Wrappers**

\`\`\`tsx
// Custom hooks: Just functions
function useAuth() {
  const user = useContext(UserContext);
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!user) navigate('/login');
  }, [user]);
  
  return user;
}

function useTheme() {
  return useContext(ThemeContext);
}

// Usage
function Dashboard() {
  const user = useAuth();
  const theme = useTheme();
  
  return <div>Dashboard content</div>;
}

// React DevTools shows:
<Dashboard />

// Just ONE component in the tree!
\`\`\`

**Why This Matters:**

**1. Cleaner Component Tree:**
\`\`\`tsx
// HOCs: Nested wrappers
<WithAuth>
  <WithTheme>
    <WithRouter>
      <WithData>
        <WithPermissions>
          <MyComponent />
        </WithPermissions>
      </WithData>
    </WithRouter>
  </WithTheme>
</WithAuth>

// Hard to debug in React DevTools
// Hard to understand component structure
// "Wrapper hell"

// Hooks: Flat
<MyComponent />

// Easy to debug
// Clear structure
// No wrappers
\`\`\`

**2. Better Performance:**
\`\`\`tsx
// HOCs: Multiple components render
// When props change:
// 1. AuthWrapper renders
// 2. ThemeWrapper renders
// 3. MyComponent renders

// Hooks: One component renders
// When props change:
// 1. MyComponent renders
// (hooks execute inside component)
\`\`\`

**3. Simpler Debugging:**
\`\`\`tsx
// Error stack with HOCs:
at MyComponent (Component.tsx:45)
  at WithData (withData.tsx:12)
    at WithRouter (withRouter.tsx:8)
      at WithTheme (withTheme.tsx:5)
        at WithAuth (withAuth.tsx:3)

// Hard to find your actual component

// Error stack with hooks:
at MyComponent (Component.tsx:45)

// Clean and simple
\`\`\`

**4. No Prop Name Collisions:**
\`\`\`tsx
// HOCs: Might have collisions
const Component = withAuth(withRouter(MyComponent));

// Both might inject a "loading" prop:
// withAuth: { user, loading }
// withRouter: { location, loading }

// Which "loading" wins? Depends on wrapping order
// Confusing!

// Hooks: Explicit naming
function MyComponent() {
  const { user, loading: authLoading } = useAuth();
  const { location, loading: routerLoading } = useRouter();
  
  // No collisions, clear names
}
\`\`\`

**5. Easier Composition:**
\`\`\`tsx
// HOCs: Nested composition
export default withAuth(
  withTheme(
    withRouter(
      withData(
        MyComponent
      )
    )
  )
);

// Need to compose in specific order
// Hard to read

// Hooks: Natural composition
function MyComponent() {
  const user = useAuth();
  const theme = useTheme();
  const location = useLocation();
  const data = useData();
  
  // Clear and linear
  // Easy to add/remove
}
\`\`\`

**Why Other Options Are Wrong:**

**"Custom hooks are faster to execute":**
- Hooks and HOCs have similar performance
- The performance benefit is from fewer components in tree (not execution speed)
- Both are fast enough for most use cases

**"Custom hooks can only be used with function components":**
- This is true but not an advantage
- It's a limitation if anything
- HOCs work with both class and function components

**"Custom hooks automatically memoize their return values":**
- Hooks don't automatically memoize
- You must explicitly use \`useMemo\` or \`useCallback\`
- HOCs don't automatically memoize either

\`\`\`tsx
// ❌ Hooks don't auto-memoize
function useExpensiveValue() {
  const value = expensiveComputation();
  return value;  // Computed every render!
}

// ✅ Must explicitly memoize
function useExpensiveValue() {
  const value = useMemo(() => expensiveComputation(), []);
  return value;  // Computed once
}
\`\`\`

**Real-World Comparison:**

**HOC Pattern (Legacy):**
\`\`\`tsx
// File: withUserData.js
function withUserData(Component) {
  return function WithUserDataWrapper(props) {
    const [user, setUser] = useState(null);
    
    useEffect(() => {
      fetchUser(props.userId).then(setUser);
    }, [props.userId]);
    
    return <Component {...props} user={user} />;
  };
}

// File: Dashboard.js
function Dashboard({ user }) {
  return <div>Welcome {user?.name}</div>;
}

export default withUserData(Dashboard);

// React tree:
<WithUserDataWrapper>
  <Dashboard />
</WithUserDataWrapper>
\`\`\`

**Hook Pattern (Modern):**
\`\`\`tsx
// File: useUserData.js
function useUserData(userId) {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]);
  
  return user;
}

// File: Dashboard.js
function Dashboard({ userId }) {
  const user = useUserData(userId);
  return <div>Welcome {user?.name}</div>;
}

export default Dashboard;

// React tree:
<Dashboard />
\`\`\`

**Measuring the Difference:**

\`\`\`tsx
// HOC approach with 5 HOCs:
// React DevTools Profiler shows:
// - 6 components rendered (5 wrappers + 1 actual)
// - Render time: 8ms

// Hook approach with 5 hooks:
// React DevTools Profiler shows:
// - 1 component rendered
// - Render time: 5ms

// Hooks are faster because fewer components to render
\`\`\`

**When HOCs Might Still Be Needed:**

1. **Wrapping third-party components you can't modify:**
\`\`\`tsx
import ThirdPartyComponent from 'some-library';

// Can't add hooks to ThirdPartyComponent (don't have source)
const WithDataComponent = withData(ThirdPartyComponent);
\`\`\`

2. **Legacy codebases with many HOCs:**
- Already using HOC patterns everywhere
- Incremental migration to hooks
- Consistency during transition

**Interview Tip:** Explaining that hooks avoid "wrapper hell" by not adding components to the tree shows understanding of React's component model. Mentioning the performance and debugging benefits demonstrates practical experience with both patterns.`,
    },
    {
      id: 'q4',
      question:
        'Which lifecycle method has NO direct equivalent in function components with hooks?',
      options: [
        'componentDidMount',
        'componentWillUnmount',
        'shouldComponentUpdate',
        'componentDidCatch',
      ],
      correctAnswer: 3,
      explanation: `The correct answer is **"componentDidCatch"**.

Error boundaries (componentDidCatch and getDerivedStateFromError) currently only work in class components. There's no hook equivalent yet.

**Error Boundaries Must Be Classes:**

\`\`\`tsx
// ✅ WORKS: Class component error boundary
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };
  
  static getDerivedStateFromError(error) {
    // Update state so next render shows fallback UI
    return { hasError: true, error };
  }
  
  componentDidCatch(error, errorInfo) {
    // Log error to error reporting service
    logErrorToService(error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong: {this.state.error.message}</h1>;
    }
    
    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <MyComponent />
</ErrorBoundary>

// ❌ DOESN'T WORK: Function component error boundary
function ErrorBoundary({ children }) {
  const [hasError, setHasError] = useState(false);
  
  // No hook for catching errors!
  // useEffect doesn't catch errors
  // try/catch doesn't catch React errors
  
  return hasError ? <h1>Error!</h1> : children;
}
\`\`\`

**Why There's No Hook for Error Boundaries:**

1. **Error boundaries catch errors in rendering, not in event handlers:**
\`\`\`tsx
function MyComponent() {
  // ❌ Error boundary catches this:
  if (Math.random() > 0.5) {
    throw new Error('Render error');
  }
  
  // ❌ Error boundary does NOT catch this:
  function handleClick() {
    throw new Error('Click error');
  }
  
  return <button onClick={handleClick}>Click</button>;
}
\`\`\`

2. **Hooks can't catch their own errors:**
- If a hook could catch errors, how would it catch errors in itself?
- Chicken and egg problem

3. **React team hasn't finalized the API:**
- Error boundaries are important for production
- React team wants to get the API right
- No rush to ship an imperfect hook

**Lifecycle Methods That DO Have Hook Equivalents:**

**componentDidMount:**
\`\`\`tsx
// Class
componentDidMount() {
  fetchData();
}

// Hook
useEffect(() => {
  fetchData();
}, []);
\`\`\`

**componentWillUnmount:**
\`\`\`tsx
// Class
componentWillUnmount() {
  clearInterval(this.timer);
}

// Hook
useEffect(() => {
  const timer = setInterval(() => {}, 1000);
  
  return () => clearInterval(timer);  // Cleanup
}, []);
\`\`\`

**shouldComponentUpdate:**
\`\`\`tsx
// Class
shouldComponentUpdate(nextProps, nextState) {
  return nextProps.userId !== this.props.userId;
}

// Hook (React.memo)
const MyComponent = React.memo(
  function MyComponent({ userId }) {
    return <div>User {userId}</div>;
  },
  (prevProps, nextProps) => {
    // Return true if props are equal (don't re-render)
    // Return false if props are different (re-render)
    return prevProps.userId === nextProps.userId;
  }
);
\`\`\`

**Working with Error Boundaries in Function Components:**

You can't create an error boundary with hooks, but you can USE them:

\`\`\`tsx
// ErrorBoundary.tsx (Class component)
class ErrorBoundary extends React.Component {
  state = { hasError: false };
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Error caught:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong</h1>;
    }
    
    return this.props.children;
  }
}

// App.tsx (Function component)
function App() {
  return (
    <ErrorBoundary>
      <Header />
      <Main />
      <Footer />
    </ErrorBoundary>
  );
}
\`\`\`

**Using react-error-boundary Library:**

Third-party library provides a ready-made error boundary:

\`\`\`tsx
import { ErrorBoundary } from 'react-error-boundary';

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div role="alert">
      <p>Something went wrong:</p>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={() => {
        // Reset app state
      }}
      onError={(error, errorInfo) => {
        // Log to error service
        logErrorToService(error, errorInfo);
      }}
    >
      <MyComponent />
    </ErrorBoundary>
  );
}
\`\`\`

**Why Error Boundaries Are Still Needed:**

Even with hooks, you need error boundaries for production apps:

\`\`\`tsx
function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={
            <ErrorBoundary>
              <Home />
            </ErrorBoundary>
          } />
          <Route path="/dashboard" element={
            <ErrorBoundary>
              <Dashboard />
            </ErrorBoundary>
          } />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

// Benefits:
// 1. App doesn't crash if one component errors
// 2. User sees helpful error message
// 3. Errors logged to service (Sentry, etc.)
// 4. User can retry
\`\`\`

**Catching Errors in Hooks:**

For errors in event handlers or async code, use try/catch:

\`\`\`tsx
function MyComponent() {
  const [error, setError] = useState(null);
  
  async function handleClick() {
    try {
      await fetchData();
    } catch (err) {
      setError(err.message);  // Manually handle error
    }
  }
  
  if (error) {
    return <div>Error: {error}</div>;
  }
  
  return <button onClick={handleClick}>Fetch</button>;
}
\`\`\`

**Future of Error Boundaries:**

React team is working on hooks for error boundaries:

\`\`\`tsx
// Proposed API (not yet available):
function MyComponent() {
  const [error, resetError] = useErrorBoundary();
  
  if (error) {
    return (
      <div>
        <p>Error: {error.message}</p>
        <button onClick={resetError}>Try again</button>
      </div>
    );
  }
  
  return <div>Content</div>;
}
\`\`\`

**Current Workaround:**

Keep one class component for error boundary:

\`\`\`tsx
// ErrorBoundary.tsx - Only class in codebase!
class ErrorBoundary extends React.Component {
  // ... error boundary code
}

// All other components can be functions
function App() { /* ... */ }
function Header() { /* ... */ }
function Footer() { /* ... */ }
\`\`\`

**Interview Tip:** Explaining that error boundaries (componentDidCatch) have no hook equivalent shows knowledge of current React limitations. Mentioning react-error-boundary library or keeping one class component demonstrates practical problem-solving. Discussing that hooks can't catch their own errors shows understanding of the technical challenge.`,
    },
    {
      id: 'q5',
      question:
        'When migrating from PropTypes to TypeScript, what is the equivalent of PropTypes.func.isRequired?',
      options: [
        '() => void',
        'Function',
        '(...args: any[]) => any',
        'All of the above are valid',
      ],
      correctAnswer: 3,
      explanation: `The correct answer is **"All of the above are valid"**.

TypeScript offers multiple ways to type functions, each with different levels of specificity. All are valid equivalents to \`PropTypes.func.isRequired\`.

**PropTypes (Legacy):**
\`\`\`tsx
import PropTypes from 'prop-types';

function Button({ onClick, label }) {
  return <button onClick={onClick}>{label}</button>;
}

Button.propTypes = {
  onClick: PropTypes.func.isRequired,  // Just says "must be a function"
  label: PropTypes.string.isRequired
};
\`\`\`

**TypeScript Equivalents:**

**Option 1: \`() => void\` - Specific, No Args**
\`\`\`tsx
interface ButtonProps {
  onClick: () => void;  // Function that takes no args, returns nothing
  label: string;
}

function Button({ onClick, label }: ButtonProps) {
  return <button onClick={onClick}>{label}</button>;
}

// Usage:
<Button onClick={() => console.log('Clicked')} label="Click me" />
\`\`\`

**Benefits:**
- ✅ Most specific
- ✅ TypeScript enforces no parameters
- ✅ Best type safety

**Drawbacks:**
- ❌ Too strict if function might take parameters

\`\`\`tsx
// ✅ This works:
<Button onClick={() => alert('Hi')} label="Click" />

// ❌ This doesn't work:
<Button onClick={(e) => console.log(e)} label="Click" />
// Error: Type '(e: any) => void' is not assignable to type '() => void'

// ❌ This also doesn't work:
function handleClick(e: React.MouseEvent) {
  console.log(e);
}
<Button onClick={handleClick} label="Click" />
// Error: Types don't match
\`\`\`

**Option 2: \`Function\` - Generic**
\`\`\`tsx
interface ButtonProps {
  onClick: Function;  // Any function
  label: string;
}

function Button({ onClick, label }: ButtonProps) {
  return <button onClick={() => onClick()}>{ label}</button>;
}

// Usage: Anything works
<Button onClick={() => {}} label="Click" />
<Button onClick={(a, b, c) => {}} label="Click" />
<Button onClick={console.log} label="Click" />
\`\`\`

**Benefits:**
- ✅ Very flexible
- ✅ Accepts any function

**Drawbacks:**
- ❌ No type safety (any function accepted)
- ❌ TypeScript can't check if you're calling it correctly
- ❌ ESLint warns: "Don't use Function type"

**Option 3: \`(...args: any[]) => any\` - Flexible with Structure**
\`\`\`tsx
interface ButtonProps {
  onClick: (...args: any[]) => any;  // Function with any args/return
  label: string;
}

function Button({ onClick, label }: ButtonProps) {
  return <button onClick={onClick}>{label}</button>;
}

// Usage: Works with any function signature
<Button onClick={() => {}} label="Click" />
<Button onClick={(e) => console.log(e)} label="Click" />
<Button onClick={(a, b) => a + b} label="Click" />
\`\`\`

**Benefits:**
- ✅ Flexible (accepts any function)
- ✅ Still maintains function structure
- ✅ Better than \`Function\` type

**Drawbacks:**
- ❌ No specific type safety
- ❌ \`any\` loses type information

**Best Practice: Be Specific**

For event handlers, use specific React types:

\`\`\`tsx
interface ButtonProps {
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  label: string;
}

function Button({ onClick, label }: ButtonProps) {
  return <button onClick={onClick}>{label}</button>;
}

// Now TypeScript knows exactly what this is:
<Button 
  onClick={(e) => {
    console.log(e.clientX, e.clientY);  // TypeScript knows these exist
  }} 
  label="Click" 
/>
\`\`\`

**Common React Event Types:**

\`\`\`tsx
interface FormProps {
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onFocus: (event: React.FocusEvent<HTMLInputElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => void;
}
\`\`\`

**Migrating PropTypes to TypeScript:**

\`\`\`tsx
// Before: PropTypes
import PropTypes from 'prop-types';

function UserCard({ user, onEdit, onDelete }) {
  return (
    <div>
      <h3>{user.name}</h3>
      <button onClick={() => onEdit(user.id)}>Edit</button>
      <button onClick={() => onDelete(user.id)}>Delete</button>
    </div>
  );
}

UserCard.propTypes = {
  user: PropTypes.shape({
    id: PropTypes.number.isRequired,
    name: PropTypes.string.isRequired,
    email: PropTypes.string.isRequired
  }).isRequired,
  onEdit: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired
};

// After: TypeScript (Basic)
interface User {
  id: number;
  name: string;
  email: string;
}

interface UserCardProps {
  user: User;
  onEdit: Function;
  onDelete: Function;
}

function UserCard({ user, onEdit, onDelete }: UserCardProps) {
  return (
    <div>
      <h3>{user.name}</h3>
      <button onClick={() => onEdit(user.id)}>Edit</button>
      <button onClick={() => onDelete(user.id)}>Delete</button>
    </div>
  );
}

// After: TypeScript (Better - Specific)
interface User {
  id: number;
  name: string;
  email: string;
}

interface UserCardProps {
  user: User;
  onEdit: (userId: number) => void;    // Specific!
  onDelete: (userId: number) => void;  // Specific!
}

function UserCard({ user, onEdit, onDelete }: UserCardProps) {
  return (
    <div>
      <h3>{user.name}</h3>
      <button onClick={() => onEdit(user.id)}>Edit</button>
      <button onClick={() => onDelete(user.id)}>Delete</button>
    </div>
  );
}

// Now TypeScript catches errors:
<UserCard 
  user={user}
  onEdit={(id) => editUser(id)}  // ✓ Correct
  onDelete={(id, extra) => deleteUser(id)}  // ✗ Error: Too many params!
/>
\`\`\`

**Advantages of TypeScript over PropTypes:**

| Aspect | PropTypes | TypeScript |
|--------|-----------|------------|
| Checking | Runtime | Compile-time |
| Performance | Overhead in production | No runtime cost |
| IDE Support | None | Autocomplete, IntelliSense |
| Refactoring | Manual updates | Auto-updates |
| Complexity | Limited types | Rich type system |
| Documentation | Separate | Built-in (types are docs) |

**Example: TypeScript Catches More:**

\`\`\`tsx
// PropTypes: Can't enforce this
Component.propTypes = {
  onSave: PropTypes.func.isRequired
};

// Usage: PropTypes won't catch this at compile time
<Component onSave={(name, age) => save(name, age)} />

// TypeScript: Enforces exact signature
interface Props {
  onSave: (data: { name: string; age: number }) => Promise<void>;
}

// Usage: TypeScript catches at compile time
<Component onSave={(name, age) => save(name, age)} />
// Error: Type '(name: any, age: any) => void' is not assignable to 
//        type '(data: { name: string; age: number }) => Promise<void>'

// Correct:
<Component onSave={(data) => save(data.name, data.age)} />
\`\`\`

**Complete Migration Example:**

\`\`\`tsx
// PropTypes
import PropTypes from 'prop-types';

function TodoList({ todos, onToggle, onDelete, filter }) {
  return (
    <ul>
      {todos
        .filter(t => filter === 'all' || t.done === (filter === 'done'))
        .map(todo => (
          <li key={todo.id}>
            <input 
              type="checkbox" 
              checked={todo.done}
              onChange={() => onToggle(todo.id)}
            />
            {todo.text}
            <button onClick={() => onDelete(todo.id)}>Delete</button>
          </li>
        ))}
    </ul>
  );
}

TodoList.propTypes = {
  todos: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.number.isRequired,
      text: PropTypes.string.isRequired,
      done: PropTypes.bool.isRequired
    })
  ).isRequired,
  onToggle: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
  filter: PropTypes.oneOf(['all', 'done', 'undone']).isRequired
};

// TypeScript
interface Todo {
  id: number;
  text: string;
  done: boolean;
}

type Filter = 'all' | 'done' | 'undone';

interface TodoListProps {
  todos: Todo[];
  onToggle: (todoId: number) => void;
  onDelete: (todoId: number) => void;
  filter: Filter;
}

function TodoList({ todos, onToggle, onDelete, filter }: TodoListProps) {
  return (
    <ul>
      {todos
        .filter(t => filter === 'all' || t.done === (filter === 'done'))
        .map(todo => (
          <li key={todo.id}>
            <input 
              type="checkbox" 
              checked={todo.done}
              onChange={() => onToggle(todo.id)}
            />
            {todo.text}
            <button onClick={() => onDelete(todo.id)}>Delete</button>
          </li>
        ))}
    </ul>
  );
}
\`\`\`

**Interview Tip:** Explaining that all three TypeScript function types are valid but discussing which is most appropriate shows understanding of TypeScript's flexibility and best practices. Mentioning specific React event types demonstrates practical TypeScript + React experience.`,
    },
  ],
};
