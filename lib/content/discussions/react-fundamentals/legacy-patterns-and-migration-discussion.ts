export const legacyPatternsAndMigrationDiscussion = {
  title: 'Legacy Patterns & Migration Discussion Questions',
  id: 'legacy-patterns-and-migration-discussion',
  sectionId: 'legacy-patterns-and-migration',
  questions: [
    {
      id: 'q1',
      question:
        "Explain how you would systematically migrate a large React codebase from class components to function components with hooks. What would be your migration strategy, what tools would you use, and how would you ensure the migration doesn't introduce bugs?",
      answer: `Migrating a large codebase requires a careful, incremental approach. Here's a comprehensive strategy:

**Phase 1: Assessment & Planning (Week 1)**

**1. Analyze the Codebase:**
\`\`\`bash
# Count class components
find src -name "*.js" -o -name "*.tsx" | xargs grep -l "extends Component" | wc -l

# Find HOCs
find src -name "*.js" -o -name "*.tsx" | xargs grep -l "return class extends"

# Find render props
find src -name "*.js" -o -name "*.tsx" | xargs grep -l "this.props.children("

# Identify PropTypes usage
find src -name "*.js" | xargs grep -l "PropTypes"
\`\`\`

**2. Categorize Components:**
\`\`\`
Total: 250 components
├─ Class components: 180
│  ├─ Simple (just state/props): 120
│  ├─ Lifecycle methods: 45
│  └─ Complex (multiple lifecycles): 15
├─ HOCs: 12
├─ Render props: 8
└─ Function components (already modern): 70
\`\`\`

**3. Create Migration Priorities:**
\`\`\`
Priority 1: Leaf components (no children) - 80 components
Priority 2: Simple components (no complex lifecycle) - 60 components
Priority 3: Components with dependencies - 30 components
Priority 4: Core/complex components - 10 components
\`\`\`

**Phase 2: Setup & Preparation (Week 2)**

**1. Add TypeScript (if not already):**
\`\`\`bash
npm install --save-dev typescript @types/react @types/react-dom

# Create tsconfig.json
npx tsc --init
\`\`\`

**2. Setup Testing Infrastructure:**
\`\`\`bash
npm install --save-dev @testing-library/react @testing-library/jest-dom

# Ensure all components have tests before migrating
\`\`\`

**3. Install Migration Tools:**
\`\`\`bash
# React codemod for automated conversions
npm install -g jscodeshift
git clone https://github.com/reactjs/react-codeshift

# ESLint rules to catch issues
npm install --save-dev eslint-plugin-react-hooks
\`\`\`

**4. Configure ESLint:**
\`\`\`javascript
// .eslintrc.js
module.exports = {
  plugins: ['react-hooks'],
  rules: {
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn'
  }
};
\`\`\`

**Phase 3: Incremental Migration (Weeks 3-12)**

**Week 3-4: Migrate Leaf Components**

Start with components that have no children and simple logic:

\`\`\`tsx
// Before: Simple class component
class Button extends React.Component {
  render() {
    return (
      <button onClick={this.props.onClick}>
        {this.props.label}
      </button>
    );
  }
}

// After: Function component (codemod can handle this!)
function Button({ onClick, label }) {
  return (
    <button onClick={onClick}>
      {label}
    </button>
  );
}
\`\`\`

**Use codemod for simple cases:**
\`\`\`bash
# Automated conversion
jscodeshift -t react-codemod/transforms/class-to-function.js src/components/Button.tsx

# Review changes
git diff

# Test
npm test
\`\`\`

**Week 5-7: Migrate Components with State**

\`\`\`tsx
// Before: Class with state
class Counter extends React.Component {
  state = { count: 0 };
  
  increment = () => {
    this.setState({ count: this.state.count + 1 });
  }
  
  render() {
    return (
      <div>
        <p>{this.state.count}</p>
        <button onClick={this.increment}>+</button>
      </div>
    );
  }
}

// After: Function with useState
function Counter() {
  const [count, setCount] = useState(0);
  
  function increment() {
    setCount(count + 1);
  }
  
  return (
    <div>
      <p>{count}</p>
      <button onClick={increment}>+</button>
    </div>
  );
}
\`\`\`

**Testing strategy:**
\`\`\`tsx
// Write test BEFORE migrating
describe('Counter', () => {
  it('increments when button clicked', () => {
    render(<Counter />);
    const button = screen.getByText('+');
    
    expect(screen.getByText('0')).toBeInTheDocument();
    fireEvent.click(button);
    expect(screen.getByText('1')).toBeInTheDocument();
  });
});

// Test passes with class component
// Migrate to function component
// Test still passes ✓
\`\`\`

**Week 8-10: Migrate Lifecycle Methods**

This is the tricky part—manual work required:

\`\`\`tsx
// Before: Complex lifecycle
class UserProfile extends React.Component {
  state = { user: null, loading: true, error: null };
  
  componentDidMount() {
    this.fetchUser();
  }
  
  componentDidUpdate(prevProps) {
    if (prevProps.userId !== this.props.userId) {
      this.fetchUser();
    }
  }
  
  componentWillUnmount() {
    this.abortController.abort();
  }
  
  fetchUser = async () => {
    this.setState({ loading: true });
    this.abortController = new AbortController();
    
    try {
      const response = await fetch(
        \`/api/users/\${this.props.userId}\`,
        { signal: this.abortController.signal }
      );
      const user = await response.json();
      this.setState({ user, loading: false });
    } catch (error) {
      if (error.name !== 'AbortError') {
        this.setState({ error: error.message, loading: false });
      }
    }
  }
  
  render() {
    const { loading, error, user } = this.state;
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    return <div>{user.name}</div>;
  }
}

// After: useEffect
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const abortController = new AbortController();
    
    async function fetchUser() {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(
          \`/api/users/\${userId}\`,
          { signal: abortController.signal }
        );
        const userData = await response.json();
        setUser(userData);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    }
    
    fetchUser();
    
    // Cleanup (componentWillUnmount equivalent)
    return () => abortController.abort();
  }, [userId]);  // Re-run when userId changes (componentDidUpdate equivalent)
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  return <div>{user?.name}</div>;
}
\`\`\`

**Week 11-12: Migrate HOCs and Render Props**

\`\`\`tsx
// Before: HOC
function withAuth(Component) {
  return class extends React.Component {
    componentDidMount() {
      if (!this.props.user) {
        this.props.navigate('/login');
      }
    }
    
    render() {
      return this.props.user ? <Component {...this.props} /> : null;
    }
  };
}

const Dashboard = withAuth(({ user }) => <div>Welcome {user.name}</div>);

// After: Custom hook
function useAuth() {
  const user = useContext(UserContext);
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);
  
  return user;
}

function Dashboard() {
  const user = useAuth();
  if (!user) return null;
  return <div>Welcome {user.name}</div>;
}
\`\`\`

**Phase 4: Quality Assurance (Ongoing)**

**1. Continuous Testing:**
\`\`\`bash
# Run tests after each migration
npm test

# Visual regression testing (if available)
npm run test:visual

# E2E tests
npm run test:e2e
\`\`\`

**2. Code Review Checklist:**
\`\`\`markdown
□ All tests pass
□ ESLint rules satisfied
□ No console errors in dev
□ No runtime warnings
□ Props/state behavior identical
□ Performance unchanged (use Profiler)
□ TypeScript types added
□ No PropTypes remaining
\`\`\`

**3. Monitor Production:**
\`\`\`javascript
// Add error boundary during migration
class ErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    // Log to error tracking service
    logErrorToService(error, errorInfo);
  }
  
  render() {
    return this.props.children;
  }
}

// Wrap migrated components
<ErrorBoundary>
  <NewlyMigratedComponent />
</ErrorBoundary>
\`\`\`

**Phase 5: Cleanup (Week 13)**

**1. Remove Legacy Dependencies:**
\`\`\`bash
# Remove PropTypes
npm uninstall prop-types

# Remove old HOC libraries
npm uninstall recompose

# Update React
npm install react@latest react-dom@latest
\`\`\`

**2. Update Documentation:**
\`\`\`markdown
# CONTRIBUTING.md

## Component Guidelines

- ✅ Use function components
- ✅ Use hooks for state/effects
- ✅ Use TypeScript for types
- ❌ No new class components
- ❌ No PropTypes (use TypeScript)
- ❌ No HOCs (use hooks)
\`\`\`

**Tools & Automation:**

**1. React Codemod:**
\`\`\`bash
# Convert class to function
npx react-codemod class-to-function src/

# Convert PropTypes to TypeScript
npx react-codemod proptypes-to-typescript src/

# Convert React.createClass to classes
npx react-codemod create-class-to-class src/
\`\`\`

**2. Custom Scripts:**
\`\`\`javascript
// scripts/find-unmigrated.js
const fs = require('fs');
const path = require('path');

function findClassComponents(dir) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      findClassComponents(fullPath);
    } else if (file.endsWith('.tsx') || file.endsWith('.jsx')) {
      const content = fs.readFileSync(fullPath, 'utf-8');
      if (content.includes('extends Component') || content.includes('extends React.Component')) {
        console.log(\`Class component found: \${fullPath}\`);
      }
    }
  });
}

findClassComponents('./src');
\`\`\`

**Avoiding Common Pitfalls:**

**1. Don't migrate everything at once**
- ❌ Big bang rewrite
- ✅ Incremental migration

**2. Always add tests first**
- ❌ Migrate → Test
- ✅ Test → Migrate → Verify tests still pass

**3. Watch for subtle behavior differences**
\`\`\`tsx
// Class: setState merges objects
this.setState({ count: 1 });
this.setState({ name: 'Alice' });
// Result: { count: 1, name: 'Alice' } ✓

// Hook: Multiple setStates don't merge
const [state, setState] = useState({});
setState({ count: 1 });
setState({ name: 'Alice' });
// Result: { name: 'Alice' } (count lost!) ❌

// Fix: Merge manually
setState(prev => ({ ...prev, count: 1 }));
setState(prev => ({ ...prev, name: 'Alice' }));
// Result: { count: 1, name: 'Alice' } ✓
\`\`\`

**Migration Metrics:**

Track progress:
\`\`\`
Week 1:  0% migrated
Week 4:  30% migrated (leaf components)
Week 7:  55% migrated (simple state)
Week 10: 80% migrated (lifecycle methods)
Week 12: 95% migrated (HOCs/render props)
Week 13: 100% migrated (cleanup)
\`\`\`

**Interview Insight:**
Explaining an incremental migration strategy with testing, automation, and risk mitigation shows senior-level project management skills. Discussing specific challenges (lifecycle mapping, HOC conversion) demonstrates technical depth.`,
    },
    {
      id: 'q2',
      question:
        'Compare Higher-Order Components (HOCs) and custom hooks for code reuse in React. What are the advantages and disadvantages of each pattern, and why did the React team recommend moving to hooks?',
      answer: `HOCs and hooks both enable code reuse, but hooks provide significant advantages. Here's a comprehensive comparison:

**Higher-Order Components (HOCs):**

A HOC is a function that takes a component and returns a new component with additional props or behavior.

**Example: Authentication HOC**
\`\`\`tsx
function withAuth(Component) {
  return function AuthComponent(props) {
    const user = useContext(UserContext);
    const navigate = useNavigate();
    
    useEffect(() => {
      if (!user) {
        navigate('/login');
      }
    }, [user, navigate]);
    
    if (!user) {
      return <div>Redirecting to login...</div>;
    }
    
    return <Component {...props} user={user} />;
  };
}

// Usage
const ProtectedDashboard = withAuth(Dashboard);

function Dashboard({ user }) {
  return <h1>Welcome, {user.name}</h1>;
}
\`\`\`

**Custom Hooks:**

A custom hook is a JavaScript function that uses other hooks.

**Example: Authentication Hook**
\`\`\`tsx
function useAuth() {
  const user = useContext(UserContext);
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);
  
  return user;
}

// Usage
function Dashboard() {
  const user = useAuth();
  
  if (!user) {
    return <div>Redirecting to login...</div>;
  }
  
  return <h1>Welcome, {user.name}</h1>;
}
\`\`\`

**Comparison:**

**1. Composability**

**HOCs: Nesting Hell**
\`\`\`tsx
const EnhancedComponent = 
  withAuth(
    withTheme(
      withRouter(
        withData(
          MyComponent
        )
      )
    )
  );

// React DevTools shows:
<AuthHOC>
  <ThemeHOC>
    <RouterHOC>
      <DataHOC>
        <MyComponent />
      </DataHOC>
    </RouterHOC>
  </ThemeHOC>
</AuthHOC>

// "Wrapper hell" - hard to debug!
\`\`\`

**Hooks: Flat Composition**
\`\`\`tsx
function MyComponent() {
  const user = useAuth();
  const theme = useTheme();
  const location = useLocation();
  const data = useData();
  
  return <div>Content</div>;
}

// React DevTools shows:
<MyComponent />  // Just one component!

// Much cleaner and easier to debug
\`\`\`

**2. Prop Collisions**

**HOCs: Can Clash**
\`\`\`tsx
const Component = withAuth(withRouter(MyComponent));

// Both HOCs might add props with same name:
// withAuth adds: { user, isAuthenticated }
// withRouter adds: { location, navigate }

// If both add "loading":
// Which one wins? Depends on wrapping order
// Hard to predict and debug
\`\`\`

**Hooks: Explicit Names**
\`\`\`tsx
function MyComponent() {
  const { user, loading: authLoading } = useAuth();
  const { data, loading: dataLoading } = useData();
  
  // Clear which loading is which
  // No collisions, full control
}
\`\`\`

**3. TypeScript Support**

**HOCs: Complex Types**
\`\`\`tsx
// Type signature for HOC is complex
type WithAuthProps = {
  user: User;
  isAuthenticated: boolean;
};

function withAuth<P extends object>(
  Component: React.ComponentType<P & WithAuthProps>
): React.ComponentType<Omit<P, keyof WithAuthProps>> {
  // Implementation
}

// Hard to type correctly
// TypeScript often struggles to infer types
// Errors can be cryptic
\`\`\`

**Hooks: Simple Types**
\`\`\`tsx
function useAuth(): { user: User | null; isAuthenticated: boolean } {
  // Implementation
}

// TypeScript infers everything automatically
// Errors are clear and helpful
\`\`\`

**4. Props Flow**

**HOCs: Hidden Props**
\`\`\`tsx
const Dashboard = withAuth(({ user, analytics }) => {
  return <div>Welcome {user.name}</div>;
});

// Where does \`analytics\` come from?
// Is it from withAuth?
// Is it passed by parent?
// Hard to tell without checking HOC implementation
\`\`\`

**Hooks: Explicit**
\`\`\`tsx
function Dashboard({ analytics }) {
  const { user } = useAuth();
  
  // Clear: user from useAuth
  // Clear: analytics from props
  // Obvious where everything comes from
}
\`\`\`

**5. Performance**

**HOCs: Extra Components**
\`\`\`tsx
// Every HOC adds a component to the tree
withAuth(
  withTheme(
    Component
  )
)

// Render cycle:
// 1. AuthHOC renders
// 2. ThemeHOC renders
// 3. Component renders

// More components = more work for React
\`\`\`

**Hooks: Single Component**
\`\`\`tsx
function Component() {
  useAuth();
  useTheme();
  
  return <div>Content</div>;
}

// Render cycle:
// 1. Component renders
// (hooks execute inside component)

// Fewer components = better performance
\`\`\`

**6. Static Analysis**

**HOCs: Hard to Analyze**
\`\`\`tsx
// Tools can't easily tell what props a HOC adds
const Enhanced = withAuth(Component);

// What props does Enhanced accept?
// Need to trace through HOC code
// Refactoring tools struggle
\`\`\`

**Hooks: Easy to Analyze**
\`\`\`tsx
function Component() {
  const { user } = useAuth();  // Clear dependency
  
  return <div>{user.name}</div>;
}

// Tools can easily extract dependencies
// Refactoring tools work great
// ESLint can warn about missing dependencies
\`\`\`

**7. Debugging**

**HOCs: Verbose Stack**
\`\`\`
// Stack trace with HOCs:
at MyComponent (Component.tsx:45)
  at WithData (withData.tsx:12)
    at WithRouter (withRouter.tsx:8)
      at WithTheme (withTheme.tsx:5)
        at WithAuth (withAuth.tsx:3)

// Hard to find your actual component
\`\`\`

**Hooks: Clean Stack**
\`\`\`
// Stack trace with hooks:
at MyComponent (Component.tsx:45)

// Easy to find your component
// Hook internals are abstracted
\`\`\`

**Why React Recommends Hooks:**

**1. Simpler Mental Model**
- No wrapper components
- Logic is just JavaScript functions
- Clearer data flow

**2. Better Code Organization**
- Group related logic together
- Split unrelated logic easily
- Extract to custom hooks naturally

**3. Easier Testing**
\`\`\`tsx
// Testing HOCs: Complex
const EnhancedComponent = withAuth(MyComponent);
// Need to mock auth provider
// Need to handle wrapper component

// Testing hooks: Simple
function MyComponent() {
  const { user } = useAuth();
  return <div>{user.name}</div>;
}

// Mock useAuth directly
jest.mock('./useAuth', () => ({
  useAuth: () => ({ user: { name: 'Test' } })
}));
\`\`\`

**4. Reusability**
\`\`\`tsx
// HOCs: Hard to share partial logic
// Must create new HOC for each variation

// Hooks: Easy to compose
function useAuthWithRedirect() {
  const user = useAuth();
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!user) navigate('/login');
  }, [user]);
  
  return user;
}

// Build on existing hooks easily
\`\`\`

**When HOCs Might Still Be Useful:**

**1. Wrapping Third-Party Components:**
\`\`\```tsx
// Can't modify third-party component to use hooks
import ThirdPartyComponent from 'some-library';

const WithData = withData(ThirdPartyComponent);
\`\`\`

**2. Legacy Codebases:**
- Already using HOCs everywhere
- Migration not yet planned
- Consistency with existing patterns

**3. Rendering Different Component:**
\`\`\`tsx
// HOC can return completely different component
function withLoadingOrError(Component) {
  return function (props) {
    if (props.loading) return <Spinner />;
    if (props.error) return <Error />;
    return <Component {...props} />;
  };
}

// Hook can't replace entire component
\`\`\`

**Migration Path: HOC → Hook**

\`\`\`tsx
// Step 1: Original HOC
function withMouse(Component) {
  return class extends React.Component {
    state = { x: 0, y: 0 };
    
    handleMove = (e) => {
      this.setState({ x: e.clientX, y: e.clientY });
    }
    
    componentDidMount() {
      window.addEventListener('mousemove', this.handleMove);
    }
    
    componentWillUnmount() {
      window.removeEventListener('mousemove', this.handleMove);
    }
    
    render() {
      return <Component {...this.props} mouse={this.state} />;
    }
  };
}

// Step 2: Convert to Hook
function useMouse() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    function handleMove(e) {
      setPosition({ x: e.clientX, y: e.clientY });
    }
    
    window.addEventListener('mousemove', handleMove);
    return () => window.removeEventListener('mousemove', handleMove);
  }, []);
  
  return position;
}

// Step 3: Update Components
// Before:
const TrackedComponent = withMouse(MyComponent);

// After:
function MyComponent() {
  const mouse = useMouse();
  return <div>Mouse: {mouse.x}, {mouse.y}</div>;
}
\`\`\`

**Summary:**

| Aspect | HOCs | Custom Hooks |
|--------|------|--------------|
| Composability | ❌ Wrapper hell | ✅ Flat composition |
| TypeScript | ❌ Complex types | ✅ Simple types |
| Props | ❌ Hidden/collisions | ✅ Explicit |
| Performance | ❌ Extra components | ✅ Single component |
| Debugging | ❌ Verbose stack | ✅ Clean stack |
| Reusability | ⚠️ OK | ✅ Excellent |
| Mental Model | ❌ Complex | ✅ Simple |
| Refactoring | ❌ Hard | ✅ Easy |

**Recommendation:** Use custom hooks for all new code. Migrate HOCs to hooks gradually as you touch that code.

**Interview Insight:**
Explaining the specific problems with HOCs (wrapper hell, prop collisions, TypeScript complexity) shows practical experience. Discussing why hooks solve these problems demonstrates understanding of React's evolution and design philosophy.`,
    },
    {
      id: 'q3',
      question:
        'Explain the challenges of converting componentDidUpdate lifecycle method to useEffect. What are the common pitfalls developers encounter, and how do you ensure equivalent behavior?',
      answer: `Converting \`componentDidUpdate\` to \`useEffect\` is one of the trickiest parts of migrating to hooks. Here's why it's challenging and how to do it correctly:

**The Core Challenge:**

\`componentDidUpdate\` runs **after every update** but gives you **previous props/state** for comparison.

\`useEffect\` runs based on **dependencies** but doesn't automatically provide previous values.

**Example 1: Basic Comparison**

**Class (componentDidUpdate):**
\`\`\`tsx
class UserProfile extends React.Component {
  componentDidUpdate(prevProps) {
    // Only fetch if userId changed
    if (prevProps.userId !== this.props.userId) {
      this.fetchUser(this.props.userId);
    }
  }
  
  fetchUser(userId) {
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(user => this.setState({ user }));
  }
  
  render() {
    return <div>{this.state.user?.name}</div>;
  }
}
\`\`\`

**Hook (useEffect) - Simple Case:**
\`\`\`tsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    // Runs when userId changes
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(setUser);
  }, [userId]);  // Dependency array handles comparison!
  
  return <div>{user?.name}</div>;
}
\`\`\`

**Why this works:** useEffect with \`[userId]\` automatically compares previous and current userId.

**Challenge 1: Preventing Initial Run**

\`\`\`tsx
// componentDidUpdate DOESN'T run on mount
componentDidUpdate(prevProps) {
  // Only runs on updates, not mount
  if (prevProps.count !== this.props.count) {
    console.log('Count changed');
  }
}

// useEffect DOES run on mount
useEffect(() => {
  // Runs on mount AND when count changes
  console.log('Count changed');
}, [count]);
\`\`\`

**Solution: Skip first run:**
\`\`\`tsx
function Component({ count }) {
  const isFirstRender = useRef(true);
  
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;  // Skip first run
    }
    
    console.log('Count changed (not initial mount)');
  }, [count]);
}

// Or use custom hook:
function useUpdateEffect(effect, deps) {
  const isFirstRender = useRef(true);
  
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    
    return effect();
  }, deps);
}

// Usage:
useUpdateEffect(() => {
  console.log('Count changed (not initial mount)');
}, [count]);
\`\`\`

**Challenge 2: Comparing Multiple Props**

\`\`\`tsx
// componentDidUpdate: Easy to compare multiple props
componentDidUpdate(prevProps) {
  if (
    prevProps.userId !== this.props.userId ||
    prevProps.filter !== this.props.filter
  ) {
    this.loadData(this.props.userId, this.props.filter);
  }
}

// useEffect: Include all deps in array
useEffect(() => {
  loadData(userId, filter);
}, [userId, filter]);  // Runs when EITHER changes
\`\`\`

**Challenge 3: Need Previous Value**

Sometimes you genuinely need the previous value:

\`\`\`tsx
// componentDidUpdate: Previous value provided
componentDidUpdate(prevProps) {
  const changed = prevProps.count !== this.props.count;
  const increased = this.props.count > prevProps.count;
  
  if (changed && increased) {
    console.log(\`Count increased from \${prevProps.count} to \${this.props.count}\`);
  }
}

// useEffect: Must track previous value manually
function Component({ count }) {
  const prevCount = usePrevious(count);
  
  useEffect(() => {
    if (prevCount !== undefined) {  // Not first render
      const increased = count > prevCount;
      
      if (increased) {
        console.log(\`Count increased from \${prevCount} to \${count}\`);
      }
    }
  }, [count, prevCount]);
}

// Custom hook to get previous value
function usePrevious(value) {
  const ref = useRef();
  
  useEffect(() => {
    ref.current = value;
  });
  
  return ref.current;
}
\`\`\`

**Challenge 4: Conditional Updates**

\`\`\`tsx
// componentDidUpdate: Can check multiple conditions
componentDidUpdate(prevProps, prevState) {
  // Complex conditional logic
  if (prevProps.userId !== this.props.userId) {
    this.fetchUser(this.props.userId);
  }
  
  if (prevState.searchQuery !== this.state.searchQuery) {
    this.performSearch(this.state.searchQuery);
  }
  
  if (
    prevProps.sortBy !== this.props.sortBy ||
    prevProps.filter !== this.props.filter
  ) {
    this.refreshData(this.props.sortBy, this.props.filter);
  }
}

// useEffect: Split into multiple effects
function Component({ userId, sortBy, filter }) {
  const [searchQuery, setSearchQuery] = useState('');
  
  // Effect 1: User ID changes
  useEffect(() => {
    fetchUser(userId);
  }, [userId]);
  
  // Effect 2: Search query changes
  useEffect(() => {
    performSearch(searchQuery);
  }, [searchQuery]);
  
  // Effect 3: Sort or filter changes
  useEffect(() => {
    refreshData(sortBy, filter);
  }, [sortBy, filter]);
}
\`\`\`

**Best Practice:** Split complex componentDidUpdate logic into multiple focused useEffect hooks.

**Challenge 5: Comparing Objects/Arrays**

\`\`\`tsx
// componentDidUpdate: Can deep compare
componentDidUpdate(prevProps) {
  if (!isEqual(prevProps.user, this.props.user)) {  // Deep comparison
    this.updateUI(this.props.user);
  }
}

// useEffect: Only shallow comparison
function Component({ user }) {
  useEffect(() => {
    updateUI(user);
  }, [user]);  // ← Compares reference, not content!
}

// Problem:
// If parent creates new user object with same values:
// { name: 'Alice', age: 30 } !== { name: 'Alice', age: 30 }
// Effect runs unnecessarily!
\`\`\`

**Solutions:**

**Option 1: Compare specific properties:**
\`\`\`tsx
useEffect(() => {
  updateUI(user);
}, [user.id, user.name, user.age]);  // Compare individual values
\`\`\`

**Option 2: Use deep comparison hook:**
\`\`\`tsx
import { useDeepCompareEffect } from 'use-deep-compare-effect';

useDeepCompareEffect(() => {
  updateUI(user);
}, [user]);  // Deep comparison
\`\`\`

**Option 3: Stringify (for simple objects):**
\`\`\```tsx
const userString = JSON.stringify(user);

useEffect(() => {
  updateUI(user);
}, [userString]);  // Compares stringified version
\`\`\`

**Challenge 6: Side Effects Based on State Changes**

\`\`\`tsx
// componentDidUpdate: Can compare state
componentDidUpdate(prevProps, prevState) {
  if (prevState.isOpen !== this.state.isOpen && this.state.isOpen) {
    document.body.style.overflow = 'hidden';  // Lock scroll
  }
  
  if (prevState.isOpen !== this.state.isOpen && !this.state.isOpen) {
    document.body.style.overflow = 'auto';  // Unlock scroll
  }
}

// useEffect: Need previous state
function Component() {
  const [isOpen, setIsOpen] = useState(false);
  const prevIsOpen = usePrevious(isOpen);
  
  useEffect(() => {
    if (prevIsOpen === false && isOpen === true) {
      document.body.style.overflow = 'hidden';
    }
    
    if (prevIsOpen === true && isOpen === false) {
      document.body.style.overflow = 'auto';
    }
  }, [isOpen, prevIsOpen]);
}

// Or simpler: Don't check previous, just set based on current
useEffect(() => {
  document.body.style.overflow = isOpen ? 'hidden' : 'auto';
  
  return () => {
    document.body.style.overflow = 'auto';  // Cleanup
  };
}, [isOpen]);
\`\`\`

**Challenge 7: Debouncing/Throttling**

\`\`\`tsx
// componentDidUpdate: Can implement debouncing
componentDidUpdate(prevProps) {
  if (prevProps.searchQuery !== this.props.searchQuery) {
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => {
      this.performSearch(this.props.searchQuery);
    }, 500);
  }
}

componentWillUnmount() {
  clearTimeout(this.searchTimer);
}

// useEffect: Use cleanup function
function Component({ searchQuery }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      performSearch(searchQuery);
    }, 500);
    
    return () => clearTimeout(timer);  // Cleanup previous timer
  }, [searchQuery]);
}
\`\`\`

**Complete Migration Example:**

\`\`\`tsx
// Before: Complex componentDidUpdate
class ProductList extends React.Component {
  state = {
    products: [],
    loading: false,
    error: null
  };
  
  componentDidUpdate(prevProps, prevState) {
    // 1. Category changed: Fetch new products
    if (prevProps.category !== this.props.category) {
      this.fetchProducts(this.props.category, this.props.sortBy);
    }
    
    // 2. Sort changed: Re-sort existing products
    if (
      prevProps.sortBy !== this.props.sortBy &&
      prevProps.category === this.props.category
    ) {
      this.sortProducts(this.props.sortBy);
    }
    
    // 3. Products loaded: Update analytics
    if (prevState.loading && !this.state.loading && !this.state.error) {
      this.trackProductsViewed(this.state.products);
    }
  }
  
  fetchProducts(category, sortBy) {
    this.setState({ loading: true });
    // ... fetch logic
  }
  
  sortProducts(sortBy) {
    // ... sort logic
  }
  
  trackProductsViewed(products) {
    // ... analytics
  }
  
  render() {
    // ... render logic
  }
}

// After: Multiple focused useEffects
function ProductList({ category, sortBy }) {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Effect 1: Fetch products when category or sortBy changes
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    fetchProducts(category, sortBy)
      .then(data => {
        setProducts(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [category, sortBy]);
  
  // Effect 2: Track when products loaded
  useEffect(() => {
    if (products.length > 0 && !loading && !error) {
      trackProductsViewed(products);
    }
  }, [products, loading, error]);
  
  // Render logic
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  return <ProductGrid products={products} />;
}
\`\`\`

**Migration Checklist:**

\`\`\`
□ Identify what triggers the update
□ Add those values to useEffect dependency array
□ If you need "did not run on mount", use useUpdateEffect
□ If you need previous values, use usePrevious hook
□ Split complex logic into multiple focused useEffect
□ For objects/arrays, compare specific properties or use deep compare
□ Ensure cleanup functions handle all side effects
□ Test that behavior matches original componentDidUpdate
\`\`\`

**Interview Insight:**
Discussing the challenges (initial run, previous values, object comparison) shows deep understanding of both lifecycle methods and hooks. Providing specific solutions (usePrevious, useUpdateEffect, splitting effects) demonstrates practical problem-solving skills.`,
    },
  ],
};
