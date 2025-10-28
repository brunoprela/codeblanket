export const legacyPatternsAndMigration = {
  title: 'Legacy Patterns & Migration',
  id: 'legacy-patterns-and-migration',
  content: `
# Legacy Patterns & Migration

## Introduction

Modern React uses **function components and hooks** (introduced in React 16.8, 2019). However, you'll encounter **legacy patterns** in existing codebases:
- **Class components**
- **Lifecycle methods**
- **PropTypes**
- **Higher-Order Components (HOCs)**
- **Render Props**

Understanding these patterns is essential for:
- Reading and maintaining legacy code
- Migrating old codebases to modern React
- Working with older libraries
- Technical interviews (legacy code is common!)

This section teaches you to **read**, **understand**, and **modernize** legacy React code.

## Class Components

Before hooks, React components were classes:

\`\`\`tsx
// Modern: Function component
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Legacy: Class component
class Greeting extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
\`\`\`

### Class Component Structure

\`\`\`tsx
import React, { Component } from 'react';

class UserProfile extends Component {
  // 1. Constructor (optional)
  constructor(props) {
    super(props);  // MUST call super(props)
    
    // 2. Initialize state
    this.state = {
      isEditing: false,
      editedName: props.name
    };
    
    // 3. Bind methods (if not using arrow functions)
    this.handleClick = this.handleClick.bind(this);
  }
  
  // 4. Lifecycle methods
  componentDidMount() {
    console.log('Component mounted');
  }
  
  componentDidUpdate(prevProps, prevState) {
    console.log('Component updated');
  }
  
  componentWillUnmount() {
    console.log('Component will unmount');
  }
  
  // 5. Event handlers
  handleClick() {
    this.setState({ isEditing: true });
  }
  
  // Or with arrow function (auto-binds this)
  handleSave = () => {
    this.setState({ isEditing: false });
  }
  
  // 6. render method (required!)
  render() {
    const { name, age } = this.props;
    const { isEditing, editedName } = this.state;
    
    return (
      <div>
        <h1>{name}</h1>
        <p>Age: {age}</p>
        {isEditing && <input value={editedName} />}
        <button onClick={this.handleClick}>Edit</button>
        <button onClick={this.handleSave}>Save</button>
      </div>
    );
  }
}
\`\`\`

### Key Differences from Function Components

| Aspect | Class | Function |
|--------|-------|----------|
| Definition | \`class X extends Component\` | \`function X()\` |
| Props access | \`this.props.name\` | \`props.name\` |
| State | \`this.state\` | \`useState()\` |
| Update state | \`this.setState()\` | \`setState()\` |
| Side effects | Lifecycle methods | \`useEffect()\` |
| Binding | \`this.method.bind(this)\` | Not needed |
| Render | \`render() {}\` method | Return JSX directly |

## Lifecycle Methods

Class components use lifecycle methods. Here's how they map to hooks:

### componentDidMount

**Class (Legacy):**
\`\`\`tsx
class Component extends React.Component {
  componentDidMount() {
    // Runs once after first render
    fetchData();
  }
  
  render() {
    return <div>Content</div>;
  }
}
\`\`\`

**Function (Modern):**
\`\`\`tsx
function Component() {
  useEffect(() => {
    // Runs once after first render
    fetchData();
  }, []);  // Empty deps = mount only
  
  return <div>Content</div>;
}
\`\`\`

### componentDidUpdate

**Class (Legacy):**
\`\`\`tsx
class Component extends React.Component {
  componentDidUpdate(prevProps, prevState) {
    // Runs after every update
    if (prevProps.userId !== this.props.userId) {
      fetchUser(this.props.userId);
    }
  }
  
  render() {
    return <div>Content</div>;
  }
}
\`\`\`

**Function (Modern):**
\`\`\`tsx
function Component({ userId }) {
  useEffect(() => {
    // Runs when userId changes
    fetchUser(userId);
  }, [userId]);  // Deps array = update trigger
  
  return <div>Content</div>;
}
\`\`\`

### componentWillUnmount

**Class (Legacy):**
\`\`\`tsx
class Component extends React.Component {
  componentDidMount() {
    this.timer = setInterval(() => {
      console.log('Tick');
    }, 1000);
  }
  
  componentWillUnmount() {
    // Cleanup before unmount
    clearInterval(this.timer);
  }
  
  render() {
    return <div>Content</div>;
  }
}
\`\`\`

**Function (Modern):**
\`\`\`tsx
function Component() {
  useEffect(() => {
    const timer = setInterval(() => {
      console.log('Tick');
    }, 1000);
    
    // Cleanup function (runs on unmount)
    return () => clearInterval(timer);
  }, []);
  
  return <div>Content</div>;
}
\`\`\`

### Lifecycle Comparison Chart

| Lifecycle Method | Hook Equivalent | Purpose |
|-----------------|-----------------|---------|
| \`constructor()\` | \`useState()\` | Initialize state |
| \`componentDidMount()\` | \`useEffect(() => {}, [])\` | Run once after mount |
| \`componentDidUpdate()\` | \`useEffect(() => {}, [deps])\` | Run on updates |
| \`componentWillUnmount()\` | \`useEffect(() => { return cleanup }, [])\` | Cleanup on unmount |
| \`shouldComponentUpdate()\` | \`React.memo()\` | Prevent unnecessary renders |
| \`getDerivedStateFromProps()\` | \`useState\` + logic in render | Derive state from props |

## PropTypes vs TypeScript

Legacy React used PropTypes for type checking:

**PropTypes (Legacy):**
\`\`\`tsx
import PropTypes from 'prop-types';

function UserCard({ name, age, isAdmin }) {
  return (
    <div>
      <h3>{name}</h3>
      <p>Age: {age}</p>
      {isAdmin && <span>Admin</span>}
    </div>
  );
}

UserCard.propTypes = {
  name: PropTypes.string.isRequired,
  age: PropTypes.number.isRequired,
  isAdmin: PropTypes.bool
};

UserCard.defaultProps = {
  isAdmin: false
};
\`\`\`

**TypeScript (Modern):**
\`\`\`tsx
interface UserCardProps {
  name: string;
  age: number;
  isAdmin?: boolean;
}

function UserCard({ name, age, isAdmin = false }: UserCardProps) {
  return (
    <div>
      <h3>{name}</h3>
      <p>Age: {age}</p>
      {isAdmin && <span>Admin</span>}
    </div>
  );
}
\`\`\`

**Advantages of TypeScript:**
- ✅ Compile-time type checking (catch errors before runtime)
- ✅ Better IDE autocomplete
- ✅ Refactoring safety
- ✅ Self-documenting code
- ✅ No runtime overhead (PropTypes checks at runtime)

## Higher-Order Components (HOCs)

HOCs were used for code reuse before hooks.

**HOC Pattern (Legacy):**
\`\`\`tsx
// HOC: Function that takes a component and returns a new component
function withAuth(Component) {
  return class extends React.Component {
    componentDidMount() {
      if (!this.props.isAuthenticated) {
        this.props.history.push('/login');
      }
    }
    
    render() {
      if (!this.props.isAuthenticated) {
        return <div>Loading...</div>;
      }
      
      return <Component {...this.props} />;
    }
  };
}

// Usage
class Dashboard extends React.Component {
  render() {
    return <h1>Dashboard</h1>;
  }
}

export default withAuth(Dashboard);
\`\`\`

**Custom Hook (Modern):**
\`\`\`tsx
// Custom hook
function useAuth() {
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    checkAuth().then(authenticated => {
      setIsAuthenticated(authenticated);
      setIsLoading(false);
      
      if (!authenticated) {
        navigate('/login');
      }
    });
  }, [navigate]);
  
  return { isAuthenticated, isLoading };
}

// Usage
function Dashboard() {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) return <div>Loading...</div>;
  if (!isAuthenticated) return null;
  
  return <h1>Dashboard</h1>;
}
\`\`\`

**Why hooks are better than HOCs:**
- ✅ Simpler: No wrapper components
- ✅ More readable: Logic is inline
- ✅ No "wrapper hell" (nested HOCs)
- ✅ Better TypeScript support
- ✅ Easier to debug

## Render Props

Render props shared logic by passing a function as a child:

**Render Props (Legacy):**
\`\`\`tsx
class Mouse extends React.Component {
  state = { x: 0, y: 0 };
  
  handleMouseMove = (event) => {
    this.setState({
      x: event.clientX,
      y: event.clientY
    });
  }
  
  componentDidMount() {
    window.addEventListener('mousemove', this.handleMouseMove);
  }
  
  componentWillUnmount() {
    window.removeEventListener('mousemove', this.handleMouseMove);
  }
  
  render() {
    return this.props.children(this.state);
  }
}

// Usage
<Mouse>
  {({ x, y }) => (
    <div>
      Mouse position: {x}, {y}
    </div>
  )}
</Mouse>
\`\`\`

**Custom Hook (Modern):**
\`\`\`tsx
function useMouse() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    function handleMouseMove(event) {
      setPosition({ x: event.clientX, y: event.clientY });
    }
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  return position;
}

// Usage
function Component() {
  const { x, y } = useMouse();
  
  return (
    <div>
      Mouse position: {x}, {y}
    </div>
  );
}
\`\`\`

**Why hooks are better than render props:**
- ✅ Simpler: No extra JSX nesting
- ✅ More composable: Use multiple hooks easily
- ✅ Better performance: No extra component renders
- ✅ Cleaner code: Logic extraction is straightforward

## Migrating Legacy Code

### Step-by-Step Migration

**1. Convert Class to Function:**

\`\`\`tsx
// Before: Class component
class Counter extends React.Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
  }
  
  increment = () => {
    this.setState({ count: this.state.count + 1 });
  }
  
  render() {
    return (
      <div>
        <p>Count: {this.state.count}</p>
        <button onClick={this.increment}>Increment</button>
      </div>
    );
  }
}

// After: Function component
function Counter() {
  const [count, setCount] = useState(0);
  
  function increment() {
    setCount(count + 1);
  }
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
    </div>
  );
}
\`\`\`

**2. Convert Lifecycle to useEffect:**

\`\`\`tsx
// Before: Lifecycle methods
class UserProfile extends React.Component {
  state = { user: null };
  
  componentDidMount() {
    fetchUser(this.props.userId).then(user => {
      this.setState({ user });
    });
  }
  
  componentDidUpdate(prevProps) {
    if (prevProps.userId !== this.props.userId) {
      fetchUser(this.props.userId).then(user => {
        this.setState({ user });
      });
    }
  }
  
  componentWillUnmount() {
    // Cancel any pending requests
  }
  
  render() {
    const { user } = this.state;
    if (!user) return <div>Loading...</div>;
    return <div>{user.name}</div>;
  }
}

// After: useEffect
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    let cancelled = false;
    
    fetchUser(userId).then(user => {
      if (!cancelled) {
        setUser(user);
      }
    });
    
    return () => {
      cancelled = true;  // Cleanup
    };
  }, [userId]);  // Re-run when userId changes
  
  if (!user) return <div>Loading...</div>;
  return <div>{user.name}</div>;
}
\`\`\`

**3. Convert HOC to Custom Hook:**

\`\`\`tsx
// Before: HOC
function withLoading(Component) {
  return class extends React.Component {
    state = { isLoading: true };
    
    componentDidMount() {
      setTimeout(() => {
        this.setState({ isLoading: false });
      }, 1000);
    }
    
    render() {
      if (this.state.isLoading) {
        return <div>Loading...</div>;
      }
      return <Component {...this.props} />;
    }
  };
}

const UserList = withLoading(({ users }) => (
  <ul>
    {users.map(user => <li key={user.id}>{user.name}</li>)}
  </ul>
));

// After: Custom hook
function useLoading(delay = 1000) {
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [delay]);
  
  return isLoading;
}

function UserList({ users }) {
  const isLoading = useLoading();
  
  if (isLoading) return <div>Loading...</div>;
  
  return (
    <ul>
      {users.map(user => <li key={user.id}>{user.name}</li>)}
    </ul>
  );
}
\`\`\`

## Reading Legacy Code

### Common Patterns You'll See

**1. Binding in Constructor:**
\`\`\`tsx
class Component extends React.Component {
  constructor(props) {
    super(props);
    // This is necessary for \`this\` to work in callbacks
    this.handleClick = this.handleClick.bind(this);
  }
  
  handleClick() {
    console.log(this.props.name);
  }
  
  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}
\`\`\`

**Why?** In JavaScript classes, methods aren't auto-bound. Without binding, \`this\` is undefined.

**Modern equivalent:** Function components don't have \`this\`, so no binding needed!

**2. State as Object:**
\`\`\`tsx
// Legacy: Single state object
this.setState({ count: 1, name: 'Alice' });

// Modern: Multiple useState calls
const [count, setCount] = useState(1);
const [name, setName] = useState('Alice');
\`\`\`

**3. Accessing Previous State:**
\`\`\`tsx
// Legacy: Function form of setState
this.setState(prevState => ({
  count: prevState.count + 1
}));

// Modern: Function form of setter
setCount(prevCount => prevCount + 1);
\`\`\`

## Best Practices for Working with Legacy Code

1. **Don't rewrite everything at once**
   - Migrate incrementally
   - Start with leaf components (no children)
   - Work your way up the tree

2. **Test thoroughly**
   - Legacy code often lacks tests
   - Add tests before migrating
   - Verify behavior unchanged

3. **Use codemods** (automated migration tools)
   \`\`\`bash
   npx react-codemod class-to-function
   \`\`\`

4. **Keep both patterns temporarily**
   - New components: Use hooks
   - Old components: Migrate when touched
   - Eventually: Full migration

5. **Document why legacy patterns exist**
   - Comment explaining "this is legacy"
   - Add TODO for migration
   - Track technical debt

## Summary: Modern vs Legacy

| Pattern | Legacy | Modern |
|---------|--------|--------|
| Components | Class components | Function components |
| State | \`this.state\`, \`this.setState()\` | \`useState()\` |
| Side effects | Lifecycle methods | \`useEffect()\` |
| Context | \`<Context.Consumer>\` | \`useContext()\` |
| Type checking | PropTypes | TypeScript |
| Code reuse | HOCs, Render Props | Custom hooks |
| Memoization | \`shouldComponentUpdate()\` | \`React.memo()\`, \`useMemo()\` |
| Refs | \`React.createRef()\` | \`useRef()\` |

## What's Next?

Congratulations! You've completed Module 1: React Fundamentals. You now understand:
- ✅ Modern React (function components, hooks)
- ✅ Legacy React (class components, lifecycle methods)
- ✅ How to read and migrate legacy code

**Next Module:** Advanced React concepts including useEffect deep dive, custom hooks, Context API, performance optimization, and more!
`,
};
