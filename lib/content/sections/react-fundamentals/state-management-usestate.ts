export const stateManagementUseState = {
  title: 'State Management with useState',
  id: 'state-management-usestate',
  content: `
# State Management with useState

## Introduction

State is what makes React components **interactive** and **dynamic**. Without state, components are just static HTML—with state, they become living, breathing parts of your application that respond to user interactions. **Understanding useState is foundational to React development**—it's the most commonly used hook and the gateway to building real applications.

### What is State?

**State** is data that changes over time and causes your component to re-render when updated. Think of state as your component's memory.

\`\`\`tsx
// Without state (static)
function Counter() {
  let count = 0;  // Regular JavaScript variable
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => count++}>Increment</button>
      {/* Clicking does nothing visible—count increments but UI doesn't update */}
    </div>
  );
}

// With state (dynamic)
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);  // React state
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      {/* Clicking updates count AND re-renders UI */}
    </div>
  );
}
\`\`\`

**Key difference**: Regular variables don't trigger re-renders. State variables do.

## useState Anatomy

\`\`\`tsx
const [value, setValue] = useState(initialValue);
//     ↑       ↑          ↑         ↑
//   Current  Setter   Hook name  Starting value
//   value    function
\`\`\`

**Breakdown**:
1. **\`useState\`** - The hook function (import from 'react')
2. **\`initialValue\`** - Starting value (only used on first render)
3. **\`[value, setValue]\`** - Array destructuring (returns array of [state, setter])
4. **\`value\`** - Current state value (read-only)
5. **\`setValue\`** - Function to update state (triggers re-render)

**Naming convention**: \`[thing, setThing]\`

\`\`\`tsx
const [count, setCount] = useState(0);
const [name, setName] = useState('');
const [isOpen, setIsOpen] = useState(false);
const [user, setUser] = useState(null);
const [items, setItems] = useState([]);
\`\`\`

## useState Basics: Primitive Types

\`\`\`tsx
import { useState } from 'react';

function ExampleComponent() {
  // Number
  const [age, setAge] = useState(25);
  
  // String
  const [name, setName] = useState('Alice');
  
  // Boolean
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  
  // Null (for data that hasn't loaded yet)
  const [user, setUser] = useState<User | null>(null);
  
  return (
    <div>
      <p>Age: {age}</p>
      <button onClick={() => setAge(age + 1)}>Birthday</button>
      
      <p>Name: {name}</p>
      <button onClick={() => setName('Bob')}>Change Name</button>
      
      <p>Status: {isLoggedIn ? 'Logged In' : 'Logged Out'}</p>
      <button onClick={() => setIsLoggedIn(!isLoggedIn)}>Toggle</button>
    </div>
  );
}
\`\`\`

## How useState Works: The Magic Behind Re-renders

\`\`\`tsx
function Counter() {
  console.log('Component rendering...');
  
  const [count, setCount] = useState(0);
  
  const handleIncrement = () => {
    setCount(count + 1);
    // What happens here?
    // 1. setCount schedules a re-render
    // 2. React re-runs this entire function
    // 3. useState returns the NEW count value
    // 4. JSX is created with new count
    // 5. React diffs Virtual DOM
    // 6. React updates only what changed in real DOM
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={handleIncrement}>Increment</button>
    </div>
  );
}

// Console output when you click 3 times:
// Component rendering... (initial render, count = 0)
// Component rendering... (1st click, count = 1)
// Component rendering... (2nd click, count = 2)
// Component rendering... (3rd click, count = 3)
\`\`\`

**Key insight**: Entire component function re-runs on every state update. State persists across re-renders, but local variables reset.

## State Updates are Asynchronous

**Critical concept**: \`setState\` doesn't update state immediately—it schedules an update.

\`\`\`tsx
function Counter() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    setCount(count + 1);
    console.log(count);  // Still 0! Not 1!
    // State hasn't updated yet—update is scheduled
    
    // If you need the new value, wait for next render:
    // Use useEffect or the functional update form
  };
  
  return <button onClick={handleClick}>Count: {count}</button>;
}
\`\`\`

**Why asynchronous?**
- React batches multiple state updates for performance
- Multiple \`setState\` calls in one event handler → one re-render

\`\`\`tsx
function MultipleUpdates() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');
  
  const handleClick = () => {
    setCount(count + 1);      // Update 1
    setName('Alice');         // Update 2
    setCount(count + 2);      // Update 3
    
    // React batches these → ONE re-render
    // Before React 18: Only batches in event handlers
    // React 18+: Batches everywhere (even in promises, timeouts)
  };
  
  return <button onClick={handleClick}>Update</button>;
}
\`\`\`

## Functional Updates: The Safe Way

**Problem**: Updating state based on previous state can go wrong.

\`\`\`tsx
// ❌ WRONG: Multiple updates don't stack correctly
function Counter() {
  const [count, setCount] = useState(0);
  
  const increment3Times = () => {
    setCount(count + 1);  // count = 0, so sets to 1
    setCount(count + 1);  // count = 0, so sets to 1 (not 2!)
    setCount(count + 1);  // count = 0, so sets to 1 (not 3!)
    // Result: count becomes 1, not 3
    // Why? All three read the same 'count' value (0)
  };
  
  return <button onClick={increment3Times}>Count: {count}</button>;
}

// ✅ CORRECT: Functional updates use latest state
function Counter() {
  const [count, setCount] = useState(0);
  
  const increment3Times = () => {
    setCount(prevCount => prevCount + 1);  // 0 + 1 = 1
    setCount(prevCount => prevCount + 1);  // 1 + 1 = 2
    setCount(prevCount => prevCount + 1);  // 2 + 1 = 3
    // Result: count becomes 3 ✓
  };
  
  return <button onClick={increment3Times}>Count: {count}</button>;
}
\`\`\`

**Rule of thumb**: Use functional updates when new state depends on old state.

\`\`\`tsx
// ✅ Use functional form
setCount(prev => prev + 1);
setItems(prev => [...prev, newItem]);
setUser(prev => ({ ...prev, name: 'Alice' }));

// ❌ Don't use direct form for dependent updates
setCount(count + 1);  // Risky if multiple updates
\`\`\`

## useState with Objects

**Critical rule**: Never mutate state directly. Always create new objects.

\`\`\`tsx
interface User {
  name: string;
  age: number;
  email: string;
}

function UserProfile() {
  const [user, setUser] = useState<User>({
    name: 'Alice',
    age: 25,
    email: 'alice@example.com'
  });
  
  // ❌ WRONG: Direct mutation (doesn't trigger re-render)
  const updateNameWrong = () => {
    user.name = 'Bob';  // Mutating original object
    setUser(user);      // Same object reference—React won't re-render
  };
  
  // ✅ CORRECT: Create new object
  const updateNameCorrect = () => {
    setUser({
      ...user,           // Copy all properties
      name: 'Bob'        // Override name
    });
  };
  
  // ✅ ALSO CORRECT: Functional update
  const incrementAge = () => {
    setUser(prevUser => ({
      ...prevUser,
      age: prevUser.age + 1
    }));
  };
  
  return (
    <div>
      <p>Name: {user.name}</p>
      <p>Age: {user.age}</p>
      <p>Email: {user.email}</p>
      <button onClick={updateNameCorrect}>Change Name</button>
      <button onClick={incrementAge}>Birthday</button>
    </div>
  );
}
\`\`\`

**Why immutability matters**:
1. React compares by reference (\`oldUser === newUser\`)
2. If same object reference → no re-render
3. New object → re-render triggered

## useState with Arrays

**Never use mutating methods** (push, pop, splice, sort, reverse). Always create new arrays.

\`\`\`tsx
function TodoList() {
  const [todos, setTodos] = useState<string[]>([]);
  
  // ❌ WRONG: Mutating array
  const addTodoWrong = (todo: string) => {
    todos.push(todo);  // Mutates original array
    setTodos(todos);   // Same array reference—no re-render
  };
  
  // ✅ CORRECT: Create new array (spread operator)
  const addTodo = (todo: string) => {
    setTodos([...todos, todo]);  // New array with added item
  };
  
  // ✅ Remove item
  const removeTodo = (index: number) => {
    setTodos(todos.filter((_, i) => i !== index));
  };
  
  // ✅ Update item
  const updateTodo = (index: number, newText: string) => {
    setTodos(todos.map((todo, i) => 
      i === index ? newText : todo
    ));
  };
  
  // ✅ Insert at beginning
  const addTodoAtStart = (todo: string) => {
    setTodos([todo, ...todos]);
  };
  
  // ✅ Sort (creates new sorted array)
  const sortTodos = () => {
    setTodos([...todos].sort());  // [...todos] creates copy first
  };
  
  return (
    <div>
      {todos.map((todo, index) => (
        <div key={index}>
          <span>{todo}</span>
          <button onClick={() => removeTodo(index)}>Remove</button>
          <button onClick={() => updateTodo(index, 'Updated!')}>Update</button>
        </div>
      ))}
      <button onClick={() => addTodo('New Todo')}>Add Todo</button>
    </div>
  );
}
\`\`\`

**Array manipulation cheat sheet**:

| Operation | ❌ Mutating | ✅ Immutable |
|-----------|------------|-------------|
| Add to end | \`arr.push(item)\` | \`[...arr, item]\` |
| Add to start | \`arr.unshift(item)\` | \`[item, ...arr]\` |
| Remove | \`arr.splice(i, 1)\` | \`arr.filter((_, index) => index !== i)\` |
| Replace | \`arr[i] = newItem\` | \`arr.map((item, index) => index === i ? newItem : item)\` |
| Sort | \`arr.sort()\` | \`[...arr].sort()\` |

## Multiple State Variables vs Single State Object

**Two approaches**:

\`\`\`tsx
// Approach 1: Multiple state variables (recommended for unrelated data)
function UserForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState(0);
  
  // Pros: Simple updates, clear naming
  // Cons: More lines of code
  
  return (
    <form>
      <input value={name} onChange={e => setName(e.target.value)} />
      <input value={email} onChange={e => setEmail(e.target.value)} />
      <input value={age} onChange={e => setAge(Number(e.target.value))} />
    </form>
  );
}

// Approach 2: Single state object (recommended for related data)
function UserForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: 0
  });
  
  const handleChange = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Pros: Single source of truth, easier to pass around
  // Cons: More complex updates (must spread)
  
  return (
    <form>
      <input 
        value={formData.name} 
        onChange={e => handleChange('name', e.target.value)} 
      />
      <input 
        value={formData.email} 
        onChange={e => handleChange('email', e.target.value)} 
      />
      <input 
        value={formData.age} 
        onChange={e => handleChange('age', Number(e.target.value))} 
      />
    </form>
  );
}
\`\`\`

**When to use which**:
- **Multiple state variables**: Unrelated data, simple updates, different update frequencies
- **Single state object**: Related data (form fields), need to pass entire state as prop, complex validation

## Derived State: Don't Store What You Can Calculate

\`\`\`tsx
// ❌ WRONG: Storing derived state
function ShoppingCart() {
  const [items, setItems] = useState([
    { id: 1, price: 10, quantity: 2 },
    { id: 2, price: 20, quantity: 1 }
  ]);
  const [total, setTotal] = useState(40);  // Redundant!
  
  const addItem = (item) => {
    setItems([...items, item]);
    // Bug risk: Forgot to update total!
  };
}

// ✅ CORRECT: Calculate derived state
function ShoppingCart() {
  const [items, setItems] = useState([
    { id: 1, price: 10, quantity: 2 },
    { id: 2, price: 20, quantity: 1 }
  ]);
  
  // Calculate total on every render (fast—no state needed)
  const total = items.reduce((sum, item) => 
    sum + item.price * item.quantity, 0
  );
  
  // No risk of total being out of sync!
}
\`\`\`

**Rule**: If you can calculate it from existing state/props, don't store it in state.

\`\`\`tsx
// ✅ GOOD: Derived values
const fullName = \`\${firstName} \${lastName}\`;
const isValid = email.includes('@') && password.length >= 8;
const filteredItems = items.filter(item => item.category === selectedCategory);

// ❌ BAD: Storing derived values in state
const [fullName, setFullName] = useState('');  // Calculate it!
const [isValid, setIsValid] = useState(false); // Calculate it!
\`\`\`

## Lazy Initialization

**Problem**: Expensive initial state calculation runs on every render.

\`\`\`tsx
// ❌ WRONG: Runs on every render
function ExpensiveComponent() {
  const [data, setData] = useState(
    JSON.parse(localStorage.getItem('data'))  // Runs on EVERY render
  );
  // Even though initial value only matters on first render
}

// ✅ CORRECT: Lazy initialization (runs only once)
function ExpensiveComponent() {
  const [data, setData] = useState(() => {
    // This function only runs on first render
    const saved = localStorage.getItem('data');
    return saved ? JSON.parse(saved) : [];
  });
}
\`\`\`

**When to use**:
- Reading from localStorage/sessionStorage
- Complex calculations
- Creating large objects/arrays

## State vs Props

**Props** and **state** are both data, but they serve different purposes.

| Feature | Props | State |
|---------|-------|-------|
| **Defined by** | Parent component | Component itself |
| **Can change?** | No (from component's perspective) | Yes |
| **Triggers re-render?** | Yes (when parent passes new value) | Yes |
| **Used for** | Passing data down | Managing component's own data |

\`\`\`tsx
// Props: Data passed from parent
interface UserCardProps {
  name: string;  // Parent controls this
  email: string;
}

function UserCard({ name, email }: UserCardProps) {
  // Can't modify name or email here
  return <div>{name} - {email}</div>;
}

// State: Component's own data
function Counter() {
  const [count, setCount] = useState(0);  // Component controls this
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}

// Combined: Props and State
function TodoItem({ initialText }: { initialText: string }) {
  const [text, setText] = useState(initialText);  // Start with prop value
  const [isEditing, setIsEditing] = useState(false);  // Component's own state
  
  return isEditing ? (
    <input value={text} onChange={e => setText(e.target.value)} />
  ) : (
    <span onClick={() => setIsEditing(true)}>{text}</span>
  );
}
\`\`\`

## Common useState Mistakes

### Mistake 1: Mutating State Directly

\`\`\`tsx
// ❌ WRONG
const [user, setUser] = useState({ name: 'Alice', age: 25 });
user.age = 26;  // Direct mutation
setUser(user);  // Won't trigger re-render (same object reference)

// ✅ CORRECT
setUser({ ...user, age: 26 });
\`\`\`

### Mistake 2: Updating State Based on Stale Value

\`\`\`tsx
// ❌ WRONG
const [count, setCount] = useState(0);
setCount(count + 1);
setCount(count + 1);  // Both use old 'count' value

// ✅ CORRECT
setCount(prev => prev + 1);
setCount(prev => prev + 1);
\`\`\`

### Mistake 3: Using State for Derived Values

\`\`\`tsx
// ❌ WRONG
const [items, setItems] = useState([...]);
const [itemCount, setItemCount] = useState(items.length);
// itemCount can get out of sync

// ✅ CORRECT
const [items, setItems] = useState([...]);
const itemCount = items.length;  // Always in sync
\`\`\`

### Mistake 4: Initializing State from Props (Usually Wrong)

\`\`\`tsx
// ❌ WRONG (unless you explicitly want to "fork" the data)
function Component({ initialCount }: { initialCount: number }) {
  const [count, setCount] = useState(initialCount);
  // If parent changes initialCount, this component ignores it
  // State is only initialized once on first render
}

// ✅ CORRECT (if you want to stay synced with parent)
function Component({ count }: { count: number }) {
  // Use prop directly—no state needed
  return <div>{count}</div>;
}

// ✅ ALSO CORRECT (if you want a local copy that can diverge)
function Component({ initialCount }: { initialCount: number }) {
  const [count, setCount] = useState(initialCount);
  // Explicitly named "initial" to signal: "I'm forking this data"
  return <div>{count}</div>;
}
\`\`\`

## Real-World Examples

### Example 1: Toggle Switch

\`\`\`tsx
interface ToggleProps {
  label: string;
  defaultValue?: boolean;
  onChange?: (value: boolean) => void;
}

function Toggle({ label, defaultValue = false, onChange }: ToggleProps) {
  const [isOn, setIsOn] = useState(defaultValue);
  
  const handleToggle = () => {
    const newValue = !isOn;
    setIsOn(newValue);
    onChange?.(newValue);  // Notify parent
  };
  
  return (
    <button 
      onClick={handleToggle}
      className={\`toggle \${isOn ? 'toggle--on' : 'toggle--off'}\`}
      aria-pressed={isOn}
    >
      {label}: {isOn ? 'ON' : 'OFF'}
    </button>
  );
}

// Usage
<Toggle label="Notifications" onChange={value => console.log(value)} />
\`\`\`

### Example 2: Form Input with Validation

\`\`\`tsx
function EmailInput() {
  const [email, setEmail] = useState('');
  const [touched, setTouched] = useState(false);
  
  // Derived state (don't store in useState)
  const isValid = email.includes('@') && email.includes('.');
  const showError = touched && !isValid;
  
  return (
    <div>
      <input
        type="email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        onBlur={() => setTouched(true)}
        className={showError ? 'input--error' : ''}
      />
      {showError && <span className="error">Invalid email</span>}
    </div>
  );
}
\`\`\`

### Example 3: Counter with Min/Max

\`\`\`tsx
interface CounterProps {
  min?: number;
  max?: number;
  step?: number;
}

function Counter({ min = 0, max = 100, step = 1 }: CounterProps) {
  const [count, setCount] = useState(min);
  
  const increment = () => {
    setCount(prev => Math.min(prev + step, max));
  };
  
  const decrement = () => {
    setCount(prev => Math.max(prev - step, min));
  };
  
  return (
    <div>
      <button onClick={decrement} disabled={count <= min}>-</button>
      <span>{count}</span>
      <button onClick={increment} disabled={count >= max}>+</button>
    </div>
  );
}
\`\`\`

### Example 4: Shopping Cart

\`\`\`tsx
interface CartItem {
  id: number;
  name: string;
  price: number;
  quantity: number;
}

function ShoppingCart() {
  const [items, setItems] = useState<CartItem[]>([]);
  
  const addItem = (item: Omit<CartItem, 'quantity'>) => {
    setItems(prev => {
      const existing = prev.find(i => i.id === item.id);
      if (existing) {
        // Increment quantity
        return prev.map(i => 
          i.id === item.id 
            ? { ...i, quantity: i.quantity + 1 }
            : i
        );
      } else {
        // Add new item
        return [...prev, { ...item, quantity: 1 }];
      }
    });
  };
  
  const removeItem = (id: number) => {
    setItems(prev => prev.filter(item => item.id !== id));
  };
  
  const updateQuantity = (id: number, quantity: number) => {
    if (quantity <= 0) {
      removeItem(id);
    } else {
      setItems(prev => 
        prev.map(item => 
          item.id === id ? { ...item, quantity } : item
        )
      );
    }
  };
  
  // Derived state
  const total = items.reduce((sum, item) => 
    sum + item.price * item.quantity, 0
  );
  
  return (
    <div>
      {items.map(item => (
        <div key={item.id}>
          <span>{item.name}</span>
          <span>\${item.price}</span>
          <input 
            type="number"
            value={item.quantity}
            onChange={e => updateQuantity(item.id, Number(e.target.value))}
          />
          <button onClick={() => removeItem(item.id)}>Remove</button>
        </div>
      ))}
      <div>Total: \${total.toFixed(2)}</div>
    </div>
  );
}
\`\`\`

## Performance Considerations

**useState is fast**, but be aware of these patterns:

\`\`\`tsx
// ❌ BAD: Expensive calculation on every render
function Component() {
  const [data] = useState(expensiveCalculation());  // Runs every render!
}

// ✅ GOOD: Lazy initialization
function Component() {
  const [data] = useState(() => expensiveCalculation());  // Runs once
}

// ❌ BAD: Large objects re-created on every render
function Component() {
  const [config] = useState({
    // 1000 lines of config
  });  // This object is re-created every render
}

// ✅ GOOD: Define outside component or use useMemo
const DEFAULT_CONFIG = {
  // 1000 lines of config
};

function Component() {
  const [config] = useState(DEFAULT_CONFIG);
}
\`\`\`

## What's Next?

Now that you understand state, you're ready to learn about **Event Handling**—how users interact with your stateful components. You'll learn:
- Event handler patterns
- Passing data with events
- Event object properties
- Preventing default behavior
- Event delegation

State + Events = Interactivity. Let's make your components truly dynamic! 🚀
`,
};
