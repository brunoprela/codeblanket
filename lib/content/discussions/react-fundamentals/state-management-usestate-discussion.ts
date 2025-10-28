export const stateManagementUseStateDiscussion = [
  {
    id: 1,
    question:
      "A junior developer on your team writes: `const [items, setItems] = useState([]); items.push(newItem); setItems(items);` to add items to an array. The component doesn't re-render. They're confused because 'we called setItems, so why doesn't it update?' Explain what's wrong, why React doesn't detect the change, how React's reconciliation works with object references, and provide the correct solution with 5 different ways to properly update arrays in React state.",
    answer: `## Comprehensive Answer:

This is one of the **most common React mistakes**—directly mutating state. Let me explain why this doesn't work and how to fix it properly.

### What's Wrong: The Mutation Problem

\`\`\`tsx
// ❌ WRONG CODE
const [items, setItems] = useState([]);

const addItem = (newItem) => {
  items.push(newItem);  // MUTATES original array
  setItems(items);      // Passes SAME array reference
};

// Why it fails:
// 1. items.push() mutates the original array IN PLACE
// 2. setItems receives the SAME array reference
// 3. React compares: oldArray === newArray → TRUE (same reference)
// 4. React thinks: "No change detected, skip re-render"
// 5. UI doesn't update even though array contents changed
\`\`\`

### How React Detects Changes: Reference Equality

React uses **shallow comparison** (reference equality) to detect state changes for performance.

\`\`\`tsx
// React's internal comparison (simplified)
function shouldComponentUpdate(oldState, newState) {
  return oldState !== newState;  // Reference comparison, NOT deep equality
}

// Example:
const arr1 = [1, 2, 3];
const arr2 = arr1;
arr2.push(4);

console.log(arr1 === arr2);  // true (same reference)
console.log(arr1);           // [1, 2, 3, 4]
console.log(arr2);           // [1, 2, 3, 4]

// Both point to same memory location!
// React sees: oldState === newState → No update
\`\`\`

**Why React uses reference comparison**:
1. **Performance**: Checking references is O(1), deep equality is O(n)
2. **Predictability**: You control when updates happen
3. **Immutability**: Encourages immutable data patterns

### The Problem Visualized

\`\`\`
Memory Layout:

Initial State:
┌─────────────────────────┐
│ Component State         │
│ items: 0x1234 ────────┐ │
└─────────────────────────┘│
                           ▼
                    ┌──────────────┐
                    │ Array        │
                    │ [item1]      │
                    │ 0x1234       │
                    └──────────────┘

After items.push(item2):
┌─────────────────────────┐
│ Component State         │
│ items: 0x1234 ────────┐ │  ← SAME REFERENCE!
└─────────────────────────┘│
                           ▼
                    ┌──────────────┐
                    │ Array        │
                    │ [item1,      │  ← Contents changed
                    │  item2]      │  ← But reference didn't!
                    │ 0x1234       │
                    └──────────────┘

React comparison:
oldItems (0x1234) === newItems (0x1234) → TRUE
Conclusion: No change → No re-render ❌
\`\`\`

### Correct Solution: Create New Array

\`\`\`
Memory Layout with Spread Operator:

Initial State:
┌─────────────────────────┐
│ Component State         │
│ items: 0x1234 ────────┐ │
└─────────────────────────┘│
                           ▼
                    ┌──────────────┐
                    │ Array        │
                    │ [item1]      │
                    │ 0x1234       │
                    └──────────────┘

After setItems([...items, item2]):
┌─────────────────────────┐
│ Component State         │
│ items: 0x5678 ────────┐ │  ← NEW REFERENCE!
└─────────────────────────┘│
                           ▼
                    ┌──────────────┐
                    │ Array (NEW)  │
                    │ [item1,      │
                    │  item2]      │
                    │ 0x5678       │  ← Different memory location
                    └──────────────┘
                           
React comparison:
oldItems (0x1234) === newItems (0x5678) → FALSE
Conclusion: Changed → Re-render! ✅
\`\`\`

### 5 Correct Ways to Update Arrays

**Method 1: Spread Operator (Most Common)**

\`\`\`tsx
const [items, setItems] = useState<string[]>([]);

// Add to end
const addItem = (newItem: string) => {
  setItems([...items, newItem]);
  // Creates: new array with all old items + new item
};

// Add to beginning
const addItemAtStart = (newItem: string) => {
  setItems([newItem, ...items]);
};

// Add at specific position
const addItemAt = (newItem: string, index: number) => {
  setItems([
    ...items.slice(0, index),
    newItem,
    ...items.slice(index)
  ]);
};

// Real-world example: Todo list
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);
  
  const addTodo = (text: string) => {
    const newTodo: Todo = {
      id: Date.now(),
      text,
      completed: false
    };
    setItems([...todos, newTodo]);  // ✅ New array
  };
  
  return (
    <div>
      {todos.map(todo => (
        <div key={todo.id}>{todo.text}</div>
      ))}
    </div>
  );
}
\`\`\`

**Method 2: Array.concat() (Functional Programming Style)**

\`\`\`tsx
const [items, setItems] = useState<string[]>([]);

// Add single item
const addItem = (newItem: string) => {
  setItems(items.concat(newItem));
  // concat returns NEW array
};

// Add multiple items
const addMultipleItems = (newItems: string[]) => {
  setItems(items.concat(newItems));
};

// Why concat is good:
// - Doesn't mutate original (unlike push)
// - Returns new array automatically
// - Works with single items or arrays
// - Functional programming style

// Example: Batch adding
function BatchAdd() {
  const [users, setUsers] = useState<User[]>([]);
  
  const importUsers = async () => {
    const response = await fetch('/api/users');
    const newUsers = await response.json();
    setUsers(users.concat(newUsers));  // Add entire batch
  };
}
\`\`\`

**Method 3: Array.filter() (For Removing Items)**

\`\`\`tsx
const [items, setItems] = useState<Todo[]>([
  { id: 1, text: 'Buy milk', completed: false },
  { id: 2, text: 'Walk dog', completed: true },
  { id: 3, text: 'Read book', completed: false }
]);

// Remove by ID
const removeItem = (idToRemove: number) => {
  setItems(items.filter(item => item.id !== idToRemove));
  // filter creates NEW array with items that pass test
};

// Remove by condition
const removeCompleted = () => {
  setItems(items.filter(item => !item.completed));
};

// Remove first occurrence
const removeFirst = (text: string) => {
  const index = items.findIndex(item => item.text === text);
  if (index !== -1) {
    setItems(items.filter((_, i) => i !== index));
  }
};

// Real-world: Delete button
function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([...]);
  
  return (
    <div>
      {todos.map(todo => (
        <div key={todo.id}>
          <span>{todo.text}</span>
          <button onClick={() => {
            setTodos(todos.filter(t => t.id !== todo.id));
          }}>
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}
\`\`\`

**Method 4: Array.map() (For Updating Items)**

\`\`\`tsx
const [items, setItems] = useState<Todo[]>([...]);

// Update single item
const toggleTodo = (idToToggle: number) => {
  setItems(items.map(item => 
    item.id === idToToggle 
      ? { ...item, completed: !item.completed }  // Update this one
      : item                                      // Keep others same
  ));
  // map creates NEW array with transformed items
};

// Update multiple items
const markAllCompleted = () => {
  setItems(items.map(item => ({ ...item, completed: true })));
};

// Update with complex logic
const updateTodoText = (id: number, newText: string) => {
  setItems(items.map(item => {
    if (item.id === id) {
      return {
        ...item,
        text: newText,
        updatedAt: new Date()  // Add timestamp
      };
    }
    return item;
  }));
};

// Real-world: Toggle checkbox
function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([...]);
  
  return (
    <div>
      {todos.map(todo => (
        <div key={todo.id}>
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() => {
              setTodos(todos.map(t => 
                t.id === todo.id 
                  ? { ...t, completed: !t.completed }
                  : t
              ));
            }}
          />
          <span>{todo.text}</span>
        </div>
      ))}
    </div>
  );
}
\`\`\`

**Method 5: Functional Updates (For State Based on Previous State)**

\`\`\`tsx
const [items, setItems] = useState<string[]>([]);

// When you need LATEST state (multiple rapid updates)
const addItem = (newItem: string) => {
  setItems(prevItems => [...prevItems, newItem]);
  // prevItems is guaranteed to be latest value
};

// Example: Multiple rapid updates
const addMultipleItems = () => {
  setItems(prev => [...prev, 'Item 1']);
  setItems(prev => [...prev, 'Item 2']);
  setItems(prev => [...prev, 'Item 3']);
  // All three updates work correctly
  // Result: ['Item 1', 'Item 2', 'Item 3']
};

// Without functional updates (WRONG):
const addMultipleItemsWrong = () => {
  setItems([...items, 'Item 1']);  // items = []
  setItems([...items, 'Item 2']);  // items = [] (still!)
  setItems([...items, 'Item 3']);  // items = [] (still!)
  // All three read OLD items value
  // Result: Only ['Item 3'] (last one wins)
};

// Real-world: Batch operations
function BatchOperations() {
  const [todos, setTodos] = useState<Todo[]>([]);
  
  const importTodos = (newTodos: Todo[]) => {
    setTodos(prev => {
      // Deduplicate by ID
      const existingIds = new Set(prev.map(t => t.id));
      const uniqueNew = newTodos.filter(t => !existingIds.has(t.id));
      return [...prev, ...uniqueNew];
    });
  };
  
  const clearCompleted = () => {
    setTodos(prev => prev.filter(t => !t.completed));
  };
}
\`\`\`

### Complete Working Example: Todo App

\`\`\`tsx
import { useState } from 'react';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [inputText, setInputText] = useState('');
  
  // ✅ Add todo (Method 1: Spread)
  const addTodo = () => {
    if (!inputText.trim()) return;
    
    const newTodo: Todo = {
      id: Date.now(),
      text: inputText,
      completed: false
    };
    
    setTodos([...todos, newTodo]);  // NEW array
    setInputText('');
  };
  
  // ✅ Remove todo (Method 3: Filter)
  const removeTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));  // NEW array
  };
  
  // ✅ Toggle todo (Method 4: Map)
  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, completed: !todo.completed }  // NEW object
        : todo
    ));  // NEW array
  };
  
  // ✅ Update todo text (Method 4: Map)
  const updateTodoText = (id: number, newText: string) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, text: newText }
        : todo
    ));
  };
  
  // ✅ Clear completed (Method 3: Filter)
  const clearCompleted = () => {
    setTodos(todos.filter(todo => !todo.completed));
  };
  
  // ✅ Mark all completed (Method 4: Map)
  const markAllCompleted = () => {
    setTodos(todos.map(todo => ({ ...todo, completed: true })));
  };
  
  return (
    <div>
      <h1>Todo List</h1>
      
      <div>
        <input
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyPress={e => e.key === 'Enter' && addTodo()}
          placeholder="Add todo..."
        />
        <button onClick={addTodo}>Add</button>
      </div>
      
      <div>
        <button onClick={markAllCompleted}>Complete All</button>
        <button onClick={clearCompleted}>Clear Completed</button>
      </div>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span style={{ 
              textDecoration: todo.completed ? 'line-through' : 'none' 
            }}>
              {todo.text}
            </span>
            <button onClick={() => removeTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
      
      <p>
        {todos.filter(t => !t.completed).length} items left
      </p>
    </div>
  );
}
\`\`\`

### Why Immutability Matters: Performance Benefits

\`\`\`tsx
// With immutability + React.memo
const TodoItem = React.memo(({ todo, onToggle, onDelete }) => {
  console.log('TodoItem rendered:', todo.id);
  
  return (
    <div>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={onToggle}
      />
      <span>{todo.text}</span>
      <button onClick={onDelete}>Delete</button>
    </div>
  );
});

function TodoList() {
  const [todos, setTodos] = useState([
    { id: 1, text: 'Todo 1', completed: false },
    { id: 2, text: 'Todo 2', completed: false },
    { id: 3, text: 'Todo 3', completed: false }
  ]);
  
  const toggleTodo = (id) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, completed: !todo.completed }  // NEW object for this todo
        : todo                                      // SAME object for others
    ));
  };
  
  return (
    <>
      {todos.map(todo => (
        <TodoItem
          key={todo.id}
          todo={todo}
          onToggle={() => toggleTodo(todo.id)}
        />
      ))}
    </>
  );
}

// When you toggle todo 1:
// Console output:
// "TodoItem rendered: 1"  ← Only todo 1 re-renders!
// Todos 2 and 3 don't re-render because their object references didn't change

// With mutation, ALL todos would re-render (slower)
\`\`\`

### Common Follow-Up Questions

**Q: "Why can't React just check the array contents?"**

**A**: Deep equality checks are expensive:

\`\`\`tsx
// Shallow comparison (React's default): O(1)
oldArray === newArray  // Instant

// Deep equality: O(n) where n = array length
function deepEqual(arr1, arr2) {
  if (arr1.length !== arr2.length) return false;
  for (let i = 0; i < arr1.length; i++) {
    if (arr1[i] !== arr2[i]) return false;  // For objects: another O(n)
  }
  return true;
}

// For 1000 items: 1000 comparisons per update
// For nested objects: Even worse (O(n²) or worse)

// Real-world impact:
// - Shallow: 0.001ms
// - Deep: 10ms for large arrays
// React prefers fast + predictable
\`\`\`

**Q: "What if I need to sort the array?"**

**A**: Create a copy first:

\`\`\`tsx
// ❌ WRONG: sort() mutates
const sortItems = () => {
  items.sort();  // Mutates original
  setItems(items);  // Same reference
};

// ✅ CORRECT: Copy, then sort
const sortItems = () => {
  const sorted = [...items].sort();  // Create copy, then sort
  setItems(sorted);  // New reference
};

// Or use functional update:
const sortItems = () => {
  setItems(prev => [...prev].sort());
};
\`\`\`

**Q: "Isn't creating new arrays expensive?"**

**A**: No, it's actually fast:

\`\`\`tsx
// Performance comparison (1000 items):

// Mutation (seems faster):
items.push(newItem);  // 0.001ms
// But: Breaks React's optimization, forces re-render of entire tree

// Immutable update:
const newItems = [...items, newItem];  // 0.01ms (10x slower)
// But: Enables React.memo, only updates what changed

// Net result: Immutable is 5-10x FASTER at scale
// Because it enables React's optimization
\`\`\`

### What to Tell the Junior Developer

"Great question! The issue is **mutation vs immutability**. Here's what's happening:

**Your code:**
\`\`\`tsx
items.push(newItem);  // Mutates array IN PLACE
setItems(items);      // Passes SAME array reference
\`\`\`

**Problem**: React compares references (oldArray === newArray). Since you mutated the SAME array, React sees no change.

**Fix**: Create a NEW array:
\`\`\`tsx
setItems([...items, newItem]);  // NEW array with added item
\`\`\`

**Why it works**: New array → different reference → React detects change → re-renders.

**Rule**: Never mutate state directly. Always create new objects/arrays.

**5 patterns to remember:**
1. Add: \`[...items, newItem]\`
2. Remove: \`items.filter(i => i.id !== id)\`
3. Update: \`items.map(i => i.id === id ? { ...i, text: newText } : i)\`
4. Sort: \`[...items].sort()\`
5. Functional: \`setItems(prev => [...prev, newItem])\`

Try the fixed version and let me know if you have questions!"

### Key Takeaways

1. **React uses reference equality** (===) for performance
2. **Mutation breaks React** because reference doesn't change
3. **Always create new arrays/objects** when updating state
4. **Spread operator** is your friend: \`[...items, newItem]\`
5. **Immutability enables optimizations** (React.memo, PureComponent)
6. **5 essential patterns**: spread, concat, filter, map, functional updates

This is one of React's most important concepts. Understand this, and you'll avoid dozens of bugs!
`,
  },
  {
    id: 2,
    question:
      "Your React application has a form with 20 fields (name, email, address, phone, etc.). A developer creates 20 separate useState calls: `const [name, setName] = useState(''); const [email, setEmail] = useState('');` ... (repeated 20 times). Another developer suggests: 'Use one useState with an object: `const [formData, setFormData] = useState({ name: '', email: '', ... })`.` Which approach is better and why? Discuss performance implications, code maintainability, validation complexity, and provide a production-ready form implementation with proper TypeScript types.",
    answer: `## Comprehensive Answer:

This is a classic React architecture decision. The answer: **It depends on your use case, but for forms, a single state object is usually better**. Let me explain why with detailed analysis.

### The Two Approaches

**Approach 1: Multiple useState Calls (20 separate states)**

\`\`\`tsx
function UserForm() {
  // 20 separate state variables
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [street, setStreet] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [zipCode, setZipCode] = useState('');
  const [country, setCountry] = useState('');
  const [dateOfBirth, setDateOfBirth] = useState('');
  const [gender, setGender] = useState('');
  const [occupation, setOccupation] = useState('');
  const [company, setCompany] = useState('');
  const [website, setWebsite] = useState('');
  const [linkedin, setLinkedin] = useState('');
  const [twitter, setTwitter] = useState('');
  const [bio, setBio] = useState('');
  const [newsletter, setNewsletter] = useState(false);
  const [terms, setTerms] = useState(false);
  const [privacy, setPrivacy] = useState(false);
  
  // 20 separate onChange handlers
  // ... gets verbose quickly
}
\`\`\`

**Approach 2: Single useState with Object**

\`\`\`tsx
interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  address: {
    street: string;
    city: string;
    state: string;
    zipCode: string;
    country: string;
  };
  dateOfBirth: string;
  gender: string;
  occupation: string;
  company: string;
  website: string;
  social: {
    linkedin: string;
    twitter: string;
  };
  bio: string;
  preferences: {
    newsletter: boolean;
    terms: boolean;
    privacy: boolean;
  };
}

function UserForm() {
  const [formData, setFormData] = useState<FormData>({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    address: {
      street: '',
      city: '',
      state: '',
      zipCode: '',
      country: ''
    },
    dateOfBirth: '',
    gender: '',
    occupation: '',
    company: '',
    website: '',
    social: {
      linkedin: '',
      twitter: ''
    },
    bio: '',
    preferences: {
      newsletter: false,
      terms: false,
      privacy: false
    }
  });
  
  // Single reusable handler
  // ... much cleaner
}
\`\`\`

### Detailed Comparison

| Aspect | Multiple useState | Single useState Object | Winner |
|--------|------------------|----------------------|--------|
| **Lines of code** | 60+ lines (20 states + 20 handlers) | 20 lines (1 state + 1 handler) | Single ✅ |
| **Re-renders** | 1 per field change | 1 per field change | Tie |
| **Type safety** | Hard to group related data | Easy with TypeScript interface | Single ✅ |
| **Validation** | Complex (20 separate validators) | Centralized validation logic | Single ✅ |
| **Form submission** | Gather 20 variables into object | Already an object | Single ✅ |
| **Reset form** | 20 setter calls | 1 setter call | Single ✅ |
| **Partial updates** | Simple (one setter) | Requires spread operator | Multiple ✅ |
| **Debugging** | Each state in DevTools | One nested object | Tie |

### Performance Analysis

**Myth: "Single object causes more re-renders"**

**Reality: Both trigger the same number of re-renders**

\`\`\`tsx
// Multiple useState: Typing in firstName
setFirstName('Alice');  // 1 re-render

// Single useState: Typing in firstName
setFormData({ ...formData, firstName: 'Alice' });  // 1 re-render

// Performance is IDENTICAL
// React batches state updates automatically (React 18+)
\`\`\`

**Actual performance consideration: Object spreading**

\`\`\`tsx
// Multiple useState: O(1) update
setFirstName('Alice');  // Instant

// Single useState: O(n) update
setFormData({ ...formData, firstName: 'Alice' });  
// Spreads all 20 properties (nanoseconds for 20 fields)

// Performance difference: 0.001ms (negligible)
// Even with 100 fields: 0.01ms (still negligible)
\`\`\`

**Benchmark:**

\`\`\`tsx
// Test: Update firstName 10,000 times

// Multiple useState:
console.time('multiple');
for (let i = 0; i < 10000; i++) {
  setFirstName(\`Name \${i}\`);
}
console.timeEnd('multiple');
// Result: ~100ms

// Single useState:
console.time('single');
for (let i = 0; i < 10000; i++) {
  setFormData(prev => ({ ...prev, firstName: \`Name \${i}\` }));
}
console.timeEnd('single');
// Result: ~105ms

// Difference: 5ms over 10,000 updates
// Per update: 0.0005ms (unnoticeable)
\`\`\`

**Conclusion**: Performance difference is negligible. Choose based on **code quality**, not performance.

### Production-Ready Implementation: Single State Object

\`\`\`tsx
import { useState, FormEvent, ChangeEvent } from 'react';

// Type definitions
interface Address {
  street: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

interface SocialLinks {
  linkedin: string;
  twitter: string;
  github: string;
}

interface Preferences {
  newsletter: boolean;
  terms: boolean;
  privacy: boolean;
}

interface FormData {
  // Personal info
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  gender: 'male' | 'female' | 'other' | '';
  
  // Address
  address: Address;
  
  // Professional
  occupation: string;
  company: string;
  website: string;
  
  // Social
  social: SocialLinks;
  
  // Additional
  bio: string;
  
  // Preferences
  preferences: Preferences;
}

interface FormErrors {
  [key: string]: string;
}

const INITIAL_FORM_DATA: FormData = {
  firstName: '',
  lastName: '',
  email: '',
  phone: '',
  dateOfBirth: '',
  gender: '',
  address: {
    street: '',
    city: '',
    state: '',
    zipCode: '',
    country: 'US'
  },
  occupation: '',
  company: '',
  website: '',
  social: {
    linkedin: '',
    twitter: '',
    github: ''
  },
  bio: '',
  preferences: {
    newsletter: false,
    terms: false,
    privacy: false
  }
};

function UserForm() {
  const [formData, setFormData] = useState<FormData>(INITIAL_FORM_DATA);
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Set<string>>(new Set());
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Generic change handler for all fields
  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    
    // Handle checkboxes
    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      
      // Handle nested preferences
      if (name.startsWith('preferences.')) {
        const prefKey = name.split('.')[1] as keyof Preferences;
        setFormData(prev => ({
          ...prev,
          preferences: {
            ...prev.preferences,
            [prefKey]: checked
          }
        }));
      }
      return;
    }
    
    // Handle nested address fields
    if (name.startsWith('address.')) {
      const addressKey = name.split('.')[1] as keyof Address;
      setFormData(prev => ({
        ...prev,
        address: {
          ...prev.address,
          [addressKey]: value
        }
      }));
      return;
    }
    
    // Handle nested social fields
    if (name.startsWith('social.')) {
      const socialKey = name.split('.')[1] as keyof SocialLinks;
      setFormData(prev => ({
        ...prev,
        social: {
          ...prev.social,
          [socialKey]: value
        }
      }));
      return;
    }
    
    // Handle top-level fields
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Track which fields have been touched (for validation)
  const handleBlur = (fieldName: string) => {
    setTouched(prev => new Set(prev).add(fieldName));
    validateField(fieldName);
  };
  
  // Validate individual field
  const validateField = (fieldName: string) => {
    const newErrors = { ...errors };
    
    switch (fieldName) {
      case 'firstName':
        if (!formData.firstName.trim()) {
          newErrors.firstName = 'First name is required';
        } else if (formData.firstName.length < 2) {
          newErrors.firstName = 'First name must be at least 2 characters';
        } else {
          delete newErrors.firstName;
        }
        break;
        
      case 'email':
        const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
        if (!formData.email) {
          newErrors.email = 'Email is required';
        } else if (!emailRegex.test(formData.email)) {
          newErrors.email = 'Invalid email format';
        } else {
          delete newErrors.email;
        }
        break;
        
      case 'phone':
        const phoneRegex = /^\\+?[1-9]\\d{1,14}$/;
        if (formData.phone && !phoneRegex.test(formData.phone.replace(/[\\s-()]/g, ''))) {
          newErrors.phone = 'Invalid phone number';
        } else {
          delete newErrors.phone;
        }
        break;
        
      case 'website':
        const urlRegex = /^https?:\\/\\/(www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b/;
        if (formData.website && !urlRegex.test(formData.website)) {
          newErrors.website = 'Invalid website URL';
        } else {
          delete newErrors.website;
        }
        break;
        
      case 'preferences.terms':
        if (!formData.preferences.terms) {
          newErrors['preferences.terms'] = 'You must accept the terms';
        } else {
          delete newErrors['preferences.terms'];
        }
        break;
    }
    
    setErrors(newErrors);
  };
  
  // Validate entire form
  const validateForm = (): boolean => {
    const fields = [
      'firstName',
      'lastName',
      'email',
      'phone',
      'website',
      'preferences.terms'
    ];
    
    fields.forEach(validateField);
    
    return Object.keys(errors).length === 0;
  };
  
  // Handle form submission
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      console.log('Form has errors:', errors);
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit form');
      }
      
      const result = await response.json();
      console.log('Form submitted successfully:', result);
      
      // Reset form
      handleReset();
      
      alert('Profile created successfully!');
    } catch (error) {
      console.error('Error submitting form:', error);
      alert('Failed to create profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Reset form to initial state
  const handleReset = () => {
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
    setTouched(new Set());
  };
  
  // Helper to check if field has error and was touched
  const showError = (fieldName: string) => {
    return touched.has(fieldName) && errors[fieldName];
  };
  
  return (
    <form onSubmit={handleSubmit} className="user-form">
      <h2>User Profile</h2>
      
      {/* Personal Information */}
      <fieldset>
        <legend>Personal Information</legend>
        
        <div className="form-group">
          <label htmlFor="firstName">
            First Name <span className="required">*</span>
          </label>
          <input
            id="firstName"
            name="firstName"
            type="text"
            value={formData.firstName}
            onChange={handleChange}
            onBlur={() => handleBlur('firstName')}
            className={showError('firstName') ? 'error' : ''}
            required
          />
          {showError('firstName') && (
            <span className="error-message">{errors.firstName}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="lastName">
            Last Name <span className="required">*</span>
          </label>
          <input
            id="lastName"
            name="lastName"
            type="text"
            value={formData.lastName}
            onChange={handleChange}
            onBlur={() => handleBlur('lastName')}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="email">
            Email <span className="required">*</span>
          </label>
          <input
            id="email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            onBlur={() => handleBlur('email')}
            className={showError('email') ? 'error' : ''}
            required
          />
          {showError('email') && (
            <span className="error-message">{errors.email}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="phone">Phone</label>
          <input
            id="phone"
            name="phone"
            type="tel"
            value={formData.phone}
            onChange={handleChange}
            onBlur={() => handleBlur('phone')}
            placeholder="+1234567890"
          />
          {showError('phone') && (
            <span className="error-message">{errors.phone}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="dateOfBirth">Date of Birth</label>
          <input
            id="dateOfBirth"
            name="dateOfBirth"
            type="date"
            value={formData.dateOfBirth}
            onChange={handleChange}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="gender">Gender</label>
          <select
            id="gender"
            name="gender"
            value={formData.gender}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
      </fieldset>
      
      {/* Address */}
      <fieldset>
        <legend>Address</legend>
        
        <div className="form-group">
          <label htmlFor="address.street">Street</label>
          <input
            id="address.street"
            name="address.street"
            type="text"
            value={formData.address.street}
            onChange={handleChange}
          />
        </div>
        
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="address.city">City</label>
            <input
              id="address.city"
              name="address.city"
              type="text"
              value={formData.address.city}
              onChange={handleChange}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="address.state">State</label>
            <input
              id="address.state"
              name="address.state"
              type="text"
              value={formData.address.state}
              onChange={handleChange}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="address.zipCode">ZIP Code</label>
            <input
              id="address.zipCode"
              name="address.zipCode"
              type="text"
              value={formData.address.zipCode}
              onChange={handleChange}
            />
          </div>
        </div>
      </fieldset>
      
      {/* Professional Information */}
      <fieldset>
        <legend>Professional Information</legend>
        
        <div className="form-group">
          <label htmlFor="occupation">Occupation</label>
          <input
            id="occupation"
            name="occupation"
            type="text"
            value={formData.occupation}
            onChange={handleChange}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="company">Company</label>
          <input
            id="company"
            name="company"
            type="text"
            value={formData.company}
            onChange={handleChange}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="website">Website</label>
          <input
            id="website"
            name="website"
            type="url"
            value={formData.website}
            onChange={handleChange}
            onBlur={() => handleBlur('website')}
            placeholder="https://example.com"
          />
          {showError('website') && (
            <span className="error-message">{errors.website}</span>
          )}
        </div>
      </fieldset>
      
      {/* Social Links */}
      <fieldset>
        <legend>Social Media</legend>
        
        <div className="form-group">
          <label htmlFor="social.linkedin">LinkedIn</label>
          <input
            id="social.linkedin"
            name="social.linkedin"
            type="url"
            value={formData.social.linkedin}
            onChange={handleChange}
            placeholder="https://linkedin.com/in/username"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="social.twitter">Twitter</label>
          <input
            id="social.twitter"
            name="social.twitter"
            type="url"
            value={formData.social.twitter}
            onChange={handleChange}
            placeholder="https://twitter.com/username"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="social.github">GitHub</label>
          <input
            id="social.github"
            name="social.github"
            type="url"
            value={formData.social.github}
            onChange={handleChange}
            placeholder="https://github.com/username"
          />
        </div>
      </fieldset>
      
      {/* Bio */}
      <div className="form-group">
        <label htmlFor="bio">Bio</label>
        <textarea
          id="bio"
          name="bio"
          value={formData.bio}
          onChange={handleChange}
          rows={4}
          maxLength={500}
        />
        <span className="character-count">
          {formData.bio.length}/500 characters
        </span>
      </div>
      
      {/* Preferences */}
      <fieldset>
        <legend>Preferences</legend>
        
        <div className="form-group checkbox">
          <input
            id="preferences.newsletter"
            name="preferences.newsletter"
            type="checkbox"
            checked={formData.preferences.newsletter}
            onChange={handleChange}
          />
          <label htmlFor="preferences.newsletter">
            Subscribe to newsletter
          </label>
        </div>
        
        <div className="form-group checkbox">
          <input
            id="preferences.terms"
            name="preferences.terms"
            type="checkbox"
            checked={formData.preferences.terms}
            onChange={handleChange}
            required
          />
          <label htmlFor="preferences.terms">
            I accept the terms and conditions <span className="required">*</span>
          </label>
          {showError('preferences.terms') && (
            <span className="error-message">{errors['preferences.terms']}</span>
          )}
        </div>
        
        <div className="form-group checkbox">
          <input
            id="preferences.privacy"
            name="preferences.privacy"
            type="checkbox"
            checked={formData.preferences.privacy}
            onChange={handleChange}
            required
          />
          <label htmlFor="preferences.privacy">
            I accept the privacy policy <span className="required">*</span>
          </label>
        </div>
      </fieldset>
      
      {/* Form Actions */}
      <div className="form-actions">
        <button
          type="button"
          onClick={handleReset}
          disabled={isSubmitting}
        >
          Reset
        </button>
        <button
          type="submit"
          disabled={isSubmitting || Object.keys(errors).length > 0}
        >
          {isSubmitting ? 'Submitting...' : 'Submit'}
        </button>
      </div>
      
      {/* Debug Info (remove in production) */}
      {process.env.NODE_ENV === 'development' && (
        <details style={{ marginTop: '20px' }}>
          <summary>Debug Info</summary>
          <pre>{JSON.stringify(formData, null, 2)}</pre>
          <pre>Errors: {JSON.stringify(errors, null, 2)}</pre>
        </details>
      )}
    </form>
  );
}

export default UserForm;
\`\`\`

### Benefits of Single State Object Approach

**1. Easier form submission**

\`\`\`tsx
// Multiple useState: Gather into object
const handleSubmit = () => {
  const data = {
    firstName,
    lastName,
    email,
    phone,
    // ... 20 fields
  };
  await api.post('/users', data);
};

// Single useState: Already an object
const handleSubmit = () => {
  await api.post('/users', formData);  // Done!
};
\`\`\`

**2. Easier form reset**

\`\`\`tsx
// Multiple useState: Reset each field
const handleReset = () => {
  setFirstName('');
  setLastName('');
  setEmail('');
  // ... 20 calls
};

// Single useState: One call
const handleReset = () => {
  setFormData(INITIAL_FORM_DATA);
};
\`\`\`

**3. Type safety**

\`\`\`tsx
// Multiple useState: No grouping
const firstName: string;
const lastName: string;
// Hard to see relationship

// Single useState: Clear structure
interface FormData {
  firstName: string;
  lastName: string;
  // Grouped logically
}
\`\`\`

**4. Validation**

\`\`\`tsx
// Multiple useState: Scattered validation
const validateFirstName = () => { ... };
const validateLastName = () => { ... };
// ... 20 validators

// Single useState: Centralized
const validateForm = (data: FormData) => {
  // All validation in one place
};
\`\`\`

### When to Use Multiple useState

**Use multiple useState when:**

1. **Unrelated data** (not a form)
\`\`\`tsx
const [theme, setTheme] = useState('light');
const [user, setUser] = useState(null);
const [isLoading, setIsLoading] = useState(false);
// These are independent—don't group
\`\`\`

2. **Different update frequencies**
\`\`\`tsx
const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });  // Updates 60fps
const [userName, setUserName] = useState('');  // Updates rarely
// Don't group—mouse updates would re-render user input
\`\`\`

3. **Simple components**
\`\`\`tsx
const [count, setCount] = useState(0);
const [isOpen, setIsOpen] = useState(false);
// 2-3 fields? Multiple useState is fine
\`\`\`

### Recommendation for 20-Field Form

**Use single useState object** because:
- ✅ **90% less code** (1 handler vs 20)
- ✅ **Type-safe** (TypeScript interface)
- ✅ **Easier validation** (centralized logic)
- ✅ **Easier submission** (already an object)
- ✅ **Easier reset** (one call)
- ✅ **Better organization** (logical grouping)
- ✅ **Performance identical** (same re-renders)

The spread operator overhead (\`{ ...formData, field: value }\`) is negligible (< 0.001ms per update). Code quality wins.

**Final answer**: For forms with 5+ fields, use single state object. For unrelated data, use multiple useState. Your second developer is correct! 🎯
`,
  },
  {
    id: 3,
    question:
      "During code review, you see: `const [count, setCount] = useState(0); setCount(count + 1); setCount(count + 1); setCount(count + 1);` and the developer expects count to be 3, but it's 1. They don't understand why. Explain React's batching behavior, the difference between React 17 and React 18 batching, how the event loop affects state updates, why functional updates solve this problem, and provide examples of scenarios where batching helps vs hurts performance.",
    answer: `## Comprehensive Answer:

This is a fundamental React concept that confuses many developers: **React batches state updates for performance, but this creates a "stale closure" problem**. Let me explain thoroughly.

### The Problem: Stale Closures

\`\`\`tsx
function Counter() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    setCount(count + 1);  // count = 0, sets to 1
    setCount(count + 1);  // count = 0 (still!), sets to 1
    setCount(count + 1);  // count = 0 (still!), sets to 1
    
    // Result: count becomes 1, not 3
    console.log(count);  // Logs 0 (state hasn't updated yet!)
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={handleClick}>Increment 3x</button>
    </div>
  );
}
\`\`\`

**Why it happens:**

1. **JavaScript closure**: All three \`setCount\` calls capture the SAME \`count\` value (0)
2. **Batching**: React queues all three updates
3. **Last update wins**: All three try to set count to 1, so final result is 1

### Visualizing the Problem

\`\`\`
Initial State:
count = 0

Button Click Event:
┌─────────────────────────────────────┐
│ handleClick() executes              │
│                                     │
│ Line 1: setCount(count + 1)        │
│         count is 0                  │
│         Queues: setState(1)         │
│                                     │
│ Line 2: setCount(count + 1)        │
│         count is STILL 0!           │
│         Queues: setState(1)         │
│                                     │
│ Line 3: setCount(count + 1)        │
│         count is STILL 0!           │
│         Queues: setState(1)         │
└─────────────────────────────────────┘

React's Update Queue:
[ setState(1), setState(1), setState(1) ]
      ↓           ↓           ↓
   All set      Same       Last one
   to 1!        value!     wins!

Final State:
count = 1 (not 3!)
\`\`\`

### The Solution: Functional Updates

\`\`\`tsx
function Counter() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    setCount(prevCount => prevCount + 1);  // 0 + 1 = 1
    setCount(prevCount => prevCount + 1);  // 1 + 1 = 2
    setCount(prevCount => prevCount + 1);  // 2 + 1 = 3
    
    // Result: count becomes 3 ✓
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={handleClick}>Increment 3x</button>
    </div>
  );
}
\`\`\`

**How it works:**

\`\`\`
Button Click Event:
┌──────────────────────────────────────────────────┐
│ handleClick() executes                           │
│                                                  │
│ Line 1: setCount(prev => prev + 1)             │
│         Queues: updateFunction1                 │
│                                                  │
│ Line 2: setCount(prev => prev + 1)             │
│         Queues: updateFunction2                 │
│                                                  │
│ Line 3: setCount(prev => prev + 1)             │
│         Queues: updateFunction3                 │
└──────────────────────────────────────────────────┘

React Processes Queue:
Step 1: currentState = 0
        updateFunction1(0) = 0 + 1 = 1

Step 2: currentState = 1
        updateFunction2(1) = 1 + 1 = 2

Step 3: currentState = 2
        updateFunction3(2) = 2 + 1 = 3

Final State:
count = 3 ✓
\`\`\`

### React's Batching Behavior

**React 17 (Old Batching):**

\`\`\`tsx
// React 17: Only batches in event handlers

function Component() {
  const [count, setCount] = useState(0);
  
  // ✅ Batched (1 re-render)
  const handleClick = () => {
    setCount(1);
    setCount(2);
    setCount(3);
    // Only re-renders once with final value (3)
  };
  
  // ❌ NOT batched (3 re-renders!)
  const fetchData = async () => {
    const data = await fetch('/api/data');
    setCount(1);  // Re-render 1
    setCount(2);  // Re-render 2
    setCount(3);  // Re-render 3
    // Re-renders 3 times (inefficient!)
  };
  
  // ❌ NOT batched (3 re-renders!)
  useEffect(() => {
    setTimeout(() => {
      setCount(1);  // Re-render 1
      setCount(2);  // Re-render 2
      setCount(3);  // Re-render 3
    }, 1000);
  }, []);
}
\`\`\`

**React 18 (Automatic Batching):**

\`\`\`tsx
// React 18: Batches EVERYWHERE

function Component() {
  const [count, setCount] = useState(0);
  
  // ✅ Batched (1 re-render)
  const handleClick = () => {
    setCount(1);
    setCount(2);
    setCount(3);
  };
  
  // ✅ NOW batched! (1 re-render)
  const fetchData = async () => {
    const data = await fetch('/api/data');
    setCount(1);
    setCount(2);
    setCount(3);
    // Only re-renders once!
  };
  
  // ✅ NOW batched! (1 re-render)
  useEffect(() => {
    setTimeout(() => {
      setCount(1);
      setCount(2);
      setCount(3);
    }, 1000);
  }, []);
  
  // ✅ NOW batched! (1 re-render)
  const handleNativeEvent = () => {
    document.addEventListener('scroll', () => {
      setCount(1);
      setCount(2);
      setCount(3);
    });
  };
}
\`\`\`

**Key difference**: React 18 batches updates in **promises, timeouts, native events**—everywhere!

### Event Loop and State Updates

**How React batching works with the event loop:**

\`\`\`tsx
function Component() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    console.log('1. Start of click handler, count:', count);
    
    setCount(count + 1);
    console.log('2. After first setState, count:', count);  // Still 0!
    
    setCount(count + 1);
    console.log('3. After second setState, count:', count);  // Still 0!
    
    setCount(count + 1);
    console.log('4. After third setState, count:', count);  // Still 0!
    
    console.log('5. End of click handler');
  };
  
  console.log('6. Component rendering, count:', count);
  
  return <button onClick={handleClick}>Click</button>;
}

// Console output when you click:
// 6. Component rendering, count: 0  (Initial render)
// 1. Start of click handler, count: 0
// 2. After first setState, count: 0  (Not updated yet!)
// 3. After second setState, count: 0  (Not updated yet!)
// 4. After third setState, count: 0  (Not updated yet!)
// 5. End of click handler
// 6. Component rendering, count: 1  (Re-render with batched result)
\`\`\`

**Event loop flow:**

\`\`\`
JavaScript Event Loop:
┌──────────────────────────────────────┐
│ 1. Run synchronous code              │
│    - handleClick() executes          │
│    - 3x setCount() queued            │
│    - count is still 0                │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 2. Event handler completes           │
│    React says: "Time to process      │
│    queued state updates"             │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 3. React batches updates             │
│    - Merges 3 setCount calls         │
│    - Calculates final state: 1       │
│    - Triggers ONE re-render          │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 4. Component re-renders              │
│    - count is now 1                  │
│    - DOM updates                     │
└──────────────────────────────────────┘
\`\`\`

### When Batching Helps Performance

**Scenario 1: Multiple state updates in one function**

\`\`\`tsx
function UserProfile() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState(0);
  
  // ✅ WITHOUT batching: 3 re-renders (slow!)
  // ✅ WITH batching: 1 re-render (fast!)
  const handleFormSubmit = async () => {
    const response = await fetch('/api/user');
    const data = await response.json();
    
    setName(data.name);    // Would trigger re-render without batching
    setEmail(data.email);  // Would trigger re-render without batching
    setAge(data.age);      // Would trigger re-render without batching
    
    // With React 18: Only 1 re-render!
    // Performance gain: 3x faster
  };
}
\`\`\`

**Scenario 2: Rapid user interactions**

\`\`\`tsx
function SearchBar() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSearch = async (q: string) => {
    setQuery(q);
    setIsLoading(true);
    setResults([]);  // Clear old results
    
    // Without batching: 3 re-renders
    // With batching: 1 re-render
    
    const data = await fetch(\`/api/search?q=\${q}\`);
    const newResults = await data.json();
    
    setResults(newResults);
    setIsLoading(false);
    
    // Without batching: 2 more re-renders (5 total!)
    // With batching: 1 more re-render (2 total)
  };
}
\`\`\`

**Performance impact:**
- Without batching: 5 re-renders × 10ms = 50ms
- With batching: 2 re-renders × 10ms = 20ms
- **Improvement: 60% faster**

### When Batching Can Cause Issues

**Issue 1: Reading state immediately after setting it**

\`\`\`tsx
// ❌ PROBLEM: Expects immediate update
function Component() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    setCount(count + 1);
    console.log(count);  // Logs 0, not 1! (State hasn't updated yet)
    
    // This fails:
    if (count === 10) {
      alert('Reached 10!');  // Never fires when clicking from 9
    }
  };
}

// ✅ SOLUTION: Use useEffect to react to state changes
function Component() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    if (count === 10) {
      alert('Reached 10!');  // Fires correctly
    }
  }, [count]);
  
  const handleClick = () => {
    setCount(count + 1);
  };
}
\`\`\`

**Issue 2: Dependent calculations**

\`\`\`tsx
// ❌ PROBLEM: total calculated before price updates
function ShoppingCart() {
  const [price, setPrice] = useState(100);
  const [tax, setTax] = useState(10);
  
  const handleUpdatePrice = (newPrice: number) => {
    setPrice(newPrice);
    setTax(newPrice * 0.1);  // Uses OLD price!
    
    // If newPrice = 200:
    // setPrice(200)
    // setTax(100 * 0.1 = 10)  ← Wrong! Should be 20
  };
}

// ✅ SOLUTION 1: Use functional updates
function ShoppingCart() {
  const [price, setPrice] = useState(100);
  const [tax, setTax] = useState(10);
  
  const handleUpdatePrice = (newPrice: number) => {
    setPrice(newPrice);
    setTax(newPrice * 0.1);  // Use newPrice directly
  };
}

// ✅ SOLUTION 2: Derive tax from price (better!)
function ShoppingCart() {
  const [price, setPrice] = useState(100);
  const tax = price * 0.1;  // Calculate on every render
  
  // No state needed for tax—it's derived!
}
\`\`\`

### Opting Out of Batching (React 18)

**Sometimes you WANT multiple re-renders:**

\`\`\`tsx
import { flushSync } from 'react-dom';

function Component() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);
  
  const handleClick = () => {
    // Force immediate re-render
    flushSync(() => {
      setCount(count + 1);
    });
    // Component re-renders here
    
    // Do something with updated DOM
    const element = document.getElementById('count');
    console.log(element.textContent);  // Shows updated count
    
    // Then another update
    flushSync(() => {
      setFlag(true);
    });
    // Component re-renders again
  };
  
  return <div id="count">{count}</div>;
}

// Use sparingly! Breaks performance optimization.
// Only use when you need to:
// - Read DOM immediately after update
// - Sync with third-party libraries
// - Trigger browser APIs that depend on DOM
\`\`\`

### Complete Example: Counter with Multiple Update Patterns

\`\`\`tsx
import { useState } from 'react';

function ComprehensiveCounter() {
  const [count, setCount] = useState(0);
  
  // ❌ WRONG: Stale closures
  const incrementWrong = () => {
    setCount(count + 1);
    setCount(count + 1);
    setCount(count + 1);
    // Result: count + 1 (not count + 3)
  };
  
  // ✅ CORRECT: Functional updates
  const incrementCorrect = () => {
    setCount(prev => prev + 1);
    setCount(prev => prev + 1);
    setCount(prev => prev + 1);
    // Result: count + 3 ✓
  };
  
  // ❌ WRONG: Can't read updated value immediately
  const incrementAndLog = () => {
    setCount(count + 1);
    console.log(\`Count is now: \${count}\`);  // Logs OLD value
  };
  
  // ✅ CORRECT: Use useEffect for side effects
  useEffect(() => {
    console.log(\`Count changed to: \${count}\`);
  }, [count]);
  
  // ✅ Mix of direct and functional updates
  const complexUpdate = () => {
    setCount(10);  // Set to specific value
    setCount(prev => prev + 1);  // Then increment
    setCount(prev => prev * 2);  // Then double
    // Result: (10 + 1) * 2 = 22
  };
  
  return (
    <div>
      <h2>Count: {count}</h2>
      <button onClick={incrementWrong}>
        Wrong: +3 (actually +1)
      </button>
      <button onClick={incrementCorrect}>
        Correct: +3
      </button>
      <button onClick={() => setCount(0)}>
        Reset
      </button>
    </div>
  );
}
\`\`\`

### What to Tell the Developer

"Great question! This is a common React gotcha. Here's what's happening:

**The Problem:**
\`\`\`tsx
setCount(count + 1);  // count = 0
setCount(count + 1);  // count = 0 (still!)
setCount(count + 1);  // count = 0 (still!)
\`\`\`

All three calls read the SAME \`count\` value (0) because:
1. State updates are asynchronous (batched)
2. Your function captures \`count\` at the time it runs
3. All three try to set count to 1

**The Solution:**
\`\`\`tsx
setCount(prev => prev + 1);  // 0 + 1 = 1
setCount(prev => prev + 1);  // 1 + 1 = 2
setCount(prev => prev + 1);  // 2 + 1 = 3
\`\`\`

The callback receives the LATEST state value, so each update builds on the previous one.

**Rule of thumb**: Use functional updates when new state depends on old state.

Try it out and let me know if you have questions!"

### Key Takeaways

1. **Batching is good**: Prevents unnecessary re-renders (performance!)
2. **Stale closures happen**: Multiple updates read same old value
3. **Functional updates solve it**: Callbacks receive latest state
4. **React 18 batches everywhere**: Promises, timeouts, native events
5. **Don't read state immediately**: Use useEffect for post-update effects
6. **flushSync opts out**: Force immediate re-render (rare!)

Understanding batching is crucial for React development. Master this, and you'll avoid countless bugs! 🎯
`,
  },
];
