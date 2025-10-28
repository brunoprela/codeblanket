export const eventHandlingDiscussion = [
  {
    id: 1,
    question:
      "A developer writes: `<button onClick={handleClick()}>Click</button>` and the function executes immediately on render, not when clicked. They're confused why adding parentheses causes this. Explain the difference between passing a function reference vs calling a function, how JavaScript closures work in this context, provide 5 different correct ways to pass arguments to event handlers, and explain when each approach is preferred.",
    answer: `## Comprehensive Answer:

This is one of the most common React mistakes—**calling a function instead of passing a function reference**. Let me explain the fundamental difference and all the solutions.

### The Problem: Function Call vs Function Reference

\`\`\`tsx
// ❌ WRONG: Calls function immediately
<button onClick={handleClick()}>Click</button>
// This calls handleClick() DURING RENDER
// Returns undefined (unless function returns another function)
// Button's onClick receives undefined → nothing happens on click

// ✅ CORRECT: Passes function reference
<button onClick={handleClick}>Click</button>
// This passes the FUNCTION ITSELF to onClick
// React calls it when button is clicked
\`\`\`

**What happens with parentheses:**

\`\`\`tsx
function Component() {
  const handleClick = () => {
    console.log('Clicked!');
  };
  
  console.log('1. Component rendering');
  
  return (
    <div>
      {/* Immediately invoked during render */}
      <button onClick={handleClick()}>
        Wrong
      </button>
      {/* Function reference, called on click */}
      <button onClick={handleClick}>
        Correct
      </button>
    </div>
  );
}

// Console output when component renders:
// 1. Component rendering
// Clicked!  ← handleClick() called immediately!

// When you click "Wrong" button: Nothing happens
// When you click "Correct" button: "Clicked!" logs
\`\`\`

### Understanding JavaScript Functions

**Functions in JavaScript are first-class citizens**—they can be:
1. Stored in variables
2. Passed as arguments
3. Returned from other functions

\`\`\`javascript
// Function reference (no parentheses)
const myFunction = () => console.log('Hello');

// Store in variable
const ref = myFunction;  // ref now points to the function

// Pass as argument
setTimeout(myFunction, 1000);  // ✅ Correct
setTimeout(myFunction(), 1000);  // ❌ Wrong - calls immediately

// Call the function (with parentheses)
myFunction();  // Executes: logs "Hello"

// What onClick expects:
onClick={myFunction}   // ✅ Function reference
onClick={myFunction()} // ❌ Function call result (usually undefined)
\`\`\`

### The Argument Problem

**Challenge**: How do you pass arguments if you can't use parentheses?

\`\`\`tsx
function TodoList() {
  const todos = [
    { id: 1, text: 'Buy milk' },
    { id: 2, text: 'Walk dog' }
  ];
  
  const handleDelete = (id: number) => {
    console.log('Deleting todo:', id);
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
          
          {/* ❌ WRONG: Calls immediately */}
          <button onClick={handleDelete(todo.id)}>
            Delete
          </button>
          {/* During render: handleDelete(1), handleDelete(2) execute
              Console logs: "Deleting todo: 1", "Deleting todo: 2"
              Clicking button does nothing */}
          
          {/* Need a way to pass todo.id WITHOUT calling function */}
        </li>
      ))}
    </ul>
  );
}
\`\`\`

### Solution 1: Arrow Function Wrapper (Most Common)

\`\`\`tsx
<button onClick={() => handleDelete(todo.id)}>
  Delete
</button>

// How it works:
// 1. Arrow function ITSELF is passed to onClick (function reference)
// 2. When button clicked, React calls the arrow function
// 3. Arrow function calls handleDelete(todo.id)
\`\`\`

**Breakdown:**
\`\`\`tsx
// What React sees:
onClick={
  () => handleDelete(todo.id)  // This is a function
}

// When clicked:
// React calls: () => handleDelete(todo.id)
// Which executes: handleDelete(todo.id)
\`\`\`

**Pros:**
- ✅ Clear and readable
- ✅ Easy to add multiple statements
- ✅ TypeScript-friendly
- ✅ Can access event object: \`(e) => handleDelete(todo.id, e)\`

**Cons:**
- ⚠️ Creates new function on every render (usually negligible)

**When to use:** 99% of cases—this is the standard pattern

### Solution 2: Currying (Higher-Order Function)

\`\`\`tsx
const handleDelete = (id: number) => (e: React.MouseEvent) => {
  console.log('Deleting todo:', id);
  console.log('Event:', e);
};

<button onClick={handleDelete(todo.id)}>
  Delete
</button>

// How it works:
// handleDelete(todo.id) returns a FUNCTION
// That returned function is passed to onClick
\`\`\`

**Breakdown:**
\`\`\`tsx
// handleDelete is a function that returns a function
const handleDelete = (id: number) => {
  // This outer function runs during render
  return (e: React.MouseEvent) => {
    // This inner function runs on click
    console.log('Deleting:', id);
  };
};

// During render:
onClick={handleDelete(1)}  // Returns: (e) => { console.log('Deleting:', 1) }

// When clicked:
// React calls the returned function
\`\`\`

**Pros:**
- ✅ No inline arrow function in JSX
- ✅ Cleaner JSX
- ✅ Can access event object

**Cons:**
- ⚠️ Less intuitive syntax
- ⚠️ Still creates function on every render
- ⚠️ Harder for beginners

**When to use:** Clean JSX is priority, team comfortable with currying

### Solution 3: bind() Method

\`\`\`tsx
const handleDelete = (id: number, e: React.MouseEvent) => {
  console.log('Deleting todo:', id);
  console.log('Event:', e);
};

<button onClick={handleDelete.bind(null, todo.id)}>
  Delete
</button>

// How bind works:
// .bind(thisArg, arg1, arg2, ...) returns a new function
// with 'this' set to thisArg and arguments pre-filled
\`\`\`

**Breakdown:**
\`\`\`tsx
// handleDelete.bind(null, 123) returns:
function boundFunction(e) {
  return handleDelete(123, e);
}

// During render:
onClick={handleDelete.bind(null, todo.id)}  // Returns bound function

// When clicked:
// React calls bound function with event
// Bound function calls handleDelete(todo.id, event)
\`\`\`

**Pros:**
- ✅ No arrow function
- ✅ Pre-fills arguments

**Cons:**
- ⚠️ Less common (unfamiliar to many developers)
- ⚠️ Still creates function on every render
- ⚠️ Null as first argument (awkward)

**When to use:** Rarely—arrow functions are clearer

### Solution 4: Data Attributes

\`\`\`tsx
const handleDelete = (e: React.MouseEvent<HTMLButtonElement>) => {
  const id = Number(e.currentTarget.dataset.id);
  console.log('Deleting todo:', id);
};

<button data-id={todo.id} onClick={handleDelete}>
  Delete
</button>

// How it works:
// 1. Store data in HTML data-* attribute
// 2. Read from event.currentTarget.dataset
\`\`\`

**Breakdown:**
\`\`\`tsx
// HTML rendered:
<button data-id="123" onclick="...">Delete</button>

// On click:
// event.currentTarget = the button element
// event.currentTarget.dataset = { id: "123" }
// Read and convert: Number("123") = 123
\`\`\`

**Pros:**
- ✅ No closure over todo.id
- ✅ Function reference (no new function)
- ✅ Works with event delegation

**Cons:**
- ⚠️ Type safety lost (all dataset values are strings)
- ⚠️ Manual conversion (Number(), parseInt())
- ⚠️ More verbose

**When to use:** Performance-critical large lists, need event delegation

### Solution 5: Inline Handler Definition

\`\`\`tsx
<button onClick={function(e) {
  handleDelete(todo.id);
}}>
  Delete
</button>

// Or with arrow function and multiple lines:
<button onClick={(e) => {
  e.preventDefault();
  console.log('Deleting:', todo.id);
  handleDelete(todo.id);
}}>
  Delete
</button>
\`\`\`

**Pros:**
- ✅ Can include multiple statements
- ✅ Access to both todo.id and event

**Cons:**
- ⚠️ Verbose in JSX
- ⚠️ Harder to test
- ⚠️ Creates new function on every render

**When to use:** Quick prototyping, one-off handlers

### Comparison Table

| Approach | Syntax | Performance | Readability | Best For |
|----------|--------|-------------|-------------|----------|
| **Arrow wrapper** | \`onClick={() => f(x)}\` | Good | Excellent | 99% of cases |
| **Currying** | \`onClick={f(x)}\` | Good | Medium | Clean JSX |
| **bind()** | \`onClick={f.bind(null, x)}\` | Good | Low | Rarely |
| **Data attributes** | \`data-x={x} onClick={f}\` | Best | Low | Large lists |
| **Inline** | \`onClick={(e) => {...}}\` | Good | Medium | Prototyping |

### Real-World Example: Todo App

\`\`\`tsx
import { useState } from 'react';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([
    { id: 1, text: 'Buy milk', completed: false },
    { id: 2, text: 'Walk dog', completed: true }
  ]);
  
  // Method 1: Arrow wrapper (recommended)
  const handleDelete = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };
  
  // Method 2: Currying
  const handleToggle = (id: number) => (e: React.MouseEvent) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };
  
  // Method 4: Data attributes
  const handleEdit = (e: React.MouseEvent<HTMLButtonElement>) => {
    const id = Number(e.currentTarget.dataset.id);
    const text = prompt('New text:');
    if (text) {
      setTodos(todos.map(todo =>
        todo.id === id ? { ...todo, text } : todo
      ));
    }
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {/* Method 2: Currying - no arrow in JSX */}
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={handleToggle(todo.id)}
          />
          
          <span style={{
            textDecoration: todo.completed ? 'line-through' : 'none'
          }}>
            {todo.text}
          </span>
          
          {/* Method 1: Arrow wrapper - most common */}
          <button onClick={() => handleDelete(todo.id)}>
            Delete
          </button>
          
          {/* Method 4: Data attributes */}
          <button data-id={todo.id} onClick={handleEdit}>
            Edit
          </button>
          
          {/* Method 5: Inline */}
          <button onClick={(e) => {
            if (confirm('Are you sure?')) {
              handleDelete(todo.id);
            }
          }}>
            Delete with Confirm
          </button>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

### Performance Deep Dive

**Question:** "Don't arrow functions create new functions every render? Isn't that slow?"

**Answer:** Usually negligible, but let's measure:

\`\`\`tsx
// Benchmark: 1000 items rendering
function LargeList() {
  const [items] = useState(Array.from({ length: 1000 }, (_, i) => i));
  
  const handleClick = (id: number) => {
    console.log('Clicked:', id);
  };
  
  console.time('render');
  const list = items.map(id => (
    <button key={id} onClick={() => handleClick(id)}>
      Item {id}
    </button>
  ));
  console.timeEnd('render');
  
  return <div>{list}</div>;
}

// Result: ~5ms to create 1000 arrow functions
// Creating functions is FAST in modern JavaScript
// This is NOT your bottleneck (DOM updates are)

// When it matters: React.memo + 10,000+ items
const MemoizedItem = React.memo(({ id, onClick }) => {
  console.log('Item rendered:', id);
  return <button onClick={onClick}>Item {id}</button>;
});

function OptimizedList() {
  const [items] = useState(Array.from({ length: 10000 }, (_, i) => i));
  
  // useCallback prevents creating new function
  const handleClick = useCallback((id: number) => {
    console.log('Clicked:', id);
  }, []);
  
  return (
    <div>
      {items.map(id => (
        <MemoizedItem
          key={id}
          id={id}
          onClick={() => handleClick(id)}  // Still creates new function!
        />
      ))}
    </div>
  );
}

// For truly optimized: Use data attributes or currying
\`\`\`

### Common Mistakes

**Mistake 1: Parentheses everywhere**
\`\`\`tsx
// ❌ Both wrong!
<button onClick={handleClick()}>Click</button>
<button onClick={() => handleClick()()}>Click</button>

// ✅ Correct
<button onClick={handleClick}>Click</button>
\`\`\`

**Mistake 2: Forgetting event parameter**
\`\`\`tsx
// ❌ Wrong: e is undefined
<button onClick={handleClick(todo.id)}>Click</button>

function handleClick(id: number, e: React.MouseEvent) {
  console.log(e);  // undefined!
}

// ✅ Correct: Pass event explicitly
<button onClick={(e) => handleClick(todo.id, e)}>Click</button>
\`\`\`

**Mistake 3: Using bind() incorrectly**
\`\`\`tsx
// ❌ Wrong: First argument is 'this' context
<button onClick={handleClick.bind(todo.id)}>Click</button>

// ✅ Correct: Use null for 'this', then arguments
<button onClick={handleClick.bind(null, todo.id)}>Click</button>
\`\`\`

### What to Tell the Developer

"Great question! This is a super common mistake. Here's what's happening:

**Your code:**
\`\`\`tsx
<button onClick={handleClick()}>Click</button>
\`\`\`

**Problem:** The parentheses \`()\` **call the function immediately** during render. It's like writing:
\`\`\`tsx
const result = handleClick();  // Executes now
<button onClick={result}>Click</button>  // result is usually undefined
\`\`\`

**Solution:** Remove parentheses to pass the function itself:
\`\`\`tsx
<button onClick={handleClick}>Click</button>
\`\`\`

**If you need arguments:**
\`\`\`tsx
<button onClick={() => handleClick(id)}>Click</button>
\`\`\`

The arrow function \`() => handleClick(id)\` is ITSELF a function that React calls on click.

**Think of it like:**
- \`handleClick\` = "Here's a function, call it later"
- \`handleClick()\` = "Call this function RIGHT NOW"

Try it and let me know if you have questions!"

### Key Takeaways

1. **Function reference vs call**: \`f\` passes function, \`f()\` calls it
2. **Arrow wrapper is standard**: \`onClick={() => f(x)}\` for 99% of cases
3. **5 valid approaches**: Arrow, currying, bind, data attributes, inline
4. **Performance rarely matters**: Creating functions is fast
5. **Choose based on readability**: Arrow wrapper wins for clarity

Understanding this distinction is fundamental to React and JavaScript event handling! 🎯
`,
  },
  {
    id: 2,
    question:
      "Your team is building a modal component. When users click inside the modal content, the modal closes because the click bubbles up to the backdrop handler. A junior developer suggests: 'Let's add onClick={null} to the content div to disable clicks.' Explain event bubbling in React, why onClick={null} won't work, the correct solution using stopPropagation(), when you should and shouldn't stop propagation, and provide 3 real-world examples where event bubbling is beneficial.",
    answer: `## Comprehensive Answer:

This is a classic event bubbling problem. The junior developer's intuition is right (stop the event) but the solution is wrong. Let me explain event bubbling, propagation, and the correct solutions.

### Understanding Event Bubbling

**Event bubbling** means events "bubble up" from child elements to parent elements.

\`\`\`tsx
function Example() {
  const handleParent = () => console.log('Parent clicked');
  const handleChild = () => console.log('Child clicked');
  const handleGrandchild = () => console.log('Grandchild clicked');
  
  return (
    <div onClick={handleParent} style={{ padding: 40, background: 'red' }}>
      Parent
      <div onClick={handleChild} style={{ padding: 40, background: 'green' }}>
        Child
        <div onClick={handleGrandchild} style={{ padding: 40, background: 'blue' }}>
          Grandchild
        </div>
      </div>
    </div>
  );
}

// Click on Grandchild logs:
// "Grandchild clicked"
// "Child clicked"
// "Parent clicked"

// Event travels: Grandchild → Child → Parent (bubbling up)
\`\`\`

**Event flow:**
\`\`\`
User clicks Grandchild:

1. Capturing phase (top-down):
   window → document → body → Parent → Child → Grandchild
   
2. Target phase:
   Grandchild element (event.target)
   
3. Bubbling phase (bottom-up):
   Grandchild → Child → Parent → body → document → window
   
React handlers run during BUBBLING phase by default
\`\`\`

### The Modal Problem

\`\`\`tsx
// ❌ PROBLEM: Clicking modal content closes modal
function Modal({ onClose, children }) {
  const handleBackdropClick = () => {
    onClose();  // Close modal
  };
  
  return (
    <div className="backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        {children}
      </div>
    </div>
  );
}

// When you click inside modal content:
// 1. Click event starts at clicked element
// 2. Bubbles up to modal-content div
// 3. Bubbles up to backdrop div
// 4. handleBackdropClick runs → modal closes!
\`\`\`

**Visual flow:**
\`\`\`
User clicks button inside modal:

[Backdrop onClick={close}]           ← Event reaches here
    └─ [Modal Content]               ← Event bubbles through here
           └─ [Button]               ← Click starts here

Result: Modal closes (unwanted!)
\`\`\`

### Why onClick={null} Doesn't Work

\`\`\`tsx
// ❌ Junior developer's attempt
function Modal({ onClose, children }) {
  return (
    <div className="backdrop" onClick={onClose}>
      {/* This doesn't prevent bubbling! */}
      <div className="modal-content" onClick={null}>
        {children}
      </div>
    </div>
  );
}

// Why it fails:
// 1. onClick={null} means "no handler on this element"
// 2. But event still bubbles up!
// 3. Event reaches backdrop → onClose fires

// Think of it like:
// - onClick={null} = "I don't want to listen to this event"
// - But event still travels up the tree!
\`\`\`

**Analogy:** Removing your ears doesn't stop sound waves from traveling.

### Solution 1: stopPropagation() (Most Common)

\`\`\`tsx
// ✅ CORRECT: Stop event from bubbling
function Modal({ onClose, children }) {
  const handleBackdropClick = () => {
    onClose();
  };
  
  const handleContentClick = (e: React.MouseEvent) => {
    e.stopPropagation();  // Stop bubbling here
  };
  
  return (
    <div className="backdrop" onClick={handleBackdropClick}>
      <div className="modal-content" onClick={handleContentClick}>
        {children}
      </div>
    </div>
  );
}

// How it works:
// 1. Click inside modal content
// 2. handleContentClick runs
// 3. e.stopPropagation() stops bubbling
// 4. handleBackdropClick NEVER runs
// 5. Modal stays open ✓
\`\`\`

**Event flow with stopPropagation:**
\`\`\`
User clicks button inside modal:

[Backdrop onClick={close}]           ← Event STOPPED, doesn't reach here
    └─ [Modal Content onClick={e.stopPropagation()}]  ← Stops bubbling
           └─ [Button]               ← Click starts here

Result: Modal stays open ✓
\`\`\`

### Solution 2: Check event.target (Also Correct)

\`\`\`tsx
// ✅ ALTERNATIVE: Check what was actually clicked
function Modal({ onClose, children }) {
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Only close if backdrop itself was clicked, not children
    if (e.target === e.currentTarget) {
      onClose();
    }
  };
  
  return (
    <div className="backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        {children}
      </div>
    </div>
  );
}

// How it works:
// e.target = actual element that was clicked
// e.currentTarget = element with the onClick handler (backdrop)
// If they're the same, user clicked backdrop directly
\`\`\`

**Comparison:**
- **e.target**: The element that was actually clicked (innermost)
- **e.currentTarget**: The element with the event handler (always the backdrop)

\`\`\`tsx
// Example:
<div className="backdrop" onClick={handleClick}>  ← e.currentTarget
  <div className="modal">
    <button>Click</button>  ← e.target (if button clicked)
  </div>
</div>

// If button clicked:
// e.target = button element
// e.currentTarget = backdrop div
// e.target !== e.currentTarget → don't close

// If backdrop clicked directly:
// e.target = backdrop div
// e.currentTarget = backdrop div
// e.target === e.currentTarget → close!
\`\`\`

### Solution 3: Separate Backdrop Element

\`\`\`tsx
// ✅ STRUCTURAL: Separate backdrop from modal
function Modal({ onClose, children }) {
  return (
    <>
      {/* Backdrop covers entire screen */}
      <div className="backdrop" onClick={onClose} />
      
      {/* Modal sits on top (CSS: position: fixed, z-index) */}
      <div className="modal-content">
        {children}
      </div>
    </>
  );
}

// CSS ensures modal appears above backdrop
// Clicking modal doesn't reach backdrop (separate elements)
\`\`\`

### When to Use stopPropagation()

**✅ Good use cases:**

1. **Modal/Overlay patterns**
\`\`\`tsx
<div className="overlay" onClick={close}>
  <div className="modal" onClick={e => e.stopPropagation()}>
    Content
  </div>
</div>
\`\`\`

2. **Dropdown menus**
\`\`\`tsx
<div onClick={closeAllDropdowns}>
  <Dropdown>
    <button onClick={e => e.stopPropagation()}>
      Don't close dropdown
    </button>
  </Dropdown>
</div>
\`\`\`

3. **Nested clickable areas**
\`\`\`tsx
<div className="card" onClick={openCard}>
  <button onClick={e => {
    e.stopPropagation();
    deleteCard();
  }}>
    Delete (don't open card)
  </button>
</div>
\`\`\`

4. **Custom context menus**
\`\`\`tsx
<div onContextMenu={e => {
  e.preventDefault();
  e.stopPropagation();
  showCustomMenu(e.clientX, e.clientY);
}}>
  Right-click me
</div>
\`\`\`

**❌ Bad use cases (don't stop propagation):**

1. **Global event handlers**
\`\`\`tsx
// ❌ Don't stop propagation
// Breaks analytics, keyboard shortcuts, accessibility
<button onClick={e => {
  e.stopPropagation();
  handleClick();
}}>
  Click
</button>
\`\`\`

2. **When event delegation is beneficial**
3. **When parent needs to know about events**
4. **When using React Portal** (events already bubble correctly)

### When Event Bubbling is Beneficial

**Example 1: Event Delegation (Performance)**

\`\`\`tsx
// ❌ WITHOUT event delegation: 1000 event listeners
function List({ items }) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={() => console.log(item.id)}>
          {item.text}
        </li>
      ))}
    </ul>
  );
}

// Each <li> gets its own onClick handler
// 1000 items = 1000 event listeners in memory
// Slower initial render, more memory

// ✅ WITH event delegation: 1 event listener
function ListOptimized({ items }) {
  const handleClick = (e: React.MouseEvent<HTMLUListElement>) => {
    const li = e.target.closest('li');
    if (li) {
      const id = li.dataset.id;
      console.log('Clicked item:', id);
    }
  };
  
  return (
    <ul onClick={handleClick}>
      {items.map(item => (
        <li key={item.id} data-id={item.id}>
          {item.text}
        </li>
      ))}
    </ul>
  );
}

// Single handler on <ul> catches all clicks
// Event bubbles from <li> to <ul>
// 10x faster for large lists
\`\`\`

**Example 2: Analytics and Monitoring**

\`\`\`tsx
// Global click tracking (leverages bubbling)
function App() {
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      // Track all clicks in app
      analytics.track('click', {
        element: e.target.tagName,
        timestamp: Date.now()
      });
    };
    
    // Single listener catches ALL clicks via bubbling
    document.addEventListener('click', handleClick);
    
    return () => document.removeEventListener('click', handleClick);
  }, []);
  
  return (
    <div>
      <button>Button 1</button>
      <button>Button 2</button>
      <a href="#">Link</a>
      {/* All clicks bubble up to document */}
    </div>
  );
}

// If everything called stopPropagation(), analytics breaks!
\`\`\`

**Example 3: Accessibility and Keyboard Navigation**

\`\`\`tsx
// Global keyboard shortcut handler
function App() {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+S: Save
      if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        save();
      }
      
      // Ctrl+F: Search
      if (e.ctrlKey && e.key === 'f') {
        e.preventDefault();
        openSearch();
      }
    };
    
    // Works anywhere in app via bubbling
    window.addEventListener('keydown', handleKeyDown);
    
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  return <div>{/* App content */}</div>;
}

// Bubbling allows global keyboard shortcuts!
\`\`\`

### Complete Modal Implementation

\`\`\`tsx
import { useEffect, useRef } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  closeOnBackdropClick?: boolean;
  closeOnEscape?: boolean;
}

function Modal({
  isOpen,
  onClose,
  children,
  closeOnBackdropClick = true,
  closeOnEscape = true
}: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  
  // Close on Escape key
  useEffect(() => {
    if (!isOpen || !closeOnEscape) return;
    
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, closeOnEscape, onClose]);
  
  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);
  
  if (!isOpen) return null;
  
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!closeOnBackdropClick) return;
    
    // Method 1: Check if backdrop itself was clicked
    if (e.target === e.currentTarget) {
      onClose();
    }
  };
  
  const handleContentClick = (e: React.MouseEvent) => {
    // Method 2: Stop propagation (alternative)
    if (closeOnBackdropClick) {
      e.stopPropagation();
    }
  };
  
  return (
    <div
      className="modal-backdrop"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={modalRef}
        className="modal-content"
        onClick={handleContentClick}
      >
        <button
          className="modal-close"
          onClick={onClose}
          aria-label="Close modal"
        >
          ×
        </button>
        {children}
      </div>
    </div>
  );
}

// Usage
function App() {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <>
      <button onClick={() => setIsOpen(true)}>
        Open Modal
      </button>
      
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        closeOnBackdropClick={true}
        closeOnEscape={true}
      >
        <h2>Modal Title</h2>
        <p>Modal content here</p>
        <button onClick={() => console.log('Button clicked')}>
          Click me (modal stays open)
        </button>
      </Modal>
    </>
  );
}
\`\`\`

### Debugging Event Propagation

\`\`\`tsx
function DebugEvents() {
  const handleEvent = (name: string) => (e: React.MouseEvent) => {
    console.log(\`\${name} clicked\`);
    console.log('target:', e.target);  // What was actually clicked
    console.log('currentTarget:', e.currentTarget);  // Element with handler
    console.log('---');
  };
  
  return (
    <div onClick={handleEvent('Grandparent')} style={{ padding: 40, background: 'red' }}>
      Grandparent
      <div onClick={handleEvent('Parent')} style={{ padding: 40, background: 'green' }}>
        Parent
        <button onClick={handleEvent('Child')}>
          Child button
        </button>
      </div>
    </div>
  );
}

// Click Child button logs:
// Child clicked
//   target: <button>
//   currentTarget: <button>
// Parent clicked
//   target: <button>
//   currentTarget: <div> (parent)
// Grandparent clicked
//   target: <button>
//   currentTarget: <div> (grandparent)
\`\`\`

### What to Tell the Junior Developer

"Great thinking! You're right that we need to stop the event, but \`onClick={null}\` won't work. Here's why:

**The problem:**
\`\`\`tsx
<div onClick={close}>        ← Backdrop
  <div onClick={null}>      ← This doesn't stop bubbling!
    {children}
  </div>
</div>
\`\`\`

\`onClick={null}\` means "no handler here", but the event still bubbles up!

**Solution:**
\`\`\`tsx
<div onClick={close}>
  <div onClick={(e) => e.stopPropagation()}>  ← Stops here
    {children}
  </div>
</div>
\`\`\`

\`e.stopPropagation()\` stops the event from bubbling to the backdrop.

**Even better:**
\`\`\`tsx
const handleBackdropClick = (e) => {
  if (e.target === e.currentTarget) {  // Only close if backdrop clicked
    close();
  }
};
\`\`\`

Try both approaches and see which you prefer!"

### Key Takeaways

1. **Events bubble up**: Child → Parent → Grandparent
2. **onClick={null} doesn't stop bubbling**: It just means "no handler"
3. **stopPropagation() stops bubbling**: Event stops at that element
4. **e.target vs e.currentTarget**: Actual click vs handler element
5. **Bubbling is usually good**: Enables analytics, event delegation, accessibility
6. **Stop propagation sparingly**: Only when necessary (modals, dropdowns, nested clickables)

Understanding event propagation is essential for building complex interactive UIs! 🎯
`,
  },
  {
    id: 3,
    question:
      "Your application has a search bar that fetches results from an API as the user types. Currently, it makes an API call on EVERY keystroke. For 'react', that's 5 API calls (r, re, rea, reac, react). Your manager says this is expensive and asks you to 'debounce' the input. Explain what debouncing is, how to implement it in React with useState and useEffect, provide a complete working example with TypeScript, and discuss the trade-offs between debouncing, throttling, and immediate search.",
    answer: `## Comprehensive Answer:

This is a very common real-world optimization problem. Making API calls on every keystroke is expensive and unnecessary—most users type faster than 200ms, so intermediate calls are wasted. **Debouncing** solves this by waiting until the user stops typing.

### The Problem: Too Many API Calls

\`\`\`tsx
// ❌ BAD: API call on every keystroke
function SearchBar() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  
  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    
    // API call immediately!
    const response = await fetch(\`/api/search?q=\${value}\`);
    const data = await response.json();
    setResults(data);
  };
  
  return <input value={query} onChange={handleChange} />;
}

// User types "react":
// Keystroke: r     → API call: /api/search?q=r
// Keystroke: e     → API call: /api/search?q=re
// Keystroke: a     → API call: /api/search?q=rea
// Keystroke: c     → API call: /api/search?q=reac
// Keystroke: t     → API call: /api/search?q=react

// Total: 5 API calls
// Problem: First 4 are wasted (user didn't want "r", "re", etc.)
// Cost: 5× server load, 5× data transfer, slower UI
\`\`\`

**Performance impact:**
- API calls: 5× unnecessary
- Server load: 5× higher
- User sees: Flickering results (r → re → rea → reac → react)
- Network cost: $$ (especially at scale)

### What is Debouncing?

**Debouncing** = Wait N milliseconds after user stops typing, THEN execute.

\`\`\`
Debouncing visualization:

User types "react":
Time: 0ms    100ms   200ms   300ms   400ms   500ms   600ms   700ms
      r      e       a       c       t       |       |       |
      ↓      ↓       ↓       ↓       ↓       ↓       ↓       ↓
Timer Reset  Reset   Reset   Reset   Reset  Wait... Wait... FIRE!
starts                                       (300ms) (300ms) (API call)

Only 1 API call at 800ms (300ms after last keystroke)
Savings: 80% (5 calls → 1 call)
\`\`\`

**Key concept**: Timer resets on every keystroke. Only fires when user pauses.

### Debouncing Implementation with useEffect

\`\`\`tsx
import { useState, useEffect } from 'react';

function DebouncedSearchBar() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Update query immediately (for UI responsiveness)
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  };
  
  // Debounce: Wait 300ms after user stops typing
  useEffect(() => {
    // Set timer
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);  // Delay in milliseconds
    
    // Cleanup: Clear timer if query changes before 300ms
    return () => {
      clearTimeout(timer);
    };
  }, [query]);  // Re-run when query changes
  
  // Search when debounced query changes
  useEffect(() => {
    if (!debouncedQuery) {
      setResults([]);
      return;
    }
    
    const searchAPI = async () => {
      setIsSearching(true);
      
      try {
        const response = await fetch(
          \`/api/search?q=\${encodeURIComponent(debouncedQuery)}\`
        );
        const data = await response.json();
        setResults(data);
      } catch (error) {
        console.error('Search failed:', error);
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    };
    
    searchAPI();
  }, [debouncedQuery]);  // Only search when debounced query changes
  
  return (
    <div>
      <input
        value={query}
        onChange={handleChange}
        placeholder="Search..."
      />
      {isSearching && <p>Searching...</p>}
      <ul>
        {results.map((result: any) => (
          <li key={result.id}>{result.title}</li>
        ))}
      </ul>
    </div>
  );
}

// How it works:
// User types "r": query="r", timer starts (300ms)
// User types "e" (50ms later): query="re", timer RESETS (300ms)
// User types "a" (50ms later): query="rea", timer RESETS (300ms)
// User types "c" (50ms later): query="reac", timer RESETS (300ms)
// User types "t" (50ms later): query="react", timer RESETS (300ms)
// User pauses (300ms pass): timer fires → debouncedQuery="react"
// useEffect sees debouncedQuery changed → API call

// Result: 1 API call instead of 5!
\`\`\`

### Custom Debounce Hook (Reusable)

\`\`\`tsx
import { useState, useEffect } from 'react';

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);
  
  return debouncedValue;
}

// Usage
function SearchBar() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);  // Custom hook!
  const [results, setResults] = useState([]);
  
  useEffect(() => {
    if (debouncedQuery) {
      fetch(\`/api/search?q=\${debouncedQuery}\`)
        .then(res => res.json())
        .then(data => setResults(data));
    }
  }, [debouncedQuery]);
  
  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <ul>
        {results.map(r => <li key={r.id}>{r.title}</li>)}
      </ul>
    </div>
  );
}

// Benefits of custom hook:
// - Reusable across components
// - Cleaner component code
// - Encapsulated logic
// - Easy to test
\`\`\`

### Complete Production Example

\`\`\`tsx
import { useState, useEffect, useCallback } from 'react';

interface SearchResult {
  id: number;
  title: string;
  description: string;
}

interface SearchState {
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  error: string | null;
}

function ProductionSearchBar() {
  const [state, setState] = useState<SearchState>({
    query: '',
    results: [],
    isSearching: false,
    error: null
  });
  
  const [debouncedQuery] = useDebounce(state.query, 300);
  
  // Abort controller for canceling in-flight requests
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setState(prev => ({
      ...prev,
      query: e.target.value,
      error: null
    }));
  };
  
  const handleClear = () => {
    setState({
      query: '',
      results: [],
      isSearching: false,
      error: null
    });
    
    // Cancel any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };
  
  // Search effect
  useEffect(() => {
    // Don't search if query is empty
    if (!debouncedQuery.trim()) {
      setState(prev => ({ ...prev, results: [], isSearching: false }));
      return;
    }
    
    // Abort previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new abort controller
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    
    const searchAPI = async () => {
      setState(prev => ({ ...prev, isSearching: true, error: null }));
      
      try {
        const response = await fetch(
          \`/api/search?q=\${encodeURIComponent(debouncedQuery)}\`,
          {
            signal: abortController.signal  // Abortable
          }
        );
        
        if (!response.ok) {
          throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
        }
        
        const data: SearchResult[] = await response.json();
        
        // Only update if not aborted
        if (!abortController.signal.aborted) {
          setState(prev => ({
            ...prev,
            results: data,
            isSearching: false,
            error: null
          }));
        }
      } catch (error: any) {
        // Ignore abort errors
        if (error.name === 'AbortError') {
          console.log('Request aborted');
          return;
        }
        
        console.error('Search failed:', error);
        setState(prev => ({
          ...prev,
          results: [],
          isSearching: false,
          error: error.message || 'Search failed'
        }));
      }
    };
    
    searchAPI();
    
    // Cleanup: Abort on unmount
    return () => {
      abortController.abort();
    };
  }, [debouncedQuery]);
  
  return (
    <div className="search-bar">
      <div className="search-input-container">
        <input
          type="text"
          value={state.query}
          onChange={handleChange}
          placeholder="Search products..."
          className="search-input"
        />
        
        {state.query && (
          <button
            onClick={handleClear}
            className="clear-button"
            aria-label="Clear search"
          >
            ×
          </button>
        )}
        
        {state.isSearching && (
          <div className="loading-spinner" />
        )}
      </div>
      
      {state.error && (
        <div className="error-message">
          {state.error}
        </div>
      )}
      
      {state.results.length > 0 && (
        <ul className="search-results">
          {state.results.map(result => (
            <li key={result.id} className="search-result-item">
              <h3>{result.title}</h3>
              <p>{result.description}</p>
            </li>
          ))}
        </ul>
      )}
      
      {!state.isSearching && state.query && state.results.length === 0 && (
        <p className="no-results">
          No results found for "{state.query}"
        </p>
      )}
    </div>
  );
}

// Custom hook
function useDebounce<T>(value: T, delay: number): [T] {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [value, delay]);
  
  return [debouncedValue];
}
\`\`\`

### Debouncing vs Throttling vs Immediate

**Debouncing**: Wait until user stops typing

\`\`\`
Keystrokes: r  e  a  c  t  [pause]
API calls:                 ↑ (only 1)

Best for: Search, autosave, validation
Delay: 300-500ms
\`\`\`

**Throttling**: Execute at most once per N milliseconds

\`\`\`
Keystrokes: r  e  a  c  t  (continuous typing)
Time:       0  100 200 300 400ms
API calls:  ↑      ↑      (every 200ms)

Best for: Scroll handlers, resize events, mousemove
Interval: 100-200ms
\`\`\`

**Immediate**: Execute on every keystroke

\`\`\`
Keystrokes: r  e  a  c  t
API calls:  ↑  ↑  ↑  ↑  ↑ (every keystroke)

Best for: Real-time collaboration, instant feedback
No delay
\`\`\`

### Comparison Table

| Approach | API Calls | User Experience | Cost | Best For |
|----------|-----------|-----------------|------|----------|
| **Immediate** | 5/5 keystrokes (100%) | Instant feedback | High | Real-time collab |
| **Debounce** | 1/5 keystrokes (20%) | 300ms delay | Low | Search, autosave |
| **Throttle** | 2-3/5 keystrokes (40-60%) | Regular updates | Medium | Scroll, resize |

### Throttling Implementation (For Comparison)

\`\`\`tsx
function useThrottle<T>(value: T, interval: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastExecuted = useRef<number>(Date.now());
  
  useEffect(() => {
    const now = Date.now();
    const timeSinceLastExecution = now - lastExecuted.current;
    
    if (timeSinceLastExecution >= interval) {
      setThrottledValue(value);
      lastExecuted.current = now;
    } else {
      const timer = setTimeout(() => {
        setThrottledValue(value);
        lastExecuted.current = Date.now();
      }, interval - timeSinceLastExecution);
      
      return () => clearTimeout(timer);
    }
  }, [value, interval]);
  
  return throttledValue;
}

// Usage: Scroll handler
function InfiniteScroll() {
  const [scrollY, setScrollY] = useState(0);
  const throttledScrollY = useThrottle(scrollY, 200);  // Update every 200ms max
  
  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  useEffect(() => {
    // Load more items when scrolled near bottom
    if (throttledScrollY > document.body.scrollHeight - window.innerHeight - 100) {
      loadMoreItems();
    }
  }, [throttledScrollY]);
}
\`\`\`

### Trade-offs Analysis

**Debouncing**
- ✅ Pros: Minimal API calls (80-95% reduction), low cost, simple to implement
- ❌ Cons: Delay before results appear, user must pause typing
- 💡 Use when: Cost is primary concern (API rate limits, server load)

**Throttling**
- ✅ Pros: Regular updates, balance between immediate and debounced
- ❌ Cons: Still makes unnecessary calls, more complex logic
- 💡 Use when: Need periodic updates (scroll position, mouse tracking)

**Immediate**
- ✅ Pros: Best UX (instant feedback), feels responsive
- ❌ Cons: Expensive, can overwhelm server, wasted requests
- 💡 Use when: Real-time collaboration, instant validation, cost not a concern

### Optimal Delay Times

\`\`\`tsx
// Recommended delays by use case:

// Search input: 300-500ms
const searchDelay = 300;  // Most common

// Autosave: 1000-2000ms
const autosaveDelay = 1500;  // Let user finish thought

// Form validation: 500-800ms
const validationDelay = 500;  // Quick feedback

// Window resize: 100-200ms
const resizeDelay = 150;  // Responsive but not excessive

// Scroll: 100ms (throttle, not debounce)
const scrollInterval = 100;  // Smooth tracking
\`\`\`

### Performance Metrics

\`\`\`tsx
// Before debouncing (user types "react"):
// API calls: 5
// Data transfer: 5 × 50KB = 250KB
// Server load: 5 requests
// Cost: $0.0005 × 5 = $0.0025

// After debouncing (300ms):
// API calls: 1
// Data transfer: 1 × 50KB = 50KB
// Server load: 1 request
// Cost: $0.0005 × 1 = $0.0005

// Savings: 80% (5 → 1)
// At 1M searches/day: $2,500/day → $500/day = $2,000/day saved!
\`\`\`

### Advanced: Debounce with Abort

\`\`\`tsx
// Cancel in-flight requests when new search starts
function useDebounceWithAbort(value: string, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  const abortController = useRef<AbortController | null>(null);
  
  useEffect(() => {
    // Abort previous request
    if (abortController.current) {
      abortController.current.abort();
    }
    
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(timer);
      if (abortController.current) {
        abortController.current.abort();
      }
    };
  }, [value, delay]);
  
  return [debouncedValue, abortController];
}

// Prevents race conditions:
// User types "react" → API call 1 starts
// User types "redux" → API call 1 aborted, API call 2 starts
// Results always match latest query
\`\`\`

### Key Takeaways

1. **Debouncing = wait until user stops**: Saves 80-95% of API calls
2. **Use useEffect + setTimeout**: Standard React pattern
3. **Custom hook for reusability**: \`useDebounce(value, delay)\`
4. **300ms is optimal**: Balance between UX and cost
5. **Throttling for continuous events**: Scroll, resize, mousemove
6. **Immediate for real-time**: Collaboration, instant validation
7. **Abort in-flight requests**: Prevent race conditions

**Recommendation for search:** Debounce with 300-500ms delay. Perfect balance of UX and cost! 🎯
`,
  },
];
