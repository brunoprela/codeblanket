export const eventHandling = {
  title: 'Event Handling',
  id: 'event-handling',
  content: `
# Event Handling

## Introduction

Event handling is how your React components respond to user interactions—clicks, typing, hovering, scrolling, and more. **Without event handlers, your app would be just static content**. Mastering event handling is essential for building interactive applications that feel responsive and intuitive.

### Why Event Handling Matters

\`\`\`tsx
// Without events: Static, unresponsive
function StaticButton() {
  return <button>Click me</button>;
  // Clicking does nothing
}

// With events: Interactive, responsive
function InteractiveButton() {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    setCount(count + 1);
  };
  
  return (
    <button onClick={handleClick}>
      Clicked {count} times
    </button>
  );
}
\`\`\`

**Real-world applications** are 90% event handling: buttons, forms, drag-and-drop, keyboard shortcuts, infinite scroll, auto-save, etc.

## React Synthetic Events

React doesn't use native browser events directly. Instead, it wraps them in **SyntheticEvent** objects.

### What are Synthetic Events?

\`\`\`tsx
function Button() {
  const handleClick = (event) => {
    console.log(event);  // SyntheticBaseEvent, not native Event
    console.log(event.nativeEvent);  // Access native event if needed
  };
  
  return <button onClick={handleClick}>Click</button>;
}
\`\`\`

**Why Synthetic Events?**
1. **Cross-browser consistency**: Same API works in all browsers
2. **Performance**: Event pooling (React 16) / automatic cleanup (React 17+)
3. **Additional features**: Better event delegation, automatic cleanup

**Key difference from native events**:
\`\`\`tsx
// Native JavaScript
button.addEventListener('click', (e) => {
  console.log(e);  // Native MouseEvent
});

// React
<button onClick={(e) => {
  console.log(e);  // React SyntheticEvent (wraps MouseEvent)
}}>
  Click
</button>
\`\`\`

## Event Handler Patterns

### Pattern 1: Inline Arrow Functions

\`\`\`tsx
function Button() {
  return (
    <button onClick={() => console.log('Clicked!')}>
      Click me
    </button>
  );
}
\`\`\`

**Pros**:
- Quick for simple handlers
- Easy to pass arguments

**Cons**:
- Creates new function on every render (slight performance cost)
- Harder to test
- Harder to reuse

**When to use**: Simple one-liners, prototyping

### Pattern 2: Defined Functions (Recommended)

\`\`\`tsx
function Button() {
  const handleClick = () => {
    console.log('Clicked!');
  };
  
  return <button onClick={handleClick}>Click me</button>;
}
\`\`\`

**Pros**:
- Easier to read and maintain
- Easier to test
- Can add complex logic
- Better for debugging (named function in stack trace)

**Cons**:
- More lines of code

**When to use**: Production code, complex logic, reusable handlers

### Pattern 3: Inline with Arguments

\`\`\`tsx
function TodoList({ todos }) {
  const handleDelete = (id: number) => {
    console.log('Deleting todo:', id);
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
          {/* Arrow function to pass argument */}
          <button onClick={() => handleDelete(todo.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

**Why the arrow function?**
\`\`\`tsx
// ❌ WRONG: Calls function immediately
<button onClick={handleDelete(todo.id)}>Delete</button>
// This calls handleDelete during render, not on click!

// ✅ CORRECT: Returns function that calls handleDelete
<button onClick={() => handleDelete(todo.id)}>Delete</button>
\`\`\`

### Pattern 4: Event Object Pattern

\`\`\`tsx
function Form() {
  const [email, setEmail] = useState('');
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);  // Access event properties
  };
  
  return <input type="email" value={email} onChange={handleChange} />;
}
\`\`\`

### Pattern 5: Currying Pattern (Advanced)

\`\`\`tsx
function TodoList({ todos }) {
  // Returns a function that returns a function
  const handleDelete = (id: number) => () => {
    console.log('Deleting:', id);
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
          {/* No arrow function needed! */}
          <button onClick={handleDelete(todo.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

**How it works**:
\`\`\`tsx
handleDelete(123)  // Returns: () => console.log('Deleting:', 123)
// Button gets the returned function, calls it on click
\`\`\`

## Common Event Types

### Mouse Events

\`\`\`tsx
function MouseEvents() {
  return (
    <div>
      <button onClick={() => console.log('Click')}>
        onClick
      </button>
      
      <button onDoubleClick={() => console.log('Double click')}>
        onDoubleClick
      </button>
      
      <div onMouseEnter={() => console.log('Mouse entered')}>
        onMouseEnter (fires once when entering)
      </div>
      
      <div onMouseLeave={() => console.log('Mouse left')}>
        onMouseLeave (fires once when leaving)
      </div>
      
      <div onMouseOver={() => console.log('Mouse over')}>
        onMouseOver (fires continuously)
      </div>
      
      <div onMouseDown={() => console.log('Mouse down')}>
        onMouseDown (fires when button pressed)
      </div>
      
      <div onMouseUp={() => console.log('Mouse up')}>
        onMouseUp (fires when button released)
      </div>
      
      <div onMouseMove={(e) => console.log(\`x: \${e.clientX}, y: \${e.clientY}\`)}>
        onMouseMove (tracks position)
      </div>
      
      <div onContextMenu={(e) => {
        e.preventDefault();  // Prevent default right-click menu
        console.log('Right clicked');
      }}>
        onContextMenu (right-click)
      </div>
    </div>
  );
}
\`\`\`

### Keyboard Events

\`\`\`tsx
function KeyboardEvents() {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    console.log('Key:', e.key);
    console.log('Code:', e.code);
    console.log('Ctrl:', e.ctrlKey);
    console.log('Shift:', e.shiftKey);
    console.log('Alt:', e.altKey);
    console.log('Meta (Cmd/Win):', e.metaKey);
    
    // Check specific keys
    if (e.key === 'Enter') {
      console.log('Enter pressed');
    }
    
    if (e.key === 'Escape') {
      console.log('Escape pressed');
    }
    
    // Keyboard shortcuts
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();  // Prevent browser save
      console.log('Ctrl+S pressed - Save!');
    }
  };
  
  return (
    <div>
      <input onKeyDown={handleKeyDown} placeholder="Type here" />
      <input onKeyUp={(e) => console.log('Key released:', e.key)} />
      <input onKeyPress={(e) => console.log('Key pressed:', e.key)} />
      {/* Note: onKeyPress is deprecated, use onKeyDown */}
    </div>
  );
}

// Real-world example: Submit on Enter
function SearchBar() {
  const [query, setQuery] = useState('');
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };
  
  const handleSearch = () => {
    console.log('Searching:', query);
  };
  
  return (
    <input
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      onKeyDown={handleKeyDown}
      placeholder="Search..."
    />
  );
}
\`\`\`

### Form Events

\`\`\`tsx
function FormEvents() {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();  // Prevent page reload
    console.log('Form submitted');
  };
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('Input changed:', e.target.value);
  };
  
  const handleFocus = () => {
    console.log('Input focused');
  };
  
  const handleBlur = () => {
    console.log('Input lost focus');
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
      />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

### Focus Events

\`\`\`tsx
function FocusEvents() {
  return (
    <div>
      <input
        onFocus={() => console.log('Input focused')}
        onBlur={() => console.log('Input lost focus')}
      />
      
      {/* Focus events with validation */}
      <input
        onFocus={(e) => e.target.select()}  // Select all text on focus
        onBlur={(e) => {
          if (!e.target.value) {
            console.log('Field is empty!');
          }
        }}
      />
    </div>
  );
}
\`\`\`

### Clipboard Events

\`\`\`tsx
function ClipboardEvents() {
  const handleCopy = (e: React.ClipboardEvent<HTMLDivElement>) => {
    console.log('Copied:', window.getSelection()?.toString());
  };
  
  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    const pastedText = e.clipboardData.getData('text');
    console.log('Pasted:', pastedText);
    
    // Prevent paste if invalid
    if (pastedText.length > 100) {
      e.preventDefault();
      alert('Text too long!');
    }
  };
  
  const handleCut = () => {
    console.log('Text cut');
  };
  
  return (
    <div>
      <div onCopy={handleCopy}>
        Select and copy this text
      </div>
      <input onPaste={handlePaste} onCut={handleCut} />
    </div>
  );
}
\`\`\`

## Passing Arguments to Event Handlers

### Problem: Passing Data

\`\`\`tsx
function TodoList() {
  const [todos, setTodos] = useState([
    { id: 1, text: 'Buy milk' },
    { id: 2, text: 'Walk dog' }
  ]);
  
  // Need to pass todo.id to handler
  const handleDelete = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
          
          {/* ❌ WRONG: Calls immediately */}
          <button onClick={handleDelete(todo.id)}>Delete</button>
          
          {/* ✅ CORRECT: Arrow function */}
          <button onClick={() => handleDelete(todo.id)}>Delete</button>
          
          {/* ✅ ALSO CORRECT: Bind */}
          <button onClick={handleDelete.bind(null, todo.id)}>Delete</button>
          
          {/* ✅ ALSO CORRECT: Data attributes */}
          <button data-id={todo.id} onClick={(e) => {
            const id = Number(e.currentTarget.dataset.id);
            handleDelete(id);
          }}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

### Solution 1: Arrow Function (Most Common)

\`\`\`tsx
<button onClick={() => handleDelete(todo.id)}>Delete</button>
\`\`\`

**Pros**: Clear, easy to read, TypeScript-friendly
**Cons**: Creates new function on every render (negligible performance cost)

### Solution 2: Currying

\`\`\`tsx
const handleDelete = (id: number) => (e: React.MouseEvent) => {
  console.log('Event:', e);
  setTodos(todos.filter(todo => todo.id !== id));
};

<button onClick={handleDelete(todo.id)}>Delete</button>
\`\`\`

**Pros**: Avoids inline arrow function, cleaner JSX
**Cons**: Less intuitive syntax

### Solution 3: Data Attributes

\`\`\`tsx
<button data-id={todo.id} onClick={(e) => {
  const id = Number(e.currentTarget.dataset.id);
  handleDelete(id);
}}>Delete</button>
\`\`\`

**Pros**: Works without closures
**Cons**: Type safety lost (all data attributes are strings)

## preventDefault() and stopPropagation()

### preventDefault()

Prevents the browser's default behavior.

\`\`\`tsx
function PreventDefaultExamples() {
  // Prevent form submission (page reload)
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Form submitted without page reload');
  };
  
  // Prevent link navigation
  const handleLinkClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    console.log('Link clicked but not navigated');
  };
  
  // Prevent context menu
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    console.log('Custom context menu');
  };
  
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="text" />
        <button type="submit">Submit</button>
      </form>
      
      <a href="https://example.com" onClick={handleLinkClick}>
        Click me (won't navigate)
      </a>
      
      <div onContextMenu={handleContextMenu}>
        Right-click me
      </div>
    </div>
  );
}
\`\`\`

### stopPropagation()

Stops event from bubbling up to parent elements.

\`\`\`tsx
function StopPropagationExample() {
  const handleParentClick = () => {
    console.log('Parent clicked');
  };
  
  const handleChildClick = (e: React.MouseEvent) => {
    e.stopPropagation();  // Stops event from reaching parent
    console.log('Child clicked');
  };
  
  return (
    <div onClick={handleParentClick} style={{ padding: 20, background: 'lightblue' }}>
      Parent div
      
      <button onClick={handleChildClick} style={{ marginLeft: 20 }}>
        Child button (click won't trigger parent)
      </button>
      
      <button style={{ marginLeft: 20 }}>
        Another button (click WILL trigger parent)
      </button>
    </div>
  );
}

// Real-world example: Modal
function Modal({ onClose, children }) {
  const handleBackdropClick = () => {
    onClose();  // Close modal when clicking backdrop
  };
  
  const handleContentClick = (e: React.MouseEvent) => {
    e.stopPropagation();  // Don't close when clicking modal content
  };
  
  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content" onClick={handleContentClick}>
        {children}
      </div>
    </div>
  );
}
\`\`\`

## Event Delegation

React handles event delegation automatically—you don't need to worry about it!

\`\`\`tsx
// React attaches ONE event listener at root level
// Then delegates to specific handlers

function List() {
  return (
    <ul>
      {/* Each onClick doesn't create separate DOM event listener */}
      <li onClick={() => console.log('Item 1')}>Item 1</li>
      <li onClick={() => console.log('Item 2')}>Item 2</li>
      <li onClick={() => console.log('Item 3')}>Item 3</li>
      {/* ... 1000 items */}
    </ul>
  );
}

// React optimizes this automatically via event delegation
// Performance: O(1) event listeners, not O(n)
\`\`\`

## TypeScript Event Types

\`\`\`tsx
import React from 'react';

function TypedEvents() {
  // Mouse events
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log('Button clicked');
  };
  
  // Keyboard events
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    console.log('Key:', e.key);
  };
  
  // Form events
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
  };
  
  // Change events
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('Value:', e.target.value);
  };
  
  // Focus events
  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    console.log('Focused');
  };
  
  // Generic events
  const handleEvent = (e: React.SyntheticEvent<HTMLElement>) => {
    console.log('Generic event');
  };
  
  return (
    <div>
      <button onClick={handleClick}>Click</button>
      <input onKeyDown={handleKeyDown} />
      <form onSubmit={handleSubmit}>
        <input onChange={handleChange} onFocus={handleFocus} />
      </form>
    </div>
  );
}
\`\`\`

### Common Event Type Patterns

| Element | Event | Type |
|---------|-------|------|
| \`<button>\` | onClick | \`React.MouseEvent<HTMLButtonElement>\` |
| \`<input>\` | onChange | \`React.ChangeEvent<HTMLInputElement>\` |
| \`<input>\` | onKeyDown | \`React.KeyboardEvent<HTMLInputElement>\` |
| \`<form>\` | onSubmit | \`React.FormEvent<HTMLFormElement>\` |
| \`<div>\` | onClick | \`React.MouseEvent<HTMLDivElement>\` |
| \`<textarea>\` | onChange | \`React.ChangeEvent<HTMLTextAreaElement>\` |
| \`<select>\` | onChange | \`React.ChangeEvent<HTMLSelectElement>\` |

## Real-World Examples

### Example 1: Search with Debouncing

\`\`\`tsx
import { useState, useEffect } from 'react';

function SearchBar() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  
  // Debounce: Wait 300ms after user stops typing
  useEffect(() => {
    const timer = setTimeout(() => {
      if (query) {
        fetch(\`/api/search?q=\${query}\`)
          .then(res => res.json())
          .then(data => setResults(data));
      }
    }, 300);
    
    return () => clearTimeout(timer);  // Cleanup
  }, [query]);
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  };
  
  return (
    <div>
      <input
        value={query}
        onChange={handleChange}
        placeholder="Search..."
      />
      <ul>
        {results.map(result => (
          <li key={result.id}>{result.title}</li>
        ))}
      </ul>
    </div>
  );
}
\`\`\`

### Example 2: Keyboard Shortcuts

\`\`\`tsx
function Editor() {
  const [content, setContent] = useState('');
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd + S: Save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      handleSave();
    }
    
    // Ctrl/Cmd + B: Bold
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
      e.preventDefault();
      handleBold();
    }
    
    // Escape: Cancel
    if (e.key === 'Escape') {
      handleCancel();
    }
  };
  
  const handleSave = () => {
    console.log('Saving...', content);
  };
  
  const handleBold = () => {
    console.log('Toggle bold');
  };
  
  const handleCancel = () => {
    console.log('Cancel');
  };
  
  return (
    <textarea
      value={content}
      onChange={(e) => setContent(e.target.value)}
      onKeyDown={handleKeyDown}
      placeholder="Type here (Ctrl+S to save, Ctrl+B for bold)"
    />
  );
}
\`\`\`

### Example 3: Double Click to Edit

\`\`\`tsx
function EditableText({ initialText, onSave }) {
  const [isEditing, setIsEditing] = useState(false);
  const [text, setText] = useState(initialText);
  
  const handleDoubleClick = () => {
    setIsEditing(true);
  };
  
  const handleBlur = () => {
    setIsEditing(false);
    onSave(text);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setIsEditing(false);
      onSave(text);
    }
    if (e.key === 'Escape') {
      setText(initialText);
      setIsEditing(false);
    }
  };
  
  return isEditing ? (
    <input
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoFocus
    />
  ) : (
    <span onDoubleClick={handleDoubleClick}>
      {text}
    </span>
  );
}
\`\`\`

## Performance Considerations

### Issue: Creating Functions in Render

\`\`\`tsx
// ❌ Creates new function on every render
function List({ items }) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id}>
          <button onClick={() => console.log(item.id)}>
            Click
          </button>
        </li>
      ))}
    </ul>
  );
}

// For 1000 items: Creates 1000 new functions every render
// Usually fine! Modern JavaScript is fast
// Only optimize if profiling shows it's slow
\`\`\`

### Solution: useCallback (If Needed)

\`\`\`tsx
import { useCallback } from 'react';

function List({ items, onItemClick }) {
  const handleClick = useCallback((id: number) => {
    onItemClick(id);
  }, [onItemClick]);
  
  return (
    <ul>
      {items.map(item => (
        <MemoizedItem
          key={item.id}
          item={item}
          onClick={handleClick}
        />
      ))}
    </ul>
  );
}

const MemoizedItem = React.memo(({ item, onClick }) => {
  return (
    <li>
      <button onClick={() => onClick(item.id)}>
        {item.text}
      </button>
    </li>
  );
});
\`\`\`

**When to optimize**: Only if profiling shows performance issues with large lists (1000+ items).

## What's Next?

Now that you understand event handling, you're ready to learn **Conditional Rendering**—how to show/hide UI based on conditions. You'll learn:
- if/else patterns
- Ternary operators
- Logical && operator
- Switch statements
- Preventing component render

Events + State + Conditional Rendering = Interactive UIs! 🚀
`,
};
