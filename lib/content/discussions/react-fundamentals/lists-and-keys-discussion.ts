export const listsAndKeysDiscussion = {
  title: 'Lists & Keys Discussion Questions',
  id: 'lists-and-keys-discussion',
  sectionId: 'lists-and-keys',
  questions: [
    {
      id: 'q1',
      question:
        "Explain why using array index as a key can cause bugs in React applications. Provide a detailed example showing how React's reconciliation algorithm behaves differently with index keys versus stable ID keys when items are reordered or removed.",
      answer: `Using array index as a key is one of the most common React anti-patterns, leading to subtle but critical bugs. Here's why:

**How React Uses Keys:**

React uses keys to identify which items have changed, been added, or been removed during reconciliation. Keys tell React the **identity** of each element across renders.

**The Problem with Index Keys:**

When you use index as key, you're saying "this element's identity is its position," not "this element's identity is this specific piece of data."

**Example Bug: Reordering**

\`\`\`tsx
function TodoList() {
  const [todos, setTodos] = useState([
    { text: 'Buy milk', done: false },
    { text: 'Walk dog', done: true },
    { text: 'Write code', done: false }
  ]);
  
  return (
    <ul>
      {todos.map((todo, index) => (
        <li key={index}>
          <input 
            type="checkbox" 
            defaultChecked={todo.done}
          />
          <span>{todo.text}</span>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

**Initial Render:**
\`\`\`
key=0: [ ] Buy milk
key=1: [✓] Walk dog
key=2: [ ] Write code
\`\`\`

**After moving "Walk dog" to top:**
\`\`\`tsx
setTodos([
  { text: 'Walk dog', done: true },   // now index 0
  { text: 'Buy milk', done: false },  // now index 1
  { text: 'Write code', done: false } // now index 2
]);
\`\`\`

**What React sees (with index keys):**
\`\`\`
Before:
  key=0: "Buy milk", done=false
  key=1: "Walk dog", done=true
  key=2: "Write code", done=false

After:
  key=0: "Walk dog", done=true
  key=1: "Buy milk", done=false
  key=2: "Write code", done=false

React thinks:
  - key=0: text changed from "Buy milk" to "Walk dog", done changed from false to true
  - key=1: text changed from "Walk dog" to "Buy milk", done changed from true to false
  - key=2: unchanged
\`\`\`

**Result:** React updates text content of ALL THREE items, even though only order changed!

**With \`defaultChecked\` (Uncontrolled Input):**

The bug gets worse:

\`\`\`tsx
<input type="checkbox" defaultChecked={todo.done} />
\`\`\`

\`defaultChecked\` only sets initial value. React won't update it on re-renders.

**What happens:**
\`\`\`
Initial:
  DOM node 0: [ ] Buy milk
  DOM node 1: [✓] Walk dog  ← checked
  DOM node 2: [ ] Write code

After reorder (React reuses DOM nodes):
  DOM node 0: [✓] Walk dog  ← Still checked! (DOM state preserved)
  DOM node 1: [ ] Buy milk  ← Unchecked now (DOM state preserved)
  DOM node 2: [ ] Write code

But text updated:
  DOM node 0: text changed to "Walk dog"
  DOM node 1: text changed to "Buy milk"
\`\`\`

**Bug:** The checkbox that was checked for "Walk dog" is now on "Walk dog" (correct by accident!), but if we had deleted "Buy milk" instead:

\`\`\`
Before:
  key=0: [ ] Buy milk
  key=1: [✓] Walk dog
  key=2: [ ] Write code

After deleting index 0:
  key=0: [✓] Walk dog  ← Reuses DOM node from old key=0!
  key=1: [ ] Write code ← Reuses DOM node from old key=1!

Result: The checkbox meant for "Walk dog" now appears on "Walk dog" 
        but only because of the reordering. In reality:
  - Old DOM node 0 (unchecked) reused for "Walk dog"
  - Old DOM node 1 (checked) reused for "Write code"
  
BUG: "Write code" appears checked even though it shouldn't be!
\`\`\`

**Correct Solution: Stable IDs**

\`\`\`tsx
const [todos, setTodos] = useState([
  { id: 1, text: 'Buy milk', done: false },
  { id: 2, text: 'Walk dog', done: true },
  { id: 3, text: 'Write code', done: false }
]);

{todos.map(todo => (
  <li key={todo.id}>
    <input 
      type="checkbox" 
      defaultChecked={todo.done}
    />
    <span>{todo.text}</span>
  </li>
))}
\`\`\`

**After reordering:**
\`\`\`
Before:
  key=1: "Buy milk", done=false
  key=2: "Walk dog", done=true
  key=3: "Write code", done=false

After:
  key=2: "Walk dog", done=true
  key=1: "Buy milk", done=false
  key=3: "Write code", done=false

React knows:
  - key=1: moved to position 1 → MOVE existing DOM node
  - key=2: moved to position 0 → MOVE existing DOM node
  - key=3: stayed at position 2 → no change

Result: React MOVES DOM nodes (with their state intact), doesn't update any content!
\`\`\`

**Performance Comparison:**

**With index keys:**
- React updates content of all items
- Expensive: text updates, attribute changes
- Wrong: state can attach to wrong items

**With stable ID keys:**
- React moves existing DOM nodes
- Cheap: just DOM reordering
- Correct: state stays with correct items

**When Index Keys are OK:**

Only when ALL of these are true:
1. List is static (never changes)
2. Items never reorder
3. Items never get deleted/inserted
4. Items don't have state (checkboxes, inputs, etc.)

**Example where index is fine:**
\`\`\`tsx
// Static navigation menu
const navItems = ['Home', 'About', 'Contact'];

<nav>
  {navItems.map((item, index) => (
    <a key={index} href={\`#\${item}\`}>{item}</a>
  ))}
</nav>
\`\`\`

**Interview Insight:**
Explaining this bug demonstrates deep understanding of React's reconciliation algorithm and the importance of component identity. Many developers use index keys without understanding the consequences—showing you know why stable IDs matter sets you apart.`,
    },
    {
      id: 'q2',
      question:
        'Discuss strategies for optimizing the performance of large lists in React (1000+ items). Compare different approaches including virtualization, memoization, and pagination. When would you use each strategy?',
      answer: `Rendering large lists efficiently is critical for performance. Here are comprehensive strategies:

**The Problem:**

\`\`\`tsx
// ❌ Rendering 10,000 items = 10,000 DOM nodes = SLOW
function UserList({ users }) {  // users.length = 10,000
  return (
    <div>
      {users.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

// Problems:
// - Initial render: ~2-5 seconds
// - Scroll: janky, laggy
// - Memory: hundreds of MB
// - Re-renders: entire list updates
\`\`\`

**Strategy 1: Virtualization (Windowing)**

**Concept:** Only render items visible in viewport + small buffer.

\`\`\`tsx
import { FixedSizeList } from 'react-window';

function VirtualizedUserList({ users }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <UserCard user={users[index]} />
    </div>
  );
  
  return (
    <FixedSizeList
      height={600}              // Visible height
      itemCount={users.length}  // Total items (10,000)
      itemSize={80}             // Height per item
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}

// Only renders ~8-10 items at a time (600px / 80px + buffer)
// Performance: instant, regardless of list size!
\`\`\`

**Benefits:**
- ✅ Constant render time (doesn't scale with list size)
- ✅ Low memory usage (only visible items in DOM)
- ✅ Smooth scrolling
- ✅ Works with millions of items

**Drawbacks:**
- ❌ Requires fixed item heights (or variable with calculations)
- ❌ Adds library dependency
- ❌ More complex than naive list
- ❌ Screen readers may struggle
- ❌ Search (Ctrl+F) won't find off-screen items

**When to use:**
- 1000+ items
- Known/fixed item heights
- Performance is critical
- Scrollable container

**Libraries:**
- \`react-window\` (lightweight, 3KB)
- \`react-virtualized\` (feature-rich, 30KB)
- \`@tanstack/react-virtual\` (modern, flexible)

**Advanced: Variable height virtualization:**
\`\`\`tsx
import { VariableSizeList } from 'react-window';

function VariableHeightList({ items }) {
  const listRef = useRef();
  const rowHeights = useRef({});
  
  function getItemSize(index) {
    return rowHeights.current[index] || 80;  // Default 80px
  }
  
  function setRowHeight(index, size) {
    rowHeights.current[index] = size;
    listRef.current.resetAfterIndex(index);  // Recalculate layout
  }
  
  const Row = ({ index, style }) => (
    <div style={style}>
      <ResizeObserverComponent
        onResize={(height) => setRowHeight(index, height)}
      >
        <ItemContent item={items[index]} />
      </ResizeObserverComponent>
    </div>
  );
  
  return (
    <VariableSizeList
      ref={listRef}
      height={600}
      itemCount={items.length}
      itemSize={getItemSize}
      width="100%"
    >
      {Row}
    </VariableSizeList>
  );
}
\`\`\`

**Strategy 2: Pagination**

**Concept:** Split list into pages, show N items at a time.

\`\`\`tsx
function PaginatedList({ items, itemsPerPage = 50 }) {
  const [page, setPage] = useState(0);
  
  const pageCount = Math.ceil(items.length / itemsPerPage);
  const currentItems = items.slice(
    page * itemsPerPage,
    (page + 1) * itemsPerPage
  );
  
  return (
    <div>
      <ul>
        {currentItems.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
      
      <div className="pagination">
        <button 
          onClick={() => setPage(p => p - 1)}
          disabled={page === 0}
        >
          Previous
        </button>
        <span>Page {page + 1} of {pageCount}</span>
        <button 
          onClick={() => setPage(p => p + 1)}
          disabled={page >= pageCount - 1}
        >
          Next
        </button>
      </div>
    </div>
  );
}
\`\`\`

**Benefits:**
- ✅ Simple to implement
- ✅ No library needed
- ✅ Works with any content
- ✅ Screen reader friendly
- ✅ Shareable page URLs (with query params)
- ✅ Backend can paginate queries

**Drawbacks:**
- ❌ Extra clicks to see more items
- ❌ Can't see full list at once
- ❌ Not great for browsing/scanning

**When to use:**
- Search results
- Tables with many rows
- API returns paginated data
- Need URL-based navigation

**Strategy 3: Infinite Scroll**

**Concept:** Load more items as user scrolls to bottom.

\`\`\`tsx
function InfiniteScrollList({ fetchItems }) {
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  
  const observerRef = useRef();
  const lastItemRef = useCallback(node => {
    if (isLoading) return;
    if (observerRef.current) observerRef.current.disconnect();
    
    observerRef.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMore) {
        setPage(p => p + 1);
      }
    });
    
    if (node) observerRef.current.observe(node);
  }, [isLoading, hasMore]);
  
  useEffect(() => {
    setIsLoading(true);
    fetchItems(page).then(newItems => {
      setItems(prev => [...prev, ...newItems]);
      setHasMore(newItems.length > 0);
      setIsLoading(false);
    });
  }, [page]);
  
  return (
    <div>
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        return (
          <div
            key={item.id}
            ref={isLast ? lastItemRef : null}
          >
            {item.name}
          </div>
        );
      })}
      {isLoading && <LoadingSpinner />}
    </div>
  );
}
\`\`\`

**Benefits:**
- ✅ Smooth UX (no pagination clicks)
- ✅ Great for browsing/scrolling
- ✅ Loads data on-demand

**Drawbacks:**
- ❌ Can't jump to end
- ❌ Hard to find specific items
- ❌ Memory grows unbounded
- ❌ Scroll position issues on navigation

**When to use:**
- Social media feeds
- Image galleries
- News feeds
- Mobile-first apps

**Strategy 4: React.memo + useCallback**

**Concept:** Prevent unnecessary re-renders of list items.

\`\`\`tsx
// ❌ Without optimization: Every item re-renders when parent updates
function TodoList() {
  const [todos, setTodos] = useState([...]);
  const [filter, setFilter] = useState('all');
  
  return (
    <>
      <FilterButtons filter={filter} setFilter={setFilter} />
      <ul>
        {todos.map(todo => (
          <TodoItem key={todo.id} todo={todo} onToggle={handleToggle} />
        ))}
      </ul>
    </>
  );
}

// When filter changes, ALL TodoItems re-render!

// ✅ With optimization: Only affected items re-render
const TodoItem = memo(function TodoItem({ todo, onToggle }) {
  return (
    <li>
      <input 
        type="checkbox" 
        checked={todo.done}
        onChange={() => onToggle(todo.id)}
      />
      {todo.text}
    </li>
  );
});

function TodoList() {
  const [todos, setTodos] = useState([...]);
  const [filter, setFilter] = useState('all');
  
  // Stable function reference
  const handleToggle = useCallback((id) => {
    setTodos(prev => prev.map(t => 
      t.id === id ? { ...t, done: !t.done } : t
    ));
  }, []);
  
  return (
    <>
      <FilterButtons filter={filter} setFilter={setFilter} />
      <ul>
        {todos.map(todo => (
          <TodoItem key={todo.id} todo={todo} onToggle={handleToggle} />
        ))}
      </ul>
    </>
  );
}

// When filter changes, TodoItems DON'T re-render (props unchanged)!
\`\`\`

**Benefits:**
- ✅ Reduces wasted renders
- ✅ No library needed
- ✅ Works with any list

**Drawbacks:**
- ❌ More complex code
- ❌ Need to memoize callbacks too
- ❌ Doesn't reduce initial render

**When to use:**
- 100-1000 items (too few: not worth it, too many: use virtualization)
- Frequent parent updates
- Expensive item render

**Strategy Comparison Table:**

| Strategy | Best For | Items | Complexity | Memory | Accessibility |
|----------|----------|-------|------------|--------|---------------|
| Virtualization | 1000+ | Unlimited | Medium | Low | Poor |
| Pagination | Search results | Unlimited | Low | Low | Excellent |
| Infinite scroll | Feeds | 10,000+ | Medium | High | Good |
| Memoization | Reduce re-renders | 100-1000 | Low | Same | Excellent |
| Combo: Virtual + Infinite | Massive datasets | Unlimited | High | Low | Poor |

**Decision Framework:**

\`\`\`
How many items?
├─ < 100: No optimization needed
├─ 100-500: Consider React.memo
├─ 500-2000: Pagination or virtualization
└─ 2000+: Virtualization (+ infinite scroll)

What's the use case?
├─ Table/structured data: Pagination + virtualization
├─ Feed/timeline: Infinite scroll (+ virtualization if needed)
├─ Search results: Pagination
└─ Form with many fields: Virtualization

Performance requirements?
├─ Must be instant: Virtualization
├─ Can have some delay: Pagination
└─ Progressive loading OK: Infinite scroll
\`\`\`

**Interview Insight:**
Discussing multiple strategies and their tradeoffs shows architectural thinking. Mentioning specific numbers (when to use each) demonstrates practical experience. Bringing up accessibility concerns shows senior-level awareness.`,
    },
    {
      id: 'q3',
      question:
        'How should you handle dynamically changing lists where items can be added, removed, or reordered? Discuss the importance of stable keys and potential pitfalls when generating keys on the fly.',
      answer: `Handling dynamic lists correctly is crucial for React apps with user interactions. The key challenge: maintaining stable, unique keys as the list changes.

**The Core Principle:**

> A key must **uniquely identify** a piece of data **consistently across renders**.

**Good Keys: From Your Data**

\`\`\`tsx
// ✅ BEST: Use database ID
const todos = [
  { id: 1, text: 'Task 1' },  // ID from backend
  { id: 2, text: 'Task 2' }
];

{todos.map(todo => <Todo key={todo.id} todo={todo} />)}

// When reordered: key=1 still refers to "Task 1"
// When deleted: React knows exactly which DOM node to remove
// When added: React creates new DOM node with new key
\`\`\`

**Bad Keys: Generated Each Render**

\`\`\`tsx
// ❌ TERRIBLE: Random key (different every render!)
{todos.map(todo => (
  <Todo key={Math.random()} todo={todo} />
))}

// Every render:
// - React sees "new" keys
// - Unmounts all components
// - Mounts "new" components
// - State is lost
// - Performance disaster
\`\`\`

**Dynamic List Patterns:**

**Pattern 1: Adding Items**

\`\`\`tsx
// ✅ Generate ID when creating item
import { v4 as uuidv4 } from 'uuid';

function TodoApp() {
  const [todos, setTodos] = useState([]);
  
  function addTodo(text) {
    const newTodo = {
      id: uuidv4(),  // Generate once, never changes
      text,
      done: false,
      createdAt: Date.now()
    };
    
    setTodos(prev => [...prev, newTodo]);
  }
  
  return (
    <>
      <AddTodoForm onAdd={addTodo} />
      <ul>
        {todos.map(todo => (
          <TodoItem key={todo.id} todo={todo} />
        ))}
      </ul>
    </>
  );
}
\`\`\`

**Why this works:**
- ID generated once when item created
- Same ID across all re-renders
- React can track item reliably

**Pattern 2: Removing Items**

\`\`\`tsx
function TodoApp() {
  const [todos, setTodos] = useState([
    { id: 'a1', text: 'Task 1' },
    { id: 'a2', text: 'Task 2' },
    { id: 'a3', text: 'Task 3' }
  ]);
  
  function removeTodo(idToRemove) {
    setTodos(prev => prev.filter(todo => todo.id !== idToRemove));
  }
  
  // When removing id='a2':
  // Before: [key=a1, key=a2, key=a3]
  // After:  [key=a1, key=a3]
  // React knows: Remove DOM node with key=a2, keep others
  
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem 
          key={todo.id}
          todo={todo} 
          onRemove={() => removeTodo(todo.id)}
        />
      ))}
    </ul>
  );
}
\`\`\`

**Pattern 3: Reordering Items**

\`\`\`tsx
function TodoApp() {
  const [todos, setTodos] = useState([...]);
  
  function moveUp(index) {
    if (index === 0) return;
    
    setTodos(prev => {
      const newTodos = [...prev];
      [newTodos[index], newTodos[index - 1]] = 
        [newTodos[index - 1], newTodos[index]];
      return newTodos;
    });
  }
  
  // Before: [key=1, key=2, key=3]
  // After:  [key=2, key=1, key=3]
  // React: Move DOM nodes (not update content!)
  
  return (
    <ul>
      {todos.map((todo, index) => (
        <TodoItem 
          key={todo.id}  // NOT index!
          todo={todo} 
          onMoveUp={() => moveUp(index)}
        />
      ))}
    </ul>
  );
}
\`\`\`

**Pattern 4: Updating Items**

\`\`\`tsx
function toggleTodo(idToToggle) {
  setTodos(prev => prev.map(todo =>
    todo.id === idToToggle
      ? { ...todo, done: !todo.done }  // New object, SAME id
      : todo
  ));
}

// Before: { id: 'a1', text: 'Task', done: false }
// After:  { id: 'a1', text: 'Task', done: true }
// React: Same key, different props → update existing DOM node
\`\`\`

**Common Pitfall: Generating Keys from Content**

\`\`\`tsx
// ❌ BAD: Key derived from content
{todos.map(todo => (
  <TodoItem key={todo.text} todo={todo} />
))}

// Problems:
// 1. Duplicate text = duplicate keys!
{[
  { text: 'Buy milk' },  // key = "Buy milk"
  { text: 'Buy milk' }   // key = "Buy milk" (DUPLICATE!)
]}

// 2. Editing text changes key!
// Before: { id: 1, text: 'Taks' }      // key = "Taks"
// After:  { id: 1, text: 'Task' }     // key = "Task"
// React sees: Item removed + new item added (WRONG!)
\`\`\`

**When You Don't Have IDs:**

**Option 1: Generate stable IDs from data + position (if order is stable)**
\`\`\`tsx
const items = ['Apple', 'Banana', 'Cherry'];

// ✅ OK if list never reorders and items are unique
{items.map(item => <Item key={item} item={item} />)}

// ⚠️ RISKY if list order changes
\`\`\`

**Option 2: Add IDs when receiving data**
\`\`\`tsx
useEffect(() => {
  fetch('/api/items')
    .then(r => r.json())
    .then(data => {
      // Add IDs if data doesn't have them
      const itemsWithIds = data.map((item, index) => ({
        ...item,
        id: item.id || \`item-\${index}-\${item.name}\`
      }));
      setItems(itemsWithIds);
    });
}, []);
\`\`\`

**Option 3: Use Map instead of Array**
\`\`\`tsx
// Store items in Map with generated IDs
const [items, setItems] = useState(new Map());

function addItem(text) {
  const id = uuidv4();
  setItems(prev => new Map(prev).set(id, { text, done: false }));
}

function removeItem(id) {
  setItems(prev => {
    const next = new Map(prev);
    next.delete(id);
    return next;
  });
}

return (
  <ul>
    {Array.from(items.entries()).map(([id, item]) => (
      <TodoItem key={id} item={item} onRemove={() => removeItem(id)} />
    ))}
  </ul>
);
\`\`\`

**Advanced: Optimistic Updates**

When adding items optimistically (before server confirms):

\`\`\`tsx
function addTodoOptimistically(text) {
  const tempId = \`temp-\${Date.now()}\`;  // Temporary ID
  const optimisticTodo = {
    id: tempId,
    text,
    done: false,
    isPending: true  // Mark as pending
  };
  
  // Add immediately to UI
  setTodos(prev => [...prev, optimisticTodo]);
  
  // Send to server
  fetch('/api/todos', {
    method: 'POST',
    body: JSON.stringify({ text })
  })
    .then(r => r.json())
    .then(serverTodo => {
      // Replace temp with real
      setTodos(prev => prev.map(t =>
        t.id === tempId ? { ...serverTodo, isPending: false } : t
      ));
    })
    .catch(err => {
      // Rollback on error
      setTodos(prev => prev.filter(t => t.id !== tempId));
      showError('Failed to add todo');
    });
}
\`\`\`

**Key Stability Checklist:**

✅ **DO:**
- Use database IDs when available
- Generate UUIDs when creating items
- Derive keys from truly unique, stable data
- Keep same key when updating item properties

❌ **DON'T:**
- Use Math.random() or Date.now() as keys
- Use array index (unless list is static)
- Generate new keys on every render
- Use mutable data as keys
- Derive keys from user-editable content

**Debugging Keys:**

\`\`\`tsx
// Add console.log to see key behavior
{todos.map(todo => {
  console.log(\`Rendering todo \${todo.id}\`);
  return <TodoItem key={todo.id} todo={todo} />;
})}

// If you see "Rendering todo X" for unchanged items, keys might be wrong

// Or use React DevTools Profiler:
// - Shows which components re-rendered
// - Shows why they re-rendered
// - Helps identify key-related issues
\`\`\`

**Interview Insight:**
Discussing stable keys, ID generation strategies, and optimistic updates shows deep understanding of React's reconciliation and production-ready patterns. Mentioning the difference between "update existing item" vs "remove + add new item" demonstrates mastery of React's mental model.`,
    },
  ],
};
