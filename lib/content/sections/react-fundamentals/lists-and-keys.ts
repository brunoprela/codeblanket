export const listsAndKeys = {
  title: 'Lists & Keys',
  id: 'lists-and-keys',
  content: `
# Lists & Keys

## Introduction

**Rendering lists** is one of the most common patterns in React—displaying arrays of data as UI elements. Whether it's a list of todos, user profiles, product catalog, or chat messages, you'll use the same core pattern: **map() with keys**.

Understanding keys is critical—they're not just a requirement to silence warnings, they're essential for React's performance and correctness.

### The Core Pattern

\`\`\`tsx
const todos = ['Learn React', 'Build app', 'Deploy'];

function TodoList() {
  return (
    <ul>
      {todos.map((todo, index) => (
        <li key={index}>{todo}</li>
      ))}
    </ul>
  );
}
\`\`\`

In this section, you'll learn:
- How to render lists with map()
- Why keys are critical
- How to choose the right keys
- Common list patterns
- Performance optimization

## Rendering Lists with map()

JavaScript's \`map()\` method transforms arrays—perfect for transforming data into JSX.

\`\`\`tsx
// Data
const users = [
  { id: 1, name: 'Alice', email: 'alice@example.com' },
  { id: 2, name: 'Bob', email: 'bob@example.com' },
  { id: 3, name: 'Charlie', email: 'charlie@example.com' }
];

// Component
function UserList() {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>
          <strong>{user.name}</strong>
          <br />
          <span>{user.email}</span>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

**How map() works:**
\`\`\`tsx
// map() takes each item and returns new value
const numbers = [1, 2, 3];
const doubled = numbers.map(n => n * 2);  // [2, 4, 6]

// With JSX, map() returns array of elements
const items = numbers.map(n => <li key={n}>{n}</li>);
// Returns: [<li>1</li>, <li>2</li>, <li>3</li>]

// React renders arrays naturally
return <ul>{items}</ul>;
\`\`\`

## Why Keys Matter

Keys tell React which items have changed, been added, or been removed. Without proper keys, React can't track list items correctly.

### Without Keys: React Gets Confused

\`\`\`tsx
// ❌ No keys
<ul>
  {users.map(user => <li>{user.name}</li>)}
</ul>

// Console warning:
// "Warning: Each child in a list should have a unique "key" prop."
\`\`\`

**What happens without keys:**
1. React can't tell items apart
2. May reuse wrong DOM nodes
3. State gets attached to wrong components
4. Poor performance on reorders
5. Bugs with controlled inputs

### With Keys: React Tracks Correctly

\`\`\`tsx
// ✅ With keys
<ul>
  {users.map(user => <li key={user.id}>{user.name}</li>)}
</ul>
\`\`\`

**Visualization of key behavior:**
\`\`\`tsx
// Initial render
<li key="1">Alice</li>
<li key="2">Bob</li>
<li key="3">Charlie</li>

// After removing Bob (key="2")
<li key="1">Alice</li>
<li key="3">Charlie</li>

// React knows:
// - key="1" unchanged → reuse DOM node
// - key="2" removed → remove DOM node
// - key="3" unchanged → reuse DOM node
\`\`\`

## The Key Problem: Index as Key

Using array index as key is common but dangerous:

\`\`\`tsx
// ❌ WRONG: Index as key
{todos.map((todo, index) => (
  <li key={index}>{todo.text}</li>
))}
\`\`\`

**Why this is bad:**

\`\`\`tsx
// Initial: 3 todos
[
  <li key="0">Learn React</li>,    // index 0
  <li key="1">Build app</li>,      // index 1
  <li key="2">Deploy</li>          // index 2
]

// After deleting "Build app":
[
  <li key="0">Learn React</li>,    // index 0 (unchanged)
  <li key="1">Deploy</li>          // index 1 (WAS index 2!)
]

// React sees:
// - key="0" unchanged → keeps DOM node
// - key="1" changed content → updates DOM node (unnecessary!)
// - key="2" removed → removes DOM node

// Problem: React updated wrong item!
\`\`\`

**Real-world bug example:**
\`\`\`tsx
function TodoList() {
  const [todos, setTodos] = useState([
    { text: 'Buy milk', done: false },
    { text: 'Walk dog', done: false },
    { text: 'Write code', done: false }
  ]);
  
  return (
    <ul>
      {todos.map((todo, index) => (
        // ❌ Using index as key
        <li key={index}>
          <input 
            type="checkbox" 
            checked={todo.done}
            onChange={() => {
              const newTodos = [...todos];
              newTodos[index].done = !newTodos[index].done;
              setTodos(newTodos);
            }}
          />
          <span>{todo.text}</span>
          <button onClick={() => {
            setTodos(todos.filter((_, i) => i !== index));
          }}>
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}

// BUG SCENARIO:
// 1. Check "Walk dog" (index 1)
// 2. Delete "Buy milk" (index 0)
// 3. Now "Walk dog" becomes index 0
// 4. But the checked checkbox stays at index 1!
// 5. "Write code" now appears checked instead!
\`\`\`

**When index as key is OK:**
- ✅ List never reorders
- ✅ List never filters
- ✅ Items never added/removed from middle
- ✅ Static list (never changes)

\`\`\`tsx
// ✅ OK: Static navigation menu
const navItems = ['Home', 'About', 'Contact'];  // Never changes

<nav>
  {navItems.map((item, index) => (
    <a key={index} href={\`#\${item.toLowerCase()}\`}>
      {item}
    </a>
  ))}
</nav>
\`\`\`

## Choosing Good Keys

**Best keys:** Stable, unique identifiers from your data.

\`\`\`tsx
// ✅ GOOD: Database ID
{users.map(user => (
  <UserCard key={user.id} user={user} />
))}

// ✅ GOOD: UUID
{posts.map(post => (
  <Post key={post.uuid} post={post} />
))}

// ✅ GOOD: Combination of fields (if truly unique)
{comments.map(comment => (
  <Comment key={\`\${comment.userId}-\${comment.timestamp}\`} comment={comment} />
))}

// ❌ BAD: Index
{items.map((item, i) => <Item key={i} item={item} />)}

// ❌ BAD: Random value (changes every render!)
{items.map(item => <Item key={Math.random()} item={item} />)}

// ❌ BAD: Object reference (not stable)
{items.map(item => <Item key={item} item={item} />)}
\`\`\`

**Key requirements:**
1. **Unique among siblings** (not globally)
2. **Stable** (same item = same key across renders)
3. **Predictable** (derivable from data, not random)

### Generating Keys for Data Without IDs

If your data lacks IDs, you have options:

**Option 1: Generate IDs when creating data**
\`\`\`tsx
import { v4 as uuidv4 } from 'uuid';

function TodoForm() {
  const [todos, setTodos] = useState([]);
  
  function addTodo(text) {
    setTodos([
      ...todos,
      { id: uuidv4(), text, done: false }  // Add ID when creating
    ]);
  }
  
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
\`\`\`

**Option 2: Use content as key (if truly unique)**
\`\`\`tsx
// If emails are unique
{users.map(user => (
  <User key={user.email} user={user} />
))}

// If names + timestamps are unique
{messages.map(msg => (
  <Message key={\`\${msg.author}-\${msg.timestamp}\`} msg={msg} />
))}
\`\`\`

**Option 3: Add IDs to existing data**
\`\`\`tsx
// Transform data once when it arrives
function addIdsToData(items) {
  return items.map((item, index) => ({
    ...item,
    id: \`\${item.name}-\${index}\`  // Stable if data order doesn't change
  }));
}

const itemsWithIds = useMemo(() => addIdsToData(rawData), [rawData]);
\`\`\`

## Keys Must Be Unique Among Siblings

Keys only need to be unique within their immediate parent:

\`\`\`tsx
// ✅ GOOD: Same keys in different lists OK
function App() {
  return (
    <>
      <ul>
        {posts.map(post => (
          <li key={post.id}>{post.title}</li>
        ))}
      </ul>
      
      <ul>
        {comments.map(comment => (
          // OK even if comment.id equals some post.id
          <li key={comment.id}>{comment.text}</li>
        ))}
      </ul>
    </>
  );
}

// ❌ BAD: Duplicate keys in same list
<ul>
  <li key="1">Alice</li>
  <li key="1">Bob</li>  {/* Same key! */}
</ul>
\`\`\`

## Keys Don't Pass to Components

Keys are special—they're used by React but not passed as props:

\`\`\`tsx
function Post({ key, title }) {  // key is undefined!
  console.log(key);  // undefined
  return <div>{title}</div>;
}

// To access the same value, pass it separately
function Post({ id, title }) {
  console.log(id);  // Works!
  return <div>{title}</div>;
}

{posts.map(post => (
  <Post 
    key={post.id}    // For React
    id={post.id}     // For component
    title={post.title} 
  />
))}
\`\`\`

## Extracting List Item Components

For readability and reusability, extract list items:

\`\`\`tsx
// ❌ Inline: Gets messy
function UserList({ users }) {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>
          <img src={user.avatar} alt="" />
          <div>
            <h3>{user.name}</h3>
            <p>{user.email}</p>
            <button>Follow</button>
          </div>
        </li>
      ))}
    </ul>
  );
}

// ✅ Extracted: Clean and reusable
function UserListItem({ user }) {
  return (
    <li>
      <img src={user.avatar} alt="" />
      <div>
        <h3>{user.name}</h3>
        <p>{user.email}</p>
        <button>Follow</button>
      </div>
    </li>
  );
}

function UserList({ users }) {
  return (
    <ul>
      {users.map(user => (
        <UserListItem key={user.id} user={user} />
      ))}
    </ul>
  );
}
\`\`\`

**Key goes on the component in map(), not the root element:**
\`\`\`tsx
// ❌ WRONG: Key on li inside component
function UserListItem({ user }) {
  return <li key={user.id}>{user.name}</li>;  // Key here doesn't work!
}

{users.map(user => <UserListItem user={user} />)}  // Missing key!

// ✅ CORRECT: Key where map() is called
function UserListItem({ user }) {
  return <li>{user.name}</li>;
}

{users.map(user => <UserListItem key={user.id} user={user} />)}
\`\`\`

## Filtering and Transforming Lists

You can chain array methods before rendering:

\`\`\`tsx
interface Todo {
  id: number;
  text: string;
  done: boolean;
  priority: 'low' | 'medium' | 'high';
}

function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <div>
      <h2>High Priority Incomplete Todos</h2>
      <ul>
        {todos
          .filter(todo => !todo.done)
          .filter(todo => todo.priority === 'high')
          .map(todo => (
            <li key={todo.id}>{todo.text}</li>
          ))}
      </ul>
    </div>
  );
}

// Or combined
{todos
  .filter(t => !t.done && t.priority === 'high')
  .sort((a, b) => a.text.localeCompare(b.text))
  .map(todo => (
    <TodoItem key={todo.id} todo={todo} />
  ))}
\`\`\`

## Empty Lists

Handle empty states gracefully:

\`\`\`tsx
function ProductList({ products }) {
  if (products.length === 0) {
    return (
      <div className="empty-state">
        <p>No products found</p>
        <button>Add Product</button>
      </div>
    );
  }
  
  return (
    <div className="product-grid">
      {products.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}

// Or inline
function ProductList({ products }) {
  return (
    <div>
      {products.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="product-grid">
          {products.map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      )}
    </div>
  );
}
\`\`\`

## Nested Lists

Lists within lists—each needs its own keys:

\`\`\`tsx
interface Category {
  id: number;
  name: string;
  items: { id: number; name: string }[];
}

function NestedList({ categories }: { categories: Category[] }) {
  return (
    <div>
      {categories.map(category => (
        <div key={category.id}>
          <h2>{category.name}</h2>
          <ul>
            {category.items.map(item => (
              <li key={item.id}>{item.name}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
\`\`\`

**Key must be unique within its siblings, not globally:**
\`\`\`tsx
const data = [
  { categoryId: 1, items: [{ id: 1, name: 'A' }] },
  { categoryId: 2, items: [{ id: 1, name: 'B' }] }  // Same item.id, different category
];

// ✅ OK: item.id is unique within each category
{data.map(cat => (
  <div key={cat.categoryId}>
    {cat.items.map(item => (
      <li key={item.id}>{item.name}</li>  // OK, unique within parent
    ))}
  </div>
))}
\`\`\`

## Multiple Elements Per Item

When mapping to multiple elements, wrap in Fragment:

\`\`\`tsx
// ❌ WRONG: Can't have key on multiple elements
{users.map(user => (
  <h3 key={user.id}>{user.name}</h3>
  <p>{user.email}</p>  // Error: Adjacent JSX elements must be wrapped
))}

// ✅ Option 1: Wrapper div
{users.map(user => (
  <div key={user.id}>
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </div>
))}

// ✅ Option 2: Fragment with key (explicit syntax)
import { Fragment } from 'react';

{users.map(user => (
  <Fragment key={user.id}>
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </Fragment>
))}

// ❌ Can't use <> shorthand with key
{users.map(user => (
  <> {/* No way to add key here */}
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </>
))}
\`\`\`

## Performance Optimization

### Memoizing List Items

Prevent unnecessary re-renders of list items:

\`\`\`tsx
import { memo } from 'react';

// Without memo: Re-renders when parent updates
function TodoItem({ todo, onToggle }) {
  console.log(\`Rendering todo: \${todo.id}\`);
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
}

// With memo: Only re-renders if props change
const TodoItem = memo(function TodoItem({ todo, onToggle }) {
  console.log(\`Rendering todo: \${todo.id}\`);
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

// Parent component
function TodoList() {
  const [todos, setTodos] = useState([...]);
  
  // ⚠️ Problem: Creates new function every render!
  const handleToggle = (id) => {
    setTodos(todos.map(t => 
      t.id === id ? { ...t, done: !t.done } : t
    ));
  };
  
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem 
          key={todo.id} 
          todo={todo} 
          onToggle={handleToggle}  // New function every time!
        />
      ))}
    </ul>
  );
}

// ✅ Solution: useCallback
import { useCallback } from 'react';

function TodoList() {
  const [todos, setTodos] = useState([...]);
  
  const handleToggle = useCallback((id) => {
    setTodos(prevTodos => 
      prevTodos.map(t => t.id === id ? { ...t, done: !t.done } : t)
    );
  }, []);  // Stable function reference
  
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem 
          key={todo.id} 
          todo={todo} 
          onToggle={handleToggle}  // Same function reference
        />
      ))}
    </ul>
  );
}
\`\`\`

### Virtual Scrolling for Large Lists

For thousands of items, render only visible ones:

\`\`\`tsx
// Example with react-window
import { FixedSizeList } from 'react-window';

function LargeList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      {items[index].name}
    </div>
  );
  
  return (
    <FixedSizeList
      height={600}          // Visible height
      itemCount={items.length}
      itemSize={50}         // Height per item
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}

// Renders only ~12 items at a time (600px / 50px)
// Even if items.length is 10,000!
\`\`\`

## Common Patterns

### List with Actions

\`\`\`tsx
function TaskList({ tasks, onEdit, onDelete }) {
  return (
    <ul>
      {tasks.map(task => (
        <li key={task.id}>
          <span>{task.title}</span>
          <button onClick={() => onEdit(task.id)}>Edit</button>
          <button onClick={() => onDelete(task.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
\`\`\`

### List with Selection

\`\`\`tsx
function SelectableList({ items }) {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  
  function toggleSelection(id: number) {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }
  
  return (
    <ul>
      {items.map(item => (
        <li 
          key={item.id}
          className={selectedIds.has(item.id) ? 'selected' : ''}
          onClick={() => toggleSelection(item.id)}
        >
          {item.name}
        </li>
      ))}
    </ul>
  );
}
\`\`\`

### Sortable List

\`\`\`tsx
function SortableList({ items: initialItems }) {
  const [items, setItems] = useState(initialItems);
  const [sortBy, setSortBy] = useState<'name' | 'date'>('name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  
  const sortedItems = useMemo(() => {
    const sorted = [...items].sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else {
        return a.date - b.date;
      }
    });
    
    return sortDir === 'asc' ? sorted : sorted.reverse();
  }, [items, sortBy, sortDir]);
  
  return (
    <div>
      <button onClick={() => setSortBy('name')}>Sort by Name</button>
      <button onClick={() => setSortBy('date')}>Sort by Date</button>
      <button onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}>
        Toggle Direction
      </button>
      
      <ul>
        {sortedItems.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}
\`\`\`

## Best Practices

1. **Always provide keys** when mapping arrays
2. **Use stable, unique IDs** as keys (not index unless static)
3. **Key goes on the outermost element** in map()
4. **Extract list items** into separate components for readability
5. **Handle empty states** explicitly
6. **Memoize expensive list items** with React.memo
7. **Use useCallback** for event handlers passed to list items
8. **Consider virtual scrolling** for large lists (1000+ items)
9. **Keys must be strings or numbers** (React converts to strings internally)

## What's Next?

Now that you understand lists and keys, you're ready to learn **Forms in React**—handling user input, controlled components, validation, and submission. Forms + lists = complete CRUD applications! 🚀
`,
};
