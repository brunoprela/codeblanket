export const listsAndKeysQuiz = {
  title: 'Lists & Keys Quiz',
  id: 'lists-and-keys-quiz',
  sectionId: 'lists-and-keys',
  questions: [
    {
      id: 'q1',
      question: 'What is the primary purpose of keys in React lists?',
      options: [
        'To make the code look more professional',
        'To help React identify which items have changed, been added, or been removed',
        'To prevent duplicate items in the list',
        'To improve SEO and accessibility',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"To help React identify which items have changed, been added, or been removed"**.

Keys are fundamental to React's reconciliation algorithm—they tell React the **identity** of each element across renders.

**How React Uses Keys:**

When you render a list, React needs to figure out:
1. Which items stayed the same?
2. Which items were added?
3. Which items were removed?
4. Which items moved to a different position?

**Without keys:**
\`\`\`tsx
// React only sees an array of elements
[<li>Item 1</li>, <li>Item 2</li>, <li>Item 3</li>]

// Next render:
[<li>Item 2</li>, <li>Item 3</li>]

// React thinks: First item content changed, third item removed
// Reality: First item removed, others unchanged
\`\`\`

**With keys:**
\`\`\`tsx
// React sees identities
[<li key="1">Item 1</li>, <li key="2">Item 2</li>, <li key="3">Item 3</li>]

// Next render:
[<li key="2">Item 2</li>, <li key="3">Item 3</li>]

// React knows: key="1" removed, others unchanged (reuse DOM nodes!)
\`\`\`

**What React Does with Keys:**

**1. Efficient Updates:**
\`\`\`tsx
// Before
<li key="a">Apple</li>
<li key="b">Banana</li>
<li key="c">Cherry</li>

// After (reordered)
<li key="b">Banana</li>
<li key="a">Apple</li>
<li key="c">Cherry</li>

// With keys: React MOVES existing DOM nodes (cheap)
// Without keys: React UPDATES content of all nodes (expensive)
\`\`\`

**2. Preserving State:**
\`\`\`tsx
function TodoItem({ todo }) {
  const [isEditing, setIsEditing] = useState(false);
  
  return (
    <li>
      {isEditing ? (
        <input defaultValue={todo.text} />
      ) : (
        <span>{todo.text}</span>
      )}
    </li>
  );
}

// With correct keys: isEditing state stays with correct item
// With wrong keys: isEditing might attach to wrong item!
\`\`\`

**3. Avoiding Bugs with Uncontrolled Components:**
\`\`\`tsx
// Uncontrolled input
<input type="checkbox" defaultChecked={item.done} />

// Without stable keys:
// - DOM node reused for different item
// - Checkbox checked state persists to wrong item
// - User checks "Task A", deletes "Task B", now "Task C" appears checked!
\`\`\`

**Real-World Example:**

\`\`\`tsx
function MessageList({ messages }) {
  return (
    <div>
      {messages.map((msg, index) => (
        // ❌ Using index as key
        <div key={index}>
          <input placeholder="Reply..." />
          <p>{msg.text}</p>
        </div>
      ))}
    </div>
  );
}

// User types "Hello" in reply to message at index 1
// Another user sends a message (inserts at top)
// Now the "Hello" appears in reply to message at index 2!
\`\`\`

**Why other answers are wrong:**

- **"Make code look professional"**: Keys are functional, not cosmetic
- **"Prevent duplicate items"**: Keys don't prevent duplicates in data, just help React track them
- **"Improve SEO/accessibility"**: Keys are React-internal, not rendered to DOM

**Performance Impact:**

\`\`\`tsx
// ❌ Without keys or with bad keys:
// - React may update wrong elements
// - Unnecessary DOM manipulations
// - Component state bugs

// ✅ With good keys:
// - React identifies items correctly
// - Minimal DOM operations
// - State preserved correctly
\`\`\`

**Interview tip:** Explaining that keys are for React's reconciliation algorithm (not for developers or accessibility) shows understanding of React's internal workings. This is a fundamental concept that separates junior from mid-level React knowledge.`,
    },
    {
      id: 'q2',
      question:
        'Which of the following is the BEST choice for a key when rendering a list of user objects fetched from an API?',
      options: [
        'Array index',
        'user.id from the database',
        'Math.random()',
        'user.name + user.email',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"user.id from the database"**.

Database IDs are ideal keys because they're:
1. **Unique:** No two users share the same ID
2. **Stable:** ID never changes for a user
3. **Predictable:** Always derived from data, not random

**Why user.id is best:**

\`\`\`tsx
const users = [
  { id: 1, name: 'Alice', email: 'alice@example.com' },
  { id: 2, name: 'Bob', email: 'bob@example.com' },
  { id: 3, name: 'Charlie', email: 'charlie@example.com' }
];

function UserList() {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>  {/* ✅ Perfect */}
          {user.name} - {user.email}
        </li>
      ))}
    </ul>
  );
}
\`\`\`

**Benefits:**
- ✅ If user edits their name/email, ID stays the same → React reuses DOM node
- ✅ If users are reordered, React knows exactly which moved → just moves DOM nodes
- ✅ If a user is deleted, React knows exactly which DOM node to remove
- ✅ Works perfectly with filters, sorts, pagination

**Why other options are bad:**

**Array Index (Option 1):**
\`\`\`tsx
{users.map((user, index) => (
  <li key={index}>{user.name}</li>
))}
\`\`\`

**Problems:**
- ❌ Keys change when list reorders
- ❌ Keys change when items deleted from middle
- ❌ State attaches to position, not item

**Example bug:**
\`\`\`
Before:
  index 0: Alice (selected)
  index 1: Bob
  index 2: Charlie

Delete Alice:
  index 0: Bob (now selected! Wrong!)
  index 1: Charlie

// "selected" state stayed at index 0, now applies to Bob
\`\`\`

**Math.random() (Option 3):**
\`\`\`tsx
{users.map(user => (
  <li key={Math.random()}>{user.name}</li>
))}
\`\`\`

**Problems:**
- ❌ **Disaster!** New key EVERY render
- ❌ React thinks all items are new
- ❌ Unmounts + remounts all components
- ❌ All state lost
- ❌ Terrible performance

**What happens:**
\`\`\`tsx
// Render 1:
<li key={0.123}>Alice</li>
<li key={0.456}>Bob</li>

// Render 2 (nothing changed, just re-rendered):
<li key={0.789}>Alice</li>  // New key!
<li key={0.234}>Bob</li>    // New key!

// React: All old items removed, all new items added
// Effect: Entire list unmounts and remounts every render
\`\`\`

**user.name + user.email (Option 4):**
\`\`\`tsx
{users.map(user => (
  <li key={\`\${user.name}-\${user.email}\`}>{user.name}</li>
))}
\`\`\`

**Problems:**
- ❌ Key changes if user edits name or email
- ❌ Not unique if two users have same name+email
- ❌ React treats edited user as deleted + new one added

**Example bug:**
\`\`\`tsx
// User edits their name
Before: { id: 1, name: 'Alic', email: 'alice@example.com' }  // key: "Alic-alice@example.com"
After:  { id: 1, name: 'Alice', email: 'alice@example.com' } // key: "Alice-alice@example.com"

// React sees:
// - "Alic-alice@example.com" removed
// - "Alice-alice@example.com" added

// Effect:
// - Old component unmounted (state lost!)
// - New component mounted
// - Looks like a flash/flicker to user
\`\`\`

**When name+email might be OK:**
- User data is read-only (never edited)
- AND you're certain name+email combos are unique
- AND you have no database IDs available

**Comparison Table:**

| Key Type | Unique | Stable | Predictable | Performance | Rating |
|----------|--------|--------|-------------|-------------|--------|
| user.id | ✓ | ✓ | ✓ | Excellent | ⭐⭐⭐⭐⭐ |
| index | ✓* | ✗ | ✓ | Poor | ⭐ |
| Math.random() | ✓ | ✗ | ✗ | Terrible | ☠️ |
| name+email | ? | ✗ | ✓ | OK | ⭐⭐ |

*Unique among siblings, but represents position not identity

**Best Practices:**

1. **Always prefer data-based IDs:**
\`\`\`tsx
// ✅ Database ID
<Item key={item.id} />

// ✅ UUID
<Item key={item.uuid} />

// ✅ Unique combination (if guaranteed unique)
<Item key={\`\${item.userId}-\${item.timestamp}\`} />
\`\`\`

2. **Generate IDs when creating data:**
\`\`\`tsx
import { v4 as uuidv4 } from 'uuid';

function addItem(text) {
  setItems(prev => [
    ...prev,
    { id: uuidv4(), text }  // Generate once
  ]);
}
\`\`\`

3. **Index is OK ONLY for static lists:**
\`\`\`tsx
// ✅ OK: List never changes
const navItems = ['Home', 'About', 'Contact'];
{navItems.map((item, i) => <a key={i} href="#">{item}</a>)}
\`\`\`

**Interview tip:** Explaining why database IDs are best and detailing the problems with each alternative shows deep understanding. Mentioning that Math.random() causes unmount/remount demonstrates knowledge of React's reconciliation behavior.`,
    },
    {
      id: 'q3',
      question: 'What will happen if you use Math.random() as a key in a list?',
      options: [
        'React will throw an error',
        'React will warn but render correctly',
        'Components will unmount and remount on every render, losing all state',
        'Nothing will happen, it works the same as any other key',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"Components will unmount and remount on every render, losing all state"**.

Using Math.random() as a key is one of the worst things you can do in React—it causes massive performance issues and state loss.

**The Problem:**

\`\`\`tsx
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={Math.random()} todo={todo} />
      ))}
    </ul>
  );
}
\`\`\`

**What happens on EVERY render:**

\`\`\`tsx
// Render 1:
<TodoItem key={0.12345} todo={todo1} />
<TodoItem key={0.67890} todo={todo2} />

// Render 2 (even if todos unchanged):
<TodoItem key={0.43210} todo={todo1} />  // Different key!
<TodoItem key={0.98765} todo={todo2} />  // Different key!

// React's reconciliation:
// "key={0.12345} is gone → unmount that component"
// "key={0.67890} is gone → unmount that component"
// "key={0.43210} is new → mount new component"
// "key={0.98765} is new → mount new component"
\`\`\`

**Consequences:**

**1. All State Lost:**
\`\`\`tsx
function TodoItem({ todo }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(todo.text);
  
  return (
    <li>
      {isEditing ? (
        <input 
          value={editText}
          onChange={e => setEditText(e.target.value)}
        />
      ) : (
        <span>{todo.text}</span>
      )}
      <button onClick={() => setIsEditing(!isEditing)}>Edit</button>
    </li>
  );
}

// User clicks "Edit", starts typing
// Parent re-renders (e.g., filter state changes)
// New Math.random() keys generated
// Component unmounts
// User's edit text lost!
\`\`\`

**2. Terrible Performance:**
\`\`\`tsx
// Every render cycle:
// 1. Unmount all list items
//    - Run all cleanup effects
//    - Remove all DOM nodes
//    - Garbage collect old components
// 2. Mount all "new" list items
//    - Create new DOM nodes
//    - Run all mount effects
//    - Attach event listeners

// With 100 items:
// - 100 unmounts + 100 mounts = 200 operations
// - Every single render!
// - App becomes unusable
\`\`\`

**3. Visual Glitches:**
\`\`\`tsx
// CSS transitions/animations restart
// Scroll position jumps
// Focus lost from inputs
// Selections cleared
\`\`\`

**Real-World Example:**

\`\`\`tsx
function ChatMessages({ messages }) {
  return (
    <div>
      {messages.map(msg => (
        <Message key={Math.random()} msg={msg} />
      ))}
    </div>
  );
}

function Message({ msg }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  useEffect(() => {
    console.log('Message mounted:', msg.id);
    return () => console.log('Message unmounted:', msg.id);
  }, []);
  
  return (
    <div>
      <p>{msg.text}</p>
      {isExpanded && <MessageDetails msg={msg} />}
    </div>
  );
}

// Console output on every render:
// Message unmounted: 1
// Message unmounted: 2
// Message unmounted: 3
// Message mounted: 1
// Message mounted: 2
// Message mounted: 3

// User experience:
// - Types in reply box
// - New message arrives
// - Parent re-renders
// - ALL messages unmount/remount
// - Reply text lost!
\`\`\`

**Why other answers are wrong:**

**"React will throw an error":**
- React doesn't throw errors for bad keys
- Keys are just props to React
- Math.random() returns a valid number/string

**"React will warn but render correctly":**
- React might warn about duplicate keys (if Math.random() collides)
- But won't warn about unstable keys
- It definitely won't render "correctly"—state will be lost

**"Nothing will happen":**
- Everything will happen! (All bad)

**What React Actually Does:**

React compares previous and current keys:
\`\`\`tsx
// Previous render keys:
prevKeys = [0.123, 0.456, 0.789]

// Current render keys:
currentKeys = [0.234, 0.567, 0.891]

// React's reconciliation:
// - Can't find 0.123 in currentKeys → remove component
// - Can't find 0.456 in currentKeys → remove component
// - Can't find 0.789 in currentKeys → remove component
// - See 0.234 not in prevKeys → add component
// - See 0.567 not in prevKeys → add component
// - See 0.891 not in prevKeys → add component

// Result: Complete list replacement every render
\`\`\`

**Correct Solutions:**

**1. Use stable IDs from data:**
\`\`\`tsx
{todos.map(todo => (
  <TodoItem key={todo.id} todo={todo} />
))}
\`\`\`

**2. Generate IDs once when creating data:**
\`\`\`tsx
import { v4 as uuidv4 } from 'uuid';

function addTodo(text) {
  const newTodo = {
    id: uuidv4(),  // Generated once, never changes
    text
  };
  setTodos(prev => [...prev, newTodo]);
}
\`\`\`

**3. If absolutely no IDs, use index (with caveats):**
\`\`\`tsx
// ⚠️ OK only if list never reorders/filters
{todos.map((todo, index) => (
  <TodoItem key={index} todo={todo} />
))}
\`\`\`

**Debugging Random Keys:**

If you suspect random keys, check React DevTools:
\`\`\`tsx
// In React DevTools:
// - Open Profiler
// - Start profiling
// - Interact with app
// - Check "Ranked" view

// With Math.random() keys:
// - ALL list items show as "Mount" on every update
// - Huge flamegraph
// - Terrible performance
\`\`\`

**Interview tip:** Explaining that Math.random() causes unmount/remount (not just poor performance) shows deep understanding of React's reconciliation algorithm. Mentioning the effect on useEffect cleanup and state loss demonstrates production experience with this bug.`,
    },
    {
      id: 'q4',
      question:
        'When mapping over an array to render list items, where should the key prop be placed?',
      options: [
        'On the root element inside the component being rendered',
        'On the component in the map() call',
        'On both the component and root element for best results',
        "It doesn't matter as long as there is a key somewhere",
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"On the component in the map() call"**.

The key must be on the **direct child of the map()**, not inside the component.

**✅ Correct:**
\`\`\`tsx
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
        //        ↑ Key on component in map()
      ))}
    </ul>
  );
}

function TodoItem({ todo }) {
  return <li>{todo.text}</li>;
  //     ↑ No key here
}
\`\`\`

**❌ Wrong:**
\`\`\`tsx
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem todo={todo} />
        // ❌ Missing key!
      ))}
    </ul>
  );
}

function TodoItem({ todo }) {
  return <li key={todo.id}>{todo.text}</li>;
  //         ↑ Key here doesn't help!
}

// React warning:
// "Warning: Each child in a list should have a unique "key" prop."
\`\`\`

**Why This Rule Exists:**

**React needs to know the key at the call site:**

\`\`\`tsx
// What map() returns:
[
  <TodoItem key={1} todo={todo1} />,
  <TodoItem key={2} todo={todo2} />,
  <TodoItem key={3} todo={todo3} />
]

// React sees: Array of elements with keys
// React can identify each element
\`\`\`

**When key is inside component:**
\`\`\`tsx
// What map() returns:
[
  <TodoItem todo={todo1} />,  // No key!
  <TodoItem todo={todo2} />,  // No key!
  <TodoItem todo={todo3} />   // No key!
]

// React sees: Array of elements WITHOUT keys
// React can't identify elements
// Keys inside component are invisible to React at this level
\`\`\`

**Key Placement Rules:**

**Rule 1: Key on the element returned by map()**
\`\`\`tsx
// ✅ Component
{items.map(item => <Item key={item.id} item={item} />)}

// ✅ DOM element
{items.map(item => <li key={item.id}>{item.name}</li>)}

// ✅ Fragment
import { Fragment } from 'react';
{items.map(item => (
  <Fragment key={item.id}>
    <dt>{item.term}</dt>
    <dd>{item.definition}</dd>
  </Fragment>
))}
\`\`\`

**Rule 2: Not on elements inside the mapped component**
\`\`\`tsx
function UserList({ users }) {
  return (
    <div>
      {users.map(user => (
        // ✅ Key here
        <UserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

function UserCard({ user }) {
  return (
    // ❌ Key here does nothing
    <div key={user.id}>
      <h3>{user.name}</h3>
      <p>{user.email}</p>
    </div>
  );
}
\`\`\`

**Multiple Elements: Use Fragment**

When mapping to multiple elements, wrap in Fragment WITH key:

\`\`\`tsx
// ❌ WRONG: Can't have multiple root elements
{users.map(user => (
  <h3 key={user.id}>{user.name}</h3>
  <p>{user.email}</p>  // Error!
))}

// ❌ WRONG: Can't use <> shorthand with key
{users.map(user => (
  <>  {/* Can't add key prop to <> */}
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </>
))}

// ✅ Option 1: Fragment with key
import { Fragment } from 'react';
{users.map(user => (
  <Fragment key={user.id}>
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </Fragment>
))}

// ✅ Option 2: Wrapper div
{users.map(user => (
  <div key={user.id}>
    <h3>{user.name}</h3>
    <p>{user.email}</p>
  </div>
))}
\`\`\`

**Nested Lists:**

Each level needs keys at its own map() call:

\`\`\`tsx
function CategoryList({ categories }) {
  return (
    <div>
      {categories.map(category => (
        // ✅ Key for category
        <div key={category.id}>
          <h2>{category.name}</h2>
          <ul>
            {category.items.map(item => (
              // ✅ Key for item (different namespace)
              <li key={item.id}>{item.name}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
\`\`\`

**Keys Don't Pass as Props:**

Keys are special—React uses them but doesn't pass them to your component:

\`\`\`tsx
function Item({ key, name }) {
  console.log(key);  // undefined!
  return <div>{name}</div>;
}

{items.map(item => (
  <Item key={item.id} name={item.name} />
))}

// To access the key value, pass it separately:
function Item({ id, name }) {
  console.log(id);  // Works!
  return <div>{name}</div>;
}

{items.map(item => (
  <Item 
    key={item.id}   // For React
    id={item.id}    // For your component
    name={item.name} 
  />
))}
\`\`\`

**Why "Both" is Wrong:**

Putting keys in both places doesn't provide "extra safety"—React only looks at the key where map() is called.

\`\`\`tsx
// ❌ Redundant and confusing
{todos.map(todo => (
  <TodoItem key={todo.id} todo={todo} />
))}

function TodoItem({ todo }) {
  return <li key={todo.id}>{todo.text}</li>;  // Ignored by React
}
\`\`\`

**Common Mistake: Extracted Component**

\`\`\`tsx
// ❌ WRONG
function TodoList({ todos }) {
  function renderTodo(todo) {
    return <TodoItem key={todo.id} todo={todo} />;
    //                ↑ Key here doesn't work!
  }
  
  return <ul>{todos.map(renderTodo)}</ul>;
}

// ✅ CORRECT
function TodoList({ todos }) {
  function renderTodo(todo) {
    return <TodoItem todo={todo} />;  // No key
  }
  
  return (
    <ul>
      {todos.map(todo => (
        <div key={todo.id}>{renderTodo(todo)}</div>
        //   ↑ Key on element in map()
      ))}
    </ul>
  );
}

// ✅ BETTER: Just use map directly
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}
\`\`\`

**Interview tip:** Explaining that keys must be on the direct child of map() (not inside the component) shows understanding of how React processes elements. Mentioning that keys don't pass as props demonstrates knowledge of React's special props (key, ref, children).`,
    },
    {
      id: 'q5',
      question:
        'What is the most performant way to render a list of 10,000 items?',
      options: [
        'Render all 10,000 items and use React.memo to prevent re-renders',
        'Use virtualization (windowing) to only render visible items',
        'Split into pages of 100 items each',
        'Use useMemo to cache the list rendering',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"Use virtualization (windowing) to only render visible items"**.

For very large lists (1000+ items), virtualization is the only technique that provides truly constant-time performance regardless of list size.

**The Problem with Large Lists:**

\`\`\`tsx
// ❌ Rendering 10,000 items
function UserList({ users }) {  // users.length = 10,000
  return (
    <div style={{ height: '600px', overflow: 'auto' }}>
      {users.map(user => (
        <div key={user.id} style={{ height: '50px' }}>
          {user.name}
        </div>
      ))}
    </div>
  );
}

// Problems:
// - 10,000 DOM nodes created
// - Initial render: 2-5 seconds
// - Memory: ~100-500 MB
// - Scroll: janky, laggy
// - Browser struggles
\`\`\`

**Why Each Option Performs Differently:**

**Option 1: React.memo (Poor for 10,000 items)**

\`\`\`tsx
const UserItem = memo(function UserItem({ user }) {
  return <div>{user.name}</div>;
});

function UserList({ users }) {
  return (
    <div>
      {users.map(user => (
        <UserItem key={user.id} user={user} />
      ))}
    </div>
  );
}

// Still renders all 10,000 on initial mount!
// memo helps with RE-renders, not initial render
// Still 10,000 DOM nodes
// Still slow
\`\`\`

**Performance:**
- Initial render: ~3 seconds (all 10,000 mount)
- Re-render (unrelated state): Fast (memo prevents updates)
- Memory: High (10,000 DOM nodes)
- Rating: ⭐⭐ (helps with re-renders but not initial render)

**Option 2: Virtualization (Excellent!) ✅**

\`\`\`tsx
import { FixedSizeList } from 'react-window';

function UserList({ users }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      {users[index].name}
    </div>
  );
  
  return (
    <FixedSizeList
      height={600}              // Viewport height
      itemCount={users.length}  // Total items: 10,000
      itemSize={50}             // Height per item
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}

// Only renders ~12 items at a time (600px / 50px)
// As user scrolls, reuses DOM nodes
\`\`\`

**How it works:**
\`\`\`
Viewport (600px visible):
┌────────────────────┐
│ Item 0             │ ← Rendered
│ Item 1             │ ← Rendered
│ Item 2             │ ← Rendered
│ Item 3             │ ← Rendered
│ Item 4             │ ← Rendered
│ Item 5             │ ← Rendered
│ Item 6             │ ← Rendered
│ Item 7             │ ← Rendered
│ Item 8             │ ← Rendered
│ Item 9             │ ← Rendered
│ Item 10            │ ← Rendered
│ Item 11            │ ← Rendered
└────────────────────┘
  [9,988 items not rendered]

When user scrolls down:
- Top items scroll out → DOM nodes recycled
- New items scroll in → reuse those DOM nodes
- Only ~12-15 DOM nodes exist at any time
\`\`\`

**Performance:**
- Initial render: ~50ms (only visible items)
- Scroll: Smooth 60fps
- Memory: Low (12 DOM nodes)
- Works with millions of items
- Rating: ⭐⭐⭐⭐⭐

**Option 3: Pagination (Good, different use case)**

\`\`\`tsx
function UserList({ users }) {
  const [page, setPage] = useState(0);
  const itemsPerPage = 100;
  const pageCount = Math.ceil(users.length / itemsPerPage);
  
  const currentUsers = users.slice(
    page * itemsPerPage,
    (page + 1) * itemsPerPage
  );
  
  return (
    <div>
      {currentUsers.map(user => (
        <div key={user.id}>{user.name}</div>
      ))}
      
      <button onClick={() => setPage(p => p - 1)} disabled={page === 0}>
        Previous
      </button>
      <span>Page {page + 1} of {pageCount}</span>
      <button onClick={() => setPage(p => p + 1)} disabled={page >= pageCount - 1}>
        Next
      </button>
    </div>
  );
}
\`\`\`

**Performance:**
- Initial render: ~200ms (100 items)
- Page change: ~200ms
- Memory: Medium (100 DOM nodes)
- Rating: ⭐⭐⭐⭐ (good, but less smooth than virtualization)

**Tradeoffs:**
- ✅ Simple to implement
- ✅ Good for tables/structured data
- ❌ Requires clicks to see more
- ❌ Can't browse entire list smoothly

**Option 4: useMemo (Doesn't help)**

\`\`\`tsx
function UserList({ users }) {
  const renderedUsers = useMemo(() => 
    users.map(user => (
      <div key={user.id}>{user.name}</div>
    ))
  , [users]);
  
  return <div>{renderedUsers}</div>;
}

// Still creates 10,000 JSX elements
// Still creates 10,000 DOM nodes
// useMemo caches JSX, but React still renders it all
\`\`\`

**Performance:**
- Initial render: ~3 seconds
- Re-render (users unchanged): Fast (cached)
- Memory: High
- Rating: ⭐⭐ (helps with re-renders, not initial)

**Performance Comparison:**

| Technique | Initial Render | Re-renders | Memory | Scroll | Items Supported |
|-----------|----------------|------------|--------|--------|-----------------|
| Plain | 3000ms | 3000ms | High | Laggy | ~1,000 |
| React.memo | 3000ms | 50ms | High | Laggy | ~1,000 |
| useMemo | 3000ms | 50ms | High | Laggy | ~1,000 |
| Pagination | 200ms | 200ms | Medium | N/A | Unlimited |
| Virtualization | **50ms** | **50ms** | **Low** | **Smooth** | **Unlimited** |

**When to Use Each:**

\`\`\`
List size < 100:
  → Plain rendering (no optimization needed)

List size 100-500:
  → React.memo + useCallback

List size 500-2000:
  → Pagination OR virtualization

List size 2000+:
  → Virtualization (only option for smooth performance)

Special cases:
  - Search results → Pagination
  - Tables → Pagination + virtualization
  - Social feeds → Infinite scroll + virtualization
  - Form with many fields → Virtualization
\`\`\`

**Real-World Example:**

\`\`\`tsx
// Twitter-like feed with 10,000 tweets
import { VariableSizeList } from 'react-window';

function TweetFeed({ tweets }) {
  const listRef = useRef();
  const rowHeights = useRef({});
  
  const getItemSize = (index) => rowHeights.current[index] || 100;
  
  const Row = ({ index, style }) => (
    <div style={style}>
      <Tweet 
        tweet={tweets[index]}
        onHeightChange={(height) => {
          rowHeights.current[index] = height;
          listRef.current?.resetAfterIndex(index);
        }}
      />
    </div>
  );
  
  return (
    <VariableSizeList
      ref={listRef}
      height={800}
      itemCount={tweets.length}
      itemSize={getItemSize}
      width="100%"
    >
      {Row}
    </VariableSizeList>
  );
}

// Performance:
// - Initial: 50ms
// - Scroll: Smooth 60fps
// - Memory: ~20 tweets rendered at once
// - Works with infinite scroll
\`\`\`

**Interview tip:** Explaining virtualization and comparing it to other techniques shows deep performance understanding. Mentioning specific numbers (render times, DOM node counts) demonstrates real-world experience. Discussing when to use each technique shows architectural judgment.`,
    },
  ],
};
