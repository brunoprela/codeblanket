export const reactDeveloperTools = {
  title: 'React Developer Tools',
  id: 'react-developer-tools',
  content: `
# React Developer Tools

## Introduction

**React Developer Tools** (React DevTools) is a browser extension that gives you superpowers for debugging React applications. It's an essential tool for every React developer—allowing you to inspect components, view props and state, track performance, and debug issues efficiently.

Think of it as the "Inspect Element" for React—but instead of just seeing HTML/CSS, you see your React component tree with all its data.

In this section, you'll learn:
- How to install and use React DevTools
- Inspecting component tree
- Viewing and editing props/state
- Using the Profiler
- Common debugging patterns
- Performance monitoring

## Installing React DevTools

**For Chrome:**
1. Visit [Chrome Web Store](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
2. Click "Add to Chrome"
3. Open any React app
4. Open DevTools (F12 or Right-click → Inspect)
5. Look for "Components" and "Profiler" tabs

**For Firefox:**
1. Visit [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)
2. Click "Add to Firefox"
3. Follow same steps as Chrome

**For Edge:**
1. Visit [Edge Add-ons](https://microsoftedge.microsoft.com/addons/detail/react-developer-tools/gpphkfbcpidddadnkolkpfckpihlkkil)
2. Click "Get"
3. Follow same steps as Chrome

**Standalone App (for React Native, Safari, etc.):**
\`\`\`bash
npm install -g react-devtools
react-devtools
\`\`\`

**Visual Indicators:**

When you open a page with React:
- **Red icon** = Production build
- **Blue icon** = Development build
- **Gray icon** = No React detected

Always use development build during development for better debugging!

## The Components Tab

The **Components** tab shows your React component tree.

### Component Tree View

\`\`\`
App
├─ Header
│  ├─ Logo
│  └─ Navigation
│     ├─ NavLink (Home)
│     ├─ NavLink (About)
│     └─ NavLink (Contact)
├─ Main
│  ├─ Sidebar
│  │  └─ FilterPanel
│  └─ Content
│     ├─ PostList
│     │  ├─ Post
│     │  ├─ Post
│     │  └─ Post
│     └─ Pagination
└─ Footer
\`\`\`

**What you see:**
- Component hierarchy
- Component names
- Props passed to each component
- Current state
- Hooks used
- Context values

### Selecting Components

**3 ways to select a component:**

1. **Click in tree** → Select component in Components tab
2. **Click "Select element" button** → Click element on page
3. **Right-click element on page** → "Inspect" → Shows in Elements tab → Switch to Components tab (auto-selected)

### Viewing Props and State

When you select a component, the right panel shows:

\`\`\`
UserProfile
├─ props
│  ├─ userId: 123
│  ├─ name: "Alice Johnson"
│  ├─ email: "alice@example.com"
│  └─ onUpdate: ƒ handleUpdate()
├─ state
│  ├─ isEditing: false
│  └─ editedName: ""
└─ hooks
   ├─ State: false
   ├─ State: ""
   └─ Effect
\`\`\`

**Prop types shown:**
- Strings: \`"value"\`
- Numbers: \`123\`
- Booleans: \`true\` / \`false\`
- Functions: \`ƒ functionName()\`
- Objects: \`{...}\` (expandable)
- Arrays: \`[...]\` (expandable)
- undefined/null: shown as is

### Editing Props and State (Live!)

You can **edit values in real-time** to test component behavior:

1. Select component
2. Find prop or state value
3. Double-click to edit
4. Press Enter
5. Component re-renders with new value!

**Example use cases:**
\`\`\`tsx
// Your component
function Post({ title, likes }) {
  return (
    <div>
      <h2>{title}</h2>
      <p>{likes} likes</p>
    </div>
  );
}

// In DevTools:
// 1. Select Post component
// 2. Edit props.likes from 42 to 999999
// 3. See "999999 likes" instantly on page
// 4. Test how component handles large numbers
\`\`\`

**Testing edge cases without code changes:**
- Empty strings: \`""\`
- Very long text: \`"Lorem ipsum..."\` (1000+ characters)
- Negative numbers: \`-1\`
- null/undefined: Test error boundaries
- Empty arrays: \`[]\`
- Large arrays: \`[1, 2, 3, ..., 1000]\`

### Filtering Components

**Search box** at top filters component tree:

- **By name**: Type \`"Button"\` → Shows only Button components
- **By prop**: Type \`"userId=123"\` → Shows components with that prop
- **By state**: Type \`"state:isOpen=true"\` → Shows components with that state
- **Regular expressions**: \`"/Post.*/"\` → Matches Post, PostList, PostItem

### Component Highlighting

When you hover over a component in DevTools:
- **Page highlights** the component with an overlay
- Shows:
  - Component name
  - Dimensions (width × height)
  - File location (if source maps available)

When you hover over an element on the page:
- **DevTools highlights** corresponding component in tree

### View Source

Click the \`< >\` icon next to component name:
- **Jumps to source code** in Sources tab
- Only works if source maps enabled (default in dev mode)

### Component Stack

Shows where component is rendered from:

\`\`\`
Button
  rendered by Header
    rendered by App
      rendered by root
\`\`\`

Useful for finding who's rendering a component.

## Hooks Inspection

Modern React apps use hooks. DevTools shows all hooks used by a component:

\`\`\`tsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);          // Hook 0: State
  const [isLoading, setIsLoading] = useState(true); // Hook 1: State
  const [error, setError] = useState(null);        // Hook 2: State
  
  useEffect(() => {                                // Hook 3: Effect
    fetchUser(userId).then(setUser);
  }, [userId]);
  
  const handleUpdate = useCallback(() => {         // Hook 4: Callback
    updateUser(user);
  }, [user]);
  
  const displayName = useMemo(() => {              // Hook 5: Memo
    return \`\${user?.firstName} \${user?.lastName}\`;
  }, [user]);
  
  return <div>{displayName}</div>;
}
\`\`\`

**DevTools shows:**
\`\`\`
hooks
├─ State: null                    (useState #1)
├─ State: true                    (useState #2)
├─ State: null                    (useState #3)
├─ Effect: ƒ ()                   (useEffect)
├─ Callback: ƒ handleUpdate()     (useCallback)
└─ Memo: "Alice Johnson"          (useMemo)
\`\`\`

**You can:**
- See current values
- Edit state values live
- See which hooks re-run
- Debug hook order issues

## Context and Providers

DevTools shows context values:

\`\`\`tsx
// Your code
const ThemeContext = createContext('light');

function App() {
  return (
    <ThemeContext.Provider value="dark">
      <Button />
    </ThemeContext.Provider>
  );
}

function Button() {
  const theme = useContext(ThemeContext);
  return <button className={theme}>Click me</button>;
}
\`\`\`

**DevTools shows:**
\`\`\`
Button
├─ props: {}
├─ hooks
│  └─ Context: "dark"  ← From ThemeContext
\`\`\`

Can see:
- Which contexts component consumes
- Current context values
- Context provider hierarchy

## The Profiler Tab

The **Profiler** measures component performance.

### Recording a Profile

1. Open Profiler tab
2. Click **Record** button (red circle)
3. Interact with your app
4. Click **Stop**
5. View performance data

### Understanding Flamegraph

Shows which components rendered and how long each took:

\`\`\`
App (2.1ms)
├─ Header (0.3ms)
│  └─ Navigation (0.8ms)
├─ Main (12.4ms)  ← SLOW!
│  ├─ Sidebar (1.2ms)
│  └─ Content (10.8ms)  ← VERY SLOW!
│     └─ PostList (9.6ms)
│        ├─ Post (0.4ms) × 24 times
└─ Footer (0.2ms)
\`\`\`

**Colors indicate render duration:**
- **Green**: Fast (< 1ms)
- **Yellow**: Moderate (1-5ms)
- **Orange**: Slow (5-10ms)
- **Red**: Very slow (> 10ms)

### Commit Timeline

Shows all renders over time:

\`\`\`
Commits:
1. Initial Mount (15ms)
2. Button click (3ms)
3. Form input (1ms)
4. Form input (1ms)
5. Form input (1ms)
6. Form submit (8ms)
\`\`\`

Click any commit to see what rendered.

### Ranked Chart

Shows components sorted by render time:

\`\`\`
1. Content      10.8ms
2. Main         12.4ms
3. PostList      9.6ms
4. Navigation    0.8ms
5. Post (×24)    0.4ms each
\`\`\`

Quickly find slowest components.

### Why Component Rendered

DevTools shows WHY a component re-rendered:

- **Props changed**: Shows which props
- **State changed**: Shows which state
- **Context changed**: Shows which context
- **Hooks changed**: Shows which hooks
- **Parent rendered**: Parent triggered re-render

\`\`\`
Post
Re-rendered because:
  • props.likes changed (42 → 43)
  • Parent <PostList> rendered
\`\`\`

## Common Debugging Workflows

### Workflow 1: "Why is this component rendering so much?"

1. Open Profiler
2. Click "Record"
3. Reproduce the issue
4. Click "Stop"
5. Look at Ranked Chart
6. Select component
7. Check "Why did this render?"
8. See: "Props changed: onUpdate" (new function every render!)
9. Fix: Wrap onUpdate in useCallback

### Workflow 2: "What's the value of this prop?"

1. Open Components tab
2. Select component (or use element picker)
3. View props in right panel
4. Expand objects/arrays to see nested values

### Workflow 3: "Component shows stale data"

1. Select component
2. Check state values
3. Check prop values
4. If stale → parent not passing updated props
5. Select parent
6. Check its state/props
7. Trace up tree until you find source

### Workflow 4: "Testing edge cases"

1. Select component
2. Edit props/state to extreme values:
   - Empty string: \`""\`
   - null
   - Very large number: \`999999\`
   - Empty array: \`[]\`
3. See how component handles it
4. Add error boundaries/validation if needed

### Workflow 5: "Which component is this element?"

1. Right-click element on page
2. Click "Inspect"
3. Switch to "Components" tab
4. Component automatically selected
5. View props/state/hooks

## Advanced Features

### Component Filters

Filter what's shown in component tree:

**Settings (gear icon) → Components:**
- Hide components from specific libraries (e.g., React Router)
- Show only Host components (DOM elements)
- Show only Hooks
- Collapse all by default

### Suspended Components

Shows components that are suspended (React Suspense):

\`\`\`tsx
<Suspense fallback={<Loading />}>
  <LazyComponent />  {/* Shows as "Suspended" if still loading */}
</Suspense>
\`\`\`

### Debug Hooks

Console logs all hook values:

1. Select component
2. Click bug icon next to hook
3. Console logs: \`console.log({ hookValue })\`

### Copy Props/State

Right-click component → "Copy props/state" → Pastes JSON to clipboard

Useful for:
- Saving test data
- Sharing bug reports
- Creating fixtures

### Owner Tree

Shows which component "owns" (created) this component:

\`\`\`
Button
  owned by Header
    owned by App
\`\`\`

Different from "rendered by" (which shows the call stack).

## Best Practices

1. **Always use Development build** while debugging
   - Production build strips names/warnings
   - Development build has full debugging info

2. **Name your components**
   \`\`\`tsx
   // ❌ BAD: Anonymous component
   export default () => <div>Hello</div>;
   
   // ✅ GOOD: Named component
   export default function Greeting() {
     return <div>Hello</div>;
   }
   \`\`\`

3. **Use Profiler to find performance issues**
   - Don't guess where slowdowns are
   - Measure with Profiler
   - Optimize only what's slow

4. **Use element picker** to quickly find components
   - Faster than manually navigating tree
   - Click button, click element, see component

5. **Edit props/state to test edge cases**
   - Don't modify code to test
   - Edit live in DevTools
   - Faster iteration

6. **Check "Why did this render?"**
   - Don't guess why re-renders happen
   - DevTools tells you exactly why
   - Fix unnecessary re-renders

7. **Use component filters**
   - Hide library components (React Router, etc.)
   - Focus on your components
   - Less noise

8. **Compare commits in Profiler**
   - Record before optimization
   - Record after optimization
   - Compare side-by-side

## Keyboard Shortcuts

- **Cmd/Ctrl + Shift + C**: Pick element
- **Cmd/Ctrl + F**: Search components
- **Cmd/Ctrl + ]**: Next tab
- **Cmd/Ctrl + [**: Previous tab
- **Arrow keys**: Navigate component tree
- **Space**: Expand/collapse component

## Troubleshooting

**Problem: "Components" tab doesn't appear**
- React not detected
- Check icon is red/blue (not gray)
- Refresh page
- Check app is actually using React

**Problem: "This page is using the production build of React"**
- You're viewing production build
- Source maps disabled
- Component names minified
- Solution: Use development build for debugging

**Problem: Component names show as "Anonymous"**
- Using arrow functions without names
- Solution: Name your components

**Problem: Can't edit props/state**
- Component unmounted
- Select component again
- Check component still exists

**Problem: Source maps not working**
- Ensure \`devtool: 'source-map'\` in webpack
- Or use Create React App (has source maps by default)

## What's Next?

You've mastered React DevTools! Next, you'll learn about **Legacy Patterns & Migration**—understanding class components, lifecycle methods, and how to read and modernize legacy React code. DevTools + Legacy Knowledge = Ready to work with any React codebase! 🚀
`,
};
