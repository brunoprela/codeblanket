export const introductionToReactJsx = {
  title: 'Introduction to React & JSX',
  id: 'introduction-to-react-jsx',
  content: `
# Introduction to React & JSX

## Introduction

React has revolutionized web development since its release by Facebook in 2013. **In 2024, React remains the most popular frontend library**, powering millions of websites from startups to Fortune 500 companies like Meta, Netflix, Airbnb, and Tesla. Understanding React isn't just about learning a library—it's about mastering the paradigm that defines modern web development.

### Why React Dominates Web Development

**Market Dominance (2024)**:
- **42%** of all websites use React (according to BuiltWith)
- **65%** of developers prefer React over other frameworks (Stack Overflow Survey 2024)
- **200,000+** open source packages built for React
- React Native enables mobile development with the same skills

**Real-World Impact**:
- **Facebook** serves 3 billion users with React
- **Netflix** reduced startup time by 70% after migrating to React
- **Airbnb** manages millions of listings with React

## What is React?

React is a **declarative**, **component-based** JavaScript library for building user interfaces. Let's break down what this means:

### Declarative vs Imperative Programming

**Imperative (Vanilla JavaScript)**:
You tell the browser *how* to update the UI step-by-step.

\`\`\`javascript
// Imperative: Manual DOM manipulation
const button = document.createElement('button');
button.textContent = 'Click me';
button.addEventListener('click', () => {
  const count = parseInt(button.dataset.count || '0');
  button.dataset.count = (count + 1).toString();
  button.textContent = \`Clicked \${count + 1} times\`;
});
document.body.appendChild(button);
\`\`\`

**Problems with imperative approach**:
- 8 lines of code for a simple button
- Error-prone (forgot to update textContent? UI is out of sync)
- Hard to maintain (logic scattered across event handlers)
- No component reusability

**Declarative (React)**:
You describe *what* the UI should look like, React handles the *how*.

\`\`\`jsx
// Declarative: Describe the UI state
function ClickButton() {
  const [count, setCount] = useState(0);
  
  return (
    <button onClick={() => setCount(count + 1)}>
      Clicked {count} times
    </button>
  );
}
\`\`\`

**Benefits of declarative approach**:
- 6 lines of code (25% less)
- UI always matches state (impossible to get out of sync)
- Easy to understand (reads like the UI it creates)
- Reusable component

### Component-Based Architecture

React applications are built from **components**—reusable, self-contained pieces of UI. Think of components like LEGO blocks: each piece is independent, but you combine them to build complex structures.

\`\`\`jsx
// Simple components
function Header() {
  return <h1>My App</h1>;
}

function SearchBar() {
  return <input type="text" placeholder="Search..." />;
}

function ProductList({ products }) {
  return (
    <div>
      {products.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}

// Complex component built from simple ones
function App() {
  return (
    <div>
      <Header />
      <SearchBar />
      <ProductList products={data} />
    </div>
  );
}
\`\`\`

**Why components matter**:
- **Reusability**: Write once, use everywhere
- **Maintainability**: Bug in \`SearchBar\`? Fix it in one place
- **Testability**: Test components in isolation
- **Collaboration**: Teams can work on different components simultaneously

## The Virtual DOM: React's Secret Weapon

One of React's key innovations is the **Virtual DOM**—the reason React is so fast.

### Traditional DOM Manipulation Problem

\`\`\`javascript
// Updating 1000 items in vanilla JavaScript
const items = [...Array(1000)].map((_, i) => ({ id: i, value: Math.random() }));

// Slow: Direct DOM manipulation
items.forEach(item => {
  const element = document.getElementById(\`item-\${item.id}\`);
  element.textContent = item.value; // Triggers browser reflow/repaint
});
// Performance: ~50ms for 1000 updates
\`\`\`

**Problem**: Each DOM update triggers browser reflow and repaint—expensive operations.

### React's Virtual DOM Solution

\`\`\`
┌─────────────────────────────────────────────────────┐
│  1. Change in state                                  │
│     const [count, setCount] = useState(0)           │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  2. React creates Virtual DOM                        │
│     { type: 'div', props: { children: count } }     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  3. React diffs old vs new Virtual DOM               │
│     Old: <div>0</div>                                │
│     New: <div>1</div>                                │
│     Diff: Only textContent changed                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  4. React updates ONLY what changed in real DOM      │
│     element.textContent = '1'; // Single update      │
└─────────────────────────────────────────────────────┘

Performance: ~5ms for 1000 updates (10x faster)
\`\`\`

**Virtual DOM workflow**:
1. **State changes** → \`setCount(1)\`
2. **Virtual DOM created** → JavaScript object representation
3. **Diffing algorithm** → React compares old vs new Virtual DOM
4. **Minimal updates** → React updates only what changed in real DOM

**Benefits**:
- **10x faster** than manual DOM manipulation
- **Batching**: React batches multiple updates into one DOM operation
- **Cross-platform**: Same code works for web (React DOM) and mobile (React Native)

## JSX: JavaScript XML

JSX is React's syntax extension that looks like HTML but is actually JavaScript.

### JSX vs HTML: Key Differences

\`\`\`jsx
// JSX (looks like HTML, but it's JavaScript)
function Greeting() {
  const name = "Alice";
  const isLoggedIn = true;
  
  return (
    <div className="container">
      <h1 style={{ color: 'blue', fontSize: '24px' }}>
        Hello, {name}!
      </h1>
      {isLoggedIn && <p>Welcome back!</p>}
      <label htmlFor="email">Email:</label>
      <input id="email" />
    </div>
  );
}
\`\`\`

### JSX vs HTML Differences Table

| HTML | JSX | Reason |
|------|-----|--------|
| \`class="container"\` | \`className="container"\` | \`class\` is a reserved word in JavaScript |
| \`for="email"\` | \`htmlFor="email"\` | \`for\` is a reserved word in JavaScript |
| \`style="color: blue;"\` | \`style={{ color: 'blue' }}\` | Style is a JavaScript object, not a string |
| \`onclick="handler()"\` | \`onClick={handler}\` | camelCase for event handlers |
| \`<!-- comment -->\` | \`{/* comment */}\` | JavaScript comments inside JSX |
| \`<input>\` | \`<input />\` | All tags must be closed (XML syntax) |

### JSX Under the Hood

**What you write** (JSX):

\`\`\`jsx
const element = <h1 className="greeting">Hello, world!</h1>;
\`\`\`

**What Babel compiles it to** (JavaScript):

\`\`\`javascript
const element = React.createElement(
  'h1',
  { className: 'greeting' },
  'Hello, world!'
);
\`\`\`

**What React.createElement returns** (Virtual DOM object):

\`\`\`javascript
{
  type: 'h1',
  props: {
    className: 'greeting',
    children: 'Hello, world!'
  },
  key: null,
  ref: null
}
\`\`\`

### JSX Expressions

You can embed any JavaScript expression in JSX using curly braces \`{}\`.

\`\`\`jsx
function ExpressionExamples() {
  const user = { name: 'Alice', age: 30 };
  const products = ['Apple', 'Banana', 'Cherry'];
  
  return (
    <div>
      {/* Variable */}
      <p>Name: {user.name}</p>
      
      {/* Expression */}
      <p>Age next year: {user.age + 1}</p>
      
      {/* Function call */}
      <p>Uppercase: {user.name.toUpperCase()}</p>
      
      {/* Ternary operator */}
      <p>{user.age >= 18 ? 'Adult' : 'Minor'}</p>
      
      {/* Array method */}
      <ul>
        {products.map((product, index) => (
          <li key={index}>{product}</li>
        ))}
      </ul>
      
      {/* Logical AND */}
      {user.age >= 21 && <p>Can drink alcohol</p>}
      
      {/* Template literal */}
      <p>{\`\${user.name} is \${user.age} years old\`}</p>
    </div>
  );
}
\`\`\`

**Important**: Only *expressions* work in \`{}\`, not *statements*.

\`\`\`jsx
// ✅ WORKS (expressions)
{2 + 2}
{user.name}
{isLoggedIn ? 'Yes' : 'No'}
{items.map(i => <div>{i}</div>)}

// ❌ DOESN'T WORK (statements)
{if (isLoggedIn) { return 'Yes' }}  // if is a statement
{const x = 5}                        // const is a statement
{for (let i = 0; i < 10; i++) {}}   // for is a statement
\`\`\`

### JSX Rules and Best Practices

**Rule 1: Return a Single Root Element**

\`\`\`jsx
// ❌ WRONG: Multiple root elements
function Wrong() {
  return (
    <h1>Title</h1>
    <p>Paragraph</p>
  );
}

// ✅ CORRECT: Wrapped in parent element
function Correct1() {
  return (
    <div>
      <h1>Title</h1>
      <p>Paragraph</p>
    </div>
  );
}

// ✅ BEST: Use Fragment (no extra DOM node)
function Correct2() {
  return (
    <>
      <h1>Title</h1>
      <p>Paragraph</p>
    </>
  );
}
\`\`\`

**Rule 2: Close All Tags**

\`\`\`jsx
// ❌ WRONG: Unclosed tags
<input>
<img>
<br>

// ✅ CORRECT: All tags closed
<input />
<img />
<br />
\`\`\`

**Rule 3: camelCase Property Names**

\`\`\`jsx
// HTML uses kebab-case
<div class="container" onclick="handleClick()"></div>

// JSX uses camelCase
<div className="container" onClick={handleClick}></div>
\`\`\`

## React vs Other Frameworks (2024)

| Feature | React | Vue | Angular | Svelte |
|---------|-------|-----|---------|--------|
| **Learning Curve** | Medium | Easy | Steep | Easy |
| **Bundle Size** | 42 KB | 34 KB | 167 KB | 5 KB |
| **Performance** | Fast | Fast | Moderate | Fastest |
| **Job Market** | 🔥 Highest | Medium | High | Growing |
| **Ecosystem** | 🏆 Largest | Large | Large | Growing |
| **Mobile (Native)** | ✅ React Native | ❌ Limited | ❌ Limited | ❌ No |
| **Popularity (2024)** | 1st | 2nd | 3rd | 4th |
| **Companies** | Meta, Netflix, Airbnb | Alibaba, GitLab | Google, Microsoft | NYT, Apple |

**When to choose React**:
- ✅ Building complex, interactive UIs
- ✅ Need a huge ecosystem (libraries, tools, community)
- ✅ Want mobile app development (React Native)
- ✅ Hiring developers (largest talent pool)
- ✅ Enterprise applications

**When to consider alternatives**:
- Vue: Need faster learning curve
- Angular: Need opinionated, all-in-one framework
- Svelte: Need smallest bundle size, simple apps

## Setting Up Your React Development Environment

### Method 1: Vite (Recommended 2024)

**Vite** is the modern standard—10x faster than Create React App.

\`\`\`bash
# Create new React + TypeScript project
npm create vite@latest my-app -- --template react-ts

# Navigate and install
cd my-app
npm install

# Start development server
npm run dev
\`\`\`

**Why Vite?**
- ⚡ **Lightning fast**: Hot Module Replacement in <50ms
- 📦 **Small bundle**: Optimized production builds
- 🛠️ **Modern**: Uses native ES modules
- 💎 **Great DX**: TypeScript, JSX, CSS out of the box

### Method 2: Next.js (For Full-Stack Apps)

\`\`\`bash
# Create Next.js app
npx create-next-app@latest my-app

# Options to select:
# ✅ TypeScript
# ✅ ESLint  
# ✅ Tailwind CSS
# ✅ App Router
\`\`\`

### Project Structure (Vite)

\`\`\`
my-app/
├── src/
│   ├── App.tsx          # Main App component
│   ├── main.tsx         # Entry point
│   ├── components/      # Reusable components
│   ├── hooks/           # Custom hooks
│   ├── utils/           # Helper functions
│   └── styles/          # CSS files
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
└── vite.config.ts       # Vite config
\`\`\`

## Your First React Component

\`\`\`tsx
// src/App.tsx
import { useState } from 'react';

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="app">
      <h1>Welcome to React</h1>
      <p>You've clicked the button {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

export default App;
\`\`\`

**Let's break down what's happening**:
1. \`import { useState }\` → Import React hook for state management
2. \`function App()\` → Define a component (just a JavaScript function)
3. \`const [count, setCount]\` → Create state variable (starts at 0)
4. \`return ( JSX )\` → Return JSX describing the UI
5. \`onClick={() => ...}\` → Event handler to update state
6. \`export default App\` → Export component for use in other files

## Common Beginner Mistakes

### Mistake 1: Forgetting \`{}\` for JavaScript

\`\`\`jsx
// ❌ WRONG: Tries to render string "count"
<p>You clicked count times</p>

// ✅ CORRECT: Evaluates count variable
<p>You clicked {count} times</p>
\`\`\`

### Mistake 2: Using Quotes with \`{}\`

\`\`\`jsx
// ❌ WRONG: Quotes turn it into a string
<button onClick="{handleClick}">Click</button>

// ✅ CORRECT: No quotes for JavaScript expressions
<button onClick={handleClick}>Click</button>
\`\`\`

### Mistake 3: Modifying State Directly

\`\`\`jsx
// ❌ WRONG: Direct mutation doesn't trigger re-render
count = count + 1;

// ✅ CORRECT: Use setter function
setCount(count + 1);
\`\`\`

### Mistake 4: Forgetting Keys in Lists

\`\`\`jsx
// ❌ WRONG: No key attribute
{items.map(item => <div>{item}</div>)}

// ✅ CORRECT: Unique key for each item
{items.map((item, index) => <div key={index}>{item}</div>)}
\`\`\`

## JSX Compilation: Behind the Scenes

Modern React uses **automatic JSX transform** (React 17+), so you don't need to import React anymore.

**Old way** (React 16 and earlier):

\`\`\`jsx
import React from 'react'; // Required!

function App() {
  return <div>Hello</div>;
}
\`\`\`

**New way** (React 17+):

\`\`\`jsx
// No import needed!
function App() {
  return <div>Hello</div>;
}
\`\`\`

**What Babel does** (simplified):

\`\`\`javascript
// Your JSX
<div className="container">
  <h1>Hello, {name}!</h1>
  <p>Welcome</p>
</div>

// Compiles to (React 17+)
import { jsx as _jsx } from 'react/jsx-runtime';

_jsx('div', {
  className: 'container',
  children: [
    _jsx('h1', { children: ['Hello, ', name, '!'] }),
    _jsx('p', { children: 'Welcome' })
  ]
});
\`\`\`

## Performance Implications of JSX

**JSX is fast** because:
1. **Compiled**: JSX → JavaScript at build time (zero runtime cost)
2. **Virtual DOM**: React minimizes real DOM updates
3. **Batch updates**: React batches state updates together

**Example**: Updating 10 state variables

\`\`\`jsx
// React batches these into one re-render
setName('Alice');
setAge(30);
setEmail('alice@example.com');
// ... 7 more updates

// Only 1 DOM update, not 10!
\`\`\`

## Interview Preparation: Common Questions

### Q1: "What is React and why use it?"

**Answer**: React is a declarative, component-based JavaScript library for building user interfaces. Key benefits: (1) Virtual DOM for performance, (2) Component reusability, (3) One-way data flow for predictability, (4) Huge ecosystem and community, (5) React Native for mobile development.

### Q2: "Explain the Virtual DOM"

**Answer**: Virtual DOM is a lightweight JavaScript representation of the real DOM. When state changes, React: (1) Creates new Virtual DOM, (2) Diffs it with previous Virtual DOM, (3) Updates only what changed in real DOM. This is faster than manual DOM manipulation because React minimizes expensive DOM operations.

### Q3: "What is JSX?"

**Answer**: JSX is a syntax extension that looks like HTML but is actually JavaScript. Babel compiles JSX to React.createElement() calls. Benefits: (1) Familiar HTML-like syntax, (2) Compile-time errors, (3) Prevents XSS attacks by escaping values, (4) More readable than createElement() calls.

### Q4: "JSX vs HTML differences?"

**Answer**: Key differences: (1) className vs class, (2) htmlFor vs for, (3) camelCase event handlers (onClick vs onclick), (4) style is object not string, (5) all tags must close, (6) comments use {/* */}.

### Q5: "Can you use JavaScript expressions in JSX?"

**Answer**: Yes, using curly braces {}. You can use: variables, function calls, ternary operators, array methods, logical operators. But NOT statements like if/for/const—only expressions.

## Best Practices Summary

1. ✅ **Use fragments** \`<>...</>\` to avoid unnecessary wrapper divs
2. ✅ **Close all tags** including self-closing tags like \`<input />\`
3. ✅ **Use camelCase** for all JSX attributes
4. ✅ **Extract complex expressions** into variables for readability
5. ✅ **Always add keys** when rendering lists
6. ✅ **Use meaningful component names** (PascalCase)
7. ✅ **Keep components small** (< 200 lines)
8. ✅ **Use TypeScript** for better developer experience

## Real-World Example: Building a Card Component

\`\`\`tsx
// components/UserCard.tsx
interface User {
  id: number;
  name: string;
  email: string;
  avatar: string;
  isOnline: boolean;
}

interface UserCardProps {
  user: User;
  onMessage: (userId: number) => void;
}

function UserCard({ user, onMessage }: UserCardProps) {
  const { id, name, email, avatar, isOnline } = user;
  
  return (
    <div className="user-card">
      <div className="avatar-container">
        <img src={avatar} alt={\`\${name}'s avatar\`} />
        {isOnline && <span className="online-badge">Online</span>}
      </div>
      
      <div className="user-info">
        <h3>{name}</h3>
        <p className="email">{email}</p>
      </div>
      
      <button 
        className="message-btn"
        onClick={() => onMessage(id)}
      >
        Send Message
      </button>
    </div>
  );
}

export default UserCard;
\`\`\`

**Why this is good code**:
- ✅ TypeScript for type safety
- ✅ Destructuring for cleaner code
- ✅ Conditional rendering for online badge
- ✅ Event handler with parameter
- ✅ Semantic JSX structure
- ✅ Accessible (alt text on image)

## What's Next?

In the next section, we'll dive deep into **Function Components & TypeScript**, covering:
- How to create components
- Props and prop types
- Component composition
- TypeScript best practices
- File organization patterns

React is a journey—you've taken the first step. The concepts in this section are foundational to everything else you'll learn. Make sure you understand JSX, the Virtual DOM, and how React components work before moving forward.

## Additional Resources

**Official Documentation**:
- [React Docs](https://react.dev) - Official React documentation
- [React Beta Docs](https://react.dev/learn) - New interactive tutorials

**Tools**:
- [Vite](https://vitejs.dev) - Build tool
- [TypeScript](https://www.typescriptlang.org) - Type safety
- [React DevTools](https://react.dev/learn/react-developer-tools) - Browser extension

**Practice**:
- [React Challenges](https://reactchallenges.live) - Hands-on exercises
- [CodeSandbox](https://codesandbox.io) - Online editor

The journey of a thousand components begins with a single \`<App />\`. 🚀
`,
};
