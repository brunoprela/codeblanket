export const functionComponentsTypescript = {
  title: 'Function Components & TypeScript',
  id: 'function-components-typescript',
  content: `
# Function Components & TypeScript

## Introduction

Function components are the modern standard for building React applications. In 2024, **99% of new React code uses function components** over class components. Combined with TypeScript, function components provide type safety, better developer experience, and cleaner code. This section covers everything from basic components to advanced patterns used in production applications.

### Why Function Components Won

**Historical context**:
- **React 0.14 (2015)**: Function components introduced (no state, "stateless functional components")
- **React 16.8 (2019)**: Hooks released → Function components gained full feature parity with classes
- **React 18 (2022)**: Server Components → Only work with function components
- **2024**: Class components considered legacy, function components are the standard

**Market reality**:
- **Meta (Facebook)**: Migrated 5M+ lines to function components
- **Netflix**: 100% function components in new features
- **Airbnb**: Completed migration in 2021
- **Your career**: Interviews focus on hooks and function components, not classes

## Anatomy of a Function Component

\`\`\`tsx
// Basic structure
function ComponentName() {
  // 1. Hooks (must be at top, before any returns)
  const [state, setState] = useState(initialValue);
  const value = useMemo(() => expensiveCalculation(), [deps]);
  
  // 2. Event handlers and helper functions
  const handleClick = () => {
    // Logic here
  };
  
  // 3. Computed values (derived from state/props)
  const fullName = \`\${firstName} \${lastName}\`;
  
  // 4. Side effects
  useEffect(() => {
    // Runs after render
    return () => {
      // Cleanup
    };
  }, [dependencies]);
  
  // 5. Early returns (guards/error states)
  if (error) {
    return <ErrorMessage />;
  }
  
  // 6. Main JSX return
  return (
    <div>
      {/* Component JSX */}
    </div>
  );
}

export default ComponentName;
\`\`\`

## TypeScript with React: Essential Setup

### Installing TypeScript in Existing Project

\`\`\`bash
# Vite (recommended)
npm create vite@latest my-app -- --template react-ts

# Existing project
npm install --save-dev typescript @types/react @types/react-dom

# Generate tsconfig.json
npx tsc --init
\`\`\`

### tsconfig.json for React

\`\`\`json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
\`\`\`

**Key settings explained**:
- \`"jsx": "react-jsx"\`: Modern JSX transform (no React import needed)
- \`"strict": true\`: Enable all strict type checking
- \`"noEmit": true\`: Vite handles compilation, TypeScript just checks types

## Props: The Foundation of Components

### Basic Props with TypeScript

\`\`\`tsx
// Define props interface
interface GreetingProps {
  name: string;
  age: number;
  isStudent: boolean;
}

// Function component with typed props
function Greeting({ name, age, isStudent }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      <p>Age: {age}</p>
      {isStudent && <p>Currently a student</p>}
    </div>
  );
}

// Usage (TypeScript checks prop types)
<Greeting name="Alice" age={25} isStudent={true} />

// ❌ TypeScript Error: Type 'number' is not assignable to type 'string'
<Greeting name={123} age={25} isStudent={true} />

// ❌ TypeScript Error: Property 'age' is missing
<Greeting name="Alice" isStudent={true} />
\`\`\`

**Benefits over PropTypes**:
- ✅ Compile-time errors (catch before deployment)
- ✅ Editor autocomplete (IntelliSense shows all props)
- ✅ Refactoring safety (rename prop → all uses updated)
- ✅ Self-documenting (interface shows exactly what's expected)

### Optional Props and Default Values

\`\`\`tsx
// Method 1: Optional props with '?'
interface UserCardProps {
  name: string;
  email: string;
  avatar?: string;  // Optional
  role?: 'admin' | 'user';  // Optional with union type
}

function UserCard({ name, email, avatar, role }: UserCardProps) {
  return (
    <div className="user-card">
      <img src={avatar || '/default-avatar.png'} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
      {role && <span className="badge">{role}</span>}
    </div>
  );
}

// Valid usage
<UserCard name="Alice" email="alice@example.com" />
<UserCard name="Bob" email="bob@example.com" avatar="/bob.jpg" role="admin" />

// Method 2: Default values with destructuring
interface ButtonProps {
  label: string;
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
}

function Button({ 
  label, 
  variant = 'primary',  // Default value
  size = 'medium', 
  disabled = false 
}: ButtonProps) {
  return (
    <button 
      className={\`btn btn-\${variant} btn-\${size}\`}
      disabled={disabled}
    >
      {label}
    </button>
  );
}

// Usage (variant defaults to 'primary')
<Button label="Click me" />
<Button label="Delete" variant="danger" size="large" />
\`\`\`

### Complex Prop Types

\`\`\`tsx
// 1. Array props
interface TodoListProps {
  todos: Array<{
    id: number;
    text: string;
    completed: boolean;
  }>;
  // Or use type alias for reusability
  // todos: Todo[];
}

type Todo = {
  id: number;
  text: string;
  completed: boolean;
};

// 2. Function props (callbacks)
interface SearchBarProps {
  onSearch: (query: string) => void;  // Function that takes string, returns nothing
  onClear: () => void;  // Function with no params
}

function SearchBar({ onSearch, onClear }: SearchBarProps) {
  const [query, setQuery] = useState('');
  
  return (
    <div>
      <input 
        value={query} 
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={() => onSearch(query)}>Search</button>
      <button onClick={onClear}>Clear</button>
    </div>
  );
}

// Usage
<SearchBar 
  onSearch={(query) => console.log('Searching:', query)}
  onClear={() => console.log('Cleared')}
/>

// 3. Union types (one of several types)
interface StatusBadgeProps {
  status: 'success' | 'warning' | 'error' | 'info';  // Only these 4 values allowed
}

function StatusBadge({ status }: StatusBadgeProps) {
  const colors = {
    success: 'green',
    warning: 'yellow',
    error: 'red',
    info: 'blue'
  };
  
  return <span className={\`badge badge-\${colors[status]}\`}>{status}</span>;
}

// ✅ Valid
<StatusBadge status="success" />

// ❌ TypeScript Error: Type '"invalid"' is not assignable to type 'success' | 'warning' | 'error' | 'info'
<StatusBadge status="invalid" />

// 4. Object props with specific shape
interface UserProfileProps {
  user: {
    id: number;
    name: string;
    email: string;
    settings: {
      theme: 'light' | 'dark';
      notifications: boolean;
    };
  };
}

// 5. Generic props (advanced)
interface TableProps<T> {
  data: T[];
  columns: Array<{
    key: keyof T;
    header: string;
    render?: (value: T[keyof T]) => React.ReactNode;
  }>;
}

function Table<T>({ data, columns }: TableProps<T>) {
  return (
    <table>
      <thead>
        <tr>
          {columns.map(col => (
            <th key={String(col.key)}>{col.header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={index}>
            {columns.map(col => (
              <td key={String(col.key)}>
                {col.render 
                  ? col.render(row[col.key]) 
                  : String(row[col.key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Usage with type inference
interface User {
  id: number;
  name: string;
  email: string;
}

const users: User[] = [
  { id: 1, name: 'Alice', email: 'alice@example.com' },
  { id: 2, name: 'Bob', email: 'bob@example.com' }
];

<Table 
  data={users}
  columns={[
    { key: 'id', header: 'ID' },
    { key: 'name', header: 'Name' },
    { key: 'email', header: 'Email' }
  ]}
/>
\`\`\`

## The Children Prop

\`\`\`tsx
// Method 1: Explicit children prop
interface CardProps {
  title: string;
  children: React.ReactNode;  // Accepts any renderable content
}

function Card({ title, children }: CardProps) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="card-body">
        {children}
      </div>
    </div>
  );
}

// Usage
<Card title="Welcome">
  <p>This is the card content</p>
  <button>Click me</button>
</Card>

// Method 2: PropsWithChildren utility type
import { PropsWithChildren } from 'react';

interface ContainerProps {
  className?: string;
}

function Container({ className, children }: PropsWithChildren<ContainerProps>) {
  return (
    <div className={\`container \${className || ''}\`}>
      {children}
    </div>
  );
}

// Method 3: Render props pattern (advanced)
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => React.ReactNode;
}

function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, [url]);
  
  return <>{children(data, loading, error)}</>;
}

// Usage
<DataFetcher<User[]> url="/api/users">
  {(data, loading, error) => {
    if (loading) return <p>Loading...</p>;
    if (error) return <p>Error: {error.message}</p>;
    return <ul>{data?.map(user => <li key={user.id}>{user.name}</li>)}</ul>;
  }}
</DataFetcher>
\`\`\`

## Component Composition

**Composition is how you build complex UIs from simple components.**

\`\`\`tsx
// Building blocks (small, focused components)
interface AvatarProps {
  src: string;
  alt: string;
  size?: 'small' | 'medium' | 'large';
}

function Avatar({ src, alt, size = 'medium' }: AvatarProps) {
  const sizes = {
    small: 'w-8 h-8',
    medium: 'w-12 h-12',
    large: 'w-16 h-16'
  };
  
  return (
    <img 
      src={src} 
      alt={alt} 
      className={\`rounded-full \${sizes[size]}\`}
    />
  );
}

interface BadgeProps {
  text: string;
  variant: 'primary' | 'secondary' | 'success';
}

function Badge({ text, variant }: BadgeProps) {
  return (
    <span className={\`badge badge-\${variant}\`}>
      {text}
    </span>
  );
}

// Composed component (combines smaller components)
interface UserCardProps {
  user: {
    name: string;
    email: string;
    avatar: string;
    role: 'admin' | 'user' | 'guest';
    isOnline: boolean;
  };
  onMessage: () => void;
}

function UserCard({ user, onMessage }: UserCardProps) {
  return (
    <div className="user-card">
      <div className="user-card-header">
        <Avatar src={user.avatar} alt={user.name} size="large" />
        {user.isOnline && (
          <span className="online-indicator">●</span>
        )}
      </div>
      
      <div className="user-card-body">
        <h3>{user.name}</h3>
        <p className="email">{user.email}</p>
        <Badge 
          text={user.role} 
          variant={user.role === 'admin' ? 'primary' : 'secondary'}
        />
      </div>
      
      <div className="user-card-footer">
        <button onClick={onMessage}>Send Message</button>
      </div>
    </div>
  );
}

// Even more complex composition
interface TeamProps {
  team: {
    name: string;
    members: Array<{
      id: number;
      name: string;
      email: string;
      avatar: string;
      role: 'admin' | 'user' | 'guest';
      isOnline: boolean;
    }>;
  };
}

function Team({ team }: TeamProps) {
  const handleMessage = (memberName: string) => {
    console.log(\`Messaging \${memberName}\`);
  };
  
  return (
    <div className="team">
      <h2>{team.name}</h2>
      <div className="team-grid">
        {team.members.map(member => (
          <UserCard 
            key={member.id}
            user={member}
            onMessage={() => handleMessage(member.name)}
          />
        ))}
      </div>
    </div>
  );
}
\`\`\`

**Why composition is powerful**:
- ✅ **Reusability**: \`Avatar\` works in \`UserCard\`, \`Navbar\`, \`CommentSection\`, etc.
- ✅ **Testability**: Test \`Avatar\` in isolation, test \`UserCard\` with mocked \`Avatar\`
- ✅ **Maintainability**: Change \`Avatar\` size logic once, updates everywhere
- ✅ **Team collaboration**: Designer updates \`Avatar\`, engineer updates \`UserCard\` independently

## File Organization & Naming Conventions

### Recommended Structure

\`\`\`
src/
├── components/
│   ├── ui/              # Reusable UI components
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   ├── Button.module.css
│   │   │   └── index.ts      # Export barrel
│   │   ├── Card/
│   │   └── Avatar/
│   │
│   ├── features/        # Feature-specific components
│   │   ├── UserProfile/
│   │   │   ├── UserProfile.tsx
│   │   │   ├── UserAvatar.tsx
│   │   │   ├── UserBio.tsx
│   │   │   └── index.ts
│   │   └── ProductList/
│   │
│   └── layout/          # Layout components
│       ├── Header.tsx
│       ├── Footer.tsx
│       └── Sidebar.tsx
│
├── types/               # TypeScript types
│   ├── user.ts
│   ├── product.ts
│   └── index.ts
│
├── hooks/               # Custom hooks
│   ├── useAuth.ts
│   └── useLocalStorage.ts
│
└── utils/               # Helper functions
    └── formatters.ts
\`\`\`

### Naming Conventions

\`\`\`tsx
// ✅ CORRECT: PascalCase for components
function UserProfile() {}
function ProductCard() {}
function OrderList() {}

// ❌ WRONG: lowercase or camelCase
function userProfile() {}  // Won't work as React component
function product_card() {}  // Not conventional

// ✅ CORRECT: Component files match component names
// File: UserProfile.tsx
export function UserProfile() {}

// ✅ CORRECT: Type/Interface naming
interface UserProfileProps {}  // Component props
interface User {}              // Data type
type ButtonVariant = 'primary' | 'secondary';

// ✅ CORRECT: Event handler naming
const handleClick = () => {};
const handleSubmit = () => {};
const handleInputChange = () => {};

// Prefix with 'handle' for handlers, 'on' for props
interface ButtonProps {
  onClick: () => void;  // Prop that receives handler
}

function Button({ onClick }: ButtonProps) {
  const handleClick = () => {  // Internal handler
    // Do something
    onClick();  // Call prop
  };
  
  return <button onClick={handleClick}>Click</button>;
}
\`\`\`

## TypeScript Best Practices for React

### 1. Type Inference vs Explicit Types

\`\`\`tsx
// ✅ GOOD: Let TypeScript infer when obvious
const [count, setCount] = useState(0);  // Inferred as number
const [name, setName] = useState('');   // Inferred as string

// ✅ GOOD: Explicit types for complex state
const [user, setUser] = useState<User | null>(null);
const [data, setData] = useState<Product[]>([]);

// ❌ BAD: Unnecessary explicit types
const [count, setCount] = useState<number>(0);  // Redundant, 0 is obviously number
const [name, setName] = useState<string>('');   // Redundant, '' is obviously string
\`\`\`

### 2. Interface vs Type

\`\`\`tsx
// ✅ Use 'interface' for object shapes (extensible)
interface UserProps {
  name: string;
  email: string;
}

interface AdminProps extends UserProps {
  permissions: string[];
}

// ✅ Use 'type' for unions, primitives, tuples
type Status = 'idle' | 'loading' | 'success' | 'error';
type Coordinates = [number, number];
type ID = string | number;

// Both work for props, but interface is preferred for consistency
interface ButtonProps {
  label: string;
}
// or
type ButtonProps = {
  label: string;
};
\`\`\`

### 3. Avoid 'any'

\`\`\`tsx
// ❌ BAD: Using 'any' defeats TypeScript
function processData(data: any) {
  return data.value.toString();  // No type safety, runtime error if data.value doesn't exist
}

// ✅ GOOD: Use proper types
interface DataResponse {
  value: number;
  timestamp: string;
}

function processData(data: DataResponse) {
  return data.value.toString();  // Type safe, editor autocomplete works
}

// ✅ GOOD: Use 'unknown' if type is truly unknown
function safeProcess(data: unknown) {
  if (typeof data === 'object' && data !== null && 'value' in data) {
    return (data as DataResponse).value.toString();
  }
  return 'Invalid data';
}
\`\`\`

## Props Destructuring Patterns

\`\`\`tsx
// Pattern 1: Inline destructuring (most common)
function UserCard({ name, email, avatar }: UserCardProps) {
  return <div>{name}</div>;
}

// Pattern 2: Destructuring with rest props
function Button({ label, ...rest }: ButtonProps & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...rest}>{label}</button>;
}

// Usage: All native button props work
<Button label="Click" onClick={handleClick} disabled type="submit" />

// Pattern 3: Destructuring with aliases (rare, only if needed)
function Card({ 
  title: cardTitle,  // Alias to avoid naming conflict
  children 
}: CardProps) {
  const title = 'Something else';  // Different variable
  return <div>{cardTitle}</div>;
}

// Pattern 4: Default values in destructuring
function Alert({ 
  message, 
  type = 'info',  // Default value
  dismissible = false 
}: AlertProps) {
  return <div className={\`alert alert-\${type}\`}>{message}</div>;
}
\`\`\`

## Interview Preparation: Common Questions

### Q1: "What's the difference between function components and class components?"

**Answer**: Function components are modern React (2019+), class components are legacy. Key differences:

| Feature | Function Components | Class Components |
|---------|-------------------|------------------|
| Syntax | Simple function | ES6 class with render() |
| State | useState hook | this.state |
| Side effects | useEffect hook | componentDidMount, etc. |
| Code | 30% less code | More verbose |
| Performance | Slight edge | Slightly slower |
| Future | All new features | No new features |

Example: Same component both ways:

\`\`\`tsx
// Function component (modern)
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}

// Class component (legacy)
class Counter extends React.Component {
  state = { count: 0 };
  render() {
    return (
      <button onClick={() => this.setState({ count: this.state.count + 1 })}>
        {this.state.count}
      </button>
    );
  }
}
\`\`\`

### Q2: "Why use TypeScript with React?"

**Answer**: Type safety catches bugs before production:
1. **Compile-time errors**: Typos caught instantly, not in production
2. **Refactoring confidence**: Rename prop → TypeScript updates all 50 uses
3. **Documentation**: Interface shows exactly what props component needs
4. **Editor support**: Autocomplete, hover documentation, go-to-definition
5. **Team collaboration**: Types are contracts between team members

Real-world: Airbnb prevented 38% of bugs with TypeScript according to their migration report.

### Q3: "What is props drilling and how do you solve it?"

**Answer**: Props drilling is passing props through components that don't need them:

\`\`\`tsx
// ❌ Props drilling (bad)
function App() {
  const user = { name: 'Alice' };
  return <PageLayout user={user} />;
}

function PageLayout({ user }) {
  return <Sidebar user={user} />;  // Doesn't use user, just passes it
}

function Sidebar({ user }) {
  return <UserMenu user={user} />;  // Doesn't use user, just passes it
}

function UserMenu({ user }) {
  return <div>{user.name}</div>;  // Finally uses user!
}

// ✅ Solution 1: Context API (Module 2)
const UserContext = createContext();

function App() {
  const user = { name: 'Alice' };
  return (
    <UserContext.Provider value={user}>
      <PageLayout />
    </UserContext.Provider>
  );
}

function UserMenu() {
  const user = useContext(UserContext);  // Direct access, no drilling
  return <div>{user.name}</div>;
}

// ✅ Solution 2: Component composition
function App() {
  const user = { name: 'Alice' };
  return (
    <PageLayout>
      <Sidebar>
        <UserMenu user={user} />  // Pass directly to component that needs it
      </Sidebar>
    </PageLayout>
  );
}
\`\`\`

## Common Mistakes and How to Avoid Them

### Mistake 1: Forgetting to export component

\`\`\`tsx
// ❌ WRONG: Component not exported
function Button() {
  return <button>Click</button>;
}

// Other file
import Button from './Button';  // Error: No default export

// ✅ CORRECT: Named export
export function Button() {
  return <button>Click</button>;
}

// Usage
import { Button } from './Button';

// ✅ ALSO CORRECT: Default export
function Button() {
  return <button>Click</button>;
}

export default Button;

// Usage
import Button from './Button';
\`\`\`

### Mistake 2: Incorrect prop types

\`\`\`tsx
// ❌ WRONG: Passing string instead of number
<UserCard age="25" />  // TypeScript error if age is typed as number

// ✅ CORRECT
<UserCard age={25} />  // Curly braces for numbers, booleans, objects

// ❌ WRONG: Passing boolean as string
<Button disabled="true" />  // This sets disabled to truthy string "true", not boolean

// ✅ CORRECT
<Button disabled={true} />  // or just <Button disabled />
\`\`\`

### Mistake 3: Not using key in lists

\`\`\`tsx
// ❌ WRONG: No key prop
users.map(user => <UserCard name={user.name} />)

// ✅ CORRECT
users.map(user => <UserCard key={user.id} name={user.name} />)
\`\`\`

## What's Next?

Now that you understand function components, props, and TypeScript, you're ready to learn about **state management with useState**. State is what makes your components interactive and dynamic—the next critical piece of React.

In the next section, you'll learn:
- How to add state to components
- When to use state vs props
- State immutability
- Common state patterns
- Performance considerations

The journey continues! 🚀
`,
};
