export const conditionalRendering = {
  title: 'Conditional Rendering',
  id: 'conditional-rendering',
  content: `
# Conditional Rendering

## Introduction

**Conditional rendering** is how you show or hide UI elements based on conditions—it's fundamental to building dynamic React applications. Whether you're showing a loading spinner, displaying user-specific content, or implementing complex UI logic, conditional rendering is the tool you need.

In React, conditional rendering works the same way conditions work in JavaScript. You can use:
- \`if/else\` statements
- Ternary operators (\`condition ? true : false\`)
- Logical AND operator (\`&&\`)
- Switch statements
- Early returns

### Why Conditional Rendering Matters

\`\`\`tsx
// Without conditional rendering: Static
function Profile() {
  return (
    <div>
      <h1>Welcome!</h1>
      <p>You are logged in</p>
      <p>You are logged out</p>
    </div>
  );
}

// With conditional rendering: Dynamic
function Profile({ isLoggedIn, user }) {
  return (
    <div>
      <h1>Welcome{isLoggedIn && \`, \${user.name}\`}!</h1>
      {isLoggedIn ? (
        <p>You are logged in</p>
      ) : (
        <p>Please log in to continue</p>
      )}
    </div>
  );
}
\`\`\`

## if/else Statements

The most straightforward approach—use JavaScript \`if/else\` before the return statement.

\`\`\`tsx
function Greeting({ isLoggedIn, username }) {
  if (isLoggedIn) {
    return (
      <div>
        <h1>Welcome back, {username}!</h1>
        <button>Logout</button>
      </div>
    );
  } else {
    return (
      <div>
        <h1>Please sign in</h1>
        <button>Login</button>
      </div>
    );
  }
}

// Or with early return (cleaner)
function Greeting({ isLoggedIn, username }) {
  if (!isLoggedIn) {
    return (
      <div>
        <h1>Please sign in</h1>
        <button>Login</button>
      </div>
    );
  }
  
  return (
    <div>
      <h1>Welcome back, {username}!</h1>
      <button>Logout</button>
    </div>
  );
}
\`\`\`

**When to use if/else**:
- ✅ Completely different UI for each condition
- ✅ Complex logic before rendering
- ✅ Multiple conditions

**Pros**:
- Very readable for complex logic
- Familiar JavaScript syntax
- Easy to debug

**Cons**:
- More verbose than ternary
- Can't use inline in JSX

## Ternary Operator

Use \`condition ? true : false\` for inline conditional rendering.

\`\`\`tsx
function Status({ isOnline }) {
  return (
    <div>
      <span className={isOnline ? 'online' : 'offline'}>
        {isOnline ? '● Online' : '○ Offline'}
      </span>
    </div>
  );
}

// Nested ternaries (use sparingly!)
function UserStatus({ user }) {
  return (
    <div>
      {user === null ? (
        <p>Loading...</p>
      ) : user.isAdmin ? (
        <p>Admin User</p>
      ) : (
        <p>Regular User</p>
      )}
    </div>
  );
}

// Better: Multiple if/else for nested conditions
function UserStatus({ user }) {
  if (user === null) return <p>Loading...</p>;
  if (user.isAdmin) return <p>Admin User</p>;
  return <p>Regular User</p>;
}
\`\`\`

**When to use ternary**:
- ✅ Simple condition with two outcomes
- ✅ Inline rendering (inside JSX)
- ✅ Short, readable alternatives

**Pros**:
- Concise
- Works inline in JSX
- No need for variables

**Cons**:
- Hard to read when nested
- Must have both branches

## Logical AND Operator (&&)

Use \`&&\` to render something **only if** a condition is true.

\`\`\`tsx
function Notifications({ notifications }) {
  return (
    <div>
      <h2>Notifications</h2>
      {notifications.length > 0 && (
        <p>You have {notifications.length} new notifications</p>
      )}
    </div>
  );
}

// Multiple conditions
function Dashboard({ user }) {
  return (
    <div>
      {user.isAdmin && <AdminPanel />}
      {user.hasMessages && <MessageIndicator count={user.messageCount} />}
      {user.isPremium && <PremiumFeatures />}
    </div>
  );
}
\`\`\`

**How && works**:
\`\`\`tsx
// JavaScript evaluates left to right
true && <Component />   // Returns: <Component />
false && <Component />  // Returns: false (renders nothing)

// React renders:
// - true/false/null/undefined → nothing
// - JSX elements → the element
// - strings/numbers → as text
\`\`\`

**⚠️ Common Pitfall with Numbers**:
\`\`\`tsx
// ❌ WRONG: Renders "0" if count is 0
function Items({ count }) {
  return (
    <div>
      {count && <p>You have {count} items</p>}
    </div>
  );
}

// When count=0: 0 && <p>... evaluates to 0
// React renders "0" (numbers are rendered as text!)

// ✅ CORRECT: Use explicit comparison
function Items({ count }) {
  return (
    <div>
      {count > 0 && <p>You have {count} items</p>}
    </div>
  );
}

// Or use ternary with null
function Items({ count }) {
  return (
    <div>
      {count ? <p>You have {count} items</p> : null}
    </div>
  );
}
\`\`\`

**When to use &&**:
- ✅ Render something OR nothing (not two alternatives)
- ✅ Simple boolean conditions
- ✅ Multiple independent conditions

**Pros**:
- Very concise
- Reads naturally ("if this, show that")

**Cons**:
- Easy to make mistakes with falsy values (0, '', NaN)
- No "else" branch

## Element Variables

Store JSX in variables for complex conditional logic.

\`\`\`tsx
function LoginControl({ isLoggedIn, user }) {
  let button;
  let greeting;
  
  if (isLoggedIn) {
    button = <button onClick={logout}>Logout</button>;
    greeting = <h1>Welcome, {user.name}!</h1>;
  } else {
    button = <button onClick={login}>Login</button>;
    greeting = <h1>Please sign in</h1>;
  }
  
  return (
    <div>
      {greeting}
      {button}
    </div>
  );
}

// With switch statement
function StatusBadge({ status }) {
  let badge;
  
  switch (status) {
    case 'pending':
      badge = <span className="badge badge-yellow">Pending</span>;
      break;
    case 'approved':
      badge = <span className="badge badge-green">Approved</span>;
      break;
    case 'rejected':
      badge = <span className="badge badge-red">Rejected</span>;
      break;
    default:
      badge = <span className="badge badge-gray">Unknown</span>;
  }
  
  return <div>{badge}</div>;
}
\`\`\`

**When to use element variables**:
- ✅ Complex conditional logic
- ✅ Reusing JSX in multiple places
- ✅ Switch statements

**Pros**:
- Very readable for complex logic
- Easy to test each branch
- Clean separation of logic and JSX

**Cons**:
- More verbose
- Extra variables in scope

## Preventing Component Render

Return \`null\` to prevent a component from rendering.

\`\`\`tsx
function Warning({ show, message }) {
  if (!show) {
    return null;  // Component renders nothing
  }
  
  return (
    <div className="warning">
      <p>{message}</p>
    </div>
  );
}

// Usage
<Warning show={hasWarning} message="Please verify your email" />

// If hasWarning is false, Warning renders nothing (not even a wrapper div)
\`\`\`

**Important**: Returning \`null\` doesn't prevent lifecycle methods or hooks from running. The component still mounts and updates.

\`\`\`tsx
function Example({ show }) {
  useEffect(() => {
    console.log('Effect runs even if returning null!');
  });
  
  if (!show) return null;
  
  return <div>Visible</div>;
}
\`\`\`

## Loading States

Classic pattern: Show loading spinner while data fetches.

\`\`\`tsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetch(\`/api/users/\${userId}\`)
      .then(res => res.json())
      .then(data => {
        setUser(data);
        setIsLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsLoading(false);
      });
  }, [userId]);
  
  // Loading state
  if (isLoading) {
    return <LoadingSpinner />;
  }
  
  // Error state
  if (error) {
    return <ErrorMessage error={error} />;
  }
  
  // No data state
  if (!user) {
    return <p>No user found</p>;
  }
  
  // Success state
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
\`\`\`

## Error States

Handle errors gracefully with conditional rendering.

\`\`\`tsx
function DataFetcher() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    fetch('/api/data')
      .then(res => {
        if (!res.ok) {
          throw new Error(\`HTTP \${res.status}: \${res.statusText}\`);
        }
        return res.json();
      })
      .then(data => {
        setData(data);
        setIsLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsLoading(false);
      });
  }, []);
  
  if (isLoading) {
    return (
      <div className="loading">
        <Spinner />
        <p>Loading data...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error">
        <p>Error: {error}</p>
        <button onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }
  
  return (
    <div>
      <h2>Data loaded successfully!</h2>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
\`\`\`

## Empty States

Show helpful UI when there's no data.

\`\`\`tsx
function TodoList({ todos }) {
  if (todos.length === 0) {
    return (
      <div className="empty-state">
        <img src="/empty-box.svg" alt="No todos" />
        <h3>No todos yet</h3>
        <p>Add your first todo to get started!</p>
        <button>Add Todo</button>
      </div>
    );
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

## Permission-Based Rendering

Show/hide features based on user permissions.

\`\`\`tsx
interface User {
  name: string;
  role: 'admin' | 'user' | 'guest';
  permissions: string[];
}

function Dashboard({ user }: { user: User }) {
  const canEdit = user.permissions.includes('edit');
  const canDelete = user.permissions.includes('delete');
  const isAdmin = user.role === 'admin';
  
  return (
    <div>
      <h1>Dashboard</h1>
      
      {isAdmin && <AdminPanel />}
      
      {canEdit && <button>Edit</button>}
      {canDelete && <button>Delete</button>}
      
      {user.role === 'guest' && (
        <div className="alert">
          <p>You have limited access. Please sign up for full features.</p>
        </div>
      )}
    </div>
  );
}

// Reusable permission wrapper
function HasPermission({ 
  user, 
  permission, 
  children 
}: { 
  user: User; 
  permission: string; 
  children: React.ReactNode;
}) {
  if (!user.permissions.includes(permission)) {
    return null;
  }
  
  return <>{children}</>;
}

// Usage
<HasPermission user={user} permission="edit">
  <button>Edit</button>
</HasPermission>
\`\`\`

## Switch Statement Pattern

Use switch for multiple distinct states.

\`\`\`tsx
type OrderStatus = 'pending' | 'processing' | 'shipped' | 'delivered' | 'cancelled';

function OrderStatusBadge({ status }: { status: OrderStatus }) {
  switch (status) {
    case 'pending':
      return (
        <span className="badge badge-yellow">
          ⏳ Pending
        </span>
      );
    
    case 'processing':
      return (
        <span className="badge badge-blue">
          🔄 Processing
        </span>
      );
    
    case 'shipped':
      return (
        <span className="badge badge-purple">
          🚚 Shipped
        </span>
      );
    
    case 'delivered':
      return (
        <span className="badge badge-green">
          ✓ Delivered
        </span>
      );
    
    case 'cancelled':
      return (
        <span className="badge badge-red">
          ✗ Cancelled
        </span>
      );
    
    default:
      return <span className="badge">Unknown</span>;
  }
}

// Or with object mapping (cleaner)
const STATUS_CONFIG = {
  pending: { icon: '⏳', label: 'Pending', color: 'yellow' },
  processing: { icon: '🔄', label: 'Processing', color: 'blue' },
  shipped: { icon: '🚚', label: 'Shipped', color: 'purple' },
  delivered: { icon: '✓', label: 'Delivered', color: 'green' },
  cancelled: { icon: '✗', label: 'Cancelled', color: 'red' }
};

function OrderStatusBadge({ status }: { status: OrderStatus }) {
  const config = STATUS_CONFIG[status];
  
  if (!config) {
    return <span className="badge">Unknown</span>;
  }
  
  return (
    <span className={\`badge badge-\${config.color}\`}>
      {config.icon} {config.label}
    </span>
  );
}
\`\`\`

## Multiple Conditions

Combine conditions with logical operators.

\`\`\`tsx
function ArticleActions({ article, user }) {
  const isAuthor = article.authorId === user.id;
  const isAdmin = user.role === 'admin';
  const canEdit = isAuthor || isAdmin;
  const canDelete = isAdmin;
  const isPublished = article.status === 'published';
  
  return (
    <div className="actions">
      {/* Can edit if author or admin */}
      {canEdit && <button>Edit</button>}
      
      {/* Can delete only if admin */}
      {canDelete && <button>Delete</button>}
      
      {/* Show publish button if unpublished and can edit */}
      {!isPublished && canEdit && <button>Publish</button>}
      
      {/* Show unpublish button if published and can edit */}
      {isPublished && canEdit && <button>Unpublish</button>}
      
      {/* Show warning if not author */}
      {!isAuthor && (
        <p className="warning">
          You are editing someone else's article
        </p>
      )}
    </div>
  );
}
\`\`\`

## Render Props Pattern

Pass render functions as props for flexible conditional rendering.

\`\`\`tsx
interface FeatureFlagProps {
  flag: string;
  children: (isEnabled: boolean) => React.ReactNode;
}

function FeatureFlag({ flag, children }: FeatureFlagProps) {
  const [isEnabled, setIsEnabled] = useState(false);
  
  useEffect(() => {
    // Check if feature flag is enabled
    fetch(\`/api/features/\${flag}\`)
      .then(res => res.json())
      .then(data => setIsEnabled(data.enabled));
  }, [flag]);
  
  return <>{children(isEnabled)}</>;
}

// Usage
<FeatureFlag flag="new-dashboard">
  {(isEnabled) => (
    isEnabled ? <NewDashboard /> : <OldDashboard />
  )}
</FeatureFlag>
\`\`\`

## Best Practices

### 1. Keep Conditions Simple

\`\`\`tsx
// ❌ BAD: Complex nested ternaries
{user ? (
  user.isPremium ? (
    user.hasAccess ? (
      <PremiumContent />
    ) : <AccessDenied />
  ) : <UpgradePrompt />
) : <LoginPrompt />}

// ✅ GOOD: Early returns
if (!user) return <LoginPrompt />;
if (!user.isPremium) return <UpgradePrompt />;
if (!user.hasAccess) return <AccessDenied />;
return <PremiumContent />;

// ✅ ALSO GOOD: Extract to helper function
function renderContent() {
  if (!user) return <LoginPrompt />;
  if (!user.isPremium) return <UpgradePrompt />;
  if (!user.hasAccess) return <AccessDenied />;
  return <PremiumContent />;
}

return <div>{renderContent()}</div>;
\`\`\`

### 2. Use Explicit Boolean Conversions

\`\`\`tsx
// ❌ BAD: Relies on truthiness (error-prone)
{items.length && <List items={items} />}
// Renders "0" when items.length === 0

// ✅ GOOD: Explicit comparison
{items.length > 0 && <List items={items} />}

// ✅ ALSO GOOD: Boolean conversion
{Boolean(items.length) && <List items={items} />}

// ✅ ALSO GOOD: Ternary with null
{items.length ? <List items={items} /> : null}
\`\`\`

### 3. Avoid Inline Arrow Functions for Complex Logic

\`\`\`tsx
// ❌ BAD: Complex logic inline
{user && user.subscription && user.subscription.isPremium && 
 user.subscription.expiresAt > Date.now() && (
  <PremiumBadge />
)}

// ✅ GOOD: Extract to variable
const isPremiumActive = 
  user?.subscription?.isPremium && 
  user.subscription.expiresAt > Date.now();

{isPremiumActive && <PremiumBadge />}
\`\`\`

### 4. Handle All States

\`\`\`tsx
// ❌ BAD: Missing error and loading states
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  
  return user ? <div>{user.name}</div> : null;
}

// ✅ GOOD: Handle all states
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!user) return <p>User not found</p>;
  return <div>{user.name}</div>;
}
\`\`\`

## Real-World Example: Complete Data Fetching Component

\`\`\`tsx
import { useState, useEffect } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

interface UserListProps {
  endpoint: string;
}

function UserList({ endpoint }: UserListProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'admin' | 'user'>('all');
  
  useEffect(() => {
    const fetchUsers = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch(endpoint);
        
        if (!response.ok) {
          throw new Error(\`Failed to fetch: \${response.statusText}\`);
        }
        
        const data = await response.json();
        setUsers(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchUsers();
  }, [endpoint]);
  
  // Filter users
  const filteredUsers = users.filter(user => 
    filter === 'all' || user.role === filter
  );
  
  // Loading state
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <p>Loading users...</p>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <div className="error-container">
        <h3>Error loading users</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }
  
  // Empty state (no users fetched)
  if (users.length === 0) {
    return (
      <div className="empty-state">
        <p>No users found</p>
      </div>
    );
  }
  
  // Success state with filter
  return (
    <div>
      {/* Filter controls */}
      <div className="filters">
        <button 
          onClick={() => setFilter('all')}
          className={filter === 'all' ? 'active' : ''}
        >
          All ({users.length})
        </button>
        <button 
          onClick={() => setFilter('admin')}
          className={filter === 'admin' ? 'active' : ''}
        >
          Admins ({users.filter(u => u.role === 'admin').length})
        </button>
        <button 
          onClick={() => setFilter('user')}
          className={filter === 'user' ? 'active' : ''}
        >
          Users ({users.filter(u => u.role === 'user').length})
        </button>
      </div>
      
      {/* Empty filter state */}
      {filteredUsers.length === 0 ? (
        <p>No users match the selected filter</p>
      ) : (
        <ul className="user-list">
          {filteredUsers.map(user => (
            <li key={user.id} className="user-item">
              <div>
                <h4>{user.name}</h4>
                <p>{user.email}</p>
              </div>
              {user.role === 'admin' && (
                <span className="badge badge-admin">Admin</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
\`\`\`

## What's Next?

Now that you understand conditional rendering, you're ready to learn **Lists & Keys**—how to render arrays of data efficiently and correctly. You'll learn:
- Rendering lists with map()
- Why keys are critical
- Key selection strategies
- List performance optimization
- Common list patterns

Conditional rendering + Lists = Dynamic, data-driven UIs! 🚀
`,
};
