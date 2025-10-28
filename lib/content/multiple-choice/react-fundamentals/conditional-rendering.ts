export const conditionalRenderingQuiz = {
  title: 'Conditional Rendering Quiz',
  id: 'conditional-rendering-quiz',
  sectionId: 'conditional-rendering',
  questions: [
    {
      id: 'q1',
      question:
        'What will render when count is 0 in this code: {count && <Badge count={count} />}?',
      options: [
        'Nothing will render',
        'The number 0 will render as text',
        'A Badge component with count={0}',
        'An error will be thrown',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"The number 0 will render as text"**.

This is one of the most common bugs in React conditional rendering:

\`\`\`tsx
{count && <Badge count={count} />}

// When count = 0:
// 0 && <Badge...> evaluates to 0
// React renders 0 as text!
\`\`\`

**Why this happens:**
1. JavaScript's && operator returns the first falsy value OR the last value
2. When count is 0, the expression short-circuits and returns 0
3. React renders numbers (including 0) as text content
4. Other falsy values (false, null, undefined) don't render

**The problem:**
\`\`\`tsx
// User sees "0" on screen when no items exist
{items.length && <p>You have {items.length} items</p>}
\`\`\`

**Solutions:**

1. **Explicit boolean comparison (recommended):**
\`\`\`tsx
{count > 0 && <Badge count={count} />}
{items.length > 0 && <p>You have {items.length} items</p>}
\`\`\`

2. **Boolean conversion:**
\`\`\`tsx
{Boolean(count) && <Badge count={count} />}
{!!count && <Badge count={count} />}
\`\`\`

3. **Ternary with null:**
\`\`\`tsx
{count ? <Badge count={count} /> : null}
\`\`\`

**Why other answers are wrong:**
- **"Nothing will render"**: This would be true if React didn't render 0, but it does
- **"A Badge component"**: The && short-circuits at 0, never evaluating the right side
- **"An error"**: This is valid JavaScript and React code, no error occurs

**Interview tip:** This is a classic React gotcha that interviewers love to test. Explaining the JavaScript evaluation order and React's rendering behavior demonstrates deep understanding.`,
    },
    {
      id: 'q2',
      question:
        'Which pattern is most appropriate for rendering one of four different components based on a status enum ("pending" | "approved" | "rejected" | "draft")?',
      options: [
        'Nested ternary operators',
        'Multiple && operators',
        'Switch statement or object mapping',
        'Four separate if statements',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"Switch statement or object mapping"**.

For discrete, mutually exclusive states (especially 3+), switch statements or object mapping are the cleanest patterns.

**❌ Bad: Nested Ternary (Hard to Read)**
\`\`\`tsx
function StatusBadge({ status }) {
  return (
    <div>
      {status === 'pending' ? <PendingBadge /> :
       status === 'approved' ? <ApprovedBadge /> :
       status === 'rejected' ? <RejectedBadge /> :
       <DraftBadge />}
    </div>
  );
}
\`\`\`
Problems: Hard to read, difficult to maintain, no exhaustiveness checking

**❌ Bad: Multiple && (Verbose and Bug-Prone)**
\`\`\`tsx
function StatusBadge({ status }) {
  return (
    <div>
      {status === 'pending' && <PendingBadge />}
      {status === 'approved' && <ApprovedBadge />}
      {status === 'rejected' && <RejectedBadge />}
      {status === 'draft' && <DraftBadge />}
    </div>
  );
}
\`\`\`
Problems: Could render multiple (if bug), no default case, wasteful evaluation

**✅ Good: Switch Statement**
\`\`\`tsx
function StatusBadge({ status }: { status: Status }) {
  switch (status) {
    case 'pending':
      return <PendingBadge />;
    case 'approved':
      return <ApprovedBadge />;
    case 'rejected':
      return <RejectedBadge />;
    case 'draft':
      return <DraftBadge />;
    default:
      return <UnknownBadge />;
  }
}
\`\`\`
Benefits:
- Very readable
- Clear intent
- TypeScript can enforce exhaustiveness
- Easy to add new cases

**✅ Better: Object Mapping (Most Scalable)**
\`\`\`tsx
type Status = 'pending' | 'approved' | 'rejected' | 'draft';

const STATUS_COMPONENTS: Record<Status, React.ComponentType> = {
  pending: PendingBadge,
  approved: ApprovedBadge,
  rejected: RejectedBadge,
  draft: DraftBadge
};

function StatusBadge({ status }: { status: Status }) {
  const BadgeComponent = STATUS_COMPONENTS[status];
  
  if (!BadgeComponent) {
    return <UnknownBadge />;
  }
  
  return <BadgeComponent />;
}
\`\`\`
Benefits:
- Most maintainable
- TypeScript enforces completeness via Record<Status, ...>
- Can be defined outside component
- Easy to test each mapping
- Scales to dozens of states

**With Props:**
\`\`\`tsx
const STATUS_CONFIG = {
  pending: { Component: Badge, icon: '⏳', color: 'yellow', label: 'Pending' },
  approved: { Component: Badge, icon: '✓', color: 'green', label: 'Approved' },
  rejected: { Component: Badge, icon: '✗', color: 'red', label: 'Rejected' },
  draft: { Component: Badge, icon: '📝', color: 'gray', label: 'Draft' }
} as const;

function StatusBadge({ status }: { status: Status }) {
  const config = STATUS_CONFIG[status];
  if (!config) return <UnknownBadge />;
  
  const { Component, ...props } = config;
  return <Component {...props} />;
}
\`\`\`

**When to use each:**
- **Switch:** 3-5 states, simple logic, want exhaustiveness checking
- **Object mapping:** 5+ states, complex per-state config, need testability
- **if/else:** Different logic per branch (not just component selection)
- **Ternary:** Binary choice only (2 options)

**Interview tip:** Discussing object mapping over switch shows advanced React patterns and scalability thinking. Mentioning TypeScript exhaustiveness checking demonstrates type-safety awareness.`,
    },
    {
      id: 'q3',
      question: 'What happens when a component returns null?',
      options: [
        'React throws an error because components must return JSX',
        'The component unmounts and all hooks stop running',
        'The component renders nothing but lifecycle methods and hooks still run',
        'React skips the component entirely and never calls it again',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"The component renders nothing but lifecycle methods and hooks still run"**.

Returning \`null\` is a valid way to conditionally prevent rendering, but the component itself still exists in React's tree.

**Example:**
\`\`\`tsx
function ConditionalWarning({ show, message }: { show: boolean; message: string }) {
  useEffect(() => {
    console.log('Effect runs!');
    return () => console.log('Cleanup runs!');
  }, []);
  
  if (!show) {
    return null;  // Renders nothing
  }
  
  return (
    <div className="warning">
      <p>{message}</p>
    </div>
  );
}

// Usage
<ConditionalWarning show={false} message="Warning!" />
// Console: "Effect runs!"
// Screen: (nothing)
\`\`\`

**What actually happens:**
1. Component function executes fully
2. All hooks run (useState, useEffect, etc.)
3. Effects are called
4. Cleanup functions run on unmount
5. React just doesn't render any DOM nodes

**This is different from conditional mounting:**
\`\`\`tsx
// Component returns null (still mounted)
<ConditionalWarning show={false} message="Warning!" />
// Component exists, effect runs, returns null

// vs

// Component not mounted at all
{showWarning && <Warning message="Warning!" />}
// When showWarning=false, component never executes
\`\`\`

**Practical implications:**

**1. Effects still run:**
\`\`\`tsx
function DataFetcher({ shouldFetch }) {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    if (shouldFetch) {
      fetch('/api/data')
        .then(r => r.json())
        .then(setData);
    }
  }, [shouldFetch]);
  
  if (!data) return null;
  
  return <DataDisplay data={data} />;
}

// Even when returning null, useEffect runs!
// This can be useful for prefetching
\`\`\`

**2. State persists:**
\`\`\`tsx
function TogglableContent({ show }) {
  const [count, setCount] = useState(0);
  
  if (!show) return null;
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>Increment</button>
    </div>
  );
}

// When show toggles false → true:
// count value persists! (component never unmounted)
\`\`\`

**3. Performance consideration:**
\`\`\`tsx
// ❌ Expensive: Component executes even when hidden
function ExpensiveComponent({ show }) {
  const expensiveValue = computeExpensiveValue();  // Always runs!
  
  if (!show) return null;
  
  return <div>{expensiveValue}</div>;
}

// ✅ Better: Don't mount when not needed
{show && <ExpensiveComponent />}
// Component never executes when show=false
\`\`\`

**When to use return null:**
- ✅ Want to preserve state while hiding UI
- ✅ Want effects to continue running
- ✅ Component is cheap to execute
- ✅ Simpler than conditional mounting logic

**When to use conditional mounting:**
- ✅ Want to reset state on hide/show
- ✅ Want to stop effects from running
- ✅ Component is expensive to execute
- ✅ Need to unmount/remount lifecycle

**Why other answers are wrong:**
- **"React throws an error"**: Returning null is explicitly allowed and documented
- **"Component unmounts"**: The component stays mounted, just renders nothing
- **"React skips the component"**: React still calls the component function every render

**Interview tip:** Understanding the difference between "returns null but mounted" vs "not mounted at all" shows deep React knowledge. Discussing when each is appropriate demonstrates performance and architecture awareness.`,
    },
    {
      id: 'q4',
      question:
        'In a component that fetches data, what is the correct order to check and render different states?',
      options: [
        'Check data → Check loading → Check error',
        'Check loading → Check error → Check data/empty',
        'Check error → Check loading → Check data',
        "The order doesn't matter",
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"Check loading → Check error → Check data/empty"**.

The order matters because states can overlap, and you want to show the most relevant state to the user.

**✅ Correct Order:**
\`\`\`tsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(data => {
        setUser(data);
        setIsLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsLoading(false);
      });
  }, [userId]);
  
  // 1. Check loading first
  if (isLoading) {
    return <LoadingSpinner />;
  }
  
  // 2. Then check error
  if (error) {
    return <ErrorMessage error={error} />;
  }
  
  // 3. Then check empty/no data
  if (!user) {
    return <p>User not found</p>;
  }
  
  // 4. Finally render success state
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
\`\`\`

**Why this order?**

**1. Loading First:**
- When loading, data might be stale from previous fetch
- Error might exist from previous attempt
- Loading takes precedence over everything
- User needs to know fetch is in progress

\`\`\`tsx
// ❌ WRONG: Checking data first
if (!user) return <EmptyState />;  // Shows during loading!
if (isLoading) return <LoadingSpinner />;
// User sees EmptyState flash before LoadingSpinner
\`\`\`

**2. Error Second:**
- After loading completes, check if it failed
- Error state should override stale data
- User needs to know why data isn't loading

\`\`\`tsx
// ❌ WRONG: Checking data before error
if (user) return <UserDisplay user={user} />;  // Shows stale data!
if (error) return <ErrorMessage error={error} />;
// User sees old data even though new fetch failed
\`\`\`

**3. Empty/No Data Third:**
- After loading completes successfully
- But no data was returned (or empty array)
- Different UX than error state

**4. Success State Last:**
- All checks passed
- Data exists and is fresh
- Safe to render

**Real-world state machine:**
\`\`\`tsx
type FetchState<T> = 
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: string };

function DataComponent() {
  const [state, setState] = useState<FetchState<User>>({ status: 'idle' });
  
  useEffect(() => {
    setState({ status: 'loading' });
    
    fetch('/api/user')
      .then(r => r.json())
      .then(data => setState({ status: 'success', data }))
      .catch(err => setState({ status: 'error', error: err.message }));
  }, []);
  
  switch (state.status) {
    case 'idle':
    case 'loading':
      return <LoadingSpinner />;
    
    case 'error':
      return <ErrorMessage error={state.error} />;
    
    case 'success':
      if (!state.data) return <EmptyState />;
      return <UserDisplay user={state.data} />;
  }
}
\`\`\`

**Why other orders fail:**

**"Check data → Check loading → Check error"**
\`\`\`tsx
if (!user) return <EmptyState />;     // Shows during loading! ❌
if (isLoading) return <LoadingSpinner />;  // Never reached when no user
if (error) return <ErrorMessage />;        // Never reached
\`\`\`
Problem: EmptyState appears immediately, then flashes to loading

**"Check error → Check loading → Check data"**
\`\`\`tsx
if (error) return <ErrorMessage />;    // Shows old error during new load! ❌
if (isLoading) return <LoadingSpinner />;
if (!user) return <EmptyState />;
\`\`\`
Problem: Old error persists when new fetch starts unless explicitly cleared

**"Order doesn't matter"**
Order absolutely matters for UX and preventing flashing states!

**Advanced: Handling concurrent state changes**
\`\`\`tsx
function UserProfile({ userId }) {
  const [state, setState] = useState({ status: 'loading' });
  
  useEffect(() => {
    setState({ status: 'loading' });
    
    let cancelled = false;  // Handle concurrent requests
    
    fetch(\`/api/users/\${userId}\`)
      .then(r => r.json())
      .then(data => {
        if (!cancelled) {  // Only update if still relevant
          setState({ status: 'success', data });
        }
      })
      .catch(err => {
        if (!cancelled) {
          setState({ status: 'error', error: err.message });
        }
      });
    
    return () => {
      cancelled = true;  // Cleanup: ignore stale requests
    };
  }, [userId]);
  
  // Now state is guaranteed to be consistent
  if (state.status === 'loading') return <Loading />;
  if (state.status === 'error') return <Error error={state.error} />;
  if (state.status === 'success') return <Display data={state.data} />;
}
\`\`\`

**Interview tip:** Explaining the correct state checking order and why it matters shows strong UX awareness and understanding of async data flows. Mentioning race conditions and cleanup demonstrates senior-level React knowledge.`,
    },
    {
      id: 'q5',
      question:
        'Which approach is best for preventing a component from rendering based on user permissions?',
      options: [
        'Check permissions inside the component and return null',
        'Use conditional rendering (&&) where the component is used',
        'Create a higher-order component or wrapper that handles permission checking',
        'All approaches are equally good',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"Create a higher-order component or wrapper that handles permission checking"**.

While all approaches technically work, creating a reusable abstraction is the most maintainable and scalable solution for permission-based rendering.

**❌ Approach 1: Check inside component and return null**
\`\`\`tsx
function AdminButton({ user }) {
  if (user.role !== 'admin') {
    return null;
  }
  
  return <button>Admin Action</button>;
}

// Usage
<AdminButton user={user} />
\`\`\`

**Problems:**
- Component executes even when user can't see it
- Permission logic couples to component
- Can't reuse permission logic
- Hard to test permissions separately
- Component must know about permissions

**❌ Approach 2: Conditional rendering at usage site**
\`\`\`tsx
// Usage
{user.role === 'admin' && <AdminButton />}
{user.permissions.includes('edit') && <EditButton />}
{user.role === 'admin' && <DeleteButton />}
\`\`\`

**Problems:**
- Permission checks scattered everywhere
- Easy to forget or implement incorrectly
- Hard to update permission logic globally
- Verbose and repetitive
- No central control

**✅ Approach 3: Wrapper component (Best)**

**Option A: Permission component**
\`\`\`tsx
interface PermissionProps {
  user: User;
  requires: string | string[];
  fallback?: React.ReactNode;
  children: React.ReactNode;
}

function Permission({ user, requires, fallback = null, children }: PermissionProps) {
  const hasPermission = checkPermission(user, requires);
  
  if (!hasPermission) {
    return <>{fallback}</>;
  }
  
  return <>{children}</>;
}

// Usage
<Permission user={user} requires="admin">
  <AdminButton />
</Permission>

<Permission 
  user={user} 
  requires={['edit', 'delete']}
  fallback={<LockedMessage />}
>
  <EditButton />
  <DeleteButton />
</Permission>
\`\`\`

**Benefits:**
- Centralized permission logic
- Reusable across app
- Easy to test
- Optional fallback UI
- Component doesn't execute if no permission
- Can be nested and combined

**Option B: Higher-Order Component**
\`\`\`tsx
function withPermission<P extends object>(
  Component: React.ComponentType<P>,
  permission: string | string[]
) {
  return function PermissionWrapped(props: P & { user: User }) {
    const { user, ...restProps } = props;
    
    if (!checkPermission(user, permission)) {
      return null;
    }
    
    return <Component {...(restProps as P)} />;
  };
}

// Usage
const AdminButton = withPermission(Button, 'admin');
const EditButton = withPermission(Button, ['read', 'edit']);

<AdminButton user={user} label="Admin Action" />
\`\`\`

**Option C: Custom Hook**
\`\`\`tsx
function usePermission(permission: string | string[]): boolean {
  const { user } = useAuth();
  return useMemo(() => checkPermission(user, permission), [user, permission]);
}

// Usage in component
function EditButton() {
  const canEdit = usePermission('edit');
  
  if (!canEdit) return null;
  
  return <button>Edit</button>;
}

// Or at usage site
function Dashboard() {
  const canEdit = usePermission('edit');
  const isAdmin = usePermission('admin');
  
  return (
    <div>
      {canEdit && <EditButton />}
      {isAdmin && <AdminPanel />}
    </div>
  );
}
\`\`\`

**Option D: Render props pattern**
\`\`\`tsx
function PermissionGate({ 
  permission, 
  children 
}: { 
  permission: string; 
  children: (hasPermission: boolean) => React.ReactNode;
}) {
  const { user } = useAuth();
  const hasPermission = checkPermission(user, permission);
  
  return <>{children(hasPermission)}</>;
}

// Usage - maximum flexibility
<PermissionGate permission="edit">
  {(canEdit) => (
    canEdit ? (
      <EditButton />
    ) : (
      <LockedMessage />
    )
  )}
</PermissionGate>
\`\`\`

**Complete permission system:**
\`\`\`tsx
// Central permission logic
type Permission = 'read' | 'edit' | 'delete' | 'admin';

interface User {
  id: string;
  role: 'user' | 'admin' | 'guest';
  permissions: Permission[];
}

function checkPermission(
  user: User | null,
  required: Permission | Permission[]
): boolean {
  if (!user) return false;
  
  // Admin has all permissions
  if (user.role === 'admin') return true;
  
  const permissions = Array.isArray(required) ? required : [required];
  
  // Check if user has ALL required permissions
  return permissions.every(p => user.permissions.includes(p));
}

// Wrapper component
function HasPermission({
  permission,
  fallback = null,
  children
}: {
  permission: Permission | Permission[];
  fallback?: React.ReactNode;
  children: React.ReactNode;
}) {
  const { user } = useAuth();
  
  if (!checkPermission(user, permission)) {
    return <>{fallback}</>;
  }
  
  return <>{children}</>;
}

// Hook
function useHasPermission(permission: Permission | Permission[]): boolean {
  const { user } = useAuth();
  return checkPermission(user, permission);
}

// Usage throughout app
function Document() {
  const canEdit = useHasPermission('edit');
  const canDelete = useHasPermission('delete');
  
  return (
    <div>
      <DocumentContent />
      
      {canEdit && <EditButton />}
      
      <HasPermission permission="delete" fallback={<LockedDeleteButton />}>
        <DeleteButton />
      </HasPermission>
      
      <HasPermission permission={['edit', 'delete']}>
        <AdminControls />
      </HasPermission>
    </div>
  );
}
\`\`\`

**Why this is better:**
1. **Single source of truth:** Permission logic in one place
2. **Maintainable:** Change logic once, affects entire app
3. **Testable:** Test permission logic independently
4. **Reusable:** Use same component/hook everywhere
5. **Type-safe:** TypeScript enforces valid permissions
6. **Flexible:** Multiple patterns for different needs
7. **Performance:** Can memoize permission checks

**When to use each pattern:**
- **\`<HasPermission>\` component:** Most common case, multiple children
- **\`useHasPermission\` hook:** Complex conditional logic in component
- **Higher-order component:** Want to wrap entire component permanently
- **Render props:** Need flexibility based on permission state

**Interview tip:** Discussing a complete permission system shows architectural thinking and understanding of cross-cutting concerns. Mentioning reusability, testability, and maintenance demonstrates senior-level engineering judgment.`,
    },
  ],
};
