export const conditionalRenderingDiscussion = {
  title: 'Conditional Rendering Discussion Questions',
  id: 'conditional-rendering-discussion',
  sectionId: 'conditional-rendering',
  questions: [
    {
      id: 'q1',
      question:
        'When should you use the && operator versus a ternary operator for conditional rendering? Discuss the tradeoffs and provide examples of when each is most appropriate. What common pitfall should you avoid with the && operator when dealing with numeric values?',
      answer: `The choice between && and ternary operators depends on whether you need an "else" branch and the nature of your condition:

**Use && operator when:**
1. You want to render something OR nothing (no alternative)
2. The condition is a simple boolean
3. You want concise, readable code

Example:
\`\`\`tsx
function Notifications({ count }) {
  return (
    <div>
      {count > 0 && <NotificationBadge count={count} />}
    </div>
  );
}
\`\`\`

**Use ternary operator when:**
1. You have two distinct alternatives (if/else scenario)
2. You need both branches to render something
3. You're assigning values conditionally

Example:
\`\`\`tsx
function Status({ isOnline }) {
  return (
    <span className={isOnline ? 'online' : 'offline'}>
      {isOnline ? '● Online' : '○ Offline'}
    </span>
  );
}
\`\`\`

**Critical Pitfall with &&: The Zero Problem**

The && operator can render unexpected values when dealing with numbers:

\`\`\`tsx
// ❌ WRONG: Renders "0" when count is 0
function ItemCount({ count }) {
  return (
    <div>
      {count && <p>You have {count} items</p>}
    </div>
  );
}

// When count = 0:
// 0 && <p>... evaluates to 0
// React renders 0 as text!
\`\`\`

**Why this happens:**
- JavaScript's && operator returns the first falsy value OR the last value
- When count is 0, the expression evaluates to 0
- React renders numbers (including 0) as text content
- Other falsy values (false, null, undefined) don't render

**Solutions:**

1. **Explicit boolean comparison (recommended):**
\`\`\`tsx
{count > 0 && <p>You have {count} items</p>}
\`\`\`

2. **Boolean conversion:**
\`\`\`tsx
{Boolean(count) && <p>You have {count} items</p>}
{!!count && <p>You have {count} items</p>}
\`\`\`

3. **Ternary with null:**
\`\`\`tsx
{count ? <p>You have {count} items</p> : null}
\`\`\`

**Tradeoffs Summary:**

**&& Operator:**
- **Pros:** Concise, reads naturally, perfect for single condition
- **Cons:** Can render 0, no else branch, potential confusion with falsy values

**Ternary Operator:**
- **Pros:** Explicit alternatives, works inline, familiar syntax, no zero problem
- **Cons:** More verbose, harder to read when nested, requires both branches

**Decision Matrix:**
- Need else branch? → Ternary
- Just showing/hiding? → &&
- Working with numbers? → Explicit comparison (count > 0 &&)
- Complex logic? → if/else or early return

**Interview Insight:**
Demonstrating knowledge of the "zero problem" shows deep understanding of both JavaScript and React rendering behavior. Many developers encounter this bug and don't understand why—explaining it clearly demonstrates expertise.`,
    },
    {
      id: 'q2',
      question:
        'Explain the different approaches to handling loading, error, and empty states in React components. How would you design a reusable component or hook to standardize this pattern across your application? What are the user experience considerations for each state?',
      answer: `Handling loading, error, and empty states is critical for good UX. Here's a comprehensive approach:

**The Three-State Pattern**

Every async operation has at least three states:
1. **Loading:** Data is being fetched
2. **Error:** Fetch failed
3. **Success:** Data loaded (which might be empty)

**Basic Pattern:**

\`\`\`tsx
function DataComponent() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetchData()
      .then(result => {
        setData(result);
        setIsLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsLoading(false);
      });
  }, []);
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!data || data.length === 0) return <EmptyState />;
  
  return <DataDisplay data={data} />;
}
\`\`\`

**Reusable Hook Pattern:**

\`\`\`tsx
interface UseAsyncStateResult<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

function useAsyncState<T>(
  fetchFn: () => Promise<T>,
  deps: any[] = []
): UseAsyncStateResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, deps);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  return { data, isLoading, error, refetch: fetchData };
}

// Usage:
function UserList() {
  const { data: users, isLoading, error, refetch } = useAsyncState(
    () => fetch('/api/users').then(r => r.json()),
    []
  );
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} onRetry={refetch} />;
  if (!users?.length) return <EmptyState />;
  
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}
\`\`\`

**Reusable State Component:**

\`\`\`tsx
interface AsyncStateProps<T> {
  isLoading: boolean;
  error: string | null;
  data: T | null;
  isEmpty?: (data: T) => boolean;
  onRetry?: () => void;
  children: (data: T) => React.ReactNode;
  loadingComponent?: React.ReactNode;
  errorComponent?: (error: string, retry?: () => void) => React.ReactNode;
  emptyComponent?: React.ReactNode;
}

function AsyncState<T>({
  isLoading,
  error,
  data,
  isEmpty = (d) => !d,
  onRetry,
  children,
  loadingComponent,
  errorComponent,
  emptyComponent
}: AsyncStateProps<T>) {
  if (isLoading) {
    return <>{loadingComponent || <DefaultLoading />}</>;
  }
  
  if (error) {
    return <>{errorComponent 
      ? errorComponent(error, onRetry)
      : <DefaultError error={error} onRetry={onRetry} />
    }</>;
  }
  
  if (isEmpty(data)) {
    return <>{emptyComponent || <DefaultEmpty />}</>;
  }
  
  return <>{children(data!)}</>;
}

// Usage:
<AsyncState
  isLoading={isLoading}
  error={error}
  data={users}
  isEmpty={(users) => users.length === 0}
  onRetry={refetch}
  emptyComponent={<EmptyUserList />}
>
  {(users) => (
    <ul>
      {users.map(u => <li key={u.id}>{u.name}</li>)}
    </ul>
  )}
</AsyncState>
\`\`\`

**UX Considerations for Each State:**

**1. Loading State:**
- **Show immediately** (don't wait more than 100-200ms)
- **Skeleton screens** > spinners for perceived performance
- **Progressive loading:** Show what you can while loading rest
- **Avoid layout shift:** Reserve space for content
- **Timeout handling:** Show message if loading takes too long

\`\`\`tsx
function LoadingState() {
  const [showTimeout, setShowTimeout] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => setShowTimeout(true), 5000);
    return () => clearTimeout(timer);
  }, []);
  
  return (
    <div>
      <Skeleton />
      {showTimeout && (
        <p>This is taking longer than expected...</p>
      )}
    </div>
  );
}
\`\`\`

**2. Error State:**
- **Clear error message:** What went wrong?
- **Actionable:** Provide retry button
- **Contextual:** Suggest what user can do
- **Don't blame user:** Avoid "You did X wrong"
- **Log details:** Send to error tracking (Sentry, etc.)

\`\`\`tsx
function ErrorState({ error, onRetry }) {
  return (
    <div className="error-state">
      <Icon name="alert-circle" />
      <h3>Unable to load data</h3>
      <p>{getHumanReadableError(error)}</p>
      <button onClick={onRetry}>Try Again</button>
      <a href="/support">Contact Support</a>
    </div>
  );
}

function getHumanReadableError(error: string): string {
  if (error.includes('Network')) {
    return 'Check your internet connection and try again.';
  }
  if (error.includes('404')) {
    return 'The requested data was not found.';
  }
  if (error.includes('403')) {
    return 'You don't have permission to view this data.';
  }
  return 'Something went wrong. Please try again.';
}
\`\`\`

**3. Empty State:**
- **Explain why it's empty:** "No items yet" vs "No results found"
- **Call to action:** Guide user on next steps
- **Visual interest:** Illustration or icon
- **Avoid negative language:** "Start creating" vs "You have nothing"

\`\`\`tsx
function EmptyTodoList({ onAddTodo }) {
  return (
    <div className="empty-state">
      <img src="/empty-checklist.svg" alt="" />
      <h3>No todos yet</h3>
      <p>Add your first todo to get started!</p>
      <button onClick={onAddTodo}>Add Todo</button>
    </div>
  );
}
\`\`\`

**Advanced: Optimistic UI**

For better UX, show success state immediately and rollback on error:

\`\`\`tsx
function TodoList() {
  const [todos, setTodos] = useState([]);
  const [error, setError] = useState(null);
  
  async function addTodo(text: string) {
    const tempId = Date.now();
    const optimisticTodo = { id: tempId, text, isTemp: true };
    
    // Immediately show in UI
    setTodos(prev => [...prev, optimisticTodo]);
    
    try {
      const newTodo = await fetch('/api/todos', {
        method: 'POST',
        body: JSON.stringify({ text })
      }).then(r => r.json());
      
      // Replace temp with real
      setTodos(prev => prev.map(t => 
        t.id === tempId ? newTodo : t
      ));
    } catch (err) {
      // Rollback on error
      setTodos(prev => prev.filter(t => t.id !== tempId));
      setError('Failed to add todo');
    }
  }
  
  return (
    <div>
      {error && <ErrorBanner error={error} />}
      <ul>
        {todos.map(todo => (
          <li key={todo.id} className={todo.isTemp ? 'saving' : ''}>
            {todo.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
\`\`\`

**Interview Insight:**
Discussing loading/error/empty states comprehensively shows you think about the full user journey, not just the happy path. Mentioning optimistic UI, skeleton screens, and progressive loading demonstrates senior-level UX awareness.`,
    },
    {
      id: 'q3',
      question:
        'Compare and contrast different conditional rendering patterns (if/else, ternary, &&, element variables, switch statements) in terms of readability, maintainability, and performance. In a large React application, how would you establish conventions for when to use each pattern?',
      answer: `Different conditional rendering patterns serve different purposes. Here's a comprehensive comparison:

**1. if/else Statements**

\`\`\`tsx
function Component({ user }) {
  if (!user) {
    return <LoginPrompt />;
  }
  
  if (user.isBlocked) {
    return <BlockedMessage />;
  }
  
  return <Dashboard />;
}
\`\`\`

**Pros:**
- Most readable for complex logic
- Early returns keep nesting minimal
- Easy to add logging/debugging
- Familiar to all developers

**Cons:**
- Can't use inline in JSX
- More verbose than alternatives
- Requires extra function/variable scope

**When to use:**
- Complex conditional logic
- Multiple related checks
- Completely different component trees
- When you need to execute code between checks

**2. Ternary Operator**

\`\`\`tsx
function Component({ isOnline }) {
  return (
    <div>
      {isOnline ? <OnlineIndicator /> : <OfflineIndicator />}
    </div>
  );
}
\`\`\`

**Pros:**
- Works inline in JSX
- Concise for simple conditions
- Both branches explicit
- No "zero problem"

**Cons:**
- Hard to read when nested
- Must provide both branches
- Can become very messy

**When to use:**
- Simple binary choice
- Inline rendering in JSX
- Short, readable alternatives
- Conditional props/classNames

**3. Logical AND (&&)**

\`\`\`tsx
function Component({ notifications }) {
  return (
    <div>
      {notifications.length > 0 && <NotificationBadge />}
    </div>
  );
}
\`\`\`

**Pros:**
- Very concise
- Reads naturally ("if this, show that")
- Perfect for optional elements

**Cons:**
- Renders 0 with numbers (common bug!)
- No else branch
- Can be confusing with complex expressions

**When to use:**
- Showing something OR nothing
- Simple boolean conditions
- Optional UI elements
- With explicit boolean conversion

**4. Element Variables**

\`\`\`tsx
function Component({ status }) {
  let content;
  
  if (status === 'loading') {
    content = <Spinner />;
  } else if (status === 'error') {
    content = <ErrorMessage />;
  } else {
    content = <DataDisplay />;
  }
  
  return <div>{content}</div>;
}
\`\`\`

**Pros:**
- Separates logic from JSX
- Easy to test each branch
- Reuse variables multiple times
- Great for complex conditions

**Cons:**
- More verbose
- Extra variables in scope
- Requires more lines

**When to use:**
- Complex branching logic
- Reusing JSX in multiple places
- When you need to manipulate JSX before returning
- Building component dynamically

**5. Switch Statements**

\`\`\`tsx
function StatusBadge({ status }) {
  switch (status) {
    case 'pending':
      return <PendingBadge />;
    case 'approved':
      return <ApprovedBadge />;
    case 'rejected':
      return <RejectedBadge />;
    default:
      return <UnknownBadge />;
  }
}
\`\`\`

**Pros:**
- Clean for many discrete states
- TypeScript can enforce exhaustiveness
- Easy to add new cases

**Cons:**
- Verbose compared to object mapping
- Easy to forget break (with blocks)
- Not as flexible as if/else

**When to use:**
- Many discrete states (3+)
- Enum-like values
- When you want exhaustiveness checking

**Performance Comparison:**

**Reality: Performance is essentially identical.**

All these patterns compile to similar JavaScript. React's reconciliation is the bottleneck, not conditional logic. However:

- **Early returns** can skip rendering entire subtrees → best for performance
- **Expensive inline functions** in conditions can impact performance
- **Memoization** (React.memo, useMemo) matters more than pattern choice

\`\`\`tsx
// ❌ BAD: Creates new object every render
<Component style={isActive ? { color: 'red' } : { color: 'blue' }} />

// ✅ GOOD: Reference stable objects
const ACTIVE_STYLE = { color: 'red' };
const INACTIVE_STYLE = { color: 'blue' };
<Component style={isActive ? ACTIVE_STYLE : INACTIVE_STYLE} />
\`\`\`

**Establishing Conventions for Large Applications:**

**Recommended Style Guide:**

\`\`\`tsx
// 1. Use early returns for guard conditions
function UserProfile({ user }) {
  if (!user) return <LoginPrompt />;
  if (user.isBlocked) return <BlockedMessage />;
  
  // Main component logic
  return <Profile user={user} />;
}

// 2. Use ternary for inline binary choices
<button className={isActive ? 'active' : 'inactive'}>
  {isActive ? 'Disable' : 'Enable'}
</button>

// 3. Use && with explicit boolean for optional elements
{count > 0 && <Badge count={count} />}
{!!message && <Alert message={message} />}

// 4. Use element variables for complex multi-branch logic
function OrderStatus({ order }) {
  let statusElement;
  
  if (order.isShipped) {
    statusElement = <ShippedStatus tracking={order.tracking} />;
  } else if (order.isPending && order.requiresAction) {
    statusElement = <ActionRequired actions={order.actions} />;
  } else {
    statusElement = <DefaultStatus status={order.status} />;
  }
  
  return <div>{statusElement}</div>;
}

// 5. Use switch/object mapping for discrete states
const STATUS_COMPONENTS = {
  pending: PendingIcon,
  approved: ApprovedIcon,
  rejected: RejectedIcon
};

function StatusIcon({ status }) {
  const Icon = STATUS_COMPONENTS[status] || UnknownIcon;
  return <Icon />;
}

// 6. Extract complex conditions to helper functions
function canEditPost(post, user) {
  return post.authorId === user.id || user.role === 'admin';
}

function Post({ post, user }) {
  return (
    <div>
      {canEditPost(post, user) && <EditButton />}
    </div>
  );
}
\`\`\`

**Team Convention Document Example:**

\`\`\`markdown
# Conditional Rendering Conventions

## Rules:
1. **Early returns** for guard conditions at component start
2. **Ternary** for simple binary choices, especially inline
3. **&&** ONLY with explicit boolean conversion (avoid number traps)
4. **Element variables** when logic is complex or JSX is reused
5. **Object mapping** preferred over switch for status/enum rendering
6. **Extract functions** when conditions exceed 2-3 operations

## Forbidden:
- ❌ Nested ternaries (>1 level)
- ❌ {count && <Component />} without comparison
- ❌ Complex inline conditions (>2 operators)

## Examples: [link to examples]
\`\`\`

**Enforcing with ESLint:**

\`\`\`javascript
// .eslintrc.js
rules: {
  // Warn on nested ternaries
  'no-nested-ternary': 'warn',
  
  // Enforce explicit boolean in conditions
  '@typescript-eslint/strict-boolean-expressions': 'error'
}
\`\`\`

**Maintainability Best Practices:**

1. **Consistency > Personal Preference:** Pick one pattern per scenario type
2. **Document in Style Guide:** Include examples and rationale
3. **Code Review Focus:** Enforce during review
4. **Linting:** Automate what you can
5. **Refactor:** Improve as patterns become complex

**Interview Insight:**
Discussing conventions and team standards shows you think beyond just writing code—you understand software engineering at scale. Mentioning ESLint enforcement and documentation demonstrates leadership and architectural thinking.`,
    },
  ],
};
