export const formsIntroductionQuiz = {
  title: 'Forms Introduction Quiz',
  id: 'forms-introduction-quiz',
  sectionId: 'forms-introduction',
  questions: [
    {
      id: 'q1',
      question:
        'What is the main difference between a controlled and an uncontrolled component in React?',
      options: [
        'Controlled components use class components, uncontrolled use function components',
        'Controlled components have their value managed by React state, uncontrolled components store their value in the DOM',
        'Controlled components validate on submit, uncontrolled components validate on change',
        'Controlled components are faster than uncontrolled components',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"Controlled components have their value managed by React state, uncontrolled components store their value in the DOM"**.

This is the fundamental difference between these two patterns:

**Controlled Component:**
\`\`\`tsx
function ControlledInput() {
  const [value, setValue] = useState('');  // State holds the value
  
  return (
    <input 
      value={value}                              // ← Value from React state
      onChange={(e) => setValue(e.target.value)} // ← Updates React state
    />
  );
}

// Value flow:
// State → Input displays value
// User types → onChange updates state
// State changes → React re-renders
// Input shows new value
\`\`\`

**Uncontrolled Component:**
\`\`\`tsx
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit() {
    const value = inputRef.current?.value;  // ← Access DOM value when needed
    console.log(value);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input ref={inputRef} defaultValue="" />  {/* ← DOM manages value */}
      <button type="submit">Submit</button>
    </form>
  );
}

// Value flow:
// User types → Value changes in DOM (not React)
// React doesn't know about changes
// Access value via ref when needed
\`\`\`

**Key Differences:**

| Aspect | Controlled | Uncontrolled |
|--------|-----------|--------------|
| **Value source** | React state | DOM |
| **Value prop** | \`value={state}\` | \`defaultValue="..."\` |
| **Change handler** | \`onChange\` required | Optional |
| **Access value** | Immediately from state | Via ref when needed |
| **React awareness** | React knows current value | React doesn't track value |

**Controlled Example:**
\`\`\`tsx
function ControlledForm() {
  const [email, setEmail] = useState('');
  
  // Can access value anytime
  console.log('Current email:', email);
  
  // Can validate in real-time
  const isValid = email.includes('@');
  
  // Can disable button based on input
  return (
    <form>
      <input 
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <button disabled={!isValid}>Submit</button>
    </form>
  );
}
\`\`\`

**Uncontrolled Example:**
\`\`\`tsx
function UncontrolledForm() {
  const emailRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    // Can only access value here (or via ref)
    const email = emailRef.current?.value;
    console.log('Submitted email:', email);
    
    // Validation only on submit
    if (!email?.includes('@')) {
      alert('Invalid email');
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input ref={emailRef} defaultValue="" />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**Why other answers are wrong:**

**"Controlled components use class components, uncontrolled use function components":**
- Both patterns work with either class or function components
- Modern React uses function components for both

**"Controlled components validate on submit, uncontrolled components validate on change":**
- This is backwards! Controlled components CAN validate on change (real-time)
- Uncontrolled typically validate on submit
- Validation timing is separate from controlled/uncontrolled

**"Controlled components are faster than uncontrolled components":**
- Actually, uncontrolled can be slightly faster (no React re-renders)
- But the difference is negligible for most forms
- Choose based on functionality needed, not performance

**Real-World Example:**

\`\`\`tsx
// Controlled: Can format input as user types
function PhoneInput() {
  const [phone, setPhone] = useState('');
  
  function formatPhone(value: string) {
    const digits = value.replace(/\\D/g, '').slice(0, 10);
    if (digits.length <= 3) return digits;
    if (digits.length <= 6) return \`(\${digits.slice(0,3)}) \${digits.slice(3)}\`;
    return \`(\${digits.slice(0,3)}) \${digits.slice(3,6)}-\${digits.slice(6)}\`;
  }
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setPhone(formatPhone(e.target.value));
  }
  
  return (
    <input 
      value={phone}
      onChange={handleChange}
      placeholder="(123) 456-7890"
    />
  );
}

// Uncontrolled: Can't format as user types
function PhoneInputUncontrolled() {
  const phoneRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const phone = phoneRef.current?.value;
    // Can only format on submit
    console.log(formatPhone(phone));
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input ref={phoneRef} />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**When to Use Each:**

**Use Controlled (Recommended Default):**
- Need real-time validation
- Need to format input
- Need conditional UI based on input
- Need to programmatically set/clear values
- Most forms

**Use Uncontrolled:**
- File inputs (required—always uncontrolled)
- Simple forms without validation
- Integrating with non-React libraries
- Very large forms (rare performance optimization)

**Interview tip:** Explaining that the difference is about "where the state lives" (React vs DOM) shows deep understanding. Mentioning that controlled gives you more control but requires more code demonstrates balanced thinking about tradeoffs.`,
    },
    {
      id: 'q2',
      question:
        'In a controlled form component, why is the onChange handler necessary?',
      options: [
        'To prevent the form from submitting automatically',
        'To validate the input before it appears on screen',
        'To update the state so the input can display the new value',
        'To improve performance by debouncing input',
      ],
      correctAnswer: 2,
      explanation: `The correct answer is **"To update the state so the input can display the new value"**.

In a controlled component, the input's value is tied to React state. Without onChange updating that state, the input would be frozen and unable to change.

**Why onChange is Required:**

\`\`\`tsx
// ❌ WITHOUT onChange: Input is frozen!
function BrokenControlled() {
  const [value, setValue] = useState('');
  
  return (
    <input value={value} />  {/* No onChange! */}
  );
}

// What happens:
// 1. Input displays value (empty string)
// 2. User types "h"
// 3. Input tries to change to "h"
// 4. React re-renders
// 5. Input value is STILL "" (state didn't change)
// 6. Input appears frozen!

// Console warning:
// "You provided a \`value\` prop to a form field without an \`onChange\` handler.
//  This will render a read-only field."
\`\`\`

**✅ WITH onChange: Input works correctly**
\`\`\`tsx
function WorkingControlled() {
  const [value, setValue] = useState('');
  
  return (
    <input 
      value={value}
      onChange={(e) => setValue(e.target.value)}
    />
  );
}

// What happens:
// 1. Input displays value (empty string)
// 2. User types "h"
// 3. onChange fires with event
// 4. setValue("h") called
// 5. State updates to "h"
// 6. React re-renders
// 7. Input displays "h" ✓
\`\`\`

**The Flow in Detail:**

\`\`\`tsx
function ControlledInput() {
  const [text, setText] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    console.log('1. User typed:', e.target.value);
    setText(e.target.value);
    console.log('2. State will update');
  }
  
  console.log('3. Rendering with text:', text);
  
  return (
    <input 
      value={text}
      onChange={handleChange}
    />
  );
}

// User types "a":
// Console logs:
// "3. Rendering with text: " (initial render)
// "1. User typed: a"
// "2. State will update"
// "3. Rendering with text: a" (re-render)
\`\`\`

**Without onChange, you get a read-only input:**

\`\`\`tsx
// If you WANT a read-only input, omit onChange:
<input value="Fixed Value" readOnly />

// This is intentional: Input shows value but can't change
\`\`\`

**Why other answers are wrong:**

**"To prevent the form from submitting automatically":**
- That's what \`e.preventDefault()\` does in onSubmit handler
- onChange has nothing to do with form submission
\`\`\`tsx
function handleSubmit(e: React.FormEvent) {
  e.preventDefault();  // ← This prevents submission
  // ...
}
\`\`\`

**"To validate the input before it appears on screen":**
- onChange doesn't validate by default
- You can add validation logic, but that's optional
- The primary purpose is updating state
\`\`\`tsx
// Validation is optional, updating state is required
function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
  setText(e.target.value);  // ← Required (update state)
  
  // Optional validation:
  if (e.target.value.length > 10) {
    setError('Too long');
  }
}
\`\`\`

**"To improve performance by debouncing input":**
- onChange doesn't debounce by default
- You can add debouncing, but that's optional
- Debouncing is a performance optimization, not the core purpose

\`\`\`tsx
// Debouncing is optional
import { debounce } from 'lodash';

const debouncedUpdate = debounce((value) => {
  setText(value);
}, 300);

function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
  debouncedUpdate(e.target.value);  // Optional optimization
}
\`\`\`

**Advanced: Controlling Without Directly Updating State**

You can transform input before updating state:

\`\`\`tsx
// Only allow numbers
function NumberInput() {
  const [value, setValue] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const newValue = e.target.value.replace(/\\D/g, '');  // Remove non-digits
    setValue(newValue);
  }
  
  return <input value={value} onChange={handleChange} />;
}

// User types "a1b2c3":
// Input displays: "123" (non-digits filtered out)
\`\`\`

\`\`\`tsx
// Uppercase input
function UppercaseInput() {
  const [value, setValue] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setValue(e.target.value.toUpperCase());
  }
  
  return <input value={value} onChange={handleChange} />;
}

// User types "hello":
// Input displays: "HELLO"
\`\`\`

\`\`\`tsx
// Limit length
function LimitedInput() {
  const [value, setValue] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const newValue = e.target.value.slice(0, 10);  // Max 10 characters
    setValue(newValue);
  }
  
  return (
    <div>
      <input value={value} onChange={handleChange} />
      <span>{value.length}/10</span>
    </div>
  );
}
\`\`\`

**Comparison with Uncontrolled:**

\`\`\`tsx
// Controlled: Must have onChange
function Controlled() {
  const [value, setValue] = useState('');
  return (
    <input 
      value={value}
      onChange={(e) => setValue(e.target.value)}  // ← Required!
    />
  );
}

// Uncontrolled: onChange is optional
function Uncontrolled() {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <input 
      ref={inputRef}
      defaultValue=""  // ← Not "value"
      // No onChange needed! DOM manages value
    />
  );
}
\`\`\`

**The React Warning:**

If you provide \`value\` without \`onChange\`, React warns:

\`\`\`
Warning: You provided a \`value\` prop to a form field without an
\`onChange\` handler. This will render a read-only field. If the field
should be mutable use \`defaultValue\`. Otherwise, set either \`onChange\`
or \`readOnly\`.
\`\`\`

**Solutions:**
\`\`\`tsx
// 1. Add onChange (make it mutable)
<input value={value} onChange={(e) => setValue(e.target.value)} />

// 2. Use defaultValue (make it uncontrolled)
<input defaultValue={value} />

// 3. Add readOnly (intentionally immutable)
<input value={value} readOnly />
\`\`\`

**Interview tip:** Explaining that onChange updates state so React can re-render the input with the new value shows understanding of React's unidirectional data flow. Mentioning that without onChange the input becomes read-only demonstrates knowledge of React's warnings and intentional design.`,
    },
    {
      id: 'q3',
      question:
        'Which prop should you use to set an initial value for an uncontrolled component?',
      options: ['value', 'defaultValue', 'initialValue', 'startValue'],
      correctAnswer: 1,
      explanation: `The correct answer is **"defaultValue"**.

For uncontrolled components, use \`defaultValue\` to set the initial value, not \`value\`.

**Why defaultValue?**

\`\`\`tsx
// ✅ CORRECT: Uncontrolled with defaultValue
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  return (
    <input 
      ref={inputRef}
      defaultValue="Initial text"  // ← Sets initial value only
    />
  );
}

// What happens:
// 1. Input starts with "Initial text"
// 2. User can type freely
// 3. DOM manages the value
// 4. React doesn't control or track value
\`\`\`

**Why not value?**

\`\`\`tsx
// ❌ WRONG: Using value without onChange
function BrokenUncontrolled() {
  return (
    <input 
      value="Initial text"  // ← No onChange!
    />
  );
}

// What happens:
// 1. Input starts with "Initial text"
// 2. User tries to type
// 3. Input appears frozen!
// 4. React warning: "You provided \`value\` without \`onChange\`"

// This creates a read-only input (probably not what you want)
\`\`\`

**The Difference:**

| Prop | Behavior | After User Types |
|------|----------|------------------|
| \`value\` | Controlled by React | React must update state |
| \`defaultValue\` | Initial value only | DOM manages changes |

**Detailed Examples:**

**defaultValue (Uncontrolled):**
\`\`\`tsx
function UncontrolledExample() {
  const nameRef = useRef<HTMLInputElement>(null);
  const emailRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    console.log({
      name: nameRef.current?.value,
      email: emailRef.current?.value
    });
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        ref={nameRef}
        defaultValue="John Doe"  // ← Initial value
      />
      <input 
        ref={emailRef}
        defaultValue=""  // ← Empty initial value
      />
      <button type="submit">Submit</button>
    </form>
  );
}

// User can type freely
// Access final values via refs on submit
\`\`\`

**value (Controlled):**
\`\`\`tsx
function ControlledExample() {
  const [name, setName] = useState('John Doe');
  const [email, setEmail] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    console.log({ name, email });  // Values already in state
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        value={name}  // ← Controlled by state
        onChange={(e) => setName(e.target.value)}
      />
      <input 
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <button type="submit">Submit</button>
    </form>
  );
}

// React controls values
// Can access values anytime from state
\`\`\`

**Different Input Types:**

**Text input:**
\`\`\`tsx
// Uncontrolled
<input type="text" defaultValue="Hello" />

// Controlled
<input type="text" value={text} onChange={(e) => setText(e.target.value)} />
\`\`\`

**Textarea:**
\`\`\`tsx
// Uncontrolled
<textarea defaultValue="Default text" />

// Controlled
<textarea value={text} onChange={(e) => setText(e.target.value)} />

// Note: In HTML, textarea uses children:
// <textarea>Default text</textarea>
// In React, use value or defaultValue prop!
\`\`\`

**Checkbox:**
\`\`\`tsx
// Uncontrolled
<input type="checkbox" defaultChecked={true} />

// Controlled
<input 
  type="checkbox" 
  checked={isChecked} 
  onChange={(e) => setIsChecked(e.target.checked)} 
/>

// Note: Checkbox uses defaultChecked/checked, not defaultValue/value!
\`\`\`

**Select:**
\`\`\`tsx
// Uncontrolled
<select defaultValue="us">
  <option value="us">United States</option>
  <option value="uk">United Kingdom</option>
</select>

// Controlled
<select value={country} onChange={(e) => setCountry(e.target.value)}>
  <option value="us">United States</option>
  <option value="uk">United Kingdom</option>
</select>
\`\`\`

**Radio:**
\`\`\`tsx
// Uncontrolled
<>
  <input type="radio" name="size" value="small" defaultChecked />
  <input type="radio" name="size" value="large" />
</>

// Controlled
<>
  <input 
    type="radio" 
    name="size" 
    value="small"
    checked={size === 'small'}
    onChange={(e) => setSize(e.target.value)}
  />
  <input 
    type="radio" 
    name="size" 
    value="large"
    checked={size === 'large'}
    onChange={(e) => setSize(e.target.value)}
  />
</>
\`\`\`

**Why other answers are wrong:**

**"value":**
- Using \`value\` makes the component controlled, not uncontrolled
- Requires \`onChange\` handler
- Without \`onChange\`, creates read-only input

**"initialValue":**
- This prop doesn't exist in React
- Would be ignored

**"startValue":**
- This prop doesn't exist in React
- Would be ignored

**Common Mistake:**

\`\`\`tsx
// ❌ WRONG: Mixing controlled and uncontrolled
function MixedComponent() {
  const [value, setValue] = useState('');
  
  return (
    <input 
      defaultValue="Initial"  // ← Uncontrolled
      value={value}           // ← Controlled
      onChange={(e) => setValue(e.target.value)}
    />
  );
}

// React warning:
// "A component is changing an uncontrolled input to be controlled."

// Pick one approach:
// Either controlled (value + onChange)
// Or uncontrolled (defaultValue + ref)
\`\`\`

**Resetting Uncontrolled Inputs:**

\`\`\`tsx
function UncontrolledForm() {
  const formRef = useRef<HTMLFormElement>(null);
  
  function handleReset() {
    formRef.current?.reset();  // Reset all inputs to defaultValue
  }
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    console.log(Object.fromEntries(formData));
    formRef.current?.reset();  // Clear after submit
  }
  
  return (
    <form ref={formRef} onSubmit={handleSubmit}>
      <input name="username" defaultValue="" />
      <input name="email" defaultValue="" />
      <button type="button" onClick={handleReset}>Reset</button>
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**Programmatically Setting Uncontrolled Value:**

\`\`\`tsx
function UncontrolledProgrammatic() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  function fillForm() {
    // Can set value via DOM
    if (inputRef.current) {
      inputRef.current.value = 'Programmatically set';
    }
  }
  
  return (
    <div>
      <input ref={inputRef} defaultValue="" />
      <button onClick={fillForm}>Fill Form</button>
    </div>
  );
}

// Note: This is DOM manipulation, not the React way
// If you need to programmatically set values, use controlled components
\`\`\`

**Decision Guide:**

\`\`\`
Need initial value?
├─ Controlled component? → value={state}
└─ Uncontrolled component? → defaultValue="..."

Need checkbox/radio initial state?
├─ Controlled → checked={state}
└─ Uncontrolled → defaultChecked={true/false}
\`\`\`

**Interview tip:** Explaining that \`defaultValue\` sets the initial value but lets the DOM manage subsequent changes shows understanding of the controlled/uncontrolled distinction. Mentioning that mixing \`value\` and \`defaultValue\` causes warnings demonstrates knowledge of React's expectations.`,
    },
    {
      id: 'q4',
      question:
        'When handling a form submission in React, what should you always do first in the submit handler?',
      options: [
        'Validate all form fields',
        'Call e.preventDefault() to prevent page reload',
        'Clear all form fields',
        'Show a loading spinner',
      ],
      correctAnswer: 1,
      explanation: `The correct answer is **"Call e.preventDefault() to prevent page reload"**.

This is the FIRST thing you should do in a React form submit handler, before any other logic.

**Why e.preventDefault() is Critical:**

\`\`\`tsx
function LoginForm() {
  const [email, setEmail] = useState('');
  
  // ❌ WITHOUT e.preventDefault(): Page reloads!
  function handleSubmitBad(e: React.FormEvent) {
    console.log('Submitting:', email);
    // Form submits → Browser reloads page
    // React app resets
    // State lost
    // Console.log never seen
  }
  
  // ✅ WITH e.preventDefault(): No reload
  function handleSubmitGood(e: React.FormEvent) {
    e.preventDefault();  // ← FIRST thing!
    console.log('Submitting:', email);
    // Form doesn't submit to server
    // Page doesn't reload
    // React handles submission
  }
  
  return (
    <form onSubmit={handleSubmitGood}>
      <input 
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <button type="submit">Login</button>
    </form>
  );
}
\`\`\`

**What Happens Without e.preventDefault():**

\`\`\`tsx
// ❌ Missing e.preventDefault()
function BadForm() {
  async function handleSubmit(e: React.FormEvent) {
    // Missing: e.preventDefault()
    
    const response = await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify({ email: 'user@example.com' })
    });
    
    // This code never runs!
    // Page reloaded before fetch completed
  }
  
  return <form onSubmit={handleSubmit}>...</form>;
}

// What happens:
// 1. User clicks submit
// 2. handleSubmit starts
// 3. Form's native submit fires
// 4. Browser navigates (page reload)
// 5. React app unmounts
// 6. Fetch never completes
// 7. User sees blank/reloaded page
\`\`\`

**✅ Correct Order of Operations:**

\`\`\`tsx
function ProperForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  async function handleSubmit(e: React.FormEvent) {
    // 1. FIRST: Prevent default behavior
    e.preventDefault();
    
    // 2. Clear previous errors
    setErrors({});
    
    // 3. Validate
    if (!email.includes('@')) {
      setErrors({ email: 'Invalid email' });
      return;
    }
    
    if (password.length < 8) {
      setErrors({ password: 'Password too short' });
      return;
    }
    
    // 4. Show loading state
    setIsSubmitting(true);
    
    try {
      // 5. Submit to server
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      // 6. Handle success
      const data = await response.json();
      console.log('Success!', data);
      
      // 7. Clear form or redirect
      setEmail('');
      setPassword('');
    } catch (err) {
      // 8. Handle error
      setErrors({ 
        general: err instanceof Error ? err.message : 'Unknown error' 
      });
    } finally {
      // 9. Hide loading state
      setIsSubmitting(false);
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        disabled={isSubmitting}
      />
      {errors.email && <p>{errors.email}</p>}
      
      <input 
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={isSubmitting}
      />
      {errors.password && <p>{errors.password}</p>}
      
      {errors.general && <p>{errors.general}</p>}
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
}
\`\`\`

**Why other options are wrong (but still important):**

**"Validate all form fields":**
- Important, but AFTER e.preventDefault()
- If you validate first without preventing default, page reloads before validation completes

\`\`\`tsx
// ❌ WRONG ORDER
function handleSubmit(e: React.FormEvent) {
  if (!email.includes('@')) {  // Validate first
    setError('Invalid email');
    return;
  }
  e.preventDefault();  // Too late! Page already reloading
}

// ✅ CORRECT ORDER
function handleSubmit(e: React.FormEvent) {
  e.preventDefault();  // First!
  if (!email.includes('@')) {  // Then validate
    setError('Invalid email');
    return;
  }
}
\`\`\`

**"Clear all form fields":**
- Should be done AFTER successful submission, not first
- Clearing first would lose the data you're trying to submit!

\`\`\`tsx
// ❌ WRONG: Clear first
function handleSubmit(e: React.FormEvent) {
  e.preventDefault();
  setEmail('');    // Cleared before sending!
  setPassword('');
  
  fetch('/api/login', {
    body: JSON.stringify({ 
      email,     // Empty!
      password   // Empty!
    })
  });
}

// ✅ CORRECT: Clear after success
async function handleSubmit(e: React.FormEvent) {
  e.preventDefault();
  
  const response = await fetch('/api/login', {
    body: JSON.stringify({ email, password })  // Has data
  });
  
  if (response.ok) {
    setEmail('');     // Clear after success
    setPassword('');
  }
}
\`\`\`

**"Show a loading spinner":**
- Should be done AFTER validation passes
- Don't show loading if validation fails

\`\`\`tsx
// ❌ WRONG: Show loading before validation
function handleSubmit(e: React.FormEvent) {
  e.preventDefault();
  setIsSubmitting(true);  // Loading shown
  
  if (!email) {
    setError('Email required');
    return;  // Oops! Loading still showing
  }
  
  fetch('/api/login', ...);
}

// ✅ CORRECT: Show loading after validation
function handleSubmit(e: React.FormEvent) {
  e.preventDefault();
  
  if (!email) {
    setError('Email required');
    return;  // Exit early, no loading
  }
  
  setIsSubmitting(true);  // Now show loading
  fetch('/api/login', ...);
}
\`\`\`

**What e.preventDefault() Actually Does:**

\`\`\`tsx
// Native browser behavior (without React):
<form action="/api/login" method="POST">
  <input name="email" />
  <input name="password" />
  <button type="submit">Login</button>
</form>

// When submitted:
// 1. Browser collects form data
// 2. Sends POST request to /api/login
// 3. Navigates to response page
// 4. Current page unloads

// In React with e.preventDefault():
<form onSubmit={(e) => {
  e.preventDefault();  // ← Stops native behavior
  // Now YOU handle submission
}}>
  ...
</form>

// When submitted:
// 1. Browser would try to submit
// 2. e.preventDefault() stops it
// 3. Your handler runs instead
// 4. Page stays loaded (SPA behavior)
\`\`\`

**Alternative: Using <button> Instead of Form Submit:**

\`\`\`tsx
// Alternative pattern (not recommended)
function AlternativePattern() {
  async function handleClick() {
    // No e.preventDefault() needed
    // But loses form semantics (Enter key won't submit, etc.)
    const response = await fetch('/api/login', ...);
  }
  
  return (
    <div>  {/* Not a form! */}
      <input type="email" value={email} onChange={...} />
      <input type="password" value={password} onChange={...} />
      <button onClick={handleClick}>Login</button>
    </div>
  );
}

// Problems:
// - Pressing Enter in input doesn't submit
// - Not semantic HTML
// - Worse accessibility
// - Form validation attributes don't work
\`\`\`

**Best Practice Pattern:**

\`\`\`tsx
function BestPracticeForm() {
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();  // ← Always first!
    
    // Early returns for validation
    if (!isValid()) return;
    
    // Set loading state
    setLoading(true);
    
    try {
      // Submit logic
      await submitForm();
    } catch (err) {
      // Error handling
      handleError(err);
    } finally {
      // Always reset loading
      setLoading(false);
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      {/* form fields */}
      <button type="submit" disabled={loading}>
        {loading ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
}
\`\`\`

**Interview tip:** Explaining that e.preventDefault() stops the browser's native form submission (which would reload the page) shows understanding of web fundamentals. Mentioning that it should always be the first line demonstrates attention to order-of-operations and potential bugs.`,
    },
    {
      id: 'q5',
      question: 'Which input type is always uncontrolled in React?',
      options: ['text', 'checkbox', 'file', 'select'],
      correctAnswer: 2,
      explanation: `The correct answer is **"file"**.

File inputs are ALWAYS uncontrolled in React due to security restrictions. You cannot programmatically set their value.

**Why File Inputs are Always Uncontrolled:**

\`\`\`tsx
// ✅ CORRECT: Uncontrolled file input
function FileUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);  // Store File object in state
    }
  }
  
  return (
    <div>
      <input 
        type="file"
        onChange={handleChange}  // Can listen to changes
        // No "value" prop! Always uncontrolled
      />
      {selectedFile && <p>Selected: {selectedFile.name}</p>}
    </div>
  );
}

// ❌ CANNOT DO: Set value programmatically
function BrokenFileInput() {
  return (
    <input 
      type="file"
      value="/path/to/file.txt"  // ← DOESN'T WORK!
    />
  );
}

// Browser security: Scripts cannot set file input values
// This prevents malicious websites from uploading files without user consent
\`\`\`

**Why This Security Restriction Exists:**

\`\`\`tsx
// Imagine if this was possible (it's not):
function MaliciousForm() {
  return (
    <form action="/steal-files" method="POST">
      <input 
        type="file"
        value="C:\\Users\\victim\\Documents\\passwords.txt"  // ← Blocked!
      />
      <button type="submit">Submit</button>
    </form>
  );
}

// Without this restriction:
// - Malicious sites could steal files
// - User never chose the file
// - Major security vulnerability

// With restriction:
// - User MUST click file input and choose file
// - Scripts cannot access file system
// - Safe
\`\`\`

**How to Work with File Inputs:**

**1. Single File Upload:**
\`\`\`tsx
function SingleFileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];
    
    if (selectedFile) {
      setFile(selectedFile);
      
      // Create preview for images
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  }
  
  async function handleUpload() {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData  // Send as multipart/form-data
    });
    
    if (response.ok) {
      console.log('Upload successful!');
    }
  }
  
  return (
    <div>
      <input 
        type="file"
        accept="image/*"  // Limit to images
        onChange={handleChange}
      />
      
      {preview && <img src={preview} alt="Preview" />}
      {file && <p>{file.name} ({file.size} bytes)</p>}
      
      <button onClick={handleUpload} disabled={!file}>
        Upload
      </button>
    </div>
  );
}
\`\`\`

**2. Multiple Files:**
\`\`\`tsx
function MultipleFileUpload() {
  const [files, setFiles] = useState<File[]>([]);
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));  // Convert FileList to Array
    }
  }
  
  function removeFile(index: number) {
    setFiles(prev => prev.filter((_, i) => i !== index));
  }
  
  return (
    <div>
      <input 
        type="file"
        multiple  // Allow multiple files
        onChange={handleChange}
      />
      
      <ul>
        {files.map((file, index) => (
          <li key={index}>
            {file.name} ({file.size} bytes)
            <button onClick={() => removeFile(index)}>Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
\`\`\`

**3. Drag and Drop:**
\`\`\`tsx
function DragDropUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  
  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files) {
      setFiles(Array.from(e.dataTransfer.files));
    }
  }
  
  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }
  
  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={() => setIsDragging(false)}
      style={{
        border: \`2px dashed \${isDragging ? 'blue' : 'gray'}\`,
        padding: '40px',
        textAlign: 'center'
      }}
    >
      {files.length === 0 ? (
        <p>Drag and drop files here or click to browse</p>
      ) : (
        <ul>
          {files.map((file, i) => (
            <li key={i}>{file.name}</li>
          ))}
        </ul>
      )}
      
      <input 
        type="file"
        multiple
        onChange={(e) => {
          if (e.target.files) {
            setFiles(Array.from(e.target.files));
          }
        }}
        style={{ display: 'none' }}
        id="file-input"
      />
      <label htmlFor="file-input">
        <button as="span">Browse Files</button>
      </label>
    </div>
  );
}
\`\`\`

**Why Other Options Can Be Controlled:**

**Text input (can be controlled):**
\`\`\`tsx
// ✅ Can be controlled
<input 
  type="text"
  value={text}
  onChange={(e) => setText(e.target.value)}
/>
\`\`\`

**Checkbox (can be controlled):**
\`\`\`tsx
// ✅ Can be controlled
<input 
  type="checkbox"
  checked={isChecked}
  onChange={(e) => setIsChecked(e.target.checked)}
/>
\`\`\`

**Select (can be controlled):**
\`\`\`tsx
// ✅ Can be controlled
<select 
  value={selected}
  onChange={(e) => setSelected(e.target.value)}
>
  <option value="a">Option A</option>
  <option value="b">Option B</option>
</select>
\`\`\`

**Common Patterns with File Inputs:**

**1. Reset file input:**
\`\`\`tsx
function ResetableFileInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  
  function handleReset() {
    setFile(null);
    if (inputRef.current) {
      inputRef.current.value = '';  // Clear file input
    }
  }
  
  return (
    <div>
      <input 
        type="file"
        ref={inputRef}
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      {file && (
        <div>
          <p>{file.name}</p>
          <button onClick={handleReset}>Clear</button>
        </div>
      )}
    </div>
  );
}
\`\`\`

**2. Validate file before upload:**
\`\`\`tsx
function ValidatedFileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    setError('');
    
    // Validate file size (max 5MB)
    if (selectedFile.size > 5 * 1024 * 1024) {
      setError('File must be smaller than 5MB');
      return;
    }
    
    // Validate file type
    if (!selectedFile.type.startsWith('image/')) {
      setError('File must be an image');
      return;
    }
    
    setFile(selectedFile);
  }
  
  return (
    <div>
      <input 
        type="file"
        accept="image/*"
        onChange={handleChange}
      />
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {file && <p>Ready to upload: {file.name}</p>}
    </div>
  );
}
\`\`\`

**Summary:**

| Input Type | Can Be Controlled? | Reason |
|------------|-------------------|--------|
| text | ✅ Yes | No security restrictions |
| textarea | ✅ Yes | No security restrictions |
| checkbox | ✅ Yes | No security restrictions |
| radio | ✅ Yes | No security restrictions |
| select | ✅ Yes | No security restrictions |
| **file** | **❌ No** | **Browser security: prevents file system access** |

**Interview tip:** Explaining that file inputs are always uncontrolled due to browser security (preventing scripts from accessing the file system) shows understanding of web security fundamentals. Mentioning how to work with file inputs using onChange and the File API demonstrates practical experience.`,
    },
  ],
};
