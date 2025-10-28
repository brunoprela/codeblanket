export const formsIntroductionDiscussion = {
  title: 'Forms Introduction Discussion Questions',
  id: 'forms-introduction-discussion',
  sectionId: 'forms-introduction',
  questions: [
    {
      id: 'q1',
      question:
        'Explain the difference between controlled and uncontrolled components in React forms. Discuss the tradeoffs of each approach and provide scenarios where you would choose one over the other. How do these patterns affect performance and user experience?',
      answer: `Controlled and uncontrolled components represent two fundamentally different approaches to managing form state in React. Understanding when to use each is critical for building effective forms.

**Controlled Components: React Owns the State**

In a controlled component, React state is the "single source of truth" for the input's value.

\`\`\`tsx
function ControlledInput() {
  const [value, setValue] = useState('');
  
  return (
    <input 
      value={value}                              // Controlled by state
      onChange={(e) => setValue(e.target.value)} // Updates state
    />
  );
}
\`\`\`

**How it works:**
1. State holds current value
2. Input displays that value
3. User types → onChange fires
4. State updates
5. React re-renders
6. Input shows new value

**Flow diagram:**
\`\`\`
User types "h"
  ↓
onChange event fires
  ↓
setValue("h") called
  ↓
Component re-renders
  ↓
Input displays "h"
  ↓
(Repeat for every keystroke)
\`\`\`

**Uncontrolled Components: DOM Owns the State**

In an uncontrolled component, the DOM manages the value. You access it via a ref when needed.

\`\`\`tsx
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit() {
    const value = inputRef.current?.value;  // Access value when needed
    console.log(value);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input ref={inputRef} defaultValue="" />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**How it works:**
1. Input manages its own value (like traditional HTML)
2. User types → value changes in DOM
3. React doesn't know about changes
4. Access value via ref when needed (e.g., on submit)

**Detailed Comparison:**

**1. State Management**

Controlled:
\`\`\`tsx
// State is explicit and visible
const [email, setEmail] = useState('');

// Can easily see current value
console.log('Current email:', email);

// Can programmatically set value
setEmail('new@example.com');

// Can derive UI from value
{email && <p>You entered: {email}</p>}
\`\`\`

Uncontrolled:
\`\`\`tsx
// State is in DOM, not visible in React
const emailRef = useRef<HTMLInputElement>(null);

// Must access DOM to see value
console.log('Current email:', emailRef.current?.value);

// Can set value via DOM
emailRef.current.value = 'new@example.com';

// Hard to derive UI from value (need to track separately)
\`\`\`

**2. Validation**

Controlled (Real-time validation):
\`\`\`tsx
function ControlledForm() {
  const [password, setPassword] = useState('');
  
  const isValid = password.length >= 8;
  const hasNumber = /\\d/.test(password);
  const hasLetter = /[a-z]/i.test(password);
  
  return (
    <div>
      <input 
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      
      {/* Instant feedback */}
      <ul>
        <li style={{ color: isValid ? 'green' : 'red' }}>
          At least 8 characters
        </li>
        <li style={{ color: hasNumber ? 'green' : 'red' }}>
          Contains a number
        </li>
        <li style={{ color: hasLetter ? 'green' : 'red' }}>
          Contains a letter
        </li>
      </ul>
      
      <button disabled={!isValid || !hasNumber || !hasLetter}>
        Submit
      </button>
    </div>
  );
}
\`\`\`

Uncontrolled (Validation on submit):
\`\`\`tsx
function UncontrolledForm() {
  const passwordRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const password = passwordRef.current?.value || '';
    
    // Validate only on submit
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    
    console.log('Valid!');
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input type="password" ref={passwordRef} />
      {error && <p>{error}</p>}
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**3. Performance**

Controlled:
\`\`\`tsx
// Re-renders on every keystroke
function ControlledInput() {
  const [value, setValue] = useState('');
  
  console.log('Render');  // Logs on every character typed!
  
  return (
    <input 
      value={value}
      onChange={(e) => setValue(e.target.value)}
    />
  );
}

// With large form: Many state updates
function LargeForm() {
  const [formData, setFormData] = useState({
    field1: '',
    field2: '',
    field3: '',
    // ... 50 fields
  });
  
  // Every keystroke in any field re-renders entire form
}
\`\`\`

Uncontrolled:
\`\`\`tsx
// No re-renders on typing
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  console.log('Render');  // Only logs on initial mount!
  
  return <input ref={inputRef} />;
}

// Large form: No performance impact from typing
\`\`\`

**However:** Modern React is very fast. Performance difference is rarely noticeable unless you have:
- Extremely large forms (50+ inputs)
- Heavy computation on each render
- Slow devices

**4. Input Formatting**

Controlled (Easy to format):
\`\`\`tsx
function PhoneInput() {
  const [phone, setPhone] = useState('');
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value.replace(/\\D/g, '');  // Remove non-digits
    
    // Format: (123) 456-7890
    let formatted = value;
    if (value.length > 3) {
      formatted = \`(\${value.slice(0, 3)}) \${value.slice(3)}\`;
    }
    if (value.length > 6) {
      formatted = \`(\${value.slice(0, 3)}) \${value.slice(3, 6)}-\${value.slice(6, 10)}\`;
    }
    
    setPhone(formatted);
  }
  
  return (
    <input 
      type="tel"
      value={phone}
      onChange={handleChange}
      placeholder="(123) 456-7890"
    />
  );
}
\`\`\`

Uncontrolled (Hard to format):
\`\`\`tsx
// Can't easily format as user types
// Would need to manipulate DOM directly (anti-pattern in React)
\`\`\`

**5. Resetting Forms**

Controlled (Trivial):
\`\`\`tsx
function ControlledForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  function handleReset() {
    setEmail('');
    setPassword('');
  }
  
  return (
    <form>
      <input value={email} onChange={(e) => setEmail(e.target.value)} />
      <input value={password} onChange={(e) => setPassword(e.target.value)} />
      <button type="button" onClick={handleReset}>Reset</button>
    </form>
  );
}
\`\`\`

Uncontrolled (Need to manipulate DOM):
\`\`\`tsx
function UncontrolledForm() {
  const formRef = useRef<HTMLFormElement>(null);
  
  function handleReset() {
    formRef.current?.reset();  // Uses native form reset
  }
  
  return (
    <form ref={formRef}>
      <input name="email" />
      <input name="password" />
      <button type="button" onClick={handleReset}>Reset</button>
    </form>
  );
}
\`\`\`

**When to Use Each:**

**Use Controlled Components When:**

1. **Need real-time validation**
\`\`\`tsx
// Disable submit until email is valid
<button disabled={!isValidEmail(email)}>Submit</button>
\`\`\`

2. **Need to format input**
\`\`\`tsx
// Format credit card: 1234 5678 9012 3456
<input value={formatCreditCard(cardNumber)} onChange={...} />
\`\`\`

3. **Need conditional UI**
\`\`\`tsx
// Show password strength meter
{password && <PasswordStrength password={password} />}
\`\`\`

4. **Need to enforce rules**
\`\`\`tsx
// Limit to 10 characters
function handleChange(e) {
  const value = e.target.value.slice(0, 10);
  setValue(value);
}
\`\`\`

5. **Need dynamic forms**
\`\`\`tsx
// Add/remove fields dynamically
{fields.map(field => <input key={field.id} value={field.value} />)}
\`\`\`

6. **Most forms** (recommended default)

**Use Uncontrolled Components When:**

1. **File uploads** (required—file inputs are always uncontrolled)
\`\`\`tsx
<input type="file" ref={fileRef} onChange={handleFileChange} />
\`\`\`

2. **Integrating with non-React libraries**
\`\`\`tsx
// Third-party date picker manages its own state
<DatePicker ref={dateRef} />
\`\`\`

3. **Simple forms without validation**
\`\`\`tsx
// Contact form that only submits
<form onSubmit={handleSubmit}>
  <input name="email" />
  <textarea name="message" />
  <button>Send</button>
</form>
\`\`\`

4. **Performance optimization** (rare, only for very large forms)

5. **Quick prototypes**

**User Experience Impact:**

**Controlled:**
- ✅ Immediate feedback (validation, character count)
- ✅ Dynamic submit button state
- ✅ Consistent behavior across browsers
- ✅ Can prevent invalid input
- ❌ Slight lag on slow devices (rare)

**Uncontrolled:**
- ✅ Fastest possible input (no React re-renders)
- ✅ Native browser behavior
- ❌ No real-time feedback
- ❌ Validation only on submit
- ❌ Harder to provide good UX

**Interview Insight:**
Explaining the state ownership difference (React vs DOM) and discussing when each is appropriate shows solid understanding of React's philosophy. Mentioning that controlled is the recommended default but acknowledging uncontrolled has valid use cases demonstrates balanced,practical thinking.`,
    },
    {
      id: 'q2',
      question:
        'How do you handle form validation in React? Compare client-side validation approaches including inline validation, validation on blur, validation on submit, and schema-based validation. What are the security implications of client-side vs server-side validation?',
      answer: `Form validation is critical for user experience and data integrity. React offers multiple approaches, each with different tradeoffs.

**Validation Approaches:**

**1. Inline Validation (As User Types)**

Validates on every keystroke.

\`\`\`tsx
function InlineValidation() {
  const [email, setEmail] = useState('');
  
  // Validate on every change
  const emailError = !email.includes('@') && email.length > 0
    ? 'Email must contain @'
    : '';
  
  return (
    <div>
      <input 
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      {emailError && <p className="error">{emailError}</p>}
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ Immediate feedback
- ✅ User knows right away if input is valid
- ✅ Good for complex rules (password strength)

**Cons:**
- ❌ Annoying (shows error before user finishes typing)
- ❌ Many re-renders
- ❌ Can be distracting

**Best for:** Password strength meters, character counts

**2. Validation on Blur (When User Leaves Field)**

Validates when input loses focus.

\`\`\`tsx
function BlurValidation() {
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');
  const [touched, setTouched] = useState(false);
  
  function validateEmail(value: string) {
    if (!value) return 'Email is required';
    if (!/\\S+@\\S+\\.\\S+/.test(value)) return 'Email is invalid';
    return '';
  }
  
  function handleBlur() {
    setTouched(true);
    setEmailError(validateEmail(email));
  }
  
  return (
    <div>
      <input 
        type="email"
        value={email}
        onChange={(e) => {
          setEmail(e.target.value);
          // Clear error when user starts typing again
          if (touched) {
            setEmailError(validateEmail(e.target.value));
          }
        }}
        onBlur={handleBlur}
      />
      {touched && emailError && <p className="error">{emailError}</p>}
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ Doesn't show error while user is typing
- ✅ Good balance of feedback and UX
- ✅ Industry standard for most fields

**Cons:**
- ❌ No feedback until user leaves field
- ❌ More complex state management

**Best for:** Most input fields (email, username, etc.)

**3. Validation on Submit**

Validates only when form is submitted.

\`\`\`tsx
interface FormData {
  email: string;
  password: string;
}

interface FormErrors {
  email?: string;
  password?: string;
}

function SubmitValidation() {
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState<FormErrors>({});
  
  function validate(): FormErrors {
    const newErrors: FormErrors = {};
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\\S+@\\S+\\.\\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }
    
    return newErrors;
  }
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    const newErrors = validate();
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    // Submit form
    console.log('Valid!', formData);
  }
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error for this field
    if (errors[name as keyof FormErrors]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input 
          name="email"
          value={formData.email}
          onChange={handleChange}
        />
        {errors.email && <p>{errors.email}</p>}
      </div>
      
      <div>
        <input 
          name="password"
          type="password"
          value={formData.password}
          onChange={handleChange}
        />
        {errors.password && <p>{errors.password}</p>}
      </div>
      
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**Pros:**
- ✅ No feedback until user tries to submit
- ✅ Simplest to implement
- ✅ Validates all fields at once

**Cons:**
- ❌ User doesn't know about errors until submit
- ❌ May need to scroll to see errors
- ❌ Less guidance during input

**Best for:** Simple forms, quick forms

**4. Schema-Based Validation (with Libraries)**

Use a validation library for complex forms.

**Option A: Zod**
\`\`\`tsx
import { z } from 'zod';

const signupSchema = z.object({
  username: z.string().min(3, 'Username must be at least 3 characters'),
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  confirmPassword: z.string()
}).refine((data) => data.password === data.confirmPassword, {
  message: 'Passwords must match',
  path: ['confirmPassword']
});

type SignupData = z.infer<typeof signupSchema>;

function SignupForm() {
  const [formData, setFormData] = useState<SignupData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    const result = signupSchema.safeParse(formData);
    
    if (!result.success) {
      const fieldErrors: Record<string, string> = {};
      result.error.errors.forEach(err => {
        if (err.path[0]) {
          fieldErrors[err.path[0] as string] = err.message;
        }
      });
      setErrors(fieldErrors);
      return;
    }
    
    console.log('Valid!', result.data);
  }
  
  return <form onSubmit={handleSubmit}>{/* ... */}</form>;
}
\`\`\`

**Option B: React Hook Form + Zod (Recommended)**
\`\`\`tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
});

type FormData = z.infer<typeof schema>;

function Form() {
  const { 
    register, 
    handleSubmit, 
    formState: { errors } 
  } = useForm<FormData>({
    resolver: zodResolver(schema)
  });
  
  function onSubmit(data: FormData) {
    console.log(data);
  }
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('email')} />
      {errors.email && <p>{errors.email.message}</p>}
      
      <input type="password" {...register('password')} />
      {errors.password && <p>{errors.password.message}</p>}
      
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**Pros:**
- ✅ Type-safe validation
- ✅ Reusable schemas
- ✅ Complex validation rules
- ✅ Server/client validation sharing
- ✅ Less boilerplate

**Cons:**
- ❌ Additional library dependency
- ❌ Learning curve

**Best for:** Complex forms, production applications

**Comparison Table:**

| Approach | Feedback Timing | UX Quality | Complexity | Best For |
|----------|----------------|------------|------------|----------|
| Inline | Immediate | Good* | Medium | Password strength, character count |
| On Blur | After leaving field | Excellent | Medium | Most fields (recommended) |
| On Submit | After submit click | Okay | Low | Simple forms |
| Schema | Configurable | Excellent | Low (with library) | Complex forms, production |

*Can be annoying if not done carefully

**Client-Side vs Server-Side Validation:**

**Client-Side Validation (JavaScript)**

\`\`\`tsx
function ClientValidation() {
  const [email, setEmail] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    // Client-side check
    if (!/\\S+@\\S+\\.\\S+/.test(email)) {
      alert('Invalid email');
      return;
    }
    
    // Send to server
    fetch('/api/signup', {
      method: 'POST',
      body: JSON.stringify({ email })
    });
  }
  
  return <form onSubmit={handleSubmit}>...</form>;
}
\`\`\`

**Pros:**
- ✅ Instant feedback (no network request)
- ✅ Better user experience
- ✅ Reduces server load
- ✅ Works offline

**Cons:**
- ❌ **Can be bypassed!** (disable JavaScript, modify request)
- ❌ Not secure
- ❌ Can get out of sync with server rules

**Server-Side Validation (Required!)**

\`\`\`tsx
// Server (Node.js/Express)
app.post('/api/signup', async (req, res) => {
  const { email, password } = req.body;
  
  // MUST validate on server!
  if (!/\\S+@\\S+\\.\\S+/.test(email)) {
    return res.status(400).json({ error: 'Invalid email' });
  }
  
  if (password.length < 8) {
    return res.status(400).json({ error: 'Password too short' });
  }
  
  // Check if email already exists
  const existing = await db.findUser(email);
  if (existing) {
    return res.status(400).json({ error: 'Email already taken' });
  }
  
  // Create user
  await db.createUser({ email, password });
  res.json({ success: true });
});
\`\`\`

**Security Implications:**

**Critical: NEVER Trust Client-Side Validation**

\`\`\`tsx
// ❌ INSECURE: Only client-side validation
function InsecureForm() {
  const [amount, setAmount] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    // Check amount is positive
    if (Number(amount) <= 0) {
      alert('Amount must be positive');
      return;
    }
    
    // Send to server WITHOUT server validation
    fetch('/api/transfer', {
      method: 'POST',
      body: JSON.stringify({ amount: Number(amount) })
    });
  }
  
  return <form onSubmit={handleSubmit}>...</form>;
}

// Attacker can bypass this:
// 1. Open DevTools
// 2. Run: fetch('/api/transfer', { method: 'POST', body: JSON.stringify({ amount: -1000000 }) })
// 3. Negative amount sent to server!
\`\`\`

**✅ SECURE: Both Client + Server Validation**

\`\`\`tsx
// Client (React)
function SecureForm() {
  const [amount, setAmount] = useState('');
  const [error, setError] = useState('');
  
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    // Client-side: Quick feedback
    if (Number(amount) <= 0) {
      setError('Amount must be positive');
      return;
    }
    
    try {
      const res = await fetch('/api/transfer', {
        method: 'POST',
        body: JSON.stringify({ amount: Number(amount) })
      });
      
      if (!res.ok) {
        const data = await res.json();
        setError(data.error);  // Show server error
      }
    } catch (err) {
      setError('Network error');
    }
  }
  
  return <form onSubmit={handleSubmit}>...</form>;
}

// Server (Node.js)
app.post('/api/transfer', (req, res) => {
  const { amount } = req.body;
  
  // Server-side: MUST validate!
  if (typeof amount !== 'number' || amount <= 0) {
    return res.status(400).json({ error: 'Invalid amount' });
  }
  
  // Check user has sufficient balance
  if (user.balance < amount) {
    return res.status(400).json({ error: 'Insufficient balance' });
  }
  
  // Process transfer
  processTransfer(amount);
  res.json({ success: true });
});
\`\`\`

**Best Practice: Shared Validation Schema**

Use the same validation on client and server:

\`\`\`tsx
// shared/schema.ts (used by both client and server)
import { z } from 'zod';

export const signupSchema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Password must be at least 8 characters')
});

// client/signup.tsx
import { signupSchema } from '../shared/schema';

function SignupForm() {
  const [data, setData] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState({});
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    const result = signupSchema.safeParse(data);
    if (!result.success) {
      setErrors(/* ... */);
      return;
    }
    
    fetch('/api/signup', { /* ... */ });
  }
  
  return <form onSubmit={handleSubmit}>...</form>;
}

// server/routes.ts
import { signupSchema } from '../shared/schema';

app.post('/api/signup', (req, res) => {
  const result = signupSchema.safeParse(req.body);
  
  if (!result.success) {
    return res.status(400).json({ errors: result.error.errors });
  }
  
  // Process signup
});
\`\`\`

**Summary:**

1. **Client-side validation** = UX (quick feedback)
2. **Server-side validation** = Security (can't bypass)
3. **Always do both**
4. **Use schema validation** for consistency
5. **Never trust the client**

**Interview Insight:**
Discussing multiple validation approaches shows depth. Emphasizing that client-side validation is for UX, not security, and explaining how attackers can bypass it demonstrates security awareness—critical for senior roles.`,
    },
    {
      id: 'q3',
      question:
        'Discuss strategies for handling complex multi-step forms in React. How would you manage state across multiple steps, handle navigation between steps, preserve data when going back, and validate individual steps before allowing progression?',
      answer: `Multi-step forms (wizards, onboarding flows) are common in modern applications but require careful state management and UX considerations.

**Core Challenges:**
1. Maintaining state across steps
2. Validating each step before progression
3. Allowing back/forward navigation
4. Preserving data when going back
5. Handling submission at the end
6. URL synchronization (optional)

**Strategy 1: Single Component with Step State**

Store all data in one component, render steps conditionally.

\`\`\`tsx
interface FormData {
  // Step 1
  email: string;
  password: string;
  // Step 2
  firstName: string;
  lastName: string;
  // Step 3
  address: string;
  city: string;
  zipCode: string;
}

function MultiStepForm() {
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    address: '',
    city: '',
    zipCode: ''
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  function updateField(field: keyof FormData, value: string) {
    setFormData(prev => ({ ...prev, [field]: value }));
  }
  
  function validateStep(step: number): boolean {
    const newErrors: Record<string, string> = {};
    
    if (step === 1) {
      if (!formData.email.includes('@')) {
        newErrors.email = 'Invalid email';
      }
      if (formData.password.length < 8) {
        newErrors.password = 'Password too short';
      }
    } else if (step === 2) {
      if (!formData.firstName) {
        newErrors.firstName = 'First name required';
      }
      if (!formData.lastName) {
        newErrors.lastName = 'Last name required';
      }
    } else if (step === 3) {
      if (!formData.address) {
        newErrors.address = 'Address required';
      }
      if (!formData.city) {
        newErrors.city = 'City required';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }
  
  function handleNext() {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  }
  
  function handleBack() {
    setCurrentStep(prev => prev - 1);
  }
  
  async function handleSubmit() {
    if (!validateStep(currentStep)) return;
    
    try {
      const response = await fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      if (response.ok) {
        console.log('Success!');
      }
    } catch (err) {
      console.error(err);
    }
  }
  
  return (
    <div>
      {/* Progress indicator */}
      <div className="progress">
        <span className={currentStep >= 1 ? 'active' : ''}>Account</span>
        <span className={currentStep >= 2 ? 'active' : ''}>Profile</span>
        <span className={currentStep >= 3 ? 'active' : ''}>Address</span>
      </div>
      
      {/* Step 1: Account */}
      {currentStep === 1 && (
        <div>
          <h2>Step 1: Create Account</h2>
          <input 
            type="email"
            value={formData.email}
            onChange={(e) => updateField('email', e.target.value)}
            placeholder="Email"
          />
          {errors.email && <p>{errors.email}</p>}
          
          <input 
            type="password"
            value={formData.password}
            onChange={(e) => updateField('password', e.target.value)}
            placeholder="Password"
          />
          {errors.password && <p>{errors.password}</p>}
        </div>
      )}
      
      {/* Step 2: Profile */}
      {currentStep === 2 && (
        <div>
          <h2>Step 2: Profile Info</h2>
          <input 
            value={formData.firstName}
            onChange={(e) => updateField('firstName', e.target.value)}
            placeholder="First Name"
          />
          {errors.firstName && <p>{errors.firstName}</p>}
          
          <input 
            value={formData.lastName}
            onChange={(e) => updateField('lastName', e.target.value)}
            placeholder="Last Name"
          />
          {errors.lastName && <p>{errors.lastName}</p>}
        </div>
      )}
      
      {/* Step 3: Address */}
      {currentStep === 3 && (
        <div>
          <h2>Step 3: Address</h2>
          <input 
            value={formData.address}
            onChange={(e) => updateField('address', e.target.value)}
            placeholder="Address"
          />
          {errors.address && <p>{errors.address}</p>}
          
          <input 
            value={formData.city}
            onChange={(e) => updateField('city', e.target.value)}
            placeholder="City"
          />
          {errors.city && <p>{errors.city}</p>}
        </div>
      )}
      
      {/* Navigation */}
      <div>
        {currentStep > 1 && (
          <button onClick={handleBack}>Back</button>
        )}
        
        {currentStep < 3 ? (
          <button onClick={handleNext}>Next</button>
        ) : (
          <button onClick={handleSubmit}>Submit</button>
        )}
      </div>
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ Simple to implement
- ✅ All data in one place
- ✅ Easy to access any step's data
- ✅ No prop drilling

**Cons:**
- ❌ Large component
- ❌ Validation logic mixed with rendering
- ❌ Hard to reuse steps

**Strategy 2: Separate Step Components**

Extract each step into its own component.

\`\`\`tsx
// Step components
function Step1({ data, updateData, errors }: StepProps) {
  return (
    <div>
      <h2>Step 1: Account</h2>
      <input 
        type="email"
        value={data.email}
        onChange={(e) => updateData('email', e.target.value)}
      />
      {errors.email && <p>{errors.email}</p>}
      
      <input 
        type="password"
        value={data.password}
        onChange={(e) => updateData('password', e.target.value)}
      />
      {errors.password && <p>{errors.password}</p>}
    </div>
  );
}

function Step2({ data, updateData, errors }: StepProps) {
  return (
    <div>
      <h2>Step 2: Profile</h2>
      <input 
        value={data.firstName}
        onChange={(e) => updateData('firstName', e.target.value)}
      />
      {errors.firstName && <p>{errors.firstName}</p>}
    </div>
  );
}

// Main wizard
function MultiStepWizard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<FormData>({ /* ... */ });
  const [errors, setErrors] = useState({});
  
  const steps = [
    <Step1 data={formData} updateData={updateField} errors={errors} />,
    <Step2 data={formData} updateData={updateField} errors={errors} />,
    <Step3 data={formData} updateData={updateField} errors={errors} />
  ];
  
  function updateField(field: string, value: string) {
    setFormData(prev => ({ ...prev, [field]: value }));
  }
  
  return (
    <div>
      {steps[currentStep]}
      {/* Navigation buttons */}
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ Cleaner separation
- ✅ Easier to test each step
- ✅ More maintainable

**Cons:**
- ❌ Prop drilling
- ❌ Still centralized state

**Strategy 3: Context + Reducer (Best for Complex Forms)**

Use Context for shared state across steps.

\`\`\`tsx
// Context
interface FormContextValue {
  data: FormData;
  updateField: (field: keyof FormData, value: string) => void;
  currentStep: number;
  nextStep: () => void;
  prevStep: () => void;
  errors: Record<string, string>;
}

const FormContext = createContext<FormContextValue | null>(null);

function useFormContext() {
  const context = useContext(FormContext);
  if (!context) throw new Error('useFormContext must be used within FormProvider');
  return context;
}

// Provider
function FormProvider({ children }: { children: React.ReactNode }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<FormData>({ /* ... */ });
  const [errors, setErrors] = useState({});
  
  function updateField(field: keyof FormData, value: string) {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error
    setErrors(prev => ({ ...prev, [field]: undefined }));
  }
  
  function validate(): boolean {
    // Validation logic
    return true;
  }
  
  function nextStep() {
    if (validate()) {
      setCurrentStep(prev => prev + 1);
    }
  }
  
  function prevStep() {
    setCurrentStep(prev => Math.max(0, prev - 1));
  }
  
  return (
    <FormContext.Provider value={{
      data: formData,
      updateField,
      currentStep,
      nextStep,
      prevStep,
      errors
    }}>
      {children}
    </FormContext.Provider>
  );
}

// Step components (no props needed!)
function Step1() {
  const { data, updateField, errors } = useFormContext();
  
  return (
    <div>
      <input 
        value={data.email}
        onChange={(e) => updateField('email', e.target.value)}
      />
      {errors.email && <p>{errors.email}</p>}
    </div>
  );
}

// Wizard
function MultiStepWizard() {
  return (
    <FormProvider>
      <WizardContent />
    </FormProvider>
  );
}

function WizardContent() {
  const { currentStep, nextStep, prevStep } = useFormContext();
  
  const steps = [<Step1 />, <Step2 />, <Step3 />];
  
  return (
    <div>
      {steps[currentStep]}
      <button onClick={prevStep} disabled={currentStep === 0}>
        Back
      </button>
      <button onClick={nextStep} disabled={currentStep === steps.length - 1}>
        Next
      </button>
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ No prop drilling
- ✅ Steps completely independent
- ✅ Easy to add/remove steps
- ✅ Shared validation logic

**Cons:**
- ❌ More setup
- ❌ Context overhead

**Strategy 4: URL-Based Steps (Best UX)**

Sync steps with URL for bookmarking and browser back/forward.

\`\`\`tsx
import { useRouter, useSearchParams } from 'next/navigation';

function MultiStepForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentStep = Number(searchParams.get('step')) || 1;
  
  const [formData, setFormData] = useState<FormData>({ /* ... */ });
  
  function goToStep(step: number) {
    router.push(\`?step=\${step}\`);
  }
  
  function handleNext() {
    if (validate()) {
      goToStep(currentStep + 1);
    }
  }
  
  function handleBack() {
    goToStep(currentStep - 1);
  }
  
  return (
    <div>
      {currentStep === 1 && <Step1 data={formData} updateData={updateField} />}
      {currentStep === 2 && <Step2 data={formData} updateData={updateField} />}
      {currentStep === 3 && <Step3 data={formData} updateData={updateField} />}
      
      <button onClick={handleBack} disabled={currentStep === 1}>
        Back
      </button>
      <button onClick={handleNext}>
        {currentStep === 3 ? 'Submit' : 'Next'}
      </button>
    </div>
  );
}
\`\`\`

**Pros:**
- ✅ Browser back/forward works
- ✅ Can bookmark/share specific step
- ✅ Better UX

**Cons:**
- ❌ More complex
- ❌ Need to persist data across page loads

**Persisting Data:**

**Option 1: Session Storage**
\`\`\`tsx
useEffect(() => {
  sessionStorage.setItem('formData', JSON.stringify(formData));
}, [formData]);

useEffect(() => {
  const saved = sessionStorage.getItem('formData');
  if (saved) {
    setFormData(JSON.parse(saved));
  }
}, []);
\`\`\`

**Option 2: Server-Side (Draft Saving)**
\`\`\`tsx
// Auto-save draft every 30 seconds
useEffect(() => {
  const interval = setInterval(() => {
    fetch('/api/save-draft', {
      method: 'POST',
      body: JSON.stringify(formData)
    });
  }, 30000);
  
  return () => clearInterval(interval);
}, [formData]);
\`\`\`

**Best Practices:**

1. **Validate before allowing next step**
2. **Show progress indicator**
3. **Allow going back without losing data**
4. **Save draft periodically**
5. **Confirm before leaving page if data entered**
6. **Use URL for step state (better UX)**
7. **Extract validation logic**
8. **Use schema validation (Zod)**

**Interview Insight:**
Discussing multiple strategies and their tradeoffs shows architectural thinking. Mentioning URL-based steps and draft saving demonstrates UX awareness. Context + reducer pattern shows advanced React patterns.`,
    },
  ],
};
