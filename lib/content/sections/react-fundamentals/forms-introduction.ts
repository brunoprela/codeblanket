export const formsIntroduction = {
  title: 'Forms Introduction',
  id: 'forms-introduction',
  content: `
# Forms Introduction

## Introduction

**Forms** are the primary way users interact with web applications—logging in, searching, creating content, checking out, and more. In React, forms work differently than traditional HTML forms, giving you precise control over form state and behavior.

This section covers React's two approaches to forms:
- **Controlled components** (React manages form state)
- **Uncontrolled components** (DOM manages form state)

You'll learn when to use each, how to handle various input types, and best practices for building robust forms.

### Why React Forms are Different

In traditional HTML, form elements maintain their own state:
\`\`\`html
<!-- Traditional HTML form -->
<form>
  <input type="text" name="username" />
  <!-- Input manages its own value -->
</form>
\`\`\`

In React, you typically want React to control the value:
\`\`\`tsx
// React controlled component
function Form() {
  const [username, setUsername] = useState('');
  
  return (
    <input 
      type="text" 
      value={username}
      onChange={(e) => setUsername(e.target.value)}
    />
  );
}
\`\`\`

## Controlled Components

A **controlled component** is an input whose value is controlled by React state.

### Basic Text Input

\`\`\`tsx
function NameForm() {
  const [name, setName] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();  // Prevent page reload
    console.log('Submitted:', name);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <label>
        Name:
        <input 
          type="text"
          value={name}                                // Controlled by state
          onChange={(e) => setName(e.target.value)}   // Update state on change
        />
      </label>
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**How it works:**
1. State holds the current value
2. Input displays that value
3. onChange updates the state
4. React re-renders with new value

**Benefits:**
- ✅ React state is single source of truth
- ✅ Can validate/transform input immediately
- ✅ Can conditionally disable submit button
- ✅ Easy to reset form
- ✅ Can derive other UI from input value

### Multiple Inputs

Handle multiple inputs with separate state variables or an object:

**Option 1: Separate state variables**
\`\`\`tsx
function SignupForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    console.log({ email, password, confirmPassword });
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
      />
      <input 
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
      />
      <input 
        type="password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
        placeholder="Confirm Password"
      />
      <button type="submit">Sign Up</button>
    </form>
  );
}
\`\`\`

**Option 2: Object state (recommended for many inputs)**
\`\`\`tsx
interface FormData {
  email: string;
  password: string;
  confirmPassword: string;
}

function SignupForm() {
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
    confirmPassword: ''
  });
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value  // Update the field that changed
    }));
  }
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    console.log(formData);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="email"
        name="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="Email"
      />
      <input 
        type="password"
        name="password"
        value={formData.password}
        onChange={handleChange}
        placeholder="Password"
      />
      <input 
        type="password"
        name="confirmPassword"
        value={formData.confirmPassword}
        onChange={handleChange}
        placeholder="Confirm Password"
      />
      <button type="submit">Sign Up</button>
    </form>
  );
}
\`\`\`

## Input Types

### Text and Textarea

\`\`\`tsx
function TextInputs() {
  const [text, setText] = useState('');
  const [bio, setBio] = useState('');
  
  return (
    <form>
      {/* Single-line text input */}
      <input 
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Name"
      />
      
      {/* Multi-line textarea */}
      <textarea 
        value={bio}
        onChange={(e) => setBio(e.target.value)}
        placeholder="Tell us about yourself"
        rows={4}
      />
    </form>
  );
}
\`\`\`

**Note:** In React, textarea uses \`value\` prop (not children like HTML).

\`\`\`tsx
// ❌ HTML way (doesn't work in React)
<textarea>Default text here</textarea>

// ✅ React way
<textarea value={bio} onChange={handleChange} />
\`\`\`

### Checkbox

\`\`\`tsx
function CheckboxExample() {
  const [agreed, setAgreed] = useState(false);
  
  return (
    <label>
      <input 
        type="checkbox"
        checked={agreed}                          // Use "checked", not "value"
        onChange={(e) => setAgreed(e.target.checked)}  // Use "checked", not "value"
      />
      I agree to terms and conditions
    </label>
  );
}

// Multiple checkboxes
function MultiCheckbox() {
  const [hobbies, setHobbies] = useState<string[]>([]);
  
  function handleToggle(hobby: string) {
    setHobbies(prev => 
      prev.includes(hobby)
        ? prev.filter(h => h !== hobby)  // Remove if present
        : [...prev, hobby]                // Add if not present
    );
  }
  
  return (
    <div>
      <label>
        <input 
          type="checkbox"
          checked={hobbies.includes('reading')}
          onChange={() => handleToggle('reading')}
        />
        Reading
      </label>
      <label>
        <input 
          type="checkbox"
          checked={hobbies.includes('gaming')}
          onChange={() => handleToggle('gaming')}
        />
        Gaming
      </label>
      <label>
        <input 
          type="checkbox"
          checked={hobbies.includes('cooking')}
          onChange={() => handleToggle('cooking')}
        />
        Cooking
      </label>
      <p>Selected: {hobbies.join(', ')}</p>
    </div>
  );
}
\`\`\`

### Radio Buttons

\`\`\`tsx
function RadioExample() {
  const [size, setSize] = useState('medium');
  
  return (
    <div>
      <p>Select size:</p>
      <label>
        <input 
          type="radio"
          name="size"
          value="small"
          checked={size === 'small'}
          onChange={(e) => setSize(e.target.value)}
        />
        Small
      </label>
      <label>
        <input 
          type="radio"
          name="size"
          value="medium"
          checked={size === 'medium'}
          onChange={(e) => setSize(e.target.value)}
        />
        Medium
      </label>
      <label>
        <input 
          type="radio"
          name="size"
          value="large"
          checked={size === 'large'}
          onChange={(e) => setSize(e.target.value)}
        />
        Large
      </label>
      <p>Selected: {size}</p>
    </div>
  );
}
\`\`\`

### Select Dropdown

\`\`\`tsx
function SelectExample() {
  const [country, setCountry] = useState('us');
  
  return (
    <select 
      value={country}
      onChange={(e) => setCountry(e.target.value)}
    >
      <option value="us">United States</option>
      <option value="uk">United Kingdom</option>
      <option value="ca">Canada</option>
      <option value="au">Australia</option>
    </select>
  );
}

// Multi-select
function MultiSelectExample() {
  const [selected, setSelected] = useState<string[]>([]);
  
  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const options = e.target.selectedOptions;
    const values = Array.from(options).map(opt => opt.value);
    setSelected(values);
  }
  
  return (
    <select 
      multiple 
      value={selected}
      onChange={handleChange}
    >
      <option value="apple">Apple</option>
      <option value="banana">Banana</option>
      <option value="cherry">Cherry</option>
    </select>
  );
}
\`\`\`

**Note:** React's select uses \`value\` prop (not \`selected\` attribute on options).

## Uncontrolled Components

**Uncontrolled components** let the DOM manage form state. You access values via refs when needed.

### Basic Uncontrolled Input

\`\`\`tsx
import { useRef } from 'react';

function UncontrolledForm() {
  const nameRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const name = nameRef.current?.value;
    console.log('Submitted:', name);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="text"
        ref={nameRef}
        defaultValue="John"  // Use defaultValue, not value
      />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

**Key differences:**
- Use \`ref\` instead of \`value\` + \`onChange\`
- Use \`defaultValue\` for initial value (not \`value\`)
- Access current value via \`ref.current.value\`

### Multiple Uncontrolled Inputs

\`\`\`tsx
function UncontrolledMultiple() {
  const emailRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const data = {
      email: emailRef.current?.value,
      password: passwordRef.current?.value
    };
    console.log(data);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input type="email" ref={emailRef} defaultValue="" />
      <input type="password" ref={passwordRef} defaultValue="" />
      <button type="submit">Submit</button>
    </form>
  );
}

// Or use FormData API
function UncontrolledFormData() {
  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data = {
      email: formData.get('email'),
      password: formData.get('password')
    };
    console.log(data);
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input type="email" name="email" defaultValue="" />
      <input type="password" name="password" defaultValue="" />
      <button type="submit">Submit</button>
    </form>
  );
}
\`\`\`

### File Input (Always Uncontrolled)

File inputs are always uncontrolled because their value is read-only for security reasons.

\`\`\`tsx
function FileUpload() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  }
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (file) {
      console.log('Uploading:', file.name);
      // Upload file logic here
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="file"
        ref={fileRef}
        onChange={handleChange}
        accept="image/*"
      />
      {file && <p>Selected: {file.name}</p>}
      <button type="submit" disabled={!file}>
        Upload
      </button>
    </form>
  );
}
\`\`\`

## Controlled vs Uncontrolled: When to Use Each

### Controlled Components (Recommended Default)

**Use when:**
- ✅ Need to validate input on every keystroke
- ✅ Need to disable submit based on input
- ✅ Need to format input (e.g., phone number)
- ✅ Need conditional UI based on input
- ✅ Have complex form logic
- ✅ Need to programmatically set/clear values

\`\`\`tsx
// Example: Disable submit if password too short
function ControlledLogin() {
  const [password, setPassword] = useState('');
  
  return (
    <form>
      <input 
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button disabled={password.length < 8}>
        Login
      </button>
      {password.length > 0 && password.length < 8 && (
        <p>Password must be at least 8 characters</p>
      )}
    </form>
  );
}
\`\`\`

### Uncontrolled Components

**Use when:**
- ✅ Simple forms without validation
- ✅ File inputs (required)
- ✅ Integrating with non-React libraries
- ✅ Form values only needed on submit
- ✅ Performance optimization (rare)

\`\`\`tsx
// Example: Simple contact form
function ContactForm() {
  const formRef = useRef<HTMLFormElement>(null);
  
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const formData = new FormData(formRef.current!);
    const data = Object.fromEntries(formData);
    console.log(data);
  }
  
  return (
    <form ref={formRef} onSubmit={handleSubmit}>
      <input name="name" defaultValue="" />
      <input name="email" defaultValue="" />
      <textarea name="message" defaultValue="" />
      <button type="submit">Send</button>
    </form>
  );
}
\`\`\`

**Comparison:**

| Aspect | Controlled | Uncontrolled |
|--------|-----------|--------------|
| State location | React state | DOM |
| Value access | Immediate | On submit or via ref |
| Validation | Real-time | On submit |
| Default value | \`value={state}\` | \`defaultValue="..."\` |
| Change handler | Required (\`onChange\`) | Optional |
| Complexity | More code | Less code |
| Re-renders | On every keystroke | Only on submit |

## Form Submission

Always prevent default form behavior and handle submission in React:

\`\`\`tsx
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();  // ← Critical! Prevents page reload
    
    setIsSubmitting(true);
    setError('');
    
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      const data = await response.json();
      console.log('Success:', data);
      // Redirect or update auth state
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsSubmitting(false);
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
      />
      <input 
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
      />
      {error && <p className="error">{error}</p>}
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
}
\`\`\`

## Complete Form Example

Here's a comprehensive form with validation, error handling, and proper TypeScript:

\`\`\`tsx
interface FormData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  agreeToTerms: boolean;
  country: string;
}

interface FormErrors {
  username?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  agreeToTerms?: string;
}

function SignupForm() {
  const [formData, setFormData] = useState<FormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    agreeToTerms: false,
    country: 'us'
  });
  
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  function handleChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) {
    const { name, value, type } = e.target;
    
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' 
        ? (e.target as HTMLInputElement).checked 
        : value
    }));
    
    // Clear error for this field
    if (errors[name as keyof FormErrors]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  }
  
  function validate(): FormErrors {
    const newErrors: FormErrors = {};
    
    if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }
    
    if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }
    
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    if (!formData.agreeToTerms) {
      newErrors.agreeToTerms = 'You must agree to terms';
    }
    
    return newErrors;
  }
  
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    const newErrors = validate();
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const response = await fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error('Signup failed');
      }
      
      console.log('Success!');
      // Redirect or show success message
    } catch (err) {
      setErrors({ 
        email: err instanceof Error ? err.message : 'Unknown error' 
      });
    } finally {
      setIsSubmitting(false);
    }
  }
  
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Username:
          <input 
            type="text"
            name="username"
            value={formData.username}
            onChange={handleChange}
          />
        </label>
        {errors.username && <p className="error">{errors.username}</p>}
      </div>
      
      <div>
        <label>
          Email:
          <input 
            type="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
          />
        </label>
        {errors.email && <p className="error">{errors.email}</p>}
      </div>
      
      <div>
        <label>
          Password:
          <input 
            type="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
          />
        </label>
        {errors.password && <p className="error">{errors.password}</p>}
      </div>
      
      <div>
        <label>
          Confirm Password:
          <input 
            type="password"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
          />
        </label>
        {errors.confirmPassword && (
          <p className="error">{errors.confirmPassword}</p>
        )}
      </div>
      
      <div>
        <label>
          Country:
          <select 
            name="country"
            value={formData.country}
            onChange={handleChange}
          >
            <option value="us">United States</option>
            <option value="uk">United Kingdom</option>
            <option value="ca">Canada</option>
          </select>
        </label>
      </div>
      
      <div>
        <label>
          <input 
            type="checkbox"
            name="agreeToTerms"
            checked={formData.agreeToTerms}
            onChange={handleChange}
          />
          I agree to the terms and conditions
        </label>
        {errors.agreeToTerms && (
          <p className="error">{errors.agreeToTerms}</p>
        )}
      </div>
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Signing up...' : 'Sign Up'}
      </button>
    </form>
  );
}
\`\`\`

## Best Practices

1. **Always use preventDefault on form submit**
2. **Use controlled components by default** (easier to manage)
3. **Validate on submit, not on every keystroke** (better UX)
4. **Show clear error messages** near the relevant input
5. **Disable submit button** while submitting
6. **Provide visual feedback** (loading state, success/error messages)
7. **Use proper input types** (email, tel, etc.) for mobile keyboards
8. **Use labels** for accessibility
9. **Clear errors** when user starts typing in that field
10. **Use TypeScript** for form data and errors

## What's Next?

You've learned the basics of React forms! Next, you'll dive into **React Developer Tools**—how to inspect components, debug state/props, profile performance, and become a React debugging expert. Forms + DevTools = Building and debugging real applications! 🚀
`,
};
