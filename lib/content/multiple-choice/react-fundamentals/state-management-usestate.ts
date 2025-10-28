import { MultipleChoiceQuestion } from '@/lib/types';

export const stateManagementUseStateMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'state-mc-1',
    question:
      'What is the PRIMARY difference between regular JavaScript variables and React state variables?',
    options: [
      'State variables are stored in memory while regular variables are garbage collected',
      'State variables persist across re-renders and trigger component re-renders when updated',
      'State variables can only store primitive types, not objects or arrays',
      'State variables are faster to access than regular variables',
    ],
    correctAnswer: 1,
    explanation:
      "The key difference is that state variables (1) PERSIST across re-renders—their values don't reset when the component function runs again, (2) TRIGGER re-renders when updated via setState. Regular variables reset to their initial value every render. Example: let count = 0; count++; in a component resets count to 0 on every render. const [count, setCount] = useState(0); setCount(count + 1); persists and triggers re-render. Both are stored in memory (option 1 false). State can store ANY type including objects/arrays (option 3 false). Performance is similar (option 4 false). React manages state persistence using a hooks array internally. This is why useState must be called at the top level—React relies on call order to track state.",
  },
  {
    id: 'state-mc-2',
    question:
      'Why does the following code only increment count by 1 instead of 3: setCount(count + 1); setCount(count + 1); setCount(count + 1);?',
    options: [
      'React limits the number of state updates per render to prevent infinite loops',
      'The count variable is a constant and cannot be changed three times',
      'All three calls read the same count value due to closure, and React batches them',
      'useState only processes the first setState call and ignores subsequent calls',
    ],
    correctAnswer: 2,
    explanation:
      "This is a JavaScript closure issue combined with React batching. When the function runs, all three setCount calls capture the SAME count value (e.g., 0). So all three call setCount(0 + 1), setting count to 1. React then batches these updates—since they're all setting count to the same value (1), the result is count = 1, not 3. Option 1 is false—no artificial limit. Option 2 misunderstands const—you can't reassign count, but setCount creates NEW state. Option 4 is false—React processes all setState calls. SOLUTION: Use functional updates: setCount(prev => prev + 1) three times. Each callback receives the LATEST state: first gets 0 → 1, second gets 1 → 2, third gets 2 → 3. Result: count = 3. This demonstrates why functional updates are critical when new state depends on old state.",
  },
  {
    id: 'state-mc-3',
    question:
      'What is the correct way to update a single property in a state object without losing other properties?',
    options: [
      'user.name = "Alice"; setUser(user);',
      'setUser({ name: "Alice" });',
      'setUser({ ...user, name: "Alice" });',
      'setUser(user => user.name = "Alice");',
    ],
    correctAnswer: 2,
    explanation:
      'Option 3 is correct: setUser({ ...user, name: "Alice" }). The spread operator (...user) copies all existing properties, then name: "Alice" overrides the name property. Options 1 and 4 mutate the original object—React won\'t detect the change because the object reference stays the same (user === user → no re-render). Option 2 replaces the entire user object with { name: "Alice" }, losing all other properties (email, age, etc.). Why immutability matters: React uses reference equality (oldState === newState) for performance. Mutation: same reference → no re-render. New object: different reference → re-render. Example: const user = { name: "Bob", age: 30 }; setUser({ ...user, name: "Alice" }) results in { name: "Alice", age: 30 }. This pattern is fundamental to React—always create new objects/arrays when updating state.',
  },
  {
    id: 'state-mc-4',
    question: 'When should you use lazy initialization in useState?',
    options: [
      'Always—lazy initialization improves performance for all useState calls',
      'When the initial state is a complex object or involves an expensive calculation',
      'Only when using TypeScript to ensure proper type inference',
      'Never—lazy initialization is deprecated in React 18+',
    ],
    correctAnswer: 1,
    explanation:
      "Use lazy initialization when computing initial state is expensive. Example: const [data, setState] = useState(() => JSON.parse(localStorage.getItem('data'))). Without lazy initialization, JSON.parse runs on EVERY render (expensive). With lazy function: (() => ...), it only runs ONCE on first render. WHEN TO USE: (1) localStorage/sessionStorage reads, (2) Complex calculations, (3) Creating large objects/arrays. WHEN NOT TO USE: Simple values like useState(0) or useState('')—the function overhead costs more than the simple value. Option 1 is false—lazy initialization adds overhead for simple values. Option 3 is false—works with or without TypeScript. Option 4 is false—not deprecated, still recommended. Real-world: const [todos, setTodos] = useState(() => JSON.parse(localStorage.getItem('todos') || '[]')); saves parsing on every render (could be 10ms+). This is a performance optimization for expensive initial state.",
  },
  {
    id: 'state-mc-5',
    question:
      'What is the best practice for managing a form with 15 related fields?',
    options: [
      'Create 15 separate useState calls for maximum flexibility',
      'Use a single useState with an object containing all 15 fields',
      'Store form data in localStorage instead of state',
      'Use useReducer instead—useState cannot handle more than 10 fields',
    ],
    correctAnswer: 1,
    explanation:
      'For forms with multiple related fields, use a single useState object: const [formData, setFormData] = useState({ name: "", email: "", ... }). Benefits: (1) Single source of truth—all form data in one place, (2) Easier submission—already an object to send to API, (3) Easier reset—setFormData(INITIAL_STATE) instead of 15 separate calls, (4) Type-safe with TypeScript interface, (5) Reusable change handler: handleChange(field, value) updates any field. Option 1 (15 useState) works but creates: 15 lines of state declarations, 15 separate handlers, verbose code. Option 3 confuses persistence with state management—localStorage is for persistence, state is for UI. Option 4 is false—useState has no field limit; useReducer is an alternative but not required. Performance is identical (both trigger one re-render per update). Choose based on code quality: For 5+ related fields, single object is cleaner.',
  },
];
