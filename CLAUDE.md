# CLAUDE.md - BinSearch Frontend Development Guide

## 🎯 Project Overview

CodeBlanket is an interactive learning platform focused on mastering algorithms and data structures through hands-on Python coding practice. All code execution happens in the browser using Pyodide (Python compiled to WebAssembly). Currently supports Binary Search and Two Pointers algorithm topics.

## 🚀 Development Commands

### Essential Commands

- **Start dev server**: `npm run dev` (opens on http://localhost:3000)
- **Build for production**: `npm run build`
- **Start production server**: `npm start`

### Code Quality

- **Type checking**: `npm run type-check` - Verify TypeScript types
- **Linting**: `npm run lint` - Check code quality (zero warnings policy)
- **Linting (auto-fix)**: `npm run lint:fix` - Fix auto-fixable issues
- **Formatting**: `npm run format` - Format all code with Prettier
- **Format check**: `npm run format-check` - Check if code is formatted
- **Full validation**: `npm run validate` - Run all checks (type + lint + format)

### Workflow

```bash
# Before committing
npm run validate

# Quick format fix
npm run format

# Fix linting issues
npm run lint:fix
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── layout.tsx                  # Root layout with Pyodide script
│   ├── page.tsx                    # Home page with problem categories
│   ├── problems/
│   │   ├── [id]/page.tsx          # Individual problem page
│   │   └── page.tsx               # All problems listing
│   ├── topics/
│   │   └── [slug]/page.tsx        # Problems by topic
│   ├── monaco-config.ts           # Monaco editor configuration
│   └── globals.css                # Global styles
├── components/
│   └── PythonCodeRunner.tsx       # Monaco editor + Pyodide test runner
├── lib/
│   ├── helpers/
│   │   └── storage.ts             # LocalStorage utilities (completion, code)
│   ├── hooks/
│   │   ├── usePyodide.ts          # Custom hook for Pyodide loading
│   │   └── useCodeStorage.ts      # Custom hook for code persistence
│   ├── utils/
│   │   └── formatText.tsx         # Text formatting utilities
│   ├── problems/
│   │   ├── binary-search.ts       # Binary search problem definitions
│   │   ├── two-pointers.ts        # Two pointers problem definitions
│   │   └── index.ts               # Problem exports and categories
│   ├── pyodide.ts                 # Pyodide singleton loader
│   └── types.ts                   # TypeScript interfaces
├── public/                         # Static assets
└── ...config files
```

## 🎨 Code Style Guidelines

### General Principles

- **Framework**: Next.js 15 with TypeScript and React 19
- **Styling**: Tailwind CSS utility-first approach
- **Formatting**: Single quotes, 2 space indentation, 80 char line width
- **Components**: React functional components with hooks
- **File Naming**: kebab-case for files, PascalCase for components

### TypeScript

- Use explicit typing with TypeScript
- Avoid `any` where possible (use `unknown` if needed)
- Prefer interfaces for object shapes
- Use type inference when obvious
- Prefix unused variables with underscore: `_unusedVar`

### React Best Practices

- One component per file
- Keep components focused (single responsibility)
- Use descriptive component names
- Extract complex logic into custom hooks
- Prefer composition over prop drilling

### Naming Conventions

- **Variables/Functions**: camelCase (`getUserData`, `isLoading`)
- **Components**: PascalCase (`PythonCodeRunner`, `ProblemCard`)
- **Types/Interfaces**: PascalCase (`Problem`, `TestResult`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_RETRIES`, `API_URL`)
- **Files**: kebab-case (`python-code-runner.tsx`, `use-pyodide.ts`)

### Imports

```typescript
// 1. External libraries (React, third-party packages)
import { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';

// 2. Internal modules (grouped by category, use @ alias)
import { Problem } from '@/lib/types';
import { getPyodide } from '@/lib/pyodide';
import { usePyodide } from '@/lib/hooks/usePyodide';
import { getCompletedProblems } from '@/lib/helpers/storage';
import { formatText } from '@/lib/utils/formatText';
import { PythonCodeRunner } from '@/components/PythonCodeRunner';

// 3. Relative imports
import './styles.css';
```

## 🏗️ Component Structure

### Recommended Pattern

```tsx
'use client'; // Only if needed (client component)

import { useState } from 'react';
import { SomeType } from '@/lib/types';

interface ComponentProps {
  title: string;
  onAction?: () => void;
}

export function ComponentName({ title, onAction }: ComponentProps) {
  // 1. Hooks
  const [state, setState] = useState<SomeType | null>(null);

  // 2. Derived state
  const isReady = state !== null;

  // 3. Event handlers
  const handleClick = () => {
    onAction?.();
  };

  // 4. Effects
  useEffect(() => {
    // Effect logic
  }, []);

  // 5. Early returns
  if (!isReady) {
    return <div>Loading...</div>;
  }

  // 6. Main render
  return (
    <div>
      <h1>{title}</h1>
      <button onClick={handleClick}>Action</button>
    </div>
  );
}
```

## 🐍 Pyodide Integration

### Loading Pyodide

```typescript
import { getPyodide } from '@/lib/pyodide';

// Always use the singleton loader
const pyodide = await getPyodide();
```

### Running Python Code

```typescript
// Execute code
await pyodide.runPythonAsync(pythonCode);

// Get result
const result = await pyodide.runPythonAsync(`
import json
result = my_function(arg1, arg2)
json.dumps(result)
`);

const actualValue = JSON.parse(result);
```

### Best Practices

- Load Pyodide once globally (using singleton pattern)
- Show loading state (first load is ~10MB)
- Handle errors gracefully
- Use Web Workers for heavy computation (future enhancement)

## 📚 Adding New Problems

### Problem Definition

```typescript
// lib/problems/binary-search.ts
const newProblem: Problem = {
  id: 'problem-slug',
  title: 'Problem Title',
  difficulty: 'Easy', // 'Easy' | 'Medium' | 'Hard'
  order: 4,
  description: `
Problem description in markdown format.
Can include **bold**, \`code\`, etc.
  `,
  examples: [
    {
      input: 'nums = [1, 2, 3], target = 2',
      output: '1',
      explanation: 'Optional explanation',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^4',
    'nums is sorted in ascending order',
  ],
  hints: [
    'Think about dividing the search space',
    'What happens when you find the middle element?',
  ],
  starterCode: `def solution(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    # Your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3], 2],
      expected: 1,
    },
    // Add more test cases
  ],
  solution: `# Optional solution for reference`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
};
```

## 🎨 Styling Guidelines

### Tailwind CSS

- Use utility classes for styling
- Prefer composition over custom CSS
- Use consistent spacing scale (4, 8, 12, 16, 24, 32...)
- Follow mobile-first responsive design

### Color Scheme (Dracula Theme)

- Background: `#282a36`
- Foreground: `#f8f8f2`
- Current Line: `#44475a`
- Comment: `#6272a4`
- Cyan: `#8be9fd`
- Green: `#50fa7b`
- Orange: `#ffb86c`
- Pink: `#ff79c6`
- Purple: `#bd93f9`
- Red: `#ff5555`
- Yellow: `#f1fa8c`

### Common Patterns

```tsx
// Card
<div className="bg-white rounded-lg shadow-md p-6">

// Button
<button className="px-6 py-2.5 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors">

// Input
<input className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500">
```

## 🐛 Error Handling

### Client Components

```typescript
// Loading states
if (isLoading) return <Spinner />;

// Error states
if (error) {
  return (
    <div className="p-4 bg-red-50 border border-red-200 rounded">
      <p className="text-red-800">Error: {error.message}</p>
      <button onClick={retry}>Retry</button>
    </div>
  );
}
```

### Pyodide Errors

```typescript
try {
  const result = await pyodide.runPythonAsync(code);
} catch (error) {
  // Python syntax errors, runtime errors, etc.
  console.error('Python execution failed:', error);
}
```

## 🚀 Performance Tips

1. **Lazy load Monaco Editor** - It's already code-split by `@monaco-editor/react`
2. **Cache Pyodide** - Use the singleton pattern (already implemented)
3. **Optimize images** - Use Next.js Image component
4. **Minimize re-renders** - Use React.memo for expensive components
5. **Code splitting** - Next.js does this automatically for routes

## 📝 Before Committing

Always run before pushing:

```bash
npm run validate
```

This ensures:

- ✅ No TypeScript errors
- ✅ No linting issues
- ✅ Code is properly formatted

## 🎓 Learning Resources

- **Binary Search**: https://en.wikipedia.org/wiki/Binary_search_algorithm
- **Next.js Docs**: https://nextjs.org/docs
- **Pyodide Docs**: https://pyodide.org/en/stable/
- **Monaco Editor**: https://microsoft.github.io/monaco-editor/
- **Tailwind CSS**: https://tailwindcss.com/docs

## 💡 Implemented Features

- ✅ User progress tracking (localStorage)
- ✅ Multiple algorithm topics (Binary Search, Two Pointers)
- ✅ Code persistence in localStorage
- ✅ Problem completion tracking with status filtering
- ✅ Search, filter, and sort problems
- ✅ Dark mode with Dracula color scheme

## 💡 Future Enhancements

Ideas for expanding the platform:

- [ ] More algorithm topics (sliding window, dynamic programming, etc.)
- [ ] Algorithm visualizations
- [ ] Solution explanations with step-by-step breakdowns
- [ ] Multiple language support (JavaScript, TypeScript)
- [ ] Community solutions
- [ ] Difficulty progression system
- [ ] Problem hints with progressive disclosure

---

**Happy Coding! 🚀** Build to learn, learn to build.
