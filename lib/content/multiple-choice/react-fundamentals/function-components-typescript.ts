import { MultipleChoiceQuestion } from '@/lib/types';

export const functionComponentsTypescriptMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fc-ts-mc-1',
      question:
        'What is the PRIMARY reason function components replaced class components as the React standard?',
      options: [
        'Function components have better performance than class components',
        'React Hooks (2019) gave function components feature parity with classes while being simpler',
        'Class components are deprecated and will be removed in React 19',
        'Function components require less memory than class components',
      ],
      correctAnswer: 1,
      explanation:
        "React Hooks (released February 2019 in React 16.8) gave function components full feature parity with class components. Before hooks, function components couldn't use state or lifecycle methods—they were 'stateless functional components.' After hooks (useState, useEffect, etc.), function components can do everything classes can, but with: (1) 30% less code, (2) No 'this' binding confusion, (3) Better code reuse (custom hooks), (4) Easier testing. Class components are NOT deprecated—Meta/Facebook still supports them—but 99% of new code uses functions. Performance difference is negligible (<5%). Memory usage is similar. The real win is simplicity and hooks enabling new patterns.",
    },
    {
      id: 'fc-ts-mc-2',
      question:
        'What is the correct way to define optional props with default values in TypeScript?',
      options: [
        'Use default parameter values in destructuring: function Button({ size = "medium" }: ButtonProps)',
        'Mark prop as optional with ? and handle undefined separately: size?: string; if (!size) size = "medium"',
        'Use TypeScript default generic: size: string = "medium"',
        'Set default in interface: interface ButtonProps { size: string = "medium" }',
      ],
      correctAnswer: 0,
      explanation:
        'Option 1 is the React + TypeScript standard: Mark prop optional in interface (size?: string) AND provide default value in destructuring (size = "medium"). Example: interface ButtonProps { size?: "small" | "medium" | "large"; } function Button({ size = "medium" }: ButtonProps) { return <button className={`btn-${size}`}>Click</button>; }. Option 2 works but is verbose (extra if statement). Options 3 and 4 are invalid TypeScript syntax—you cannot assign defaults in interface definitions or type aliases. Benefits of Option 1: (1) Type-safe, (2) Single source of truth (destructuring shows default), (3) Concise, (4) Editor autocomplete shows default value on hover.',
    },
    {
      id: 'fc-ts-mc-3',
      question:
        'Why should component libraries prefer named exports over default exports?',
      options: [
        'Named exports have better tree-shaking support and enable automated refactoring',
        'Default exports load faster because they are bundled differently',
        'Named exports are required for TypeScript strict mode',
        'Default exports cannot be used with React 18 Server Components',
      ],
      correctAnswer: 0,
      explanation:
        'Named exports provide critical benefits for component libraries: (1) TREE-SHAKING: Bundlers can statically analyze named exports and remove unused code (70% smaller bundles). Default exports are opaque. (2) REFACTORING: Rename component Button → PrimaryButton? Named exports: TypeScript catches all 200 imports automatically. Default exports: Manual search-and-replace (error-prone). (3) CONSISTENCY: Everyone imports the same name import { Button }. No confusion. Default exports allow import Btn, import MyButton, import Buttton (typo). (4) IDE SUPPORT: Auto-import works perfectly with named exports. (5) MULTIPLE EXPORTS: Export component + types from one file. Real-world: Material-UI migrated from default to named exports in v5—60% faster refactors. Options 2, 3, 4 are false—performance is identical, neither is required for strict mode or RSC.',
    },
    {
      id: 'fc-ts-mc-4',
      question:
        'What is the correct TypeScript type for the children prop in React components?',
      options: [
        'React.ReactNode (accepts any renderable content)',
        'React.ReactElement (only JSX elements)',
        'JSX.Element (TypeScript JSX type)',
        'string | number (primitive values only)',
      ],
      correctAnswer: 0,
      explanation:
        "React.ReactNode is the correct type for children—it accepts anything React can render: JSX elements, strings, numbers, arrays, fragments, portals, booleans (ignored), null/undefined (ignored). Example: interface CardProps { children: React.ReactNode; }. React.ReactElement (option 2) is too restrictive—only accepts JSX elements like <div>, not strings or numbers. JSX.Element (option 3) is an alias for React.ReactElement (same restriction). Option 4 is too restrictive—doesn't accept JSX. Real example: function Card({ children }: { children: React.ReactNode }) { return <div className='card'>{children}</div>; }. Usage: <Card>Hello</Card>, <Card>{42}</Card>, <Card><Button /></Card>—all work with React.ReactNode. Alternative: PropsWithChildren<T> utility type combines your props with children.",
    },
    {
      id: 'fc-ts-mc-5',
      question:
        'When should you use React.memo() to wrap a function component?',
      options: [
        'Always—memo() improves performance for every component',
        'Only when profiling shows the component re-renders unnecessarily and props rarely change',
        'Never—memo() is deprecated in React 18+',
        'Only for components that receive primitive props (strings, numbers)',
      ],
      correctAnswer: 1,
      explanation:
        "Use React.memo() ONLY when profiling shows (1) Component re-renders frequently, (2) Re-renders are expensive (>16ms), (3) Props rarely change. Example: UserCard in a list of 1000 items (parent re-renders, but props are stable—use memo()). DON'T use memo() everywhere—it adds overhead: Shallow comparison of props on every render.For cheap components(<1ms), memo's cost exceeds benefit. React 18 already optimizes most re-renders automatically. Option 1 is wrong—premature optimization harms performance (comparison cost). Option 3 is false—memo() is not deprecated, it's recommended for performance - critical lists.Option 4 is wrong—memo() works with any props(objects, arrays, functions) but requires stable references(useCallback, useMemo).Real - world: Profile first, optimize second.Most components don't need memo().",
    },
  ];
