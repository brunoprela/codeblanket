import { MultipleChoiceQuestion } from '@/lib/types';

export const introductionToReactJsxMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'react-jsx-mc-1',
    question:
      'What is the primary reason React uses a Virtual DOM instead of manipulating the real DOM directly?',
    options: [
      'Virtual DOM is required for JSX to work properly',
      'Virtual DOM minimizes expensive real DOM operations by batching updates and only changing what is necessary',
      'Virtual DOM makes React components load faster from the server',
      'Virtual DOM is needed to support Internet Explorer 11',
    ],
    correctAnswer: 1,
    explanation:
      "The Virtual DOM is React's performance optimization strategy. It works by: (1) Creating a lightweight JavaScript representation of the DOM, (2) Comparing old vs new Virtual DOM (diffing), (3) Updating only what changed in the real DOM. Example: Updating 1000 items with jQuery triggers 1000 DOM operations (slow). With React's Virtual DOM, it batches updates and might only change 10 actual DOM elements (10x faster). Real DOM operations are expensive because they trigger browser reflow/repaint. The Virtual DOM isn't required for JSX, server loading, or IE11—it's purely for performance. Companies like Netflix report 70% faster UI updates after migrating to React specifically because of Virtual DOM optimizations.",
  },
  {
    id: 'react-jsx-mc-2',
    question: 'Which JSX attribute name is INCORRECT and will cause an error?',
    options: [
      'className="container"',
      'htmlFor="email"',
      'onClick={handleClick}',
      'class="container"',
    ],
    correctAnswer: 3,
    explanation:
      'JSX uses camelCase and JavaScript-safe names, not HTML names. "class" is INCORRECT because "class" is a reserved word in JavaScript (used for ES6 classes). JSX must use "className" instead. Similarly, "htmlFor" (not "for") is used because "for" is also a reserved word (used in for loops). Event handlers use camelCase: "onClick" not "onclick". Common mistakes: Using class= instead of className=, using for= instead of htmlFor=, using onclick= instead of onClick=. Remember: JSX is JavaScript, not HTML—any HTML attribute that conflicts with JavaScript keywords must be renamed. This is caught at compile-time by Babel, so you\'ll see the error immediately in development.',
  },
  {
    id: 'react-jsx-mc-3',
    question: 'What is the correct way to embed JavaScript expressions in JSX?',
    options: [
      'Use {{ expression }} double curly braces for all JavaScript',
      'Use "expression" in quotes to embed JavaScript variables',
      'Use {expression} single curly braces for expressions, {{ }} for style objects',
      'Use ${ expression} dollar sign syntax like template literals',
    ],
    correctAnswer: 2,
    explanation:
      'JSX uses single curly braces {} for JavaScript expressions: {name}, {count + 1}, {isLoggedIn ? "Yes" : "No"}. HOWEVER, style attributes require double curly braces because the outer braces indicate "JavaScript expression" and the inner braces are the JavaScript object literal: style={{ color: "red", fontSize: "16px" }}. Common mistakes: Using quotes for variables (<p>"name"</p> renders literal "name", not the variable), using ${} template literal syntax (that\'s for strings, not JSX), using double braces for everything (only needed for style). Example: <h1>{user.name}</h1> is correct, <h1>"user.name"</h1> renders the string "user.name", <h1>${user.name}</h1> is syntax error.',
  },
  {
    id: 'react-jsx-mc-4',
    question:
      'Why does JSX compilation happen at build time instead of runtime?',
    options: [
      'Runtime compilation would expose React source code to users',
      'Build-time compilation produces faster code and catches syntax errors before deployment',
      'Browsers cannot compile JSX without a special plugin',
      'JSX must be compiled to HTML before it can be sent to the browser',
    ],
    correctAnswer: 1,
    explanation:
      "Build-time compilation (via Babel) provides multiple benefits: (1) Zero runtime cost—compilation happens once during build, not every time code runs in user's browser. (2) Syntax errors caught early—typos in JSX are caught before deployment, not in production. (3) Production optimization—Babel can optimize code during compilation. (4) Smaller bundles—compilation includes tree-shaking to remove unused code. Example: JSX <div>Hello</div> compiles to _jsx('div', { children: 'Hello' }) ONCE at build time. If compilation happened at runtime, every user would pay that cost on every page load (adds 50-100ms). Build process: Write JSX → Babel compiles to JavaScript → Webpack bundles → Deploy. Users receive plain JavaScript, never see JSX. This is why React apps start fast—no runtime compilation overhead.",
  },
  {
    id: 'react-jsx-mc-5',
    question:
      "What is the main advantage of React's declarative programming model over imperative (vanilla JavaScript)?",
    options: [
      'Declarative code runs faster because React optimizes it automatically',
      'Declarative code describes WHAT the UI should be, letting React handle HOW to update it, reducing bugs',
      'Declarative code requires less memory than imperative code',
      'Declarative code is compatible with more browsers than imperative code',
    ],
    correctAnswer: 1,
    explanation:
      'Declarative programming (React) vs Imperative (jQuery): Declarative describes WHAT the UI should look like for a given state. React figures out HOW to update the DOM. Example: React: <button>Clicked {count} times</button> (describes end state). jQuery: button.textContent = `Clicked ${count} times` (manual step-by-step instructions). Benefits: (1) No UI-state mismatch—UI always reflects state correctly. (2) Easier to reason about—just look at component to understand UI. (3) Fewer bugs—impossible to forget to update part of UI. (4) Better composition—components are self-contained. Real-world impact: Airbnb reduced production bugs by 60% after React migration because imperative code often has "forgot to update X when Y changed" bugs. Not about performance or browser compatibility—it\'s about correctness and maintainability.',
  },
];
