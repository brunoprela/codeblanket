import { MultipleChoiceQuestion } from '@/lib/types';

export const eventHandlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'event-mc-1',
    question:
      'What is the difference between <button onClick={handleClick}> and <button onClick={handleClick()}>?',
    options: [
      'They are identical—both call handleClick when the button is clicked',
      'The first passes the function reference; the second calls it immediately during render',
      'The second is the correct TypeScript syntax; the first causes type errors',
      'The first only works with arrow functions; the second works with regular functions',
    ],
    correctAnswer: 1,
    explanation:
      'Option 1 (onClick={handleClick}) passes the FUNCTION ITSELF to React, which calls it on click. Option 2 (onClick={handleClick()}) CALLS the function immediately during render and passes the RETURN VALUE (usually undefined) to onClick. Example: const handleClick = () => console.log("Clicked"); <button onClick={handleClick()}>Click</button> logs "Clicked" during render, not on click. The button\'s onClick receives undefined → nothing happens when clicked. Correct: onClick={handleClick} or onClick={() => handleClick()} for passing arguments. This is JavaScript fundamentals: handleClick is a reference, handleClick() is a call. No TypeScript difference. Works with all function types. Understanding this prevents one of the most common React bugs.',
  },
  {
    id: 'event-mc-2',
    question:
      'How do you correctly pass an argument to an event handler in React?',
    options: [
      'onClick={handleClick(id)}',
      'onClick={() => handleClick(id)}',
      'onClick={handleClick.bind(id)}',
      'onClick={"handleClick(id)"}',
    ],
    correctAnswer: 1,
    explanation:
      "Option 2 is correct: onClick={() => handleClick(id)}. The arrow function is passed to onClick, and React calls it on click, which then calls handleClick(id). Option 1 (onClick={handleClick(id)}) calls handleClick immediately during render—wrong. Option 3 (onClick={handleClick.bind(id)}) is invalid syntax—bind requires null as first arg: handleClick.bind(null, id). Option 4 treats function call as string—never works. Alternative valid syntaxes: Currying: const handleClick = (id) => () => {...}; onClick={handleClick(id)}. Data attributes: <button data-id={id} onClick={(e) => handleClick(e.currentTarget.dataset.id)}>. Arrow wrapper is standard (99% of cases) because it's clear, TypeScript-friendly, and allows multiple statements.",
  },
  {
    id: 'event-mc-3',
    question: 'What does e.stopPropagation() do in a React event handler?',
    options: [
      'Prevents the default browser behavior (like form submission)',
      'Stops the event from bubbling up to parent elements',
      'Cancels all other event handlers on the same element',
      'Stops React from re-rendering the component',
    ],
    correctAnswer: 1,
    explanation:
      "e.stopPropagation() stops event bubbling—prevents the event from traveling up the DOM tree to parent elements. Example: <div onClick={handleParent}><button onClick={(e) => { e.stopPropagation(); handleChild(); }}>Click</button></div> → clicking button calls handleChild() but NOT handleParent(). Without stopPropagation(), both fire (event bubbles from button to div). Option 1 describes e.preventDefault() (different method). Option 3 is false—other handlers on same element still run. Option 4 is false—stopPropagation doesn't affect React rendering. Use cases: Modals (click modal content, don't close), dropdowns, nested clickable areas. DON'T overuse—breaks analytics, accessibility, event delegation. Use when child needs different behavior than parent.",
  },
  {
    id: 'event-mc-4',
    question: 'What is debouncing and why is it useful for search inputs?',
    options: [
      'Making multiple API calls simultaneously to get results faster',
      'Waiting a short delay after user stops typing before executing the search',
      'Caching search results locally to avoid API calls',
      'Converting user input to lowercase before searching',
    ],
    correctAnswer: 1,
    explanation:
      'Debouncing = wait N milliseconds after user stops typing, THEN execute. Example: User types "react" (5 keystrokes). WITHOUT debounce: 5 API calls (r, re, rea, reac, react). WITH 300ms debounce: 1 API call (react, after 300ms pause). How it works: setTimeout() starts on each keystroke, resets if user types again. Only fires if 300ms pass without typing. Implementation: useEffect(() => { const timer = setTimeout(() => setDebouncedQuery(query), 300); return () => clearTimeout(timer); }, [query]). Benefits: 80-95% fewer API calls, lower server load, lower cost, prevents flickering results. Typical delay: 300-500ms (balances UX and cost). NOT caching (option 3), NOT simultaneous calls (option 1), NOT text transformation (option 4). Essential for search, autosave, validation.',
  },
  {
    id: 'event-mc-5',
    question:
      'In a React event handler, what is the difference between e.target and e.currentTarget?',
    options: [
      'e.target is the element that was clicked; e.currentTarget is the element with the onClick handler',
      'e.target is for mouse events; e.currentTarget is for keyboard events',
      'They are aliases for the same thing—no difference',
      'e.target is the parent element; e.currentTarget is the child element',
    ],
    correctAnswer: 0,
    explanation:
      "e.target = the element that was ACTUALLY CLICKED (innermost element). e.currentTarget = the element WITH THE EVENT HANDLER (the element you attached onClick to). Example: <div onClick={handleClick}><button>Click</button></div> → If button clicked: e.target = button, e.currentTarget = div. If div clicked directly: e.target = div, e.currentTarget = div. Use case: Modal backdrop: <div onClick={(e) => { if (e.target === e.currentTarget) close(); }}> → only closes if backdrop itself clicked, not children. Options 2, 3, 4 are false. Both properties work for all event types. They're different objects. Understanding this is critical for: event delegation, modal/dropdown patterns, preventing unwanted parent triggers. e.target can be ANY descendant, e.currentTarget is always the element you attached the handler to.",
  },
];
