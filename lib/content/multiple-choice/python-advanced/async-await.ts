/**
 * Multiple choice questions for Async/Await & Asynchronous Programming section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const asyncawaitMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does the await keyword do?',
    options: [
      'Starts a new thread',
      'Pauses execution until the awaited coroutine completes',
      'Makes code run in parallel',
      'Blocks all other code',
    ],
    correctAnswer: 1,
    explanation:
      'await pauses the current coroutine and yields control to the event loop, which can run other tasks. Execution resumes when the awaited coroutine completes.',
  },
  {
    id: 'mc2',
    question:
      'What is the difference between asyncio.gather() and creating tasks with asyncio.create_task()?',
    options: [
      'No difference',
      'gather() runs tasks sequentially, create_task() runs concurrently',
      'gather() runs multiple tasks concurrently and waits for all, create_task() starts a task in background',
      'create_task() is deprecated',
    ],
    correctAnswer: 2,
    explanation:
      'asyncio.gather() runs multiple tasks concurrently and waits for all to complete. asyncio.create_task() schedules a single task to run in background and returns immediately, allowing you to do other work.',
  },
  {
    id: 'mc3',
    question: 'When should you use async/await?',
    options: [
      'For CPU-intensive calculations',
      'For I/O-bound operations like network requests',
      'For simple scripts with no I/O',
      'Always, it makes code faster',
    ],
    correctAnswer: 1,
    explanation:
      "Async/await is ideal for I/O-bound operations where tasks spend time waiting (network, databases, files). It doesn't help with CPU-bound operations.",
  },
  {
    id: 'mc4',
    question: 'What happens if you forget to await an async function?',
    options: [
      'It runs synchronously',
      'It returns a coroutine object without executing',
      'It raises an error immediately',
      'It runs in background automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Forgetting await returns a coroutine object without executing the function. You\'ll see a warning: "coroutine was never awaited".',
  },
  {
    id: 'mc5',
    question: 'What is asyncio.run() used for?',
    options: [
      'To run multiple tasks concurrently',
      'To run an async function from synchronous code',
      'To create a new event loop',
      'To convert sync functions to async',
    ],
    correctAnswer: 1,
    explanation:
      'asyncio.run() is the main entry point that runs an async function from synchronous code, creating an event loop, running the coroutine, and cleaning up.',
  },
];
