/**
 * Python Async Module
 * Module 6: Asynchronous Python Mastery
 * Complete async/await, asyncio, and concurrent programming
 */

import { Module } from '../../types';

// Import sections (all 16 completed!)
import { concurrencyFundamentals } from '../sections/python-async/concurrency-fundamentals';
import { eventLoopDeepDive } from '../sections/python-async/event-loop-deep-dive';
import { coroutinesAsyncAwait } from '../sections/python-async/coroutines-async-await';
import { tasksAndFutures } from '../sections/python-async/tasks-and-futures';
import { asyncContextManagersGenerators } from '../sections/python-async/async-context-managers-generators';
import { asyncioBuiltinFunctions } from '../sections/python-async/asyncio-builtin-functions';
import { asyncHttpAiohttp } from '../sections/python-async/async-http-aiohttp';
import { asyncDatabaseOperations } from '../sections/python-async/async-database-operations';
import { asyncFileIO } from '../sections/python-async/async-file-io';
import { errorHandlingAsync } from '../sections/python-async/error-handling-async';
import { threadingVsMultiprocessingVsAsync } from '../sections/python-async/threading-vs-multiprocessing-vs-async';
import { concurrentFuturesModule } from '../sections/python-async/concurrent-futures-module';
import { raceConditionsSynchronization } from '../sections/python-async/race-conditions-synchronization';
import { debuggingAsyncApplications } from '../sections/python-async/debugging-async-applications';
import { asyncDesignPatterns } from '../sections/python-async/async-design-patterns';
import { productionAsyncPatterns } from '../sections/python-async/production-async-patterns';

// Import quizzes (all 16 completed!)
import { concurrencyFundamentalsQuiz } from '../quizzes/python-async/concurrency-fundamentals';
import { eventLoopDeepDiveQuiz } from '../quizzes/python-async/event-loop-deep-dive';
import { coroutinesAsyncAwaitQuiz } from '../quizzes/python-async/coroutines-async-await';
import { tasksAndFuturesQuiz } from '../quizzes/python-async/tasks-and-futures';
import { asyncContextManagersGeneratorsQuiz } from '../quizzes/python-async/async-context-managers-generators';
import { asyncioBuiltinFunctionsQuiz } from '../quizzes/python-async/asyncio-builtin-functions';
import { asyncHttpAiohttpQuiz } from '../quizzes/python-async/async-http-aiohttp';
import { asyncDatabaseOperationsQuiz } from '../quizzes/python-async/async-database-operations';
import { asyncFileIOQuiz } from '../quizzes/python-async/async-file-io';
import { errorHandlingAsyncQuiz } from '../quizzes/python-async/error-handling-async';
import { threadingVsMultiprocessingVsAsyncQuiz } from '../quizzes/python-async/threading-vs-multiprocessing-vs-async';
import { concurrentFuturesModuleQuiz } from '../quizzes/python-async/concurrent-futures-module';
import { raceConditionsSynchronizationQuiz } from '../quizzes/python-async/race-conditions-synchronization';
import { debuggingAsyncApplicationsQuiz } from '../quizzes/python-async/debugging-async-applications';
import { asyncDesignPatternsQuiz } from '../quizzes/python-async/async-design-patterns';
import { productionAsyncPatternsQuiz } from '../quizzes/python-async/production-async-patterns';

// Import multiple choice (all 16 completed!)
import { concurrencyFundamentalsMultipleChoice } from '../multiple-choice/python-async/concurrency-fundamentals';
import { eventLoopDeepDiveMultipleChoice } from '../multiple-choice/python-async/event-loop-deep-dive';
import { coroutinesAsyncAwaitMultipleChoice } from '../multiple-choice/python-async/coroutines-async-await';
import { tasksAndFuturesMultipleChoice } from '../multiple-choice/python-async/tasks-and-futures';
import { asyncContextManagersGeneratorsMultipleChoice } from '../multiple-choice/python-async/async-context-managers-generators';
import { asyncioBuiltinFunctionsMultipleChoice } from '../multiple-choice/python-async/asyncio-builtin-functions';
import { asyncHttpAiohttpMultipleChoice } from '../multiple-choice/python-async/async-http-aiohttp';
import { asyncDatabaseOperationsMultipleChoice } from '../multiple-choice/python-async/async-database-operations';
import { asyncFileIOMultipleChoice } from '../multiple-choice/python-async/async-file-io';
import { errorHandlingAsyncMultipleChoice } from '../multiple-choice/python-async/error-handling-async';
import { threadingVsMultiprocessingVsAsyncMultipleChoice } from '../multiple-choice/python-async/threading-vs-multiprocessing-vs-async';
import { concurrentFuturesModuleMultipleChoice } from '../multiple-choice/python-async/concurrent-futures-module';
import { raceConditionsSynchronizationMultipleChoice } from '../multiple-choice/python-async/race-conditions-synchronization';
import { debuggingAsyncApplicationsMultipleChoice } from '../multiple-choice/python-async/debugging-async-applications';
import { asyncDesignPatternsMultipleChoice } from '../multiple-choice/python-async/async-design-patterns';
import { productionAsyncPatternsMultipleChoice } from '../multiple-choice/python-async/production-async-patterns';

export const pythonAsyncModule: Module = {
  id: 'python-async',
  title: 'Asynchronous Python Mastery',
  description:
    'Master async/await, asyncio, and concurrent programming for high-performance I/O-bound applications. Build production-ready async web servers, APIs, and data processing systems.',
  category: 'python',
  difficulty: 'advanced',
  estimatedTime: '3 weeks',
  prerequisites: ['python-intermediate', 'python-oop'],
  icon: '⚡',
  keyTakeaways: [
    'Concurrency vs parallelism: async for I/O (10K+ connections), multiprocessing for CPU (true parallelism)',
    'Event loop orchestrates coroutines—manages tasks, monitors I/O, switches at await points',
    'async/await enables non-blocking I/O: 100 concurrent requests in ~1s vs 100s sequential',
    "Tasks run immediately when created: asyncio.create_task() starts execution, don't await until ready",
    'Gather for all results, wait for control, as_completed for progressive processing',
    'Cancel with task.cancel(), handle CancelledError for cleanup, always re-raise',
    'Async patterns: gather() for concurrency, Semaphore for rate limiting, timeout for safety',
    'Production async: uvloop (2-4× faster), connection pooling, graceful shutdown, monitoring',
  ],
  learningObjectives: [
    'Understand concurrency fundamentals and when to use async vs threading vs multiprocessing',
    'Master the event loop architecture and task scheduling mechanisms',
    'Write efficient async code with async/await syntax and coroutines',
    'Manage concurrent tasks with gather(), wait(), and as_completed()',
    'Handle errors, timeouts, and cancellation in async applications',
    'Build async HTTP clients and servers with aiohttp and FastAPI',
    'Perform async database operations with asyncpg and async ORMs',
    'Debug and optimize async applications for production',
  ],
  sections: [
    {
      ...concurrencyFundamentals,
      quiz: concurrencyFundamentalsQuiz,
      multipleChoice: concurrencyFundamentalsMultipleChoice,
    },
    {
      ...eventLoopDeepDive,
      quiz: eventLoopDeepDiveQuiz,
      multipleChoice: eventLoopDeepDiveMultipleChoice,
    },
    {
      ...coroutinesAsyncAwait,
      quiz: coroutinesAsyncAwaitQuiz,
      multipleChoice: coroutinesAsyncAwaitMultipleChoice,
    },
    {
      ...tasksAndFutures,
      quiz: tasksAndFuturesQuiz,
      multipleChoice: tasksAndFuturesMultipleChoice,
    },
    {
      ...asyncContextManagersGenerators,
      quiz: asyncContextManagersGeneratorsQuiz,
      multipleChoice: asyncContextManagersGeneratorsMultipleChoice,
    },
    {
      ...asyncioBuiltinFunctions,
      quiz: asyncioBuiltinFunctionsQuiz,
      multipleChoice: asyncioBuiltinFunctionsMultipleChoice,
    },
    {
      ...asyncHttpAiohttp,
      quiz: asyncHttpAiohttpQuiz,
      multipleChoice: asyncHttpAiohttpMultipleChoice,
    },
    {
      ...asyncDatabaseOperations,
      quiz: asyncDatabaseOperationsQuiz,
      multipleChoice: asyncDatabaseOperationsMultipleChoice,
    },
    {
      ...asyncFileIO,
      quiz: asyncFileIOQuiz,
      multipleChoice: asyncFileIOMultipleChoice,
    },
    {
      ...errorHandlingAsync,
      quiz: errorHandlingAsyncQuiz,
      multipleChoice: errorHandlingAsyncMultipleChoice,
    },
    {
      ...threadingVsMultiprocessingVsAsync,
      quiz: threadingVsMultiprocessingVsAsyncQuiz,
      multipleChoice: threadingVsMultiprocessingVsAsyncMultipleChoice,
    },
    {
      ...concurrentFuturesModule,
      quiz: concurrentFuturesModuleQuiz,
      multipleChoice: concurrentFuturesModuleMultipleChoice,
    },
    {
      ...raceConditionsSynchronization,
      quiz: raceConditionsSynchronizationQuiz,
      multipleChoice: raceConditionsSynchronizationMultipleChoice,
    },
    {
      ...debuggingAsyncApplications,
      quiz: debuggingAsyncApplicationsQuiz,
      multipleChoice: debuggingAsyncApplicationsMultipleChoice,
    },
    {
      ...asyncDesignPatterns,
      quiz: asyncDesignPatternsQuiz,
      multipleChoice: asyncDesignPatternsMultipleChoice,
    },
    {
      ...productionAsyncPatterns,
      quiz: productionAsyncPatternsQuiz,
      multipleChoice: productionAsyncPatternsMultipleChoice,
    },
  ],
};
