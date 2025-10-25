/**
 * Multiple choice questions for Canvas: Chains, Groups, Chords section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const canvasChainsGroupsChordsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What does chain(A.s(), B.s(), C.s()) do?',
      options: [
        'Executes A, B, C in parallel',
        'Executes A then B then C sequentially',
        'Executes only A',
        'Cancels all tasks',
      ],
      correctAnswer: 1,
      explanation:
        "chain() executes tasks sequentially: A completes, result passed to B, B completes, result passed to C. Example: chain(download.s(url), process.s(), upload.s()) downloads → processes → uploads. Use for dependent tasks where B needs A's output.",
    },
    {
      id: 'mc2',
      question: 'What does group([A.s(), B.s(), C.s()]) do?',
      options: [
        'Executes A then B then C',
        'Executes A, B, C in parallel',
        'Executes only first task',
        'Combines task results',
      ],
      correctAnswer: 1,
      explanation:
        'group() executes tasks in parallel. All tasks (A, B, C) run simultaneously on different workers. Returns list of results. Example: group([process.s(i) for i in range(100)]) processes 100 items in parallel. Use for independent tasks.',
    },
    {
      id: 'mc3',
      question: 'What does chord([A.s(), B.s()], C.s()) do?',
      options: [
        'Executes A then B then C',
        'Executes A and B in parallel, then C with results',
        'Executes only C',
        'Cancels all tasks',
      ],
      correctAnswer: 1,
      explanation:
        'chord() executes header tasks ([A, B]) in parallel, then passes all results to callback (C). Example: chord([process.s(i) for i in range(10)], sum_results.s()) processes items in parallel, then sums. Use for parallel processing with aggregation.',
    },
    {
      id: 'mc4',
      question: 'How do you combine chain and group?',
      options: [
        'Not possible',
        'chain(group([A.s(), B.s()]), C.s()) - parallel then sequential',
        'Only one primitive at a time',
        'Use different Celery apps',
      ],
      correctAnswer: 1,
      explanation:
        'Combine primitives: chain(group([A.s(), B.s()]), C.s()) executes A and B in parallel, then C. Or group([chain(A.s(), B.s()), chain(C.s(), D.s())]) executes two chains in parallel. Canvas primitives compose flexibly for complex workflows.',
    },
    {
      id: 'mc5',
      question: 'What does .s() mean in A.s()?',
      options: [
        'Start task',
        'Signature - creates task without executing',
        'Stop task',
        'Save task',
      ],
      correctAnswer: 1,
      explanation:
        ".s() creates a signature (promise to execute task later). A.s(5) creates signature with arg 5 but doesn't execute. Used in canvas: chain(A.s(5), B.s()) creates workflow. A.delay(5) executes immediately. A.s(5) creates signature for later execution in workflow.",
    },
  ];
