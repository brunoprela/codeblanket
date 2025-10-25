/**
 * Multiple choice questions for Meta-Prompting & Self-Improvement section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const metapromptingselfimprovementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'peo-meta-mc-1',
      question: 'What is meta-prompting?',
      options: [
        'Writing very detailed prompts',
        'Using LLMs to generate or improve prompts',
        'Prompting about metadata',
        'Using multiple prompts simultaneously',
      ],
      correctAnswer: 1,
      explanation:
        'Meta-prompting is using LLMs to write or improve prompts for themselves or other LLMs. The AI generates optimized prompts based on task descriptions, often producing better results than manual prompt engineering.',
    },
    {
      id: 'peo-meta-mc-2',
      question: 'What is self-critique in prompt systems?',
      options: [
        'Users critiquing prompts',
        'The LLM generating output, then critiquing it, then improving it',
        'Checking grammar in prompts',
        'Comparing different models',
      ],
      correctAnswer: 1,
      explanation:
        'Self-critique is when the LLM generates a response, critiques its own output, identifies issues, then generates an improved version. This iterative refinement often produces higher quality results.',
    },
    {
      id: 'peo-meta-mc-3',
      question: 'What is prompt evolution?',
      options: [
        'Prompts getting longer over time',
        'Using genetic algorithms to generate and test prompt variants',
        'Training prompts on data',
        'Version controlling prompts',
      ],
      correctAnswer: 1,
      explanation:
        'Prompt evolution applies genetic algorithms: generate variants, test fitness (accuracy), keep best, mutate to create new variants, repeat. This explores the prompt space to find optimal formulations.',
    },
    {
      id: 'peo-meta-mc-4',
      question:
        'When should a self-improving system trigger prompt optimization?',
      options: [
        'Every hour automatically',
        'When failure rate exceeds threshold or success rate drops below target',
        'Never - manual only',
        'On user request only',
      ],
      correctAnswer: 1,
      explanation:
        'Trigger optimization when performance degrades: failure rate exceeds threshold (e.g., >10%) or success rate drops below target (e.g., <90%). This ensures prompt stays effective as usage patterns evolve.',
    },
    {
      id: 'peo-meta-mc-5',
      question: 'What is the main risk of fully automated prompt improvement?',
      options: [
        'It costs too much',
        "It\'s too slow",
        'Generated prompts might not align with business priorities or safety requirements',
        "It doesn't work",
      ],
      correctAnswer: 2,
      explanation:
        'Automated systems can optimize for metrics without understanding business context or safety implications. Human oversight ensures prompts align with priorities, maintain safety standards, and consider factors beyond raw accuracy.',
    },
  ];
