/**
 * Multiple choice questions for Chain-of-Thought Prompting section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const chainofthoughtpromptingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'peo-cot-mc-1',
    question: 'What is the main benefit of Chain-of-Thought prompting?',
    options: [
      'Faster response times',
      'Lower token costs',
      'Better accuracy on complex reasoning tasks',
      'Simpler prompts',
    ],
    correctAnswer: 2,
    explanation:
      'CoT improves accuracy on complex reasoning tasks by 30-50% by encouraging step-by-step thinking. While it costs more tokens and takes longer, the accuracy improvement is worth it for complex problems like math, logic, and analysis.',
  },
  {
    id: 'peo-cot-mc-2',
    question:
      'Which phrase most effectively triggers Chain-of-Thought reasoning?',
    options: [
      '"Answer quickly"',
      '"Let\'s think step by step"',
      '"Be concise"',
      '"Answer directly"',
    ],
    correctAnswer: 1,
    explanation:
      '"Let\'s think step by step" is the most effective trigger phrase. This simple addition prompts the model to show its reasoning process, significantly improving performance on complex tasks.',
  },
  {
    id: 'peo-cot-mc-3',
    question: 'What is the ReAct pattern?',
    options: [
      'Reasoning without actions',
      'Interleaving Reasoning (Thought) and Acting (tool use)',
      'Reacting to user feedback',
      'Rapid action without thinking',
    ],
    correctAnswer: 1,
    explanation:
      'ReAct (Reasoning + Acting) interleaves thinking and action steps. The model thinks about what to do, executes an action (tool call), observes the result, and repeats. This enables complex multi-step workflows with tool use.',
  },
  {
    id: 'peo-cot-mc-4',
    question: 'For which task type should you skip Chain-of-Thought prompting?',
    options: [
      'Complex mathematical problems',
      'Multi-step reasoning tasks',
      'Simple sentiment classification',
      'Code debugging',
    ],
    correctAnswer: 2,
    explanation:
      "Simple tasks like sentiment classification don't benefit from CoT and the extra tokens are wasted. CoT is most valuable for complex reasoning, math, logic, debugging, and analysis where step-by-step thinking helps.",
  },
  {
    id: 'peo-cot-mc-5',
    question: 'What is self-consistency in CoT?',
    options: [
      'Using the same example every time',
      'Generating multiple reasoning paths and voting on the answer',
      'Checking outputs for grammatical consistency',
      'Keeping temperature at 0',
    ],
    correctAnswer: 1,
    explanation:
      'Self-consistency generates multiple independent reasoning paths (typically 5-10) and takes the majority vote answer. This catches errors and improves reliability, though it costs N times more tokens.',
  },
];
