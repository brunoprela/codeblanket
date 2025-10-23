/**
 * Multiple choice questions for Chat Completions & Message Formats section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const chatcompletionsmessagesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the three main message roles in chat completions?',
    options: [
      'user, bot, admin',
      'system, user, assistant',
      'prompt, response, context',
      'input, output, history',
    ],
    correctAnswer: 1,
    explanation:
      'The three message roles are: system (sets behavior and context), user (the human input), and assistant (the AI responses). These roles structure the conversation for optimal model understanding.',
  },
  {
    id: 'mc2',
    question: 'When does a conversation need truncation?',
    options: [
      'After every 10 messages',
      "When approaching the model's context window limit",
      'Only when the user requests it',
      'Never, models handle this automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Conversations need truncation when approaching the context window limit (e.g., 80% of 16K tokens). Models do NOT handle this automatically - exceeding limits causes errors. Proactive truncation ensures reliable operation.',
  },
  {
    id: 'mc3',
    question: 'What is the purpose of the system message in chat completions?',
    options: [
      'To provide user authentication',
      'To set model behavior and provide instructions',
      'To log the conversation for debugging',
      'To specify the model version',
    ],
    correctAnswer: 1,
    explanation:
      "The system message sets consistent behavior, provides instructions, defines tone/style, and specifies output format. It's sent with every request to ensure the model behaves as desired without repeating instructions in user messages.",
  },
  {
    id: 'mc4',
    question:
      'Which truncation strategy preserves the most important information?',
    options: [
      'Sliding window (keep recent messages)',
      'Delete oldest messages first',
      'Importance-based (keep tagged/critical messages)',
      'Randomly remove messages',
    ],
    correctAnswer: 2,
    explanation:
      'Importance-based truncation preserves critical information by keeping messages tagged as important (questions, errors, key data) plus the system message and recent turns. Sliding window blindly keeps recent messages which might drop crucial early context.',
  },
  {
    id: 'mc5',
    question:
      'How should conversation history be stored for multi-session conversations?',
    options: [
      'Only in memory, cleared on restart',
      'In database for persistence, with cache for performance',
      'In the system prompt',
      'Sent to the LLM provider for storage',
    ],
    correctAnswer: 1,
    explanation:
      'Production apps should store conversations in a database (unlimited storage, persistence across sessions) while caching recent conversations in Redis for fast access. This provides durability and performance. LLM providers do NOT store conversation history.',
  },
];
