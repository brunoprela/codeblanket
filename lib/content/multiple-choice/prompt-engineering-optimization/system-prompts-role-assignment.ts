/**
 * Multiple choice questions for System Prompts & Role Assignment section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systempromptsroleassignmentMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-system-mc-1',
        question:
            'What is the main advantage of system prompts over user prompts?',
        options: [
            'They cost fewer tokens',
            'They persist across the entire conversation',
            'They generate faster responses',
            'They work with all models',
        ],
        correctAnswer: 1,
        explanation:
            'System prompts persist across the entire conversation, establishing consistent behavior without repeating instructions. This makes them ideal for defining AI personality, capabilities, and constraints that apply to all interactions.',
    },
    {
        id: 'peo-system-mc-2',
        question:
            'When defining an AI role in a system prompt, what level of specificity is most effective?',
        options: [
            'Very general (e.g., "helpful assistant")',
            'Somewhat specific (e.g., "coding assistant")',
            'Highly specific (e.g., "senior Python developer specializing in backend APIs")',
            'Specificity doesn\'t matter',
        ],
        correctAnswer: 2,
        explanation:
            'Highly specific role definitions produce more consistent and appropriate outputs. Specifying expertise, experience level, and domain helps the model understand exactly what behavior and knowledge level is expected.',
    },
    {
        id: 'peo-system-mc-3',
        question:
            'What is the recommended approach for A/B testing system prompts?',
        options: [
            'Test on internal team only',
            'Split production traffic and measure statistically significant differences',
            'Switch entirely to new prompt immediately',
            'Ask users which they prefer',
        ],
        correctAnswer: 1,
        explanation:
            'Splitting production traffic (e.g., 50/50) and measuring performance with statistical rigor is the gold standard. This provides real-world data and ensures improvements are significant before full deployment.',
    },
    {
        id: 'peo-system-mc-4',
        question:
            'Which element should NOT typically be in a system prompt?',
        options: [
            'Role and expertise definition',
            'Behavioral guidelines',
            'Specific user queries',
            'Output format requirements',
        ],
        correctAnswer: 2,
        explanation:
            'Specific user queries belong in user messages, not system prompts. System prompts define persistent behavior, role, and guidelines that apply across all queries, while user messages contain specific tasks.',
    },
    {
        id: 'peo-system-mc-5',
        question:
            'How does Cursor likely use system prompts?',
        options: [
            'To store user code',
            'To define code-editing behavior and file context understanding',
            'To execute code',
            'To manage API keys',
        ],
        correctAnswer: 1,
        explanation:
            'Cursor uses system prompts to establish its role as a code editor, define how to understand file contexts, specify diff generation behavior, and set guidelines for code quality - all persistent behaviors needed across all editing tasks.',
    },
];

