/**
 * Multiple choice questions for Output Format Control section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const outputformatcontrolMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-format-mc-1',
        question:
            'What is the main advantage of using JSON mode in OpenAI\'s API?',
        options: [
            'Faster responses',
            'Guaranteed valid JSON output',
            'Lower costs',
            'Better reasoning',
        ],
        correctAnswer: 1,
        explanation:
            'JSON mode guarantees the output will be valid JSON that can be parsed. This eliminates parsing errors and makes outputs reliable and consistent, which is critical for production automation.',
    },
    {
        id: 'peo-format-mc-2',
        question:
            'What does the Instructor library provide?',
        options: [
            'Training for prompt engineers',
            'Type-safe LLM outputs with Pydantic validation',
            'Database connections',
            'API key management',
        ],
        correctAnswer: 1,
        explanation:
            'Instructor wraps the OpenAI API with Pydantic models, providing type-safe outputs with automatic validation. You define a schema and get typed objects with guaranteed fields, eliminating manual parsing and validation.',
    },
    {
        id: 'peo-format-mc-3',
        question:
            'When an LLM output fails validation, what is the best strategy?',
        options: [
            'Give up and show error to user',
            'Try to parse it anyway',
            'Retry with error feedback explaining what was wrong',
            'Switch to a different model',
        ],
        correctAnswer: 2,
        explanation:
            'Retrying with specific error feedback telling the LLM what was wrong (e.g., "missing field X, expected type Y") successfully corrects the output in most cases. This is more effective than blind retries or giving up.',
    },
    {
        id: 'peo-format-mc-4',
        question:
            'Why is JSON preferred over XML for LLM outputs?',
        options: [
            'JSON is newer',
            'JSON is simpler, less verbose, and more widely supported',
            'XML doesn\'t work with LLMs',
            'JSON has better security',
        ],
        correctAnswer: 1,
        explanation:
            'JSON is simpler syntax, less verbose than XML, has parsers in every language, and LLMs produce it more reliably. XML is better for mixed content (text + structure), but JSON is ideal for structured data.',
    },
    {
        id: 'peo-format-mc-5',
        question:
            'What is the maximum recommended number of retries for format validation failures?',
        options: [
            '1 retry',
            '3 retries',
            '10 retries',
            'Unlimited retries',
        ],
        correctAnswer: 1,
        explanation:
            '3 retries is optimal - it catches most correctable errors without excessive cost. First retry succeeds ~60% of time, second ~80%, third ~95%. Beyond 3, diminishing returns and better to fallback to human review.',
    },
];

