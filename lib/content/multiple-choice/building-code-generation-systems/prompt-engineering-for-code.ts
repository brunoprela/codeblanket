/**
 * Multiple choice questions for Prompt Engineering for Code section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const promptengineeringforcodeMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'bcgs-prompteng-mc-1',
        question:
            'In the "context sandwich" pattern, why are constraints placed LAST in the prompt?',
        options: [
            'To make them easy to find',
            'Because of recency bias - LLMs remember last things better',
            'To separate them from context',
            'They are least important',
        ],
        correctAnswer: 1,
        explanation:
            'Placing constraints last leverages recency bias - LLMs tend to remember and follow instructions that come at the end of the prompt better. This ensures constraints like "add type hints" or "use error handling" are fresh when generating.',
    },
    {
        id: 'bcgs-prompteng-mc-2',
        question:
            'When providing a large file (2000 lines) as context, what is the most important information to preserve?',
        options: [
            'All code without any truncation',
            'The middle section with most logic',
            'Imports + edit location context + function signatures',
            'Just the function being edited',
        ],
        correctAnswer: 2,
        explanation:
            'You must preserve: imports (critical for dependencies), context around edit location (~50 lines), and function/class signatures (understanding structure). This gives the LLM what it needs without exceeding context limits.',
    },
    {
        id: 'bcgs-prompteng-mc-3',
        question:
            'What is the main benefit of including function signatures from related files in your prompt?',
        options: [
            'It makes the prompt longer and more detailed',
            'It prevents the LLM from hallucinating non-existent function interfaces',
            'It helps with style consistency',
            'It\'s required by the API',
        ],
        correctAnswer: 1,
        explanation:
            'Including actual function signatures prevents the LLM from inventing non-existent parameters or return types. Without signatures, the LLM might guess that get_user() returns a dict when it actually returns a User object.',
    },
    {
        id: 'bcgs-prompteng-mc-4',
        question:
            'How should you handle token budget when building prompts with multiple context sources?',
        options: [
            'Include everything and let the API truncate',
            'Prioritize: required context first, then optional context that fits',
            'Always summarize everything to fit',
            'Remove code and only include descriptions',
        ],
        correctAnswer: 1,
        explanation:
            'Use priority-based allocation: add required context (current file, task), then add optional context (related files, examples) only if it fits within token budget. This ensures critical info is always included.',
    },
    {
        id: 'bcgs-prompteng-mc-5',
        question:
            'What is the recommended approach for handling file imports in prompts?',
        options: [
            'Ignore imports, they\'re not important',
            'Always include all imports at the start',
            'Only include imports if editing import-related code',
            'Summarize imports as "various libraries"',
        ],
        correctAnswer: 1,
        explanation:
            'Always include imports at the start of the prompt. Imports are critical for the LLM to know what libraries are available and what can be used. Missing imports leads to hallucinated functions.',
    },
];

