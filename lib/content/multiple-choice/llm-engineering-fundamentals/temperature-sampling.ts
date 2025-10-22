/**
 * Multiple choice questions for Temperature, Top-P & Sampling Parameters section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const temperaturesamplingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What temperature value ensures completely deterministic outputs?',
        options: [
            '0.0',
            '0.5',
            '1.0',
            'Temperature does not affect determinism'
        ],
        correctAnswer: 0,
        explanation:
            'Temperature=0.0 ensures deterministic output by always selecting the highest probability token. The same input will always produce the same output. Any temperature >0 introduces randomness, making outputs non-deterministic.'
    },
    {
        id: 'mc2',
        question: 'For code generation tasks, what is the recommended temperature?',
        options: [
            '0.0 for deterministic, correct code',
            '0.7 for creative code',
            '1.5 for diverse solutions',
            '1.0 for balanced output'
        ],
        correctAnswer: 0,
        explanation:
            'Code generation should use temperature=0.0 to ensure consistent, deterministic output. Code must be correct and predictable - creativity in code generation often means bugs and syntax errors. Save higher temperatures for creative writing tasks.'
    },
    {
        id: 'mc3',
        question: 'What is the relationship between temperature and top_p?',
        options: [
            'They are completely independent',
            'Both control randomness and should not be adjusted together',
            'Top_p must always be higher than temperature',
            'Temperature is ignored when top_p is set'
        ],
        correctAnswer: 1,
        explanation:
            'Both temperature and top_p control randomness, and OpenAI recommends adjusting one OR the other, not both. Combining them creates complex, unpredictable interactions. Choose temperature (easier to understand) or top_p (filters unlikely tokens), but not both.'
    },
    {
        id: 'mc4',
        question: 'What does frequency_penalty do?',
        options: [
            'Increases response speed',
            'Reduces token repetition based on how often they appear',
            'Limits total tokens generated',
            'Controls temperature automatically'
        ],
        correctAnswer: 1,
        explanation:
            'frequency_penalty penalizes tokens based on how frequently they have already appeared in the generation, reducing repetitive word use. A penalty of 0.5-1.0 encourages more varied vocabulary without forcing unnatural language. Useful for creative writing.'
    },
    {
        id: 'mc5',
        question: 'Why might you use a stop sequence parameter?',
        options: [
            'To prevent the model from generating',
            'To end generation at a specific string (e.g., "\\n\\n")',
            'To limit cost',
            'To improve quality'
        ],
        correctAnswer: 1,
        explanation:
            'Stop sequences tell the model to stop generating when it encounters specific strings (e.g., "\\n\\n" for paragraph breaks, "```" for code blocks, "}" for JSON). This provides precise control over output length and format without relying on max_tokens alone.'
    }
];

