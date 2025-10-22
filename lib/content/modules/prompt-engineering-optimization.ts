/**
 * Prompt Engineering & Optimization Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { promptengineeringfundamentalsSection } from '../sections/prompt-engineering-optimization/prompt-engineering-fundamentals';
import { systempromptsroleassignmentSection } from '../sections/prompt-engineering-optimization/system-prompts-role-assignment';
import { fewshotlearningexamplesSection } from '../sections/prompt-engineering-optimization/few-shot-learning-examples';
import { chainofthoughtpromptingSection } from '../sections/prompt-engineering-optimization/chain-of-thought-prompting';
import { promptoptimizationtechniquesSection } from '../sections/prompt-engineering-optimization/prompt-optimization-techniques';
import { outputformatcontrolSection } from '../sections/prompt-engineering-optimization/output-format-control';
import { contextmanagementtruncationSection } from '../sections/prompt-engineering-optimization/context-management-truncation';
import { negativepromptingconstraintsSection } from '../sections/prompt-engineering-optimization/negative-prompting-constraints';
import { promptinjectionsecuritySection } from '../sections/prompt-engineering-optimization/prompt-injection-security';
import { metapromptingselfimprovementSection } from '../sections/prompt-engineering-optimization/meta-prompting-self-improvement';

// Import quizzes
import { promptengineeringfundamentalsQuiz } from '../quizzes/prompt-engineering-optimization/prompt-engineering-fundamentals';
import { systempromptsroleassignmentQuiz } from '../quizzes/prompt-engineering-optimization/system-prompts-role-assignment';
import { fewshotlearningexamplesQuiz } from '../quizzes/prompt-engineering-optimization/few-shot-learning-examples';
import { chainofthoughtpromptingQuiz } from '../quizzes/prompt-engineering-optimization/chain-of-thought-prompting';
import { promptoptimizationtechniquesQuiz } from '../quizzes/prompt-engineering-optimization/prompt-optimization-techniques';
import { outputformatcontrolQuiz } from '../quizzes/prompt-engineering-optimization/output-format-control';
import { contextmanagementtruncationQuiz } from '../quizzes/prompt-engineering-optimization/context-management-truncation';
import { negativepromptingconstraintsQuiz } from '../quizzes/prompt-engineering-optimization/negative-prompting-constraints';
import { promptinjectionsecurityQuiz } from '../quizzes/prompt-engineering-optimization/prompt-injection-security';
import { metapromptingselfimprovementQuiz } from '../quizzes/prompt-engineering-optimization/meta-prompting-self-improvement';

// Import multiple choice
import { promptengineeringfundamentalsMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/prompt-engineering-fundamentals';
import { systempromptsroleassignmentMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/system-prompts-role-assignment';
import { fewshotlearningexamplesMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/few-shot-learning-examples';
import { chainofthoughtpromptingMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/chain-of-thought-prompting';
import { promptoptimizationtechniquesMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/prompt-optimization-techniques';
import { outputformatcontrolMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/output-format-control';
import { contextmanagementtruncationMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/context-management-truncation';
import { negativepromptingconstraintsMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/negative-prompting-constraints';
import { promptinjectionsecurityMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/prompt-injection-security';
import { metapromptingselfimprovementMultipleChoice } from '../multiple-choice/prompt-engineering-optimization/meta-prompting-self-improvement';

export const promptEngineeringOptimizationModule: Module = {
    id: 'applied-ai-prompt-engineering',
    title: 'Prompt Engineering & Optimization',
    description:
        'Master the art and science of prompt engineering to build reliable, production-ready AI applications. Learn systematic approaches to write, test, optimize, and secure prompts at scale.',
    category: 'Applied AI',
    difficulty: 'Beginner',
    estimatedTime: '14 hours',
    icon: '📝',
    sections: [
        {
            ...promptengineeringfundamentalsSection,
            quiz: promptengineeringfundamentalsQuiz,
            multipleChoice: promptengineeringfundamentalsMultipleChoice,
        },
        {
            ...systempromptsroleassignmentSection,
            quiz: systempromptsroleassignmentQuiz,
            multipleChoice: systempromptsroleassignmentMultipleChoice,
        },
        {
            ...fewshotlearningexamplesSection,
            quiz: fewshotlearningexamplesQuiz,
            multipleChoice: fewshotlearningexamplesMultipleChoice,
        },
        {
            ...chainofthoughtpromptingSection,
            quiz: chainofthoughtpromptingQuiz,
            multipleChoice: chainofthoughtpromptingMultipleChoice,
        },
        {
            ...promptoptimizationtechniquesSection,
            quiz: promptoptimizationtechniquesQuiz,
            multipleChoice: promptoptimizationtechniquesMultipleChoice,
        },
        {
            ...outputformatcontrolSection,
            quiz: outputformatcontrolQuiz,
            multipleChoice: outputformatcontrolMultipleChoice,
        },
        {
            ...contextmanagementtruncationSection,
            quiz: contextmanagementtruncationQuiz,
            multipleChoice: contextmanagementtruncationMultipleChoice,
        },
        {
            ...negativepromptingconstraintsSection,
            quiz: negativepromptingconstraintsQuiz,
            multipleChoice: negativepromptingconstraintsMultipleChoice,
        },
        {
            ...promptinjectionsecuritySection,
            quiz: promptinjectionsecurityQuiz,
            multipleChoice: promptinjectionsecurityMultipleChoice,
        },
        {
            ...metapromptingselfimprovementSection,
            quiz: metapromptingselfimprovementQuiz,
            multipleChoice: metapromptingselfimprovementMultipleChoice,
        },
    ],
    keyTakeaways: [
        'Prompt engineering is software engineering - treat prompts like code with version control, testing, and systematic optimization',
        'Structure prompts with Role, Task, Constraints, Format, and Examples for reliable production outputs',
        'Few-shot learning with 3-5 examples dramatically improves consistency over zero-shot prompting',
        'Chain-of-Thought prompting improves complex reasoning by 30-50% but costs more tokens',
        'System prompts establish persistent behavior and have higher authority than user prompts',
        'Output format control with JSON and Pydantic ensures parseable, type-safe LLM outputs',
        'Context management and smart truncation handle large documents efficiently within token limits',
        'Negative prompts and constraints prevent specific failure modes and set clear boundaries',
        'Prompt injection is real security threat - use delimiters, sanitization, and output validation',
        'Meta-prompting and self-improvement enable automated prompt optimization at scale',
    ],
    learningObjectives: [
        'Write production-ready prompts with proper structure and comprehensive testing',
        'Design effective system prompts that define AI behavior and capabilities',
        'Select and order examples for optimal few-shot learning performance',
        'Apply Chain-of-Thought and ReAct patterns for complex reasoning tasks',
        'Systematically optimize prompts using A/B testing and failure analysis',
        'Enforce structured output formats with JSON Schema and Pydantic validation',
        'Manage context windows and truncate large documents intelligently',
        'Implement constraints and guardrails for safe, bounded AI behavior',
        'Secure prompts against injection attacks with defense-in-depth strategies',
        'Build self-improving systems that learn from failures and optimize automatically',
    ],
    prerequisites: [
        'Basic Python programming',
        'Understanding of APIs',
        'Familiarity with JSON',
    ],
};
