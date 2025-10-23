/**
 * Building Code Generation Systems Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { codegenerationfundamentalsSection } from '../sections/building-code-generation-systems/code-generation-fundamentals';
import { promptengineeringforcodeSection } from '../sections/building-code-generation-systems/prompt-engineering-for-code';
import { singlefilecodegenerationSection } from '../sections/building-code-generation-systems/single-file-code-generation';
import { codeeditingdiffgenerationSection } from '../sections/building-code-generation-systems/code-editing-diff-generation';
import { multifilecodegenerationSection } from '../sections/building-code-generation-systems/multi-file-code-generation';
import { coderefactoringllmsSection } from '../sections/building-code-generation-systems/code-refactoring-llms';
import { testgenerationSection } from '../sections/building-code-generation-systems/test-generation';
import { codecommentdocumentationgenerationSection } from '../sections/building-code-generation-systems/code-comment-documentation-generation';
import { codereviewbugdetectionSection } from '../sections/building-code-generation-systems/code-review-bug-detection';
import { interactivecodeeditingSection } from '../sections/building-code-generation-systems/interactive-code-editing';
import { codeexecutionvalidationSection } from '../sections/building-code-generation-systems/code-execution-validation';
import { languagespecificgenerationSection } from '../sections/building-code-generation-systems/language-specific-generation';
import { buildingcompletecodeeditorSection } from '../sections/building-code-generation-systems/building-complete-code-editor';

// Import quizzes
import { codegenerationfundamentalsQuiz } from '../quizzes/building-code-generation-systems/code-generation-fundamentals';
import { promptengineeringforcodeQuiz } from '../quizzes/building-code-generation-systems/prompt-engineering-for-code';
import { singlefilecodegenerationQuiz } from '../quizzes/building-code-generation-systems/single-file-code-generation';
import { codeeditingdiffgenerationQuiz } from '../quizzes/building-code-generation-systems/code-editing-diff-generation';
import { multifilecodegenerationQuiz } from '../quizzes/building-code-generation-systems/multi-file-code-generation';
import { coderefactoringllmsQuiz } from '../quizzes/building-code-generation-systems/code-refactoring-llms';
import { testgenerationQuiz } from '../quizzes/building-code-generation-systems/test-generation';
import { codecommentdocumentationgenerationQuiz } from '../quizzes/building-code-generation-systems/code-comment-documentation-generation';
import { codereviewbugdetectionQuiz } from '../quizzes/building-code-generation-systems/code-review-bug-detection';
import { interactivecodeeditingQuiz } from '../quizzes/building-code-generation-systems/interactive-code-editing';
import { codeexecutionvalidationQuiz } from '../quizzes/building-code-generation-systems/code-execution-validation';
import { languagespecificgenerationQuiz } from '../quizzes/building-code-generation-systems/language-specific-generation';
import { buildingcompletecodeeditorQuiz } from '../quizzes/building-code-generation-systems/building-complete-code-editor';

// Import multiple choice
import { codegenerationfundamentalsMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-generation-fundamentals';
import { promptengineeringforcodeMultipleChoice } from '../multiple-choice/building-code-generation-systems/prompt-engineering-for-code';
import { singlefilecodegenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/single-file-code-generation';
import { codeeditingdiffgenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-editing-diff-generation';
import { multifilecodegenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/multi-file-code-generation';
import { coderefactoringllmsMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-refactoring-llms';
import { testgenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/test-generation';
import { codecommentdocumentationgenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-comment-documentation-generation';
import { codereviewbugdetectionMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-review-bug-detection';
import { interactivecodeeditingMultipleChoice } from '../multiple-choice/building-code-generation-systems/interactive-code-editing';
import { codeexecutionvalidationMultipleChoice } from '../multiple-choice/building-code-generation-systems/code-execution-validation';
import { languagespecificgenerationMultipleChoice } from '../multiple-choice/building-code-generation-systems/language-specific-generation';
import { buildingcompletecodeeditorMultipleChoice } from '../multiple-choice/building-code-generation-systems/building-complete-code-editor';

export const buildingCodeGenerationSystemsModule: Module = {
  id: 'applied-ai-code-generation',
  title: 'Building Code Generation Systems',
  description:
    'Master the art of building production-grade code generation systems: from prompt engineering for code to multi-file generation, refactoring, test generation, and complete code editor integration.',
  category: 'Applied AI',
  difficulty: 'Advanced',
  estimatedTime: '15 hours',
  prerequisites: [
    'LLM Engineering Fundamentals',
    'Python proficiency',
    'Code editor/IDE concepts',
  ],
  icon: '💻',
  keyTakeaways: [
    'Design effective prompts specifically for code generation',
    'Generate complete, working code files with proper structure',
    'Implement smart code editing using diffs and search-replace',
    'Coordinate changes across multiple interdependent files',
    'Refactor code while preserving functionality',
    'Automatically generate comprehensive test suites',
    'Create documentation and comments that explain WHY, not WHAT',
    'Build automated code review with security and quality checks',
    'Enable interactive editing with streaming and validation',
    'Execute and validate generated code safely in sandboxes',
    'Adapt generation patterns to different programming languages',
    'Integrate all capabilities into production code editors',
  ],
  learningObjectives: [
    'Understand LLM capabilities and limitations for code generation',
    'Master prompt engineering techniques specific to code tasks',
    'Generate single-file code with proper imports and structure',
    'Apply precise edits using diffs, search-replace, and AST manipulation',
    'Manage dependencies and coordinate multi-file changes',
    'Perform behavior-preserving refactorings with validation',
    'Generate unit tests with edge cases and proper mocking',
    'Create docstrings and comments following language conventions',
    'Implement automated code review for bugs, security, and quality',
    'Build responsive interactive editors with streaming',
    'Execute generated code securely with resource limits',
    'Customize generation for language-specific idioms and patterns',
    'Architect complete code generation systems with all features',
  ],
  sections: [
    {
      ...codegenerationfundamentalsSection,
      quiz: codegenerationfundamentalsQuiz,
      multipleChoice: codegenerationfundamentalsMultipleChoice,
    },
    {
      ...promptengineeringforcodeSection,
      quiz: promptengineeringforcodeQuiz,
      multipleChoice: promptengineeringforcodeMultipleChoice,
    },
    {
      ...singlefilecodegenerationSection,
      quiz: singlefilecodegenerationQuiz,
      multipleChoice: singlefilecodegenerationMultipleChoice,
    },
    {
      ...codeeditingdiffgenerationSection,
      quiz: codeeditingdiffgenerationQuiz,
      multipleChoice: codeeditingdiffgenerationMultipleChoice,
    },
    {
      ...multifilecodegenerationSection,
      quiz: multifilecodegenerationQuiz,
      multipleChoice: multifilecodegenerationMultipleChoice,
    },
    {
      ...coderefactoringllmsSection,
      quiz: coderefactoringllmsQuiz,
      multipleChoice: coderefactoringllmsMultipleChoice,
    },
    {
      ...testgenerationSection,
      quiz: testgenerationQuiz,
      multipleChoice: testgenerationMultipleChoice,
    },
    {
      ...codecommentdocumentationgenerationSection,
      quiz: codecommentdocumentationgenerationQuiz,
      multipleChoice: codecommentdocumentationgenerationMultipleChoice,
    },
    {
      ...codereviewbugdetectionSection,
      quiz: codereviewbugdetectionQuiz,
      multipleChoice: codereviewbugdetectionMultipleChoice,
    },
    {
      ...interactivecodeeditingSection,
      quiz: interactivecodeeditingQuiz,
      multipleChoice: interactivecodeeditingMultipleChoice,
    },
    {
      ...codeexecutionvalidationSection,
      quiz: codeexecutionvalidationQuiz,
      multipleChoice: codeexecutionvalidationMultipleChoice,
    },
    {
      ...languagespecificgenerationSection,
      quiz: languagespecificgenerationQuiz,
      multipleChoice: languagespecificgenerationMultipleChoice,
    },
    {
      ...buildingcompletecodeeditorSection,
      quiz: buildingcompletecodeeditorQuiz,
      multipleChoice: buildingcompletecodeeditorMultipleChoice,
    },
  ],
};
