/**
 * React Fundamentals Module
 * Module 1: Foundational React concepts including components, JSX, props, state, and hooks
 */

import { Module } from '../../types';

// Import sections
import { introductionToReactJsx } from '../sections/react-fundamentals/introduction-to-react-jsx';
import { functionComponentsTypescript } from '../sections/react-fundamentals/function-components-typescript';
import { stateManagementUsestate } from '../sections/react-fundamentals/state-management-usestate';
import { eventHandling } from '../sections/react-fundamentals/event-handling';
import { conditionalRendering } from '../sections/react-fundamentals/conditional-rendering';
import { listsAndKeys } from '../sections/react-fundamentals/lists-and-keys';
import { formsIntroduction } from '../sections/react-fundamentals/forms-introduction';
import { reactDeveloperTools } from '../sections/react-fundamentals/react-developer-tools';
import { legacyPatternsAndMigration } from '../sections/react-fundamentals/legacy-patterns-and-migration';

// Import discussions
import { introductionToReactJsxDiscussion } from '../discussions/react-fundamentals/introduction-to-react-jsx-discussion';
import { functionComponentsTypescriptDiscussion } from '../discussions/react-fundamentals/function-components-typescript-discussion';
import { stateManagementUsestateDiscussion } from '../discussions/react-fundamentals/state-management-usestate-discussion';
import { eventHandlingDiscussion } from '../discussions/react-fundamentals/event-handling-discussion';
import { conditionalRenderingDiscussion } from '../discussions/react-fundamentals/conditional-rendering-discussion';
import { listsAndKeysDiscussion } from '../discussions/react-fundamentals/lists-and-keys-discussion';
import { formsIntroductionDiscussion } from '../discussions/react-fundamentals/forms-introduction-discussion';
import { reactDeveloperToolsDiscussion } from '../discussions/react-fundamentals/react-developer-tools-discussion';
import { legacyPatternsAndMigrationDiscussion } from '../discussions/react-fundamentals/legacy-patterns-and-migration-discussion';

// Import multiple choice
import { introductionToReactJsxQuiz } from '../multiple-choice/react-fundamentals/introduction-to-react-jsx';
import { functionComponentsTypescriptQuiz } from '../multiple-choice/react-fundamentals/function-components-typescript';
import { stateManagementUsestateQuiz } from '../multiple-choice/react-fundamentals/state-management-usestate';
import { eventHandlingQuiz } from '../multiple-choice/react-fundamentals/event-handling';
import { conditionalRenderingQuiz } from '../multiple-choice/react-fundamentals/conditional-rendering';
import { listsAndKeysQuiz } from '../multiple-choice/react-fundamentals/lists-and-keys';
import { formsIntroductionQuiz } from '../multiple-choice/react-fundamentals/forms-introduction';
import { reactDeveloperToolsQuiz } from '../multiple-choice/react-fundamentals/react-developer-tools';
import { legacyPatternsAndMigrationQuiz } from '../multiple-choice/react-fundamentals/legacy-patterns-and-migration';

export const reactFundamentalsModule: Module = {
  id: 'react-fundamentals',
  title: 'React Fundamentals',
  description:
    'Master the core concepts of modern React including components, JSX, props, state, and hooks',
  objectives: [
    "Understand React's component-based architecture and Virtual DOM",
    'Write modern function components with TypeScript',
    'Manage component state with the useState hook',
    'Handle events and user interactions effectively',
    'Implement conditional rendering patterns',
    'Render lists efficiently with proper key usage',
    'Build controlled and uncontrolled forms',
    'Debug React applications with React DevTools',
    'Read and migrate legacy React code to modern patterns',
  ],
  sections: [
    {
      ...introductionToReactJsx,
      quiz: introductionToReactJsxDiscussion,
      multipleChoice: introductionToReactJsxQuiz,
    },
    {
      ...functionComponentsTypescript,
      quiz: functionComponentsTypescriptDiscussion,
      multipleChoice: functionComponentsTypescriptQuiz,
    },
    {
      ...stateManagementUsestate,
      quiz: stateManagementUsestateDiscussion,
      multipleChoice: stateManagementUsestateQuiz,
    },
    {
      ...eventHandling,
      quiz: eventHandlingDiscussion,
      multipleChoice: eventHandlingQuiz,
    },
    {
      ...conditionalRendering,
      quiz: conditionalRenderingDiscussion,
      multipleChoice: conditionalRenderingQuiz,
    },
    {
      ...listsAndKeys,
      quiz: listsAndKeysDiscussion,
      multipleChoice: listsAndKeysQuiz,
    },
    {
      ...formsIntroduction,
      quiz: formsIntroductionDiscussion,
      multipleChoice: formsIntroductionQuiz,
    },
    {
      ...reactDeveloperTools,
      quiz: reactDeveloperToolsDiscussion,
      multipleChoice: reactDeveloperToolsQuiz,
    },
    {
      ...legacyPatternsAndMigration,
      quiz: legacyPatternsAndMigrationDiscussion,
      multipleChoice: legacyPatternsAndMigrationQuiz,
    },
  ],
};
