import { Module } from '../../types';

// Import sections
import { timeValueOfMoney } from '@/lib/content/sections/corporate-finance/time-value-of-money';
import { npvIrrCapitalBudgeting } from '@/lib/content/sections/corporate-finance/npv-irr-capital-budgeting';
import { costOfCapitalWacc } from '@/lib/content/sections/corporate-finance/cost-of-capital-wacc';
import { capmBeta } from '@/lib/content/sections/corporate-finance/capm-beta';
import { capitalStructureLeverage } from '@/lib/content/sections/corporate-finance/capital-structure-leverage';
import { valuationBasics } from '@/lib/content/sections/corporate-finance/valuation-basics';
import { dividendsShareBuybacks } from '@/lib/content/sections/corporate-finance/dividends-share-buybacks';
import { mergersAcquisitions } from '@/lib/content/sections/corporate-finance/mergers-acquisitions';
import { leveragedBuyouts } from '@/lib/content/sections/corporate-finance/leveraged-buyouts';
import { workingCapitalManagement } from '@/lib/content/sections/corporate-finance/working-capital-management';
import { moduleProject } from '@/lib/content/sections/corporate-finance/module-project';

// Import quizzes
import { timeValueOfMoneyQuiz } from '@/lib/content/quizzes/corporate-finance/time-value-of-money';
import { npvIrrCapitalBudgetingQuiz } from '@/lib/content/quizzes/corporate-finance/npv-irr-capital-budgeting';
import { costOfCapitalWaccQuiz } from '@/lib/content/quizzes/corporate-finance/cost-of-capital-wacc';
import { capmBetaQuiz } from '@/lib/content/quizzes/corporate-finance/capm-beta';
import { capitalStructureLeverageQuiz } from '@/lib/content/quizzes/corporate-finance/capital-structure-leverage';
import { valuationBasicsQuiz } from '@/lib/content/quizzes/corporate-finance/valuation-basics';
import { dividendsShareBuybacksQuiz } from '@/lib/content/quizzes/corporate-finance/dividends-share-buybacks';
import { mergersAcquisitionsQuiz } from '@/lib/content/quizzes/corporate-finance/mergers-acquisitions';
import { leveragedBuyoutsQuiz } from '@/lib/content/quizzes/corporate-finance/leveraged-buyouts';
import { workingCapitalManagementQuiz } from '@/lib/content/quizzes/corporate-finance/working-capital-management';
import { moduleProjectQuiz } from '@/lib/content/quizzes/corporate-finance/module-project';

// Import multiple choice
import { timeValueOfMoneyMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/time-value-of-money';
import { npvIrrCapitalBudgetingMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/npv-irr-capital-budgeting';
import { costOfCapitalWaccMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/cost-of-capital-wacc';
import { capmBetaMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/capm-beta';
import { capitalStructureLeverageMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/capital-structure-leverage';
import { valuationBasicsMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/valuation-basics';
import { dividendsShareBuybacksMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/dividends-share-buybacks';
import { mergersAcquisitionsMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/mergers-acquisitions';
import { leveragedBuyoutsMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/leveraged-buyouts';
import { workingCapitalManagementMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/working-capital-management';
import { moduleProjectMultipleChoice } from '@/lib/content/multiple-choice/corporate-finance/module-project';

// Transform functions
const transformQuiz = (quiz: any) => ({
  id: quiz.id,
  question: quiz.question,
  sampleAnswer: quiz.sampleAnswer,
  keyPoints: quiz.keyPoints,
});

const transformMC = (mc: any) => ({
  id: mc.id,
  question: mc.question,
  options: mc.options,
  correctAnswer: mc.correctAnswer,
  explanation: mc.explanation,
});

export const corporateFinanceModule: Module = {
  id: 'corporate-finance',
  title: 'Corporate Finance Fundamentals',
  description:
    'Master valuation, capital structure, M&A, LBO modeling, and core corporate finance concepts',
  sections: [
    {
      id: timeValueOfMoney.id,
      title: timeValueOfMoney.title,
      content: timeValueOfMoney.content,
    },
    {
      id: npvIrrCapitalBudgeting.id,
      title: npvIrrCapitalBudgeting.title,
      content: npvIrrCapitalBudgeting.content,
    },
    {
      id: costOfCapitalWacc.id,
      title: costOfCapitalWacc.title,
      content: costOfCapitalWacc.content,
    },
    {
      id: capmBeta.id,
      title: capmBeta.title,
      content: capmBeta.content,
    },
    {
      id: capitalStructureLeverage.id,
      title: capitalStructureLeverage.title,
      content: capitalStructureLeverage.content,
    },
    {
      id: valuationBasics.id,
      title: valuationBasics.title,
      content: valuationBasics.content,
    },
    {
      id: dividendsShareBuybacks.id,
      title: dividendsShareBuybacks.title,
      content: dividendsShareBuybacks.content,
    },
    {
      id: mergersAcquisitions.id,
      title: mergersAcquisitions.title,
      content: mergersAcquisitions.content,
    },
    {
      id: leveragedBuyouts.id,
      title: leveragedBuyouts.title,
      content: leveragedBuyouts.content,
    },
    {
      id: workingCapitalManagement.id,
      title: workingCapitalManagement.title,
      content: workingCapitalManagement.content,
    },
    {
      id: moduleProject.id,
      title: moduleProject.title,
      content: moduleProject.content,
    },
  ],
  quizzes: [
    ...timeValueOfMoneyQuiz.map(transformQuiz),
    ...npvIrrCapitalBudgetingQuiz.map(transformQuiz),
    ...costOfCapitalWaccQuiz.map(transformQuiz),
    ...capmBetaQuiz.map(transformQuiz),
    ...capitalStructureLeverageQuiz.map(transformQuiz),
    ...valuationBasicsQuiz.map(transformQuiz),
    ...dividendsShareBuybacksQuiz.map(transformQuiz),
    ...mergersAcquisitionsQuiz.map(transformQuiz),
    ...leveragedBuyoutsQuiz.map(transformQuiz),
    ...workingCapitalManagementQuiz.map(transformQuiz),
    ...moduleProjectQuiz.map(transformQuiz),
  ],
  multipleChoiceQuestions: [
    ...timeValueOfMoneyMultipleChoice.map(transformMC),
    ...npvIrrCapitalBudgetingMultipleChoice.map(transformMC),
    ...costOfCapitalWaccMultipleChoice.map(transformMC),
    ...capmBetaMultipleChoice.map(transformMC),
    ...capitalStructureLeverageMultipleChoice.map(transformMC),
    ...valuationBasicsMultipleChoice.map(transformMC),
    ...dividendsShareBuybacksMultipleChoice.map(transformMC),
    ...mergersAcquisitionsMultipleChoice.map(transformMC),
    ...leveragedBuyoutsMultipleChoice.map(transformMC),
    ...workingCapitalManagementMultipleChoice.map(transformMC),
    ...moduleProjectMultipleChoice.map(transformMC),
  ],
};
