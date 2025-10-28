import { Module } from '../../types';

// Import sections
import { financialSystemOverview } from '@/lib/content/sections/finance-foundations/financial-system-overview';
import { typesOfFinancialInstitutions } from '@/lib/content/sections/finance-foundations/types-of-financial-institutions';
import { careerPathsEngineersFinance } from '@/lib/content/sections/finance-foundations/career-paths-engineers-finance';
import { financialMarketsExplained } from '@/lib/content/sections/finance-foundations/financial-markets-explained';
import { investmentVehiclesProducts } from '@/lib/content/sections/finance-foundations/investment-vehicles-products';
import { howTradingWorks } from '@/lib/content/sections/finance-foundations/how-trading-works';
import { regulatoryLandscapeEngineers } from '@/lib/content/sections/finance-foundations/regulatory-landscape-engineers';
import { financeTerminologyDevelopers } from '@/lib/content/sections/finance-foundations/finance-terminology-developers';
import { mathematicsForFinance } from '@/lib/content/sections/finance-foundations/mathematics-finance';
import { readingFinancialNewsData } from '@/lib/content/sections/finance-foundations/reading-financial-news-data';
import { learningEnvironment } from '@/lib/content/sections/finance-foundations/learning-environment';
import { moduleProjectPersonalFinanceDashboard } from '@/lib/content/sections/finance-foundations/module-project-personal-finance-dashboard';

// Import quizzes
import { financialSystemOverviewQuiz } from '@/lib/content/quizzes/finance-foundations/financial-system-overview';
import { typesOfFinancialInstitutionsQuiz } from '@/lib/content/quizzes/finance-foundations/types-of-financial-institutions';
import { careerPathsEngineersFinanceQuiz } from '@/lib/content/quizzes/finance-foundations/career-paths-engineers-finance';
import { financialMarketsExplainedQuiz } from '@/lib/content/quizzes/finance-foundations/financial-markets-explained';
import { investmentVehiclesProductsQuiz } from '@/lib/content/quizzes/finance-foundations/investment-vehicles-products';
import { howTradingWorksQuiz } from '@/lib/content/quizzes/finance-foundations/how-trading-works';
import { regulatoryLandscapeEngineersQuiz } from '@/lib/content/quizzes/finance-foundations/regulatory-landscape-engineers';
import { financeTerminologyDevelopersQuiz } from '@/lib/content/quizzes/finance-foundations/finance-terminology-developers';
import { mathematicsFinanceQuiz } from '@/lib/content/quizzes/finance-foundations/mathematics-finance';
import { readingFinancialNewsDataQuiz } from '@/lib/content/quizzes/finance-foundations/reading-financial-news-data';
import { learningEnvironmentQuiz } from '@/lib/content/quizzes/finance-foundations/learning-environment';
import { moduleProjectPersonalFinanceDashboardQuiz } from '@/lib/content/quizzes/finance-foundations/module-project-personal-finance-dashboard';

// Import multiple choice
import { financialSystemOverviewMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/financial-system-overview';
import { typesOfFinancialInstitutionsMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/types-of-financial-institutions';
import { careerPathsEngineersFinanceMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/career-paths-engineers-finance';
import { financialMarketsExplainedMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/financial-markets-explained';
import { investmentVehiclesProductsMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/investment-vehicles-products';
import { howTradingWorksMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/how-trading-works';
import { regulatoryLandscapeEngineersMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/regulatory-landscape-engineers';
import { financeTerminologyDevelopersMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/finance-terminology-developers';
import { mathematicsFinanceMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/mathematics-finance';
import { readingFinancialNewsDataMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/reading-financial-news-data';
import { learningEnvironmentMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/learning-environment';
import { moduleProjectPersonalFinanceDashboardMultipleChoice } from '@/lib/content/multiple-choice/finance-foundations/module-project-personal-finance-dashboard';

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

export const financeFoundationsModule: Module = {
  id: 'finance-foundations',
  title: 'Finance Foundations for Engineers',
  description:
    'Understanding the financial landscape, career paths, and why finance matters for engineers',
  icon: '🎓',
  sections: [
    {
      id: financialSystemOverview.id,
      title: financialSystemOverview.title,
      content: financialSystemOverview.content,
      quiz: financialSystemOverviewQuiz.map(transformQuiz),
      multipleChoice: financialSystemOverviewMultipleChoice.map(transformMC),
    },
    {
      id: typesOfFinancialInstitutions.id,
      title: typesOfFinancialInstitutions.title,
      content: typesOfFinancialInstitutions.content,
      quiz: typesOfFinancialInstitutionsQuiz.map(transformQuiz),
      multipleChoice:
        typesOfFinancialInstitutionsMultipleChoice.map(transformMC),
    },
    {
      id: careerPathsEngineersFinance.id,
      title: careerPathsEngineersFinance.title,
      content: careerPathsEngineersFinance.content,
      quiz: careerPathsEngineersFinanceQuiz.map(transformQuiz),
      multipleChoice:
        careerPathsEngineersFinanceMultipleChoice.map(transformMC),
    },
    {
      id: financialMarketsExplained.id,
      title: financialMarketsExplained.title,
      content: financialMarketsExplained.content,
      quiz: financialMarketsExplainedQuiz.map(transformQuiz),
      multipleChoice: financialMarketsExplainedMultipleChoice.map(transformMC),
    },
    {
      id: investmentVehiclesProducts.id,
      title: investmentVehiclesProducts.title,
      content: investmentVehiclesProducts.content,
      quiz: investmentVehiclesProductsQuiz.map(transformQuiz),
      multipleChoice: investmentVehiclesProductsMultipleChoice.map(transformMC),
    },
    {
      id: howTradingWorks.id,
      title: howTradingWorks.title,
      content: howTradingWorks.content,
      quiz: howTradingWorksQuiz.map(transformQuiz),
      multipleChoice: howTradingWorksMultipleChoice.map(transformMC),
    },
    {
      id: regulatoryLandscapeEngineers.id,
      title: regulatoryLandscapeEngineers.title,
      content: regulatoryLandscapeEngineers.content,
      quiz: regulatoryLandscapeEngineersQuiz.map(transformQuiz),
      multipleChoice:
        regulatoryLandscapeEngineersMultipleChoice.map(transformMC),
    },
    {
      id: financeTerminologyDevelopers.id,
      title: financeTerminologyDevelopers.title,
      content: financeTerminologyDevelopers.content,
      quiz: financeTerminologyDevelopersQuiz.map(transformQuiz),
      multipleChoice:
        financeTerminologyDevelopersMultipleChoice.map(transformMC),
    },
    {
      id: mathematicsForFinance.id,
      title: mathematicsForFinance.title,
      content: mathematicsForFinance.content,
      quiz: mathematicsFinanceQuiz.map(transformQuiz),
      multipleChoice: mathematicsFinanceMultipleChoice.map(transformMC),
    },
    {
      id: readingFinancialNewsData.id,
      title: readingFinancialNewsData.title,
      content: readingFinancialNewsData.content,
      quiz: readingFinancialNewsDataQuiz.map(transformQuiz),
      multipleChoice: readingFinancialNewsDataMultipleChoice.map(transformMC),
    },
    {
      id: learningEnvironment.id,
      title: learningEnvironment.title,
      content: learningEnvironment.content,
      quiz: learningEnvironmentQuiz.map(transformQuiz),
      multipleChoice: learningEnvironmentMultipleChoice.map(transformMC),
    },
    {
      id: moduleProjectPersonalFinanceDashboard.id,
      title: moduleProjectPersonalFinanceDashboard.title,
      content: moduleProjectPersonalFinanceDashboard.content,
      quiz: moduleProjectPersonalFinanceDashboardQuiz.map(transformQuiz),
      multipleChoice:
        moduleProjectPersonalFinanceDashboardMultipleChoice.map(transformMC),
    },
  ],
  quizzes: [
    ...financialSystemOverviewQuiz.map(transformQuiz),
    ...typesOfFinancialInstitutionsQuiz.map(transformQuiz),
    ...careerPathsEngineersFinanceQuiz.map(transformQuiz),
    ...financialMarketsExplainedQuiz.map(transformQuiz),
    ...investmentVehiclesProductsQuiz.map(transformQuiz),
    ...howTradingWorksQuiz.map(transformQuiz),
    ...regulatoryLandscapeEngineersQuiz.map(transformQuiz),
    ...financeTerminologyDevelopersQuiz.map(transformQuiz),
    ...mathematicsFinanceQuiz.map(transformQuiz),
    ...readingFinancialNewsDataQuiz.map(transformQuiz),
    ...learningEnvironmentQuiz.map(transformQuiz),
    ...moduleProjectPersonalFinanceDashboardQuiz.map(transformQuiz),
  ],
  multipleChoiceQuestions: [
    ...financialSystemOverviewMultipleChoice.map(transformMC),
    ...typesOfFinancialInstitutionsMultipleChoice.map(transformMC),
    ...careerPathsEngineersFinanceMultipleChoice.map(transformMC),
    ...financialMarketsExplainedMultipleChoice.map(transformMC),
    ...investmentVehiclesProductsMultipleChoice.map(transformMC),
    ...howTradingWorksMultipleChoice.map(transformMC),
    ...regulatoryLandscapeEngineersMultipleChoice.map(transformMC),
    ...financeTerminologyDevelopersMultipleChoice.map(transformMC),
    ...mathematicsFinanceMultipleChoice.map(transformMC),
    ...readingFinancialNewsDataMultipleChoice.map(transformMC),
    ...learningEnvironmentMultipleChoice.map(transformMC),
    ...moduleProjectPersonalFinanceDashboardMultipleChoice.map(transformMC),
  ],
};
