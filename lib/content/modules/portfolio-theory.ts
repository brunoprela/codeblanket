import { Module } from '../../types';

// Import sections
import { modernPortfolioTheory } from '@/lib/content/sections/portfolio-theory/modern-portfolio-theory';
import { riskReturnMetrics } from '@/lib/content/sections/portfolio-theory/risk-return-metrics';
import { efficientFrontier } from '@/lib/content/sections/portfolio-theory/efficient-frontier';
import { capitalMarketLine } from '@/lib/content/sections/portfolio-theory/capital-market-line';
import { sharpeRatioPerformance } from '@/lib/content/sections/portfolio-theory/sharpe-ratio-performance';
import { meanVarianceOptimization } from '@/lib/content/sections/portfolio-theory/mean-variance-optimization';
import { blackLittermanModel } from '@/lib/content/sections/portfolio-theory/black-litterman-model';
import { assetAllocationStrategies } from '@/lib/content/sections/portfolio-theory/asset-allocation-strategies';
import { rebalancingStrategies } from '@/lib/content/sections/portfolio-theory/rebalancing-strategies';
import { factorModels } from '@/lib/content/sections/portfolio-theory/factor-models';
import { riskBudgeting } from '@/lib/content/sections/portfolio-theory/risk-budgeting';
import { portfolioConstraints } from '@/lib/content/sections/portfolio-theory/portfolio-constraints';
import { backtestingPortfolios } from '@/lib/content/sections/portfolio-theory/backtesting-portfolios';
import { portfolioOptimizationProject } from '@/lib/content/sections/portfolio-theory/portfolio-optimization-project';

// Import quizzes
import { modernPortfolioTheoryQuiz } from '@/lib/content/quizzes/portfolio-theory/modern-portfolio-theory';
import { riskReturnMetricsQuiz } from '@/lib/content/quizzes/portfolio-theory/risk-return-metrics';
import { efficientFrontierQuiz } from '@/lib/content/quizzes/portfolio-theory/efficient-frontier';
import { capitalMarketLineQuiz } from '@/lib/content/quizzes/portfolio-theory/capital-market-line';
import { sharpeRatioPerformanceQuiz } from '@/lib/content/quizzes/portfolio-theory/sharpe-ratio-performance';
import { meanVarianceOptimizationQuiz } from '@/lib/content/quizzes/portfolio-theory/mean-variance-optimization';
import { blackLittermanModelQuiz } from '@/lib/content/quizzes/portfolio-theory/black-litterman-model';
import { assetAllocationStrategiesQuiz } from '@/lib/content/quizzes/portfolio-theory/asset-allocation-strategies';
import { rebalancingStrategiesQuiz } from '@/lib/content/quizzes/portfolio-theory/rebalancing-strategies';
import { factorModelsQuiz } from '@/lib/content/quizzes/portfolio-theory/factor-models';
import { riskBudgetingQuiz } from '@/lib/content/quizzes/portfolio-theory/risk-budgeting';
import { portfolioConstraintsQuiz } from '@/lib/content/quizzes/portfolio-theory/portfolio-constraints';
import { backtestingPortfoliosQuiz } from '@/lib/content/quizzes/portfolio-theory/backtesting-portfolios';
import { portfolioOptimizationProjectQuiz } from '@/lib/content/quizzes/portfolio-theory/portfolio-optimization-project';

// Import multiple choice
import { modernPortfolioTheoryMC } from '@/lib/content/multiple-choice/portfolio-theory/modern-portfolio-theory';
import { riskReturnMetricsMC } from '@/lib/content/multiple-choice/portfolio-theory/risk-return-metrics';
import { efficientFrontierMC } from '@/lib/content/multiple-choice/portfolio-theory/efficient-frontier';
import { capitalMarketLineMC } from '@/lib/content/multiple-choice/portfolio-theory/capital-market-line';
import { sharpeRatioPerformanceMC } from '@/lib/content/multiple-choice/portfolio-theory/sharpe-ratio-performance';
import { meanVarianceOptimizationMC } from '@/lib/content/multiple-choice/portfolio-theory/mean-variance-optimization';
import { blackLittermanModelMC } from '@/lib/content/multiple-choice/portfolio-theory/black-litterman-model';
import { assetAllocationStrategiesMC } from '@/lib/content/multiple-choice/portfolio-theory/asset-allocation-strategies';
import { rebalancingStrategiesMC } from '@/lib/content/multiple-choice/portfolio-theory/rebalancing-strategies';
import { factorModelsMC } from '@/lib/content/multiple-choice/portfolio-theory/factor-models';
import { riskBudgetingMC } from '@/lib/content/multiple-choice/portfolio-theory/risk-budgeting';
import { portfolioConstraintsMC } from '@/lib/content/multiple-choice/portfolio-theory/portfolio-constraints';
import { backtestingPortfoliosMC } from '@/lib/content/multiple-choice/portfolio-theory/backtesting-portfolios';
import { portfolioOptimizationProjectMC } from '@/lib/content/multiple-choice/portfolio-theory/portfolio-optimization-project';

// Transform functions for quizzes and multiple choice
const transformQuizQuestion = (q: any) => ({
  id: q.id,
  text: q.text,
  type: q.type,
  sampleAnswer: q.sampleAnswer,
  keyPoints: q.keyPoints,
});

const transformMCQuestion = (q: any) => ({
  id: q.id,
  type: q.type,
  question: q.question,
  options: q.options,
  correctAnswer: q.correctAnswer,
  explanation: q.explanation,
});

export const portfolioTheoryModule: Module = {
  id: 'portfolio-theory',
  title: 'Portfolio Theory & Asset Allocation',
  description:
    'Master Modern Portfolio Theory, optimization techniques, factor models, and institutional portfolio management',
  sections: [
    {
      id: modernPortfolioTheory.id,
      title: modernPortfolioTheory.title,
      content: modernPortfolioTheory.content,
      quiz: modernPortfolioTheoryQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        modernPortfolioTheoryMC.questions.map(transformMCQuestion),
    },
    {
      id: riskReturnMetrics.id,
      title: riskReturnMetrics.title,
      content: riskReturnMetrics.content,
      quiz: riskReturnMetricsQuiz.questions.map(transformQuizQuestion),
      multipleChoice: riskReturnMetricsMC.questions.map(transformMCQuestion),
    },
    {
      id: efficientFrontier.id,
      title: efficientFrontier.title,
      content: efficientFrontier.content,
      quiz: efficientFrontierQuiz.questions.map(transformQuizQuestion),
      multipleChoice: efficientFrontierMC.questions.map(transformMCQuestion),
    },
    {
      id: capitalMarketLine.id,
      title: capitalMarketLine.title,
      content: capitalMarketLine.content,
      quiz: capitalMarketLineQuiz.questions.map(transformQuizQuestion),
      multipleChoice: capitalMarketLineMC.questions.map(transformMCQuestion),
    },
    {
      id: sharpeRatioPerformance.id,
      title: sharpeRatioPerformance.title,
      content: sharpeRatioPerformance.content,
      quiz: sharpeRatioPerformanceQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        sharpeRatioPerformanceMC.questions.map(transformMCQuestion),
    },
    {
      id: meanVarianceOptimization.id,
      title: meanVarianceOptimization.title,
      content: meanVarianceOptimization.content,
      quiz: meanVarianceOptimizationQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        meanVarianceOptimizationMC.questions.map(transformMCQuestion),
    },
    {
      id: blackLittermanModel.id,
      title: blackLittermanModel.title,
      content: blackLittermanModel.content,
      quiz: blackLittermanModelQuiz.questions.map(transformQuizQuestion),
      multipleChoice: blackLittermanModelMC.questions.map(transformMCQuestion),
    },
    {
      id: assetAllocationStrategies.id,
      title: assetAllocationStrategies.title,
      content: assetAllocationStrategies.content,
      quiz: assetAllocationStrategiesQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        assetAllocationStrategiesMC.questions.map(transformMCQuestion),
    },
    {
      id: rebalancingStrategies.id,
      title: rebalancingStrategies.title,
      content: rebalancingStrategies.content,
      quiz: rebalancingStrategiesQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        rebalancingStrategiesMC.questions.map(transformMCQuestion),
    },
    {
      id: factorModels.id,
      title: factorModels.title,
      content: factorModels.content,
      quiz: factorModelsQuiz.questions.map(transformQuizQuestion),
      multipleChoice: factorModelsMC.questions.map(transformMCQuestion),
    },
    {
      id: riskBudgeting.id,
      title: riskBudgeting.title,
      content: riskBudgeting.content,
      quiz: riskBudgetingQuiz.questions.map(transformQuizQuestion),
      multipleChoice: riskBudgetingMC.questions.map(transformMCQuestion),
    },
    {
      id: portfolioConstraints.id,
      title: portfolioConstraints.title,
      content: portfolioConstraints.content,
      quiz: portfolioConstraintsQuiz.questions.map(transformQuizQuestion),
      multipleChoice: portfolioConstraintsMC.questions.map(transformMCQuestion),
    },
    {
      id: backtestingPortfolios.id,
      title: backtestingPortfolios.title,
      content: backtestingPortfolios.content,
      quiz: backtestingPortfoliosQuiz.questions.map(transformQuizQuestion),
      multipleChoice:
        backtestingPortfoliosMC.questions.map(transformMCQuestion),
    },
    {
      id: portfolioOptimizationProject.id,
      title: portfolioOptimizationProject.title,
      content: portfolioOptimizationProject.content,
      quiz: portfolioOptimizationProjectQuiz.questions.map(
        transformQuizQuestion,
      ),
      multipleChoice:
        portfolioOptimizationProjectMC.questions.map(transformMCQuestion),
    },
  ],
  quizzes: [
    ...modernPortfolioTheoryQuiz.questions.map(transformQuizQuestion),
    ...riskReturnMetricsQuiz.questions.map(transformQuizQuestion),
    ...efficientFrontierQuiz.questions.map(transformQuizQuestion),
    ...capitalMarketLineQuiz.questions.map(transformQuizQuestion),
    ...sharpeRatioPerformanceQuiz.questions.map(transformQuizQuestion),
    ...meanVarianceOptimizationQuiz.questions.map(transformQuizQuestion),
    ...blackLittermanModelQuiz.questions.map(transformQuizQuestion),
    ...assetAllocationStrategiesQuiz.questions.map(transformQuizQuestion),
    ...rebalancingStrategiesQuiz.questions.map(transformQuizQuestion),
    ...factorModelsQuiz.questions.map(transformQuizQuestion),
    ...riskBudgetingQuiz.questions.map(transformQuizQuestion),
    ...portfolioConstraintsQuiz.questions.map(transformQuizQuestion),
    ...backtestingPortfoliosQuiz.questions.map(transformQuizQuestion),
    ...portfolioOptimizationProjectQuiz.questions.map(transformQuizQuestion),
  ],
  multipleChoiceQuestions: [
    ...modernPortfolioTheoryMC.questions.map(transformMCQuestion),
    ...riskReturnMetricsMC.questions.map(transformMCQuestion),
    ...efficientFrontierMC.questions.map(transformMCQuestion),
    ...capitalMarketLineMC.questions.map(transformMCQuestion),
    ...sharpeRatioPerformanceMC.questions.map(transformMCQuestion),
    ...meanVarianceOptimizationMC.questions.map(transformMCQuestion),
    ...blackLittermanModelMC.questions.map(transformMCQuestion),
    ...assetAllocationStrategiesMC.questions.map(transformMCQuestion),
    ...rebalancingStrategiesMC.questions.map(transformMCQuestion),
    ...factorModelsMC.questions.map(transformMCQuestion),
    ...riskBudgetingMC.questions.map(transformMCQuestion),
    ...portfolioConstraintsMC.questions.map(transformMCQuestion),
    ...backtestingPortfoliosMC.questions.map(transformMCQuestion),
    ...portfolioOptimizationProjectMC.questions.map(transformMCQuestion),
  ],
};
