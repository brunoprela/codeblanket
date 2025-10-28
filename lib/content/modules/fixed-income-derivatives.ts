import { bondPricingFundamentals } from '../sections/fixed-income-derivatives/bond-pricing-fundamentals';
import { yieldCurvesTermStructure } from '../sections/fixed-income-derivatives/yield-curves-term-structure';
import { durationConvexity } from '../sections/fixed-income-derivatives/duration-convexity';
import { creditRiskSpreads } from '../sections/fixed-income-derivatives/credit-risk-spreads';
import { corporateBonds } from '../sections/fixed-income-derivatives/corporate-bonds';
import { governmentSecurities } from '../sections/fixed-income-derivatives/government-securities';
import { derivativesOverview } from '../sections/fixed-income-derivatives/derivatives-overview';
import { forwardFuturesContracts } from '../sections/fixed-income-derivatives/forward-futures-contracts';
import { swaps } from '../sections/fixed-income-derivatives/swaps';
import { creditDefaultSwaps } from '../sections/fixed-income-derivatives/credit-default-swaps';
import { exoticDerivatives } from '../sections/fixed-income-derivatives/exotic-derivatives';
import { hedgingStrategies } from '../sections/fixed-income-derivatives/hedging-strategies';
import { fixedIncomePortfolioManagement } from '../sections/fixed-income-derivatives/fixed-income-portfolio-management';
import { derivativeRiskManagement } from '../sections/fixed-income-derivatives/derivative-risk-management';
import { fixedIncomeAnalyticsPlatform } from '../sections/fixed-income-derivatives/fixed-income-analytics-platform';

import { bondPricingFundamentalsQuiz } from '../quizzes/fixed-income-derivatives/bond-pricing-fundamentals';
import { yieldCurvesTermStructureQuiz } from '../quizzes/fixed-income-derivatives/yield-curves-term-structure';
import { durationConvexityQuiz } from '../quizzes/fixed-income-derivatives/duration-convexity';
import { creditRiskSpreadsQuiz } from '../quizzes/fixed-income-derivatives/credit-risk-spreads';
import { corporateBondsQuiz } from '../quizzes/fixed-income-derivatives/corporate-bonds';
import { governmentSecuritiesQuiz } from '../quizzes/fixed-income-derivatives/government-securities';
import { derivativesOverviewQuiz } from '../quizzes/fixed-income-derivatives/derivatives-overview';
import { forwardFuturesContractsQuiz } from '../quizzes/fixed-income-derivatives/forward-futures-contracts';
import { swapsQuiz } from '../quizzes/fixed-income-derivatives/swaps';
import { creditDefaultSwapsQuiz } from '../quizzes/fixed-income-derivatives/credit-default-swaps';
import { exoticDerivativesQuiz } from '../quizzes/fixed-income-derivatives/exotic-derivatives';
import { hedgingStrategiesQuiz } from '../quizzes/fixed-income-derivatives/hedging-strategies';
import { fixedIncomePortfolioManagementQuiz } from '../quizzes/fixed-income-derivatives/fixed-income-portfolio-management';
import { derivativeRiskManagementQuiz } from '../quizzes/fixed-income-derivatives/derivative-risk-management';
import { fixedIncomeAnalyticsPlatformQuiz } from '../quizzes/fixed-income-derivatives/fixed-income-analytics-platform';

import { bondPricingFundamentalsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/bond-pricing-fundamentals';
import { yieldCurvesTermStructureMultipleChoice } from '../multiple-choice/fixed-income-derivatives/yield-curves-term-structure';
import { durationConvexityMultipleChoice } from '../multiple-choice/fixed-income-derivatives/duration-convexity';
import { creditRiskSpreadsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/credit-risk-spreads';
import { corporateBondsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/corporate-bonds';
import { governmentSecuritiesMultipleChoice } from '../multiple-choice/fixed-income-derivatives/government-securities';
import { derivativesOverviewMultipleChoice } from '../multiple-choice/fixed-income-derivatives/derivatives-overview';
import { forwardFuturesContractsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/forward-futures-contracts';
import { swapsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/swaps';
import { creditDefaultSwapsMultipleChoice } from '../multiple-choice/fixed-income-derivatives/credit-default-swaps';
import { exoticDerivativesMultipleChoice } from '../multiple-choice/fixed-income-derivatives/exotic-derivatives';
import { hedgingStrategiesMultipleChoice } from '../multiple-choice/fixed-income-derivatives/hedging-strategies';
import { fixedIncomePortfolioManagementMultipleChoice } from '../multiple-choice/fixed-income-derivatives/fixed-income-portfolio-management';
import { derivativeRiskManagementMultipleChoice } from '../multiple-choice/fixed-income-derivatives/derivative-risk-management';
import { fixedIncomeAnalyticsPlatformMultipleChoice } from '../multiple-choice/fixed-income-derivatives/fixed-income-analytics-platform';

import { Module } from '@/lib/types';

const transformQuiz = (questions: any[]) => {
  return questions.map((q) => ({
    id: q.id,
    question: q.question,
    sampleAnswer: q.sampleAnswer || '',
    keyPoints: q.keyPoints || [],
  }));
};

const transformMC = (questions: any[]) => {
  return questions.map((q) => ({
    id: q.id,
    question: q.question,
    options: q.options || [],
    correctAnswer: q.correctAnswer ?? 0,
    explanation: q.explanation || '',
  }));
};

export const fixedIncomeDerivativesModule: Module = {
  id: 'fixed-income-derivatives',
  title: 'Fixed Income & Derivatives',
  description:
    'Master bond pricing, yield curves, duration/convexity, derivatives (swaps, CDS, options), hedging strategies, portfolio management, and risk management for fixed income markets.',
  icon: '💵',
  sections: [
    {
      ...bondPricingFundamentals,
      quiz: transformQuiz(bondPricingFundamentalsQuiz),
      multipleChoice: transformMC(bondPricingFundamentalsMultipleChoice),
    },
    {
      ...yieldCurvesTermStructure,
      quiz: transformQuiz(yieldCurvesTermStructureQuiz),
      multipleChoice: transformMC(yieldCurvesTermStructureMultipleChoice),
    },
    {
      ...durationConvexity,
      quiz: transformQuiz(durationConvexityQuiz),
      multipleChoice: transformMC(durationConvexityMultipleChoice),
    },
    {
      ...creditRiskSpreads,
      quiz: transformQuiz(creditRiskSpreadsQuiz),
      multipleChoice: transformMC(creditRiskSpreadsMultipleChoice),
    },
    {
      ...corporateBonds,
      quiz: transformQuiz(corporateBondsQuiz),
      multipleChoice: transformMC(corporateBondsMultipleChoice),
    },
    {
      ...governmentSecurities,
      quiz: transformQuiz(governmentSecuritiesQuiz),
      multipleChoice: transformMC(governmentSecuritiesMultipleChoice),
    },
    {
      ...derivativesOverview,
      quiz: transformQuiz(derivativesOverviewQuiz),
      multipleChoice: transformMC(derivativesOverviewMultipleChoice),
    },
    {
      ...forwardFuturesContracts,
      quiz: transformQuiz(forwardFuturesContractsQuiz),
      multipleChoice: transformMC(forwardFuturesContractsMultipleChoice),
    },
    {
      ...swaps,
      quiz: transformQuiz(swapsQuiz),
      multipleChoice: transformMC(swapsMultipleChoice),
    },
    {
      ...creditDefaultSwaps,
      quiz: transformQuiz(creditDefaultSwapsQuiz),
      multipleChoice: transformMC(creditDefaultSwapsMultipleChoice),
    },
    {
      ...exoticDerivatives,
      quiz: transformQuiz(exoticDerivativesQuiz),
      multipleChoice: transformMC(exoticDerivativesMultipleChoice),
    },
    {
      ...hedgingStrategies,
      quiz: transformQuiz(hedgingStrategiesQuiz),
      multipleChoice: transformMC(hedgingStrategiesMultipleChoice),
    },
    {
      ...fixedIncomePortfolioManagement,
      quiz: transformQuiz(fixedIncomePortfolioManagementQuiz),
      multipleChoice: transformMC(fixedIncomePortfolioManagementMultipleChoice),
    },
    {
      ...derivativeRiskManagement,
      quiz: transformQuiz(derivativeRiskManagementQuiz),
      multipleChoice: transformMC(derivativeRiskManagementMultipleChoice),
    },
    {
      ...fixedIncomeAnalyticsPlatform,
      quiz: transformQuiz(fixedIncomeAnalyticsPlatformQuiz),
      multipleChoice: transformMC(fixedIncomeAnalyticsPlatformMultipleChoice),
    },
  ],
};
