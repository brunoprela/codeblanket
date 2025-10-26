/**
 * Module: Quantitative Finance Fundamentals
 */

import { Module } from '../../types';

// Section imports
import { optionsFundamentals } from '@/lib/content/sections/quantitative-finance/options-fundamentals';
import { blackScholesModel } from '@/lib/content/sections/quantitative-finance/black-scholes-model';
import { theGreeks } from '@/lib/content/sections/quantitative-finance/the-greeks';
import { portfolioTheory } from '@/lib/content/sections/quantitative-finance/portfolio-theory';
import { factorModels } from '@/lib/content/sections/quantitative-finance/factor-models';
import { fixedIncomeAndBonds } from '@/lib/content/sections/quantitative-finance/fixed-income-and-bonds';
import { derivativesPricing } from '@/lib/content/sections/quantitative-finance/derivatives-pricing';
import { riskMeasures } from '@/lib/content/sections/quantitative-finance/risk-measures';
import { marketMicrostructure } from '@/lib/content/sections/quantitative-finance/market-microstructure';
import { statisticalArbitrage } from '@/lib/content/sections/quantitative-finance/statistical-arbitrage';
import { quantTradingStrategies } from '@/lib/content/sections/quantitative-finance/quant-trading-strategies';
import { alternativeInvestments } from '@/lib/content/sections/quantitative-finance/alternative-investments';

// Quiz imports
import { optionsFundamentalsQuiz } from '@/lib/content/quizzes/quantitative-finance/options-fundamentals';
import { blackScholesModelQuiz } from '@/lib/content/quizzes/quantitative-finance/black-scholes-model';
import { theGreeksQuiz } from '@/lib/content/quizzes/quantitative-finance/the-greeks';
import { portfolioTheoryQuiz } from '@/lib/content/quizzes/quantitative-finance/portfolio-theory';
import { factorModelsQuiz } from '@/lib/content/quizzes/quantitative-finance/factor-models';
import { fixedIncomeAndBondsQuiz } from '@/lib/content/quizzes/quantitative-finance/fixed-income-and-bonds';
import { derivativesPricingQuiz } from '@/lib/content/quizzes/quantitative-finance/derivatives-pricing';
import { riskMeasuresQuiz } from '@/lib/content/quizzes/quantitative-finance/risk-measures';
import { marketMicrostructureQuiz } from '@/lib/content/quizzes/quantitative-finance/market-microstructure';
import { statisticalArbitrageQuiz } from '@/lib/content/quizzes/quantitative-finance/statistical-arbitrage';
import { quantTradingStrategiesQuiz } from '@/lib/content/quizzes/quantitative-finance/quant-trading-strategies';
import { alternativeInvestmentsQuiz } from '@/lib/content/quizzes/quantitative-finance/alternative-investments';

// Multiple choice imports
import { optionsFundamentalsMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/options-fundamentals';
import { blackScholesModelMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/black-scholes-model';
import { theGreeksMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/the-greeks';
import { portfolioTheoryMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/portfolio-theory';
import { factorModelsMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/factor-models';
import { fixedIncomeAndBondsMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/fixed-income-and-bonds';
import { derivativesPricingMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/derivatives-pricing';
import { riskMeasuresMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/risk-measures';
import { marketMicrostructureMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/market-microstructure';
import { statisticalArbitrageMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/statistical-arbitrage';
import { quantTradingStrategiesMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/quant-trading-strategies';
import { alternativeInvestmentsMultipleChoice } from '@/lib/content/multiple-choice/quantitative-finance/alternative-investments';

// Helper to transform quiz format (quizzes with wrapped questions)
const transformQuiz = (
  quiz:
    | {
        questions: Array<{
          id: string;
          question: string;
          sampleAnswer?: string;
          answer?: string;
          keyPoints?: string[];
        }>;
      }
    | Array<{
        id: string;
        question: string;
        sampleAnswer?: string;
        answer?: string;
        keyPoints?: string[];
      }>,
) => {
  if (Array.isArray(quiz)) {
    return quiz;
  }
  return quiz.questions;
};

// Helper to transform multiple choice format (mc with wrapped questions)
const transformMC = (
  mc:
    | {
        questions: Array<{
          id: string | number;
          question: string;
          options: string[];
          correctAnswer: number;
          explanation: string;
        }>;
      }
    | Array<{
        id: string | number;
        question: string;
        options: string[];
        correctAnswer: number;
        explanation: string;
      }>,
) => {
  if (Array.isArray(mc)) {
    return mc;
  }
  return mc.questions;
};

export const quantitativeFinanceModule: Module = {
  id: 'quantitative-finance',
  title: 'Quantitative Finance Fundamentals',
  description:
    'Master options pricing, derivatives, portfolio theory, and quantitative finance for professional trading. Learn Black-Scholes, Greeks, factor models, risk measures, and build a complete quantitative trading foundation.',
  icon: '💰',
  sections: [
    {
      ...optionsFundamentals,
      quiz: transformQuiz(optionsFundamentalsQuiz),
      multipleChoice: transformMC(optionsFundamentalsMultipleChoice),
    },
    {
      ...blackScholesModel,
      quiz: transformQuiz(blackScholesModelQuiz),
      multipleChoice: transformMC(blackScholesModelMultipleChoice),
    },
    {
      ...theGreeks,
      quiz: transformQuiz(theGreeksQuiz),
      multipleChoice: transformMC(theGreeksMultipleChoice),
    },
    {
      ...portfolioTheory,
      quiz: transformQuiz(portfolioTheoryQuiz),
      multipleChoice: transformMC(portfolioTheoryMultipleChoice),
    },
    {
      ...factorModels,
      quiz: transformQuiz(factorModelsQuiz),
      multipleChoice: transformMC(factorModelsMultipleChoice),
    },
    {
      ...fixedIncomeAndBonds,
      quiz: transformQuiz(fixedIncomeAndBondsQuiz),
      multipleChoice: transformMC(fixedIncomeAndBondsMultipleChoice),
    },
    {
      ...derivativesPricing,
      quiz: transformQuiz(derivativesPricingQuiz),
      multipleChoice: transformMC(derivativesPricingMultipleChoice),
    },
    {
      ...riskMeasures,
      quiz: transformQuiz(riskMeasuresQuiz),
      multipleChoice: transformMC(riskMeasuresMultipleChoice),
    },
    {
      ...marketMicrostructure,
      quiz: transformQuiz(marketMicrostructureQuiz),
      multipleChoice: transformMC(marketMicrostructureMultipleChoice),
    },
    {
      ...statisticalArbitrage,
      quiz: transformQuiz(statisticalArbitrageQuiz),
      multipleChoice: transformMC(statisticalArbitrageMultipleChoice),
    },
    {
      ...quantTradingStrategies,
      quiz: transformQuiz(quantTradingStrategiesQuiz),
      multipleChoice: transformMC(quantTradingStrategiesMultipleChoice),
    },
    {
      ...alternativeInvestments,
      quiz: transformQuiz(alternativeInvestmentsQuiz),
      multipleChoice: transformMC(alternativeInvestmentsMultipleChoice),
    },
  ],
  keyTakeaways: [
    'Master options fundamentals: calls, puts, strategies, and payoff diagrams',
    'Understand Black-Scholes pricing model, implied volatility, and volatility smile/skew',
    'Calculate and interpret Greeks (Delta, Gamma, Theta, Vega, Rho) for risk management',
    'Apply modern portfolio theory, CAPM, and efficient frontier optimization',
    'Analyze factor models (Fama-French) and smart beta strategies',
    'Price fixed income securities: bonds, duration, convexity, yield curve',
    'Value derivatives: forwards, futures, swaps, and exotic options',
    'Implement risk measures: VaR, CVaR, volatility estimation, stress testing',
    'Understand market microstructure: bid-ask spread, order flow, liquidity',
    'Build statistical arbitrage strategies: pairs trading, cointegration, mean reversion',
    'Develop quantitative trading strategies with proper risk management',
    'Diversify with alternative investments: hedge funds, private equity, commodities',
  ],
  learningObjectives: [
    'Price options using Black-Scholes and understand its assumptions and limitations',
    'Calculate implied volatility and interpret volatility smile/skew patterns',
    'Manage option positions using Greeks: delta hedging, gamma scalping, vega trading',
    'Construct optimal portfolios using mean-variance optimization and factor models',
    'Analyze and price fixed income securities with duration and convexity',
    'Value derivatives contracts: forwards, futures, swaps, and structured products',
    'Implement professional risk management: VaR, stress testing, scenario analysis',
    'Understand market microstructure and its impact on trading strategies',
    'Build pairs trading and statistical arbitrage strategies',
    'Develop factor-based trading strategies (momentum, value, quality)',
    'Evaluate alternative investments and their role in portfolio diversification',
    'Apply quantitative methods to real-world trading with Python implementations',
  ],
  prerequisites: [
    'Calculus fundamentals (derivatives, optimization)',
    'Linear algebra (matrices, eigenvalues)',
    'Probability and statistics (distributions, hypothesis testing)',
    'Python programming with NumPy, Pandas, Matplotlib',
    'Understanding of financial markets and basic trading concepts',
  ],
};
