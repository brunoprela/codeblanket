/**
 * Module: Quantitative Interview Preparation
 */

import { Module } from '../../types';

// Section imports
import { probabilityPuzzles } from '@/lib/content/sections/quant-interview-prep/probability-puzzles';
import { optionsPricingMentalMath } from '@/lib/content/sections/quant-interview-prep/options-pricing-mental-math';
import { combinatoricsCounting } from '@/lib/content/sections/quant-interview-prep/combinatorics-counting';
import { calculusIntegrationPuzzles } from '@/lib/content/sections/quant-interview-prep/calculus-integration-puzzles';
import { linearAlgebraProblems } from '@/lib/content/sections/quant-interview-prep/linear-algebra-problems';
import { statisticsInference } from '@/lib/content/sections/quant-interview-prep/statistics-inference';
import { financialMathPuzzles } from '@/lib/content/sections/quant-interview-prep/financial-math-puzzles';
import { marketMicrostructurePuzzles } from '@/lib/content/sections/quant-interview-prep/market-microstructure-puzzles';
import { codingChallenges } from '@/lib/content/sections/quant-interview-prep/coding-challenges-quant';
import { fermiEstimation } from '@/lib/content/sections/quant-interview-prep/fermi-estimation-market-sense';
import { mockInterviewProblems } from '@/lib/content/sections/quant-interview-prep/mock-interview-problems';
import { tradingGamesSimulations } from '@/lib/content/sections/quant-interview-prep/trading-games-simulations';

// Quiz imports
import { probabilityPuzzlesQuiz } from '@/lib/content/quizzes/quant-interview-prep/probability-puzzles';
import { optionsPricingMentalMathQuiz } from '@/lib/content/quizzes/quant-interview-prep/options-pricing-mental-math';
import { combinatoricsCountingQuiz } from '@/lib/content/quizzes/quant-interview-prep/combinatorics-counting';
import { calculusIntegrationPuzzlesQuiz } from '@/lib/content/quizzes/quant-interview-prep/calculus-integration-puzzles';
import { linearAlgebraProblemsQuiz } from '@/lib/content/quizzes/quant-interview-prep/linear-algebra-problems';
import { statisticsInferenceQuiz } from '@/lib/content/quizzes/quant-interview-prep/statistics-inference';
import { financialMathPuzzlesQuiz } from '@/lib/content/quizzes/quant-interview-prep/financial-math-puzzles';
import { marketMicrostructurePuzzlesQuiz } from '@/lib/content/quizzes/quant-interview-prep/market-microstructure-puzzles';
import { codingChallengesQuiz } from '@/lib/content/quizzes/quant-interview-prep/coding-challenges-quant';
import { fermiEstimationQuiz } from '@/lib/content/quizzes/quant-interview-prep/fermi-estimation-market-sense';
import { mockInterviewProblemsQuiz } from '@/lib/content/quizzes/quant-interview-prep/mock-interview-problems';
import { tradingGamesSimulationsQuiz } from '@/lib/content/quizzes/quant-interview-prep/trading-games-simulations';

// Multiple choice imports
import { probabilityPuzzlesMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/probability-puzzles';
import { optionsPricingMentalMathMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/options-pricing-mental-math';
import { combinatoricsCountingMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/combinatorics-counting';
import { calculusIntegrationPuzzlesMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/calculus-integration-puzzles';
import { linearAlgebraProblemsMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/linear-algebra-problems';
import { statisticsInferenceMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/statistics-inference';
import { financialMathPuzzlesMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/financial-math-puzzles';
import { marketMicrostructurePuzzlesMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/market-microstructure-puzzles';
import { codingChallengesMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/coding-challenges-quant';
import { fermiEstimationMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/fermi-estimation-market-sense';
import { mockInterviewProblemsMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/mock-interview-problems';
import { tradingGamesSimulationsMultipleChoice } from '@/lib/content/multiple-choice/quant-interview-prep/trading-games-simulations';

// Helper to transform quiz format
const transformQuiz = (
  quiz:
    | {
        questions: Array<{
          id: string;
          question: string;
          sampleAnswer: string;
          keyPoints: string[];
        }>;
      }
    | Array<{
        id: string;
        question: string;
        sampleAnswer: string;
        keyPoints: string[];
      }>,
) => {
  if (Array.isArray(quiz)) {
    return quiz;
  }
  return quiz.questions;
};

// Helper to transform multiple choice format
const transformMC = (
  mc:
    | {
        questions: Array<{
          id: string;
          question: string;
          options: string[];
          correctAnswer: number;
          explanation: string;
        }>;
      }
    | Array<{
        id: string;
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

export const quantInterviewPrepModule: Module = {
  id: 'quant-interview-prep',
  title: 'Quantitative Interview Preparation',
  description:
    'Master quantitative trading interviews at top firms (Citadel, Jane Street, Two Sigma, Jump Trading, HRT). Comprehensive coverage of probability puzzles, options pricing, combinatorics, coding challenges, market sense, and trading games with 150+ problems and complete solutions.',
  icon: '🎯',
  sections: [
    {
      ...probabilityPuzzles,
      quiz: transformQuiz(probabilityPuzzlesQuiz),
      multipleChoice: transformMC(probabilityPuzzlesMultipleChoice),
    },
    {
      ...optionsPricingMentalMath,
      quiz: transformQuiz(optionsPricingMentalMathQuiz),
      multipleChoice: transformMC(optionsPricingMentalMathMultipleChoice),
    },
    {
      ...combinatoricsCounting,
      quiz: transformQuiz(combinatoricsCountingQuiz),
      multipleChoice: transformMC(combinatoricsCountingMultipleChoice),
    },
    {
      ...calculusIntegrationPuzzles,
      quiz: transformQuiz(calculusIntegrationPuzzlesQuiz),
      multipleChoice: transformMC(calculusIntegrationPuzzlesMultipleChoice),
    },
    {
      ...linearAlgebraProblems,
      quiz: transformQuiz(linearAlgebraProblemsQuiz),
      multipleChoice: transformMC(linearAlgebraProblemsMultipleChoice),
    },
    {
      ...statisticsInference,
      quiz: transformQuiz(statisticsInferenceQuiz),
      multipleChoice: transformMC(statisticsInferenceMultipleChoice),
    },
    {
      ...financialMathPuzzles,
      quiz: transformQuiz(financialMathPuzzlesQuiz),
      multipleChoice: transformMC(financialMathPuzzlesMultipleChoice),
    },
    {
      ...marketMicrostructurePuzzles,
      quiz: transformQuiz(marketMicrostructurePuzzlesQuiz),
      multipleChoice: transformMC(marketMicrostructurePuzzlesMultipleChoice),
    },
    {
      ...codingChallenges,
      quiz: transformQuiz(codingChallengesQuiz),
      multipleChoice: transformMC(codingChallengesMultipleChoice),
    },
    {
      ...fermiEstimation,
      quiz: transformQuiz(fermiEstimationQuiz),
      multipleChoice: transformMC(fermiEstimationMultipleChoice),
    },
    {
      ...mockInterviewProblems,
      quiz: transformQuiz(mockInterviewProblemsQuiz),
      multipleChoice: transformMC(mockInterviewProblemsMultipleChoice),
    },
    {
      ...tradingGamesSimulations,
      quiz: transformQuiz(tradingGamesSimulationsQuiz),
      multipleChoice: transformMC(tradingGamesSimulationsMultipleChoice),
    },
  ],
  keyTakeaways: [
    'Master 100+ probability puzzles from classic (Monty Hall) to advanced (martingales)',
    'Price options mentally: Black-Scholes calculations, Greeks, implied volatility',
    'Solve combinatorics problems: permutations, combinations, generating functions',
    'Handle calculus puzzles: optimization, integration, differential equations',
    'Apply linear algebra: matrix operations, eigenvalues, portfolio optimization',
    'Perform statistical inference: hypothesis testing, confidence intervals, power analysis',
    'Solve financial math: present value, bond pricing, arbitrage detection',
    'Understand market microstructure: order books, market impact, bid-ask spreads',
    'Code algorithmic solutions: option pricers, order books, backtesting systems',
    'Estimate with confidence: Fermi problems, market sizing, quick approximations',
    'Handle mock interviews: multi-part problems combining multiple skill areas',
    'Excel at trading games: market making, Kelly criterion, risk management',
  ],
  learningObjectives: [
    'Solve probability puzzles using multiple approaches (intuition, rigorous math, simulation)',
    'Calculate Black-Scholes prices and Greeks mentally for common scenarios',
    'Apply combinatorial methods to counting problems and derive closed-form solutions',
    'Optimize functions using calculus and solve integration problems analytically',
    'Perform matrix operations mentally and understand eigenvalue applications in finance',
    'Design hypothesis tests, construct confidence intervals, and calculate statistical power',
    'Solve time-value problems and detect arbitrage opportunities in derivatives markets',
    'Analyze order book dynamics and estimate transaction costs for large trades',
    'Implement financial algorithms in Python with proper testing and complexity analysis',
    'Make reasonable estimates with limited information using structured approaches',
    'Integrate multiple skills to solve realistic interview problems under time pressure',
    'Make optimal trading decisions in games with uncertainty and risk management',
  ],
  prerequisites: [
    'Strong foundation in probability and statistics',
    'Comfort with mental math and numerical approximation',
    'Understanding of financial markets and derivatives',
    'Python programming skills (for coding sections)',
    'Linear algebra and calculus fundamentals',
    'Basic knowledge of options, bonds, and trading concepts',
  ],
};
