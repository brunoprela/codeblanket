/**
 * Algorithmic Trading Strategies Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { algorithmicTradingOverview } from '../sections/algorithmic-trading-strategies/algorithmic-trading-overview';
import { trendFollowingStrategies } from '../sections/algorithmic-trading-strategies/trend-following-strategies';
import { meanReversionStrategies } from '../sections/algorithmic-trading-strategies/mean-reversion-strategies';
import { statisticalArbitrage } from '../sections/algorithmic-trading-strategies/statistical-arbitrage';
import { pairsTrading } from '../sections/algorithmic-trading-strategies/pairs-trading';
import { momentumStrategies } from '../sections/algorithmic-trading-strategies/momentum-strategies';
import { marketMakingStrategies } from '../sections/algorithmic-trading-strategies/market-making-strategies';
import { executionAlgorithms } from '../sections/algorithmic-trading-strategies/execution-algorithms-vwap-twap-pov';
import { newsBasedTrading } from '../sections/algorithmic-trading-strategies/news-based-trading';
import { sentimentAnalysisTrading } from '../sections/algorithmic-trading-strategies/sentiment-analysis-trading';
import { factorInvestingStrategies } from '../sections/algorithmic-trading-strategies/factor-investing-strategies';
import { multiAssetStrategies } from '../sections/algorithmic-trading-strategies/multi-asset-strategies';
import { strategyPerformanceAttribution } from '../sections/algorithmic-trading-strategies/strategy-performance-attribution';
import { projectMultiStrategyTradingSystem } from '../sections/algorithmic-trading-strategies/project-multi-strategy-trading-system';

// Import quizzes
import { algorithmicTradingOverviewQuiz } from '../quizzes/algorithmic-trading-strategies/algorithmic-trading-overview';
import { trendFollowingStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/trend-following-strategies';
import { meanReversionStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/mean-reversion-strategies';
import { statisticalArbitrageQuiz } from '../quizzes/algorithmic-trading-strategies/statistical-arbitrage';
import { pairsTradingQuiz } from '../quizzes/algorithmic-trading-strategies/pairs-trading';
import { momentumStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/momentum-strategies';
import { marketMakingStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/market-making-strategies';
import { executionAlgorithmsQuiz } from '../quizzes/algorithmic-trading-strategies/execution-algorithms-vwap-twap-pov';
import { newsBasedTradingQuiz } from '../quizzes/algorithmic-trading-strategies/news-based-trading';
import { sentimentAnalysisTradingQuiz } from '../quizzes/algorithmic-trading-strategies/sentiment-analysis-trading';
import { factorInvestingStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/factor-investing-strategies';
import { multiAssetStrategiesQuiz } from '../quizzes/algorithmic-trading-strategies/multi-asset-strategies';
import { strategyPerformanceAttributionQuiz } from '../quizzes/algorithmic-trading-strategies/strategy-performance-attribution';
import { projectMultiStrategyTradingSystemQuiz } from '../quizzes/algorithmic-trading-strategies/project-multi-strategy-trading-system';

// Import multiple choice
import { algorithmicTradingOverviewMC } from '../multiple-choice/algorithmic-trading-strategies/algorithmic-trading-overview';
import { trendFollowingStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/trend-following-strategies';
import { meanReversionStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/mean-reversion-strategies';
import { statisticalArbitrageMC } from '../multiple-choice/algorithmic-trading-strategies/statistical-arbitrage';
import { pairsTradingMC } from '../multiple-choice/algorithmic-trading-strategies/pairs-trading';
import { momentumStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/momentum-strategies';
import { marketMakingStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/market-making-strategies';
import { executionAlgorithmsMC } from '../multiple-choice/algorithmic-trading-strategies/execution-algorithms-vwap-twap-pov';
import { newsBasedTradingMC } from '../multiple-choice/algorithmic-trading-strategies/news-based-trading';
import { sentimentAnalysisTradingMC } from '../multiple-choice/algorithmic-trading-strategies/sentiment-analysis-trading';
import { factorInvestingStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/factor-investing-strategies';
import { multiAssetStrategiesMC } from '../multiple-choice/algorithmic-trading-strategies/multi-asset-strategies';
import { strategyPerformanceAttributionMC } from '../multiple-choice/algorithmic-trading-strategies/strategy-performance-attribution';
import { projectMultiStrategyTradingSystemMC } from '../multiple-choice/algorithmic-trading-strategies/project-multi-strategy-trading-system';

export const algorithmicTradingStrategiesModule: Module = {
    id: 'algorithmic-trading-strategies',
    title: 'Algorithmic Trading Strategies',
    description:
        'Master systematic trading strategies from trend following to multi-strategy systems with production-ready implementations',
    estimatedHours: 35,
    prerequisites: ['financial-markets-instruments', 'backtesting-strategy-development'],
    learningObjectives: [
        'Implement trend following, mean reversion, and statistical arbitrage strategies',
        'Build pairs trading systems with cointegration analysis',
        'Design market making and execution algorithms (VWAP, TWAP, POV)',
        'Trade on news and sentiment using NLP and alternative data',
        'Construct multi-factor and multi-asset portfolios',
        'Attribute performance to alpha, beta, and factor exposures',
        'Build production-ready multi-strategy trading systems',
    ],
    sections: [
        {
            ...algorithmicTradingOverview,
            quiz: algorithmicTradingOverviewQuiz,
            multipleChoice: algorithmicTradingOverviewMC,
        },
        {
            ...trendFollowingStrategies,
            quiz: trendFollowingStrategiesQuiz,
            multipleChoice: trendFollowingStrategiesMC,
        },
        {
            ...meanReversionStrategies,
            quiz: meanReversionStrategiesQuiz,
            multipleChoice: meanReversionStrategiesMC,
        },
        {
            ...statisticalArbitrage,
            quiz: statisticalArbitrageQuiz,
            multipleChoice: statisticalArbitrageMC,
        },
        {
            ...pairsTrading,
            quiz: pairsTradingQuiz,
            multipleChoice: pairsTradingMC,
        },
        {
            ...momentumStrategies,
            quiz: momentumStrategiesQuiz,
            multipleChoice: momentumStrategiesMC,
        },
        {
            ...marketMakingStrategies,
            quiz: marketMakingStrategiesQuiz,
            multipleChoice: marketMakingStrategiesMC,
        },
        {
            ...executionAlgorithms,
            quiz: executionAlgorithmsQuiz,
            multipleChoice: executionAlgorithmsMC,
        },
        {
            ...newsBasedTrading,
            quiz: newsBasedTradingQuiz,
            multipleChoice: newsBasedTradingMC,
        },
        {
            ...sentimentAnalysisTrading,
            quiz: sentimentAnalysisTradingQuiz,
            multipleChoice: sentimentAnalysisTradingMC,
        },
        {
            ...factorInvestingStrategies,
            quiz: factorInvestingStrategiesQuiz,
            multipleChoice: factorInvestingStrategiesMC,
        },
        {
            ...multiAssetStrategies,
            quiz: multiAssetStrategiesQuiz,
            multipleChoice: multiAssetStrategiesMC,
        },
        {
            ...strategyPerformanceAttribution,
            quiz: strategyPerformanceAttributionQuiz,
            multipleChoice: strategyPerformanceAttributionMC,
        },
        {
            ...projectMultiStrategyTradingSystem,
            quiz: projectMultiStrategyTradingSystemQuiz,
            multipleChoice: projectMultiStrategyTradingSystemMC,
        },
    ],
};

