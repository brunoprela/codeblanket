// Module 10: Backtesting & Strategy Development
// Complete index file exporting all sections

import backtestingFundamentals from './10-1-backtesting-fundamentals';
import backtestingFundamentalsQuiz from '../../../quizzes/backtesting-strategy-development/10-1-backtesting-fundamentals-quiz';
import backtestingFundamentalsDiscussion from '../../../discussions/backtesting-strategy-development/10-1-backtesting-fundamentals-discussion';

import historicalDataManagement from './10-2-historical-data-management';
import historicalDataManagementQuiz from '../../../quizzes/backtesting-strategy-development/10-2-historical-data-management-quiz';
import historicalDataManagementDiscussion from '../../../discussions/backtesting-strategy-development/10-2-historical-data-management-discussion';

import eventDrivenBacktestingArchitecture from './10-3-event-driven-backtesting-architecture';
import eventDrivenBacktestingArchitectureQuiz from '../../../quizzes/backtesting-strategy-development/10-3-event-driven-backtesting-architecture-quiz';
import eventDrivenBacktestingArchitectureDiscussion from '../../../discussions/backtesting-strategy-development/10-3-event-driven-backtesting-architecture-discussion';

import performanceMetricsForTrading from './10-4-performance-metrics-for-trading';
import performanceMetricsForTradingQuiz from '../../../quizzes/backtesting-strategy-development/10-4-performance-metrics-for-trading-quiz';
import performanceMetricsForTradingDiscussion from '../../../discussions/backtesting-strategy-development/10-4-performance-metrics-for-trading-discussion';

import transactionCostsAndSlippage from './10-5-transaction-costs-and-slippage';
import transactionCostsAndSlippageQuiz from '../../../quizzes/backtesting-strategy-development/10-5-transaction-costs-and-slippage-quiz';
import transactionCostsAndSlippageDiscussion from '../../../discussions/backtesting-strategy-development/10-5-transaction-costs-and-slippage-discussion';

import walkForwardAnalysis from './10-6-walk-forward-analysis';
import walkForwardAnalysisQuiz from '../../../quizzes/backtesting-strategy-development/10-6-walk-forward-analysis-quiz';
import walkForwardAnalysisDiscussion from '../../../discussions/backtesting-strategy-development/10-6-walk-forward-analysis-discussion';

import monteCarloSimulation from './10-7-monte-carlo-simulation';
import monteCarloSimulationQuiz from '../../../quizzes/backtesting-strategy-development/10-7-monte-carlo-simulation-quiz';
import monteCarloSimulationDiscussion from '../../../discussions/backtesting-strategy-development/10-7-monte-carlo-simulation-discussion';

import overfittingAndDataMining from './10-8-overfitting-and-data-mining';
import overfittingAndDataMiningQuiz from '../../../quizzes/backtesting-strategy-development/10-8-overfitting-and-data-mining-quiz';
import overfittingAndDataMiningDiscussion from '../../../discussions/backtesting-strategy-development/10-8-overfitting-and-data-mining-discussion';

import statisticalSignificanceTesting from './10-9-statistical-significance-testing';
import statisticalSignificanceTestingQuiz from '../../../quizzes/backtesting-strategy-development/10-9-statistical-significance-testing-quiz';
import statisticalSignificanceTestingDiscussion from '../../../discussions/backtesting-strategy-development/10-9-statistical-significance-testing-discussion';

import outOfSampleTesting from './10-10-out-of-sample-testing';
import outOfSampleTestingQuiz from '../../../quizzes/backtesting-strategy-development/10-10-out-of-sample-testing-quiz';
import outOfSampleTestingDiscussion from '../../../discussions/backtesting-strategy-development/10-10-out-of-sample-testing-discussion';

import timeSeriesCrossValidation from './10-11-time-series-cross-validation';
import timeSeriesCrossValidationQuiz from '../../../quizzes/backtesting-strategy-development/10-11-time-series-cross-validation-quiz';
import timeSeriesCrossValidationDiscussion from '../../../discussions/backtesting-strategy-development/10-11-time-series-cross-validation-discussion';

import benchmarkComparison from './10-12-benchmark-comparison';
import benchmarkComparisonQuiz from '../../../quizzes/backtesting-strategy-development/10-12-benchmark-comparison-quiz';
import benchmarkComparisonDiscussion from '../../../discussions/backtesting-strategy-development/10-12-benchmark-comparison-discussion';

import buildingBacktestingFramework from './10-13-building-backtesting-framework';
import buildingBacktestingFrameworkQuiz from '../../../quizzes/backtesting-strategy-development/10-13-building-backtesting-framework-quiz';
import buildingBacktestingFrameworkDiscussion from '../../../discussions/backtesting-strategy-development/10-13-building-backtesting-framework-discussion';

import paperTradingVsLive from './10-14-paper-trading-vs-live';
import paperTradingVsLiveQuiz from '../../../quizzes/backtesting-strategy-development/10-14-paper-trading-vs-live-quiz';
import paperTradingVsLiveDiscussion from '../../../discussions/backtesting-strategy-development/10-14-paper-trading-vs-live-discussion';

import strategyParameterOptimization from './10-15-strategy-parameter-optimization';
import strategyParameterOptimizationQuiz from '../../../quizzes/backtesting-strategy-development/10-15-strategy-parameter-optimization-quiz';
import strategyParameterOptimizationDiscussion from '../../../discussions/backtesting-strategy-development/10-15-strategy-parameter-optimization-discussion';

import productionBacktestingEngineProject from './10-16-production-backtesting-engine-project';
import productionBacktestingEngineProjectQuiz from '../../../quizzes/backtesting-strategy-development/10-16-production-backtesting-engine-project-quiz';
import productionBacktestingEngineProjectDiscussion from '../../../discussions/backtesting-strategy-development/10-16-production-backtesting-engine-project-discussion';

export const backtestingStrategyDevelopment = {
  title: 'Backtesting & Strategy Development',
  description:
    'Master the complete backtesting lifecycle from data management through production deployment, including walk-forward analysis, parameter optimization, and building production-grade backtesting infrastructure',
  sections: [
    {
      id: '10-1',
      ...backtestingFundamentals,
      quiz: backtestingFundamentalsQuiz,
      discussion: backtestingFundamentalsDiscussion,
    },
    {
      id: '10-2',
      ...historicalDataManagement,
      quiz: historicalDataManagementQuiz,
      discussion: historicalDataManagementDiscussion,
    },
    {
      id: '10-3',
      ...eventDrivenBacktestingArchitecture,
      quiz: eventDrivenBacktestingArchitectureQuiz,
      discussion: eventDrivenBacktestingArchitectureDiscussion,
    },
    {
      id: '10-4',
      ...performanceMetricsForTrading,
      quiz: performanceMetricsForTradingQuiz,
      discussion: performanceMetricsForTradingDiscussion,
    },
    {
      id: '10-5',
      ...transactionCostsAndSlippage,
      quiz: transactionCostsAndSlippageQuiz,
      discussion: transactionCostsAndSlippageDiscussion,
    },
    {
      id: '10-6',
      ...walkForwardAnalysis,
      quiz: walkForwardAnalysisQuiz,
      discussion: walkForwardAnalysisDiscussion,
    },
    {
      id: '10-7',
      ...monteCarloSimulation,
      quiz: monteCarloSimulationQuiz,
      discussion: monteCarloSimulationDiscussion,
    },
    {
      id: '10-8',
      ...overfittingAndDataMining,
      quiz: overfittingAndDataMiningQuiz,
      discussion: overfittingAndDataMiningDiscussion,
    },
    {
      id: '10-9',
      ...statisticalSignificanceTesting,
      quiz: statisticalSignificanceTestingQuiz,
      discussion: statisticalSignificanceTestingDiscussion,
    },
    {
      id: '10-10',
      ...outOfSampleTesting,
      quiz: outOfSampleTestingQuiz,
      discussion: outOfSampleTestingDiscussion,
    },
    {
      id: '10-11',
      ...timeSeriesCrossValidation,
      quiz: timeSeriesCrossValidationQuiz,
      discussion: timeSeriesCrossValidationDiscussion,
    },
    {
      id: '10-12',
      ...benchmarkComparison,
      quiz: benchmarkComparisonQuiz,
      discussion: benchmarkComparisonDiscussion,
    },
    {
      id: '10-13',
      ...buildingBacktestingFramework,
      quiz: buildingBacktestingFrameworkQuiz,
      discussion: buildingBacktestingFrameworkDiscussion,
    },
    {
      id: '10-14',
      ...paperTradingVsLive,
      quiz: paperTradingVsLiveQuiz,
      discussion: paperTradingVsLiveDiscussion,
    },
    {
      id: '10-15',
      ...strategyParameterOptimization,
      quiz: strategyParameterOptimizationQuiz,
      discussion: strategyParameterOptimizationDiscussion,
    },
    {
      id: '10-16',
      ...productionBacktestingEngineProject,
      quiz: productionBacktestingEngineProjectQuiz,
      discussion: productionBacktestingEngineProjectDiscussion,
    },
  ],
};

export default backtestingStrategyDevelopment;
