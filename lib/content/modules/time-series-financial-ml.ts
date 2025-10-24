/**
 * Module: Time Series & Financial Machine Learning
 * Module 15 of ML/AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { timeSeriesFundamentals } from '../sections/time-series-financial-ml/time-series-fundamentals';
import { classicalTimeSeriesModels } from '../sections/time-series-financial-ml/classical-time-series-models';
import { advancedTimeSeriesModels } from '../sections/time-series-financial-ml/advanced-time-series-models';
import { deepLearningForTimeSeries } from '../sections/time-series-financial-ml/deep-learning-for-time-series';
import { financialDataSourcesAPIs } from '../sections/time-series-financial-ml/financial-data-sources-apis';
import { technicalIndicators } from '../sections/time-series-financial-ml/technical-indicators';
import { fundamentalAnalysisML } from '../sections/time-series-financial-ml/fundamental-analysis-ml';
import { predictiveModelingTrading } from '../sections/time-series-financial-ml/predictive-modeling-trading';
import { portfolioOptimization } from '../sections/time-series-financial-ml/portfolio-optimization';
import { tradingStrategyDevelopment } from '../sections/time-series-financial-ml/trading-strategy-development';
import { backtestingSimulation } from '../sections/time-series-financial-ml/backtesting-simulation';
import { riskManagementPositionSizing } from '../sections/time-series-financial-ml/risk-management-position-sizing';
import { marketMicrostructure } from '../sections/time-series-financial-ml/market-microstructure';
import { reinforcementLearningTrading } from '../sections/time-series-financial-ml/reinforcement-learning-trading';
import { marketRegimesAdaptiveStrategies } from '../sections/time-series-financial-ml/market-regimes-adaptive-strategies';
import { advancedRiskManagement } from '../sections/time-series-financial-ml/advanced-risk-management';
import { strategyPerformanceEvaluation } from '../sections/time-series-financial-ml/strategy-performance-evaluation';
import { orderExecutionTradingInfrastructure } from '../sections/time-series-financial-ml/order-execution-trading-infrastructure';
import { liveTradingPaperTrading } from '../sections/time-series-financial-ml/live-trading-paper-trading';
import { cryptocurrencyTrading } from '../sections/time-series-financial-ml/cryptocurrency-trading';

// Quiz imports
import { timeSeriesFundamentalsQuiz } from '../quizzes/time-series-financial-ml/time-series-fundamentals';
import { classicalTimeSeriesModelsQuiz } from '../quizzes/time-series-financial-ml/classical-time-series-models';
import { advancedTimeSeriesModelsQuiz } from '../quizzes/time-series-financial-ml/advanced-time-series-models';
import { deepLearningForTimeSeriesQuiz } from '../quizzes/time-series-financial-ml/deep-learning-for-time-series';
import { financialDataSourcesAPIsQuiz } from '../quizzes/time-series-financial-ml/financial-data-sources-apis';
import { technicalIndicatorsQuiz } from '../quizzes/time-series-financial-ml/technical-indicators';
import { fundamentalAnalysisMLQuiz } from '../quizzes/time-series-financial-ml/fundamental-analysis-ml';
import { predictiveModelingTradingQuiz } from '../quizzes/time-series-financial-ml/predictive-modeling-trading';
import { portfolioOptimizationQuiz } from '../quizzes/time-series-financial-ml/portfolio-optimization';
import { tradingStrategyDevelopmentQuiz } from '../quizzes/time-series-financial-ml/trading-strategy-development';
import { backtestingSimulationQuiz } from '../quizzes/time-series-financial-ml/backtesting-simulation';
import { riskManagementPositionSizingQuiz } from '../quizzes/time-series-financial-ml/risk-management-position-sizing';
import { marketMicrostructureQuiz } from '../quizzes/time-series-financial-ml/market-microstructure';
import { reinforcementLearningTradingQuiz } from '../quizzes/time-series-financial-ml/reinforcement-learning-trading';
import { marketRegimesAdaptiveStrategiesQuiz } from '../quizzes/time-series-financial-ml/market-regimes-adaptive-strategies';
import { advancedRiskManagementQuiz } from '../quizzes/time-series-financial-ml/advanced-risk-management';
import { strategyPerformanceEvaluationQuiz } from '../quizzes/time-series-financial-ml/strategy-performance-evaluation';
import { orderExecutionTradingInfrastructureQuiz } from '../quizzes/time-series-financial-ml/order-execution-trading-infrastructure';
import { liveTradingPaperTradingQuiz } from '../quizzes/time-series-financial-ml/live-trading-paper-trading';
import { cryptocurrencyTradingQuiz } from '../quizzes/time-series-financial-ml/cryptocurrency-trading';

// Multiple choice imports
import { timeSeriesFundamentalsMultipleChoice } from '../multiple-choice/time-series-financial-ml/time-series-fundamentals';
import { classicalTimeSeriesModelsMultipleChoice } from '../multiple-choice/time-series-financial-ml/classical-time-series-models';
import { advancedTimeSeriesModelsMultipleChoice } from '../multiple-choice/time-series-financial-ml/advanced-time-series-models';
import { deepLearningForTimeSeriesMultipleChoice } from '../multiple-choice/time-series-financial-ml/deep-learning-for-time-series';
import { financialDataSourcesAPIsMultipleChoice } from '../multiple-choice/time-series-financial-ml/financial-data-sources-apis';
import { technicalIndicatorsMultipleChoice } from '../multiple-choice/time-series-financial-ml/technical-indicators';
import { fundamentalAnalysisMLMultipleChoice } from '../multiple-choice/time-series-financial-ml/fundamental-analysis-ml';
import { predictiveModelingTradingMultipleChoice } from '../multiple-choice/time-series-financial-ml/predictive-modeling-trading';
import { portfolioOptimizationMultipleChoice } from '../multiple-choice/time-series-financial-ml/portfolio-optimization';
import { tradingStrategyDevelopmentMultipleChoice } from '../multiple-choice/time-series-financial-ml/trading-strategy-development';
import { backtestingSimulationMultipleChoice } from '../multiple-choice/time-series-financial-ml/backtesting-simulation';
import { riskManagementPositionSizingMultipleChoice } from '../multiple-choice/time-series-financial-ml/risk-management-position-sizing';
import { marketMicrostructureMultipleChoice } from '../multiple-choice/time-series-financial-ml/market-microstructure';
import { reinforcementLearningTradingMultipleChoice } from '../multiple-choice/time-series-financial-ml/reinforcement-learning-trading';
import { marketRegimesAdaptiveStrategiesMultipleChoice } from '../multiple-choice/time-series-financial-ml/market-regimes-adaptive-strategies';
import { advancedRiskManagementMultipleChoice } from '../multiple-choice/time-series-financial-ml/advanced-risk-management';
import { strategyPerformanceEvaluationMultipleChoice } from '../multiple-choice/time-series-financial-ml/strategy-performance-evaluation';
import { orderExecutionTradingInfrastructureMultipleChoice } from '../multiple-choice/time-series-financial-ml/order-execution-trading-infrastructure';
import { liveTradingPaperTradingMultipleChoice } from '../multiple-choice/time-series-financial-ml/live-trading-paper-trading';
import { cryptocurrencyTradingMultipleChoice } from '../multiple-choice/time-series-financial-ml/cryptocurrency-trading';

export const timeSeriesFinancialMlModule: Module = {
  id: 'time-series-financial-ml',
  title: 'Time Series & Financial Machine Learning',
  description:
    'Master time series analysis and financial ML from fundamentals to production trading systems. Learn classical models (ARIMA, SARIMA), deep learning (LSTM, Transformers), and complete trading pipeline: data acquisition, feature engineering, predictive modeling, portfolio optimization, backtesting, risk management, and live execution. Build professional trading strategies with proper validation, realistic testing, and infrastructure.',
  icon: '📈',
  keyTakeaways: [
    'Understand time series components: trend, seasonality, noise, and stationarity',
    'Master classical models: ARIMA, SARIMA, exponential smoothing, and VAR',
    'Apply deep learning: LSTM, 1D CNNs, Transformers for time series forecasting',
    'Acquire financial data from APIs: Yahoo Finance, Alpha Vantage, Polygon',
    'Engineer 50+ trading features: technical indicators, volume, volatility, trends',
    'Build predictive models with walk-forward validation (55-60% accuracy achievable)',
    'Optimize portfolios: Markowitz, Risk Parity, Black-Litterman, Kelly criterion',
    'Develop trading strategies: momentum, mean reversion, ML-based, adaptive regimes',
    'Backtest properly: avoid lookahead bias, survivorship bias, realistic costs',
    'Manage risk: position sizing (2% rule), Kelly criterion, VaR, CVaR, stop losses',
    'Understand market microstructure: order types, bid-ask spread, order flow',
    'Apply reinforcement learning: DQN agents for trading decisions',
    'Detect market regimes: HMM, volatility clustering, adaptive strategy switching',
    'Evaluate strategies: Sharpe > 1, Max DD < 20%, Calmar, alpha, information ratio',
    'Execute orders: broker APIs (Alpaca), EMS, monitoring, slippage tracking',
    'Transition to live: paper trading validation, risk controls, scaling gradually',
    'Trade cryptocurrency: 24/7 markets, on-chain metrics, higher volatility management',
    'Build production systems: robust infrastructure, monitoring, failsafes',
  ],
  sections: [
    {
      ...timeSeriesFundamentals,
      quiz: timeSeriesFundamentalsQuiz,
      multipleChoice: timeSeriesFundamentalsMultipleChoice,
    },
    {
      ...classicalTimeSeriesModels,
      quiz: classicalTimeSeriesModelsQuiz,
      multipleChoice: classicalTimeSeriesModelsMultipleChoice,
    },
    {
      ...advancedTimeSeriesModels,
      quiz: advancedTimeSeriesModelsQuiz,
      multipleChoice: advancedTimeSeriesModelsMultipleChoice,
    },
    {
      ...deepLearningForTimeSeries,
      quiz: deepLearningForTimeSeriesQuiz,
      multipleChoice: deepLearningForTimeSeriesMultipleChoice,
    },
    {
      ...financialDataSourcesAPIs,
      quiz: financialDataSourcesAPIsQuiz,
      multipleChoice: financialDataSourcesAPIsMultipleChoice,
    },
    {
      ...technicalIndicators,
      quiz: technicalIndicatorsQuiz,
      multipleChoice: technicalIndicatorsMultipleChoice,
    },
    {
      ...fundamentalAnalysisML,
      quiz: fundamentalAnalysisMLQuiz,
      multipleChoice: fundamentalAnalysisMLMultipleChoice,
    },
    {
      ...predictiveModelingTrading,
      quiz: predictiveModelingTradingQuiz,
      multipleChoice: predictiveModelingTradingMultipleChoice,
    },
    {
      ...portfolioOptimization,
      quiz: portfolioOptimizationQuiz,
      multipleChoice: portfolioOptimizationMultipleChoice,
    },
    {
      ...tradingStrategyDevelopment,
      quiz: tradingStrategyDevelopmentQuiz,
      multipleChoice: tradingStrategyDevelopmentMultipleChoice,
    },
    {
      ...backtestingSimulation,
      quiz: backtestingSimulationQuiz,
      multipleChoice: backtestingSimulationMultipleChoice,
    },
    {
      ...riskManagementPositionSizing,
      quiz: riskManagementPositionSizingQuiz,
      multipleChoice: riskManagementPositionSizingMultipleChoice,
    },
    {
      ...marketMicrostructure,
      quiz: marketMicrostructureQuiz,
      multipleChoice: marketMicrostructureMultipleChoice,
    },
    {
      ...reinforcementLearningTrading,
      quiz: reinforcementLearningTradingQuiz,
      multipleChoice: reinforcementLearningTradingMultipleChoice,
    },
    {
      ...marketRegimesAdaptiveStrategies,
      quiz: marketRegimesAdaptiveStrategiesQuiz,
      multipleChoice: marketRegimesAdaptiveStrategiesMultipleChoice,
    },
    {
      ...advancedRiskManagement,
      quiz: advancedRiskManagementQuiz,
      multipleChoice: advancedRiskManagementMultipleChoice,
    },
    {
      ...strategyPerformanceEvaluation,
      quiz: strategyPerformanceEvaluationQuiz,
      multipleChoice: strategyPerformanceEvaluationMultipleChoice,
    },
    {
      ...orderExecutionTradingInfrastructure,
      quiz: orderExecutionTradingInfrastructureQuiz,
      multipleChoice: orderExecutionTradingInfrastructureMultipleChoice,
    },
    {
      ...liveTradingPaperTrading,
      quiz: liveTradingPaperTradingQuiz,
      multipleChoice: liveTradingPaperTradingMultipleChoice,
    },
    {
      ...cryptocurrencyTrading,
      quiz: cryptocurrencyTradingQuiz,
      multipleChoice: cryptocurrencyTradingMultipleChoice,
    },
  ],
};
