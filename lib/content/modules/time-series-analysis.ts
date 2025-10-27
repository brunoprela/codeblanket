/**
 * Time Series Analysis for Finance Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { timeSeriesFundamentals } from '../sections/time-series-analysis/9-1-time-series-fundamentals';
import { stationarityUnitRoots } from '../sections/time-series-analysis/9-2-stationarity-unit-roots';
import { autocorrelationPartialAutocorrelation } from '../sections/time-series-analysis/9-3-autocorrelation-partial-autocorrelation';
import { armaModels } from '../sections/time-series-analysis/9-4-arma-models';
import { arimaModels } from '../sections/time-series-analysis/9-5-arima-models';
import { garchModels } from '../sections/time-series-analysis/9-6-garch-models';
import { cointegrationPairsTrading } from '../sections/time-series-analysis/9-7-cointegration-pairs-trading';
import { vectorAutoregression } from '../sections/time-series-analysis/9-8-vector-autoregression';
import { kalmanFilters } from '../sections/time-series-analysis/9-9-kalman-filters';
import { stateSpaceModels } from '../sections/time-series-analysis/9-10-state-space-models';
import { regimeSwitching } from '../sections/time-series-analysis/9-11-regime-switching';
import { forecastingEvaluation } from '../sections/time-series-analysis/9-12-forecasting-evaluation';
import { highFrequencyTimeSeries } from '../sections/time-series-analysis/9-13-high-frequency-time-series';
import { finalProjectForecastingSystem } from '../sections/time-series-analysis/9-14-final-project-forecasting-system';

// Import quizzes
import { timeSeriesFundamentalsQuiz } from '../quizzes/time-series-analysis/9-1-time-series-fundamentals';
import { stationarityUnitRootsQuiz } from '../quizzes/time-series-analysis/9-2-stationarity-unit-roots';
import { autocorrelationPartialAutocorrelationQuiz } from '../quizzes/time-series-analysis/9-3-autocorrelation-partial-autocorrelation';
import { armaModelsQuiz } from '../quizzes/time-series-analysis/9-4-arma-models';
import { arimaModelsQuiz } from '../quizzes/time-series-analysis/9-5-arima-models';
import { garchModelsQuiz } from '../quizzes/time-series-analysis/9-6-garch-models';
import { cointegrationPairsTradingQuiz } from '../quizzes/time-series-analysis/9-7-cointegration-pairs-trading';
import { vectorAutoregressionQuiz } from '../quizzes/time-series-analysis/9-8-vector-autoregression';
import { kalmanFiltersQuiz } from '../quizzes/time-series-analysis/9-9-kalman-filters';
import { stateSpaceModelsQuiz } from '../quizzes/time-series-analysis/9-10-state-space-models';
import { regimeSwitchingQuiz } from '../quizzes/time-series-analysis/9-11-regime-switching';
import { forecastingEvaluationQuiz } from '../quizzes/time-series-analysis/9-12-forecasting-evaluation';
import { highFrequencyTimeSeriesQuiz } from '../quizzes/time-series-analysis/9-13-high-frequency-time-series';
import { finalProjectForecastingSystemQuiz } from '../quizzes/time-series-analysis/9-14-final-project-forecasting-system';

// Import multiple choice
import { timeSeriesFundamentalsMultipleChoice } from '../multiple-choice/time-series-analysis/9-1-time-series-fundamentals';
import { stationarityUnitRootsMultipleChoice } from '../multiple-choice/time-series-analysis/9-2-stationarity-unit-roots';
import { autocorrelationPartialAutocorrelationMultipleChoice } from '../multiple-choice/time-series-analysis/9-3-autocorrelation-partial-autocorrelation';
import { armaModelsMultipleChoice } from '../multiple-choice/time-series-analysis/9-4-arma-models';
import { arimaModelsMultipleChoice } from '../multiple-choice/time-series-analysis/9-5-arima-models';
import { garchModelsMultipleChoice } from '../multiple-choice/time-series-analysis/9-6-garch-models';
import { cointegrationPairsTradingMultipleChoice } from '../multiple-choice/time-series-analysis/9-7-cointegration-pairs-trading';
import { vectorAutoregressionMultipleChoice } from '../multiple-choice/time-series-analysis/9-8-vector-autoregression';
import { kalmanFiltersMultipleChoice } from '../multiple-choice/time-series-analysis/9-9-kalman-filters';
import { stateSpaceModelsMultipleChoice } from '../multiple-choice/time-series-analysis/9-10-state-space-models';
import { regimeSwitchingMultipleChoice } from '../multiple-choice/time-series-analysis/9-11-regime-switching';
import { forecastingEvaluationMultipleChoice } from '../multiple-choice/time-series-analysis/9-12-forecasting-evaluation';
import { highFrequencyTimeSeriesMultipleChoice } from '../multiple-choice/time-series-analysis/9-13-high-frequency-time-series';
import { finalProjectForecastingSystemMultipleChoice } from '../multiple-choice/time-series-analysis/9-14-final-project-forecasting-system';

export const timeSeriesAnalysisModule: Module = {
  id: 'time-series-analysis',
  title: 'Time Series Analysis for Finance',
  description:
    'Master time series modeling for financial forecasting, volatility prediction, and algorithmic trading strategies',
  estimatedHours: 45,
  prerequisites: ['financial-statements-analysis', 'quantitative-finance'],
  learningObjectives: [
    'Understand stationarity, unit roots, and time series transformations',
    'Build and validate ARIMA, GARCH, and cointegration models',
    'Implement pairs trading strategies using statistical arbitrage',
    'Master Kalman filters and state space models for dynamic estimation',
    'Apply high-frequency time series analysis to trading systems',
  ],
  sections: [
    {
      ...timeSeriesFundamentals,
      quiz: timeSeriesFundamentalsQuiz,
      multipleChoice: timeSeriesFundamentalsMultipleChoice,
    },
    {
      ...stationarityUnitRoots,
      quiz: stationarityUnitRootsQuiz,
      multipleChoice: stationarityUnitRootsMultipleChoice,
    },
    {
      ...autocorrelationPartialAutocorrelation,
      quiz: autocorrelationPartialAutocorrelationQuiz,
      multipleChoice: autocorrelationPartialAutocorrelationMultipleChoice,
    },
    {
      ...armaModels,
      quiz: armaModelsQuiz,
      multipleChoice: armaModelsMultipleChoice,
    },
    {
      ...arimaModels,
      quiz: arimaModelsQuiz,
      multipleChoice: arimaModelsMultipleChoice,
    },
    {
      ...garchModels,
      quiz: garchModelsQuiz,
      multipleChoice: garchModelsMultipleChoice,
    },
    {
      ...cointegrationPairsTrading,
      quiz: cointegrationPairsTradingQuiz,
      multipleChoice: cointegrationPairsTradingMultipleChoice,
    },
    {
      ...vectorAutoregression,
      quiz: vectorAutoregressionQuiz,
      multipleChoice: vectorAutoregressionMultipleChoice,
    },
    {
      ...kalmanFilters,
      quiz: kalmanFiltersQuiz,
      multipleChoice: kalmanFiltersMultipleChoice,
    },
    {
      ...stateSpaceModels,
      quiz: stateSpaceModelsQuiz,
      multipleChoice: stateSpaceModelsMultipleChoice,
    },
    {
      ...regimeSwitching,
      quiz: regimeSwitchingQuiz,
      multipleChoice: regimeSwitchingMultipleChoice,
    },
    {
      ...forecastingEvaluation,
      quiz: forecastingEvaluationQuiz,
      multipleChoice: forecastingEvaluationMultipleChoice,
    },
    {
      ...highFrequencyTimeSeries,
      quiz: highFrequencyTimeSeriesQuiz,
      multipleChoice: highFrequencyTimeSeriesMultipleChoice,
    },
    {
      ...finalProjectForecastingSystem,
      quiz: finalProjectForecastingSystemQuiz,
      multipleChoice: finalProjectForecastingSystemMultipleChoice,
    },
  ],
};
