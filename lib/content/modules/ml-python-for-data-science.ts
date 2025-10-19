/**
 * Module: Python for Data Science
 * Topic: Quantitative Programming
 *
 * Comprehensive coverage of NumPy and Pandas for data science
 */

import { Module } from '../../types';

// Import sections
import { numpyFundamentals } from '../sections/python-for-data-science/numpy-fundamentals';
import { numpyOperations } from '../sections/python-for-data-science/numpy-operations';
import { pandasSeriesDataFrames } from '../sections/python-for-data-science/pandas-series-dataframes';
import { dataManipulationPandas } from '../sections/python-for-data-science/data-manipulation-pandas';
import { dataCleaning } from '../sections/python-for-data-science/data-cleaning';
import { dataAggregationGrouping } from '../sections/python-for-data-science/data-aggregation-grouping';
import { mergingJoiningData } from '../sections/python-for-data-science/merging-joining-data';
import { timeSeriesPandas } from '../sections/python-for-data-science/time-series-pandas';
import { dataVisualization } from '../sections/python-for-data-science/data-visualization';
import { performanceOptimization } from '../sections/python-for-data-science/performance-optimization';

// Import quizzes (discussion questions)
import { numpyFundamentalsQuiz } from '../quizzes/python-for-data-science/numpy-fundamentals';
import { numpyoperationsQuiz } from '../quizzes/python-for-data-science/numpy-operations';
import { pandasseriesdataframesQuiz } from '../quizzes/python-for-data-science/pandas-series-dataframes';
import { datamanipulationpandasQuiz } from '../quizzes/python-for-data-science/data-manipulation-pandas';
import { datacleaningQuiz } from '../quizzes/python-for-data-science/data-cleaning';
import { dataaggregationgroupingQuiz } from '../quizzes/python-for-data-science/data-aggregation-grouping';
import { mergingjoiningdataQuiz } from '../quizzes/python-for-data-science/merging-joining-data';
import { timeseriespandasQuiz } from '../quizzes/python-for-data-science/time-series-pandas';
import { datavisualizationQuiz } from '../quizzes/python-for-data-science/data-visualization';
import { performanceoptimizationQuiz } from '../quizzes/python-for-data-science/performance-optimization';

// Import multiple choice questions
import { numpyFundamentalsMultipleChoice } from '../multiple-choice/python-for-data-science/numpy-fundamentals';
import { numpyoperationsMultipleChoice } from '../multiple-choice/python-for-data-science/numpy-operations';
import { pandasseriesdataframesMultipleChoice } from '../multiple-choice/python-for-data-science/pandas-series-dataframes';
import { datamanipulationpandasMultipleChoice } from '../multiple-choice/python-for-data-science/data-manipulation-pandas';
import { datacleaningMultipleChoice } from '../multiple-choice/python-for-data-science/data-cleaning';
import { dataaggregationgroupingMultipleChoice } from '../multiple-choice/python-for-data-science/data-aggregation-grouping';
import { mergingjoiningdataMultipleChoice } from '../multiple-choice/python-for-data-science/merging-joining-data';
import { timeseriespandasMultipleChoice } from '../multiple-choice/python-for-data-science/time-series-pandas';
import { datavisualizationMultipleChoice } from '../multiple-choice/python-for-data-science/data-visualization';
import { performanceoptimizationMultipleChoice } from '../multiple-choice/python-for-data-science/performance-optimization';

export const mlPythonForDataScience: Module = {
  id: 'ml-python-for-data-science',
  title: 'Python for Data Science',
  description:
    'Master NumPy and Pandas for efficient data manipulation, analysis, and visualization',
  category: 'undefined',
  difficulty: 'easy',
  estimatedTime: 'undefined',
  prerequisites: [
    'Module 1: Mathematical Foundations',
    'Module 4: Probability Theory',
  ],
  icon: '🐼',
  keyTakeaways: [
    'NumPy arrays provide efficient storage and operations for numerical data',
    'Broadcasting enables operations on arrays of different shapes without explicit loops',
    'Pandas Series and DataFrames are built on NumPy and add labels/indexes',
    'Vectorization is 100-1000x faster than Python loops',
    'Boolean indexing and loc/iloc enable powerful data filtering',
    'GroupBy operations split-apply-combine for aggregations',
    'Merge and join combine datasets like SQL operations',
    'DatetimeIndex and resampling handle time series data efficiently',
    'Categorical dtype reduces memory usage 10-100x for repeated strings',
    'Rolling windows compute moving statistics for technical indicators',
    'Matplotlib provides fine control, Seaborn adds statistical visualization',
    'Memory optimization (dtypes, categorical, chunking) enables working with large datasets',
    'Avoid iterrows/apply when vectorization is possible',
    'Time series require special handling: resampling, rolling, timezone awareness',
    'Data cleaning: missing data strategies (ffill, interpolation) depend on data type',
    'Performance profiling identifies bottlenecks before optimization',
  ],
  learningObjectives: [
    'Create and manipulate NumPy arrays with indexing, slicing, and reshaping',
    'Apply broadcasting rules for efficient array operations',
    'Perform linear algebra operations (dot products, matrix multiplication, eigenvalues)',
    'Construct Pandas Series and DataFrames with various data types',
    'Filter, sort, and transform data using boolean indexing and query methods',
    'Handle missing data with forward fill, interpolation, and imputation strategies',
    'Aggregate data using groupby, pivot tables, and crosstabs',
    'Merge, join, and concatenate datasets with different strategies',
    'Work with time series data: DatetimeIndex, resampling, rolling windows',
    'Create visualizations with matplotlib, seaborn, and pandas plotting',
    'Optimize memory usage with appropriate dtypes and categorical data',
    'Write vectorized code avoiding Python loops for 100x speedups',
    'Profile code to identify performance bottlenecks',
    'Build technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)',
  ],
  sections: [
    {
      ...numpyFundamentals,
      quiz: numpyFundamentalsQuiz,
      multipleChoice: numpyFundamentalsMultipleChoice,
    },
    {
      ...numpyOperations,
      quiz: numpyoperationsQuiz,
      multipleChoice: numpyoperationsMultipleChoice,
    },
    {
      ...pandasSeriesDataFrames,
      quiz: pandasseriesdataframesQuiz,
      multipleChoice: pandasseriesdataframesMultipleChoice,
    },
    {
      ...dataManipulationPandas,
      quiz: datamanipulationpandasQuiz,
      multipleChoice: datamanipulationpandasMultipleChoice,
    },
    {
      ...dataCleaning,
      quiz: datacleaningQuiz,
      multipleChoice: datacleaningMultipleChoice,
    },
    {
      ...dataAggregationGrouping,
      quiz: dataaggregationgroupingQuiz,
      multipleChoice: dataaggregationgroupingMultipleChoice,
    },
    {
      ...mergingJoiningData,
      quiz: mergingjoiningdataQuiz,
      multipleChoice: mergingjoiningdataMultipleChoice,
    },
    {
      ...timeSeriesPandas,
      quiz: timeseriespandasQuiz,
      multipleChoice: timeseriespandasMultipleChoice,
    },
    {
      ...dataVisualization,
      quiz: datavisualizationQuiz,
      multipleChoice: datavisualizationMultipleChoice,
    },
    {
      ...performanceOptimization,
      quiz: performanceoptimizationQuiz,
      multipleChoice: performanceoptimizationMultipleChoice,
    },
  ],
};
