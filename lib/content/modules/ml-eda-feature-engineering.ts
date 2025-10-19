/**
 * Exploratory Data Analysis & Feature Engineering Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { edaframeworkSection } from '../sections/ml-eda-feature-engineering/eda-framework';
import { univariateanalysisSection } from '../sections/ml-eda-feature-engineering/univariate-analysis';
import { bivariateanalysisSection } from '../sections/ml-eda-feature-engineering/bivariate-analysis';
import { multivariateanalysisSection } from '../sections/ml-eda-feature-engineering/multivariate-analysis';
import { advancedvisualizationSection } from '../sections/ml-eda-feature-engineering/advanced-visualization';
import { featureengineeringfundamentalsSection } from '../sections/ml-eda-feature-engineering/feature-engineering-fundamentals';
import { numericalfeatureengineeringSection } from '../sections/ml-eda-feature-engineering/numerical-feature-engineering';
import { categoricalfeatureengineeringSection } from '../sections/ml-eda-feature-engineering/categorical-feature-engineering';
import { timebasedfeaturesSection } from '../sections/ml-eda-feature-engineering/time-based-features';
import { advancedfeatureengineeringSection } from '../sections/ml-eda-feature-engineering/advanced-feature-engineering';

// Import quizzes
import { edaframeworkQuiz } from '../quizzes/ml-eda-feature-engineering/eda-framework';
import { univariateanalysisQuiz } from '../quizzes/ml-eda-feature-engineering/univariate-analysis';
import { bivariateanalysisQuiz } from '../quizzes/ml-eda-feature-engineering/bivariate-analysis';
import { multivariateanalysisQuiz } from '../quizzes/ml-eda-feature-engineering/multivariate-analysis';
import { advancedvisualizationQuiz } from '../quizzes/ml-eda-feature-engineering/advanced-visualization';
import { featureengineeringfundamentalsQuiz } from '../quizzes/ml-eda-feature-engineering/feature-engineering-fundamentals';
import { numericalfeatureengineeringQuiz } from '../quizzes/ml-eda-feature-engineering/numerical-feature-engineering';
import { categoricalfeatureengineeringQuiz } from '../quizzes/ml-eda-feature-engineering/categorical-feature-engineering';
import { timebasedfeaturesQuiz } from '../quizzes/ml-eda-feature-engineering/time-based-features';
import { advancedfeatureengineeringQuiz } from '../quizzes/ml-eda-feature-engineering/advanced-feature-engineering';

// Import multiple choice
import { edaframeworkMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/eda-framework';
import { univariateanalysisMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/univariate-analysis';
import { bivariateanalysisMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/bivariate-analysis';
import { multivariateanalysisMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/multivariate-analysis';
import { advancedvisualizationMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/advanced-visualization';
import { featureengineeringfundamentalsMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/feature-engineering-fundamentals';
import { numericalfeatureengineeringMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/numerical-feature-engineering';
import { categoricalfeatureengineeringMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/categorical-feature-engineering';
import { timebasedfeaturesMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/time-based-features';
import { advancedfeatureengineeringMultipleChoice } from '../multiple-choice/ml-eda-feature-engineering/advanced-feature-engineering';

export const mlEdaFeatureEngineeringModule: Module = {
  id: 'ml-eda-feature-engineering',
  title: 'Exploratory Data Analysis & Feature Engineering',
  description:
    'Master data exploration, visualization, and feature engineering techniques to extract maximum value from data and dramatically improve model performance',
  category: 'machine-learning',
  difficulty: 'intermediate',
  estimatedTime: '15 hours',
  prerequisites: ['Python for Data Science', 'Statistics Fundamentals'],
  icon: '🔍',
  keyTakeaways: [
    'EDA is critical first step - understand data before modeling',
    'Univariate analysis examines each feature independently for distributions and outliers',
    'Bivariate analysis reveals relationships between features and target',
    'Multivariate analysis uncovers complex patterns and multicollinearity',
    'Advanced visualizations communicate insights effectively',
    'Feature engineering often provides more improvement than algorithm choice',
    'Proper scaling and transformation essential for linear models',
    'Categorical encoding strategy depends on cardinality and model type',
    'Time-based features capture seasonality, trends, and temporal patterns',
    'Advanced techniques combine automation with domain knowledge',
  ],
  learningObjectives: [
    'Conduct systematic exploratory data analysis on any dataset',
    'Create effective visualizations for data understanding and communication',
    'Identify and handle data quality issues (missing values, outliers, duplicates)',
    'Analyze univariate distributions and apply appropriate transformations',
    'Discover bivariate relationships using correlation and statistical tests',
    'Detect multicollinearity and reduce dimensionality with PCA',
    'Engineer numerical features through scaling, binning, and transformations',
    'Encode categorical features appropriately for different model types',
    'Create time-based features for temporal data and forecasting',
    'Apply advanced feature engineering techniques and automated feature generation',
    'Perform feature selection to identify most impactful predictors',
    'Build reproducible feature engineering pipelines for production',
  ],
  sections: [
    {
      ...edaframeworkSection,
      quiz: edaframeworkQuiz,
      multipleChoice: edaframeworkMultipleChoice,
    },
    {
      ...univariateanalysisSection,
      quiz: univariateanalysisQuiz,
      multipleChoice: univariateanalysisMultipleChoice,
    },
    {
      ...bivariateanalysisSection,
      quiz: bivariateanalysisQuiz,
      multipleChoice: bivariateanalysisMultipleChoice,
    },
    {
      ...multivariateanalysisSection,
      quiz: multivariateanalysisQuiz,
      multipleChoice: multivariateanalysisMultipleChoice,
    },
    {
      ...advancedvisualizationSection,
      quiz: advancedvisualizationQuiz,
      multipleChoice: advancedvisualizationMultipleChoice,
    },
    {
      ...featureengineeringfundamentalsSection,
      quiz: featureengineeringfundamentalsQuiz,
      multipleChoice: featureengineeringfundamentalsMultipleChoice,
    },
    {
      ...numericalfeatureengineeringSection,
      quiz: numericalfeatureengineeringQuiz,
      multipleChoice: numericalfeatureengineeringMultipleChoice,
    },
    {
      ...categoricalfeatureengineeringSection,
      quiz: categoricalfeatureengineeringQuiz,
      multipleChoice: categoricalfeatureengineeringMultipleChoice,
    },
    {
      ...timebasedfeaturesSection,
      quiz: timebasedfeaturesQuiz,
      multipleChoice: timebasedfeaturesMultipleChoice,
    },
    {
      ...advancedfeatureengineeringSection,
      quiz: advancedfeatureengineeringQuiz,
      multipleChoice: advancedfeatureengineeringMultipleChoice,
    },
  ],
};
