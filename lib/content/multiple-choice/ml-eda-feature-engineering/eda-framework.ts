/**
 * Multiple choice questions for EDA Framework section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const edaframeworkMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the PRIMARY purpose of Exploratory Data Analysis (EDA) in machine learning projects?',
    options: [
      'To train machine learning models on the data',
      'To understand data structure, quality, and relationships before modeling',
      'To deploy models into production',
      'To collect more data from external sources',
    ],
    correctAnswer: 1,
    explanation:
      'EDA is about understanding your data before building models. It helps identify data quality issues, understand distributions, discover relationships, and inform feature engineering - all BEFORE training models.',
  },
  {
    id: 'mc2',
    question:
      'You have a dataset with 100,000 rows and discover that 45% of values in a key feature are missing. What is the MOST appropriate first step?',
    options: [
      'Immediately drop the column since it has too much missing data',
      'Fill all missing values with the mean',
      "Investigate the pattern of missingness to understand if it's random or systematic",
      'Ignore the missing values and proceed with modeling',
    ],
    correctAnswer: 2,
    explanation:
      "Before deciding how to handle missing data, you must understand WHY it's missing. If missing data follows a pattern (e.g., high-value customers don't provide info), dropping or imputing could introduce bias. Investigation comes first.",
  },
  {
    id: 'mc3',
    question:
      'During EDA, you find that two features have a correlation coefficient of 0.95. What does this suggest?',
    options: [
      'These features are completely independent',
      'One feature causes changes in the other',
      'These features contain redundant information (multicollinearity)',
      'Both features must be removed from the dataset',
    ],
    correctAnswer: 2,
    explanation:
      "A correlation of 0.95 indicates strong multicollinearity - the features contain very similar information. This doesn't prove causation, but suggests redundancy. You might remove one, combine them, or use regularization, but don't necessarily remove both.",
  },
  {
    id: 'mc4',
    question: 'What is the correct order for conducting EDA?',
    options: [
      'Multivariate analysis → Univariate analysis → Data quality checks → Problem definition',
      'Problem definition → Data quality checks → Univariate analysis → Bivariate analysis → Multivariate analysis',
      'Feature engineering → Model training → EDA → Deployment',
      'Data quality checks → Model training → Univariate analysis',
    ],
    correctAnswer: 1,
    explanation:
      'EDA should progress logically: define the problem, check data quality, then analyze complexity incrementally (univariate → bivariate → multivariate). Start simple and build understanding progressively.',
  },
  {
    id: 'mc5',
    question:
      'You notice that a numeric feature has dtype "object" in pandas. What does this typically indicate?',
    options: [
      'The feature is correctly formatted',
      'The feature likely contains non-numeric characters (strings, symbols) mixed with numbers',
      'The feature contains only integers',
      'The feature has been properly encoded for machine learning',
    ],
    correctAnswer: 1,
    explanation:
      'Dtype "object" in pandas typically means the column contains strings or mixed types. For a numeric feature, this suggests data quality issues like special characters, missing values coded as strings ("N/A"), or inconsistent formatting that needs cleaning.',
  },
];
