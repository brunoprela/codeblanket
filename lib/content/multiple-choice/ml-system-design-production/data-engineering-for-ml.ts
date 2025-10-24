import { MultipleChoiceQuestion } from '@/lib/types';

export const dataEngineeringForMlMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'deml-mc-1',
    question: 'What is the main difference between ETL and ELT?',
    options: [
      'ETL is faster',
      'ETL transforms before loading, ELT loads then transforms',
      'ELT is always cheaper',
      'They are the same',
    ],
    correctAnswer: 1,
    explanation:
      'ETL (Extract-Transform-Load): Extract from source → Transform in pipeline (Spark) → Load into warehouse. Transform before storage. ELT (Extract-Load-Transform): Extract → Load raw into warehouse (Snowflake/BigQuery) → Transform on-demand with SQL. Transform after storage. ELT advantages for ML: (1) Flexibility (query raw data for any feature without re-ETL), (2) Rapid experimentation (write SQL, test immediately), (3) Keep raw data (can always reprocess). Use ELT for ML: Feature discovery common, need to try many transformations. ETL: When transformations fixed, reduce storage (only keep aggregated).',
  },
  {
    id: 'deml-mc-2',
    question: 'What is point-in-time correctness in feature stores?',
    options: [
      'Features must be computed quickly',
      'Features at time t use only data available before t (no data leakage)',
      'Features must be stored with timestamps',
      'Features must be updated in real-time',
    ],
    correctAnswer: 1,
    explanation:
      'Point-in-time correctness: Training uses features available at prediction time, not future data. Example: Predict conversion at 10:00. Use features computed before 10:00. BAD: Join by user_id (gets latest features, including after 10:00 → data leakage). GOOD: Temporal join—match user_id AND timestamp <= 10:00. Feast handles automatically: get_historical_features(entity_df, features) performs temporal joins. Prevents overfitting to future. Critical for valid training. Real-time serving: Use latest features (no future concern). Timestamped storage: Required but not sufficient (need temporal joins).',
  },
  {
    id: 'deml-mc-3',
    question:
      'What is the recommended approach for handling missing data in production ML pipelines?',
    options: [
      'Always drop rows with missing values',
      'Always fill with mean/median',
      'Forward fill for time series gaps (<2 days), validate critical fields, never impute unrealistic values',
      'Fill all missing with zeros',
    ],
    correctAnswer: 2,
    explanation:
      "Handle missing data based on context: TIME SERIES: Forward fill (ffill) for short gaps (1-2 days, holidays/weekends). Remove long gaps (>7 days, data issue). Never interpolate across market close (unrealistic). CRITICAL FIELDS: Null validation—reject data if user_id, timestamp, or critical features null (data quality issue). IMPUTATION: Mean/median introduces unrealistic values (don't use for financial data where actual=NA is meaningful). For ML features: Create is_missing indicator (model learns missing patterns). Example: Transaction data missing merchant_category → is_merchant_category_missing=1 (captures info). Don't blindly fill (destroys signal). Drop rows: Only if <1% missing (acceptable loss).",
  },
  {
    id: 'deml-mc-4',
    question: 'What tool is best for versioning large ML datasets?',
    options: ['Git', 'DVC (Data Version Control)', 'Docker', 'MLflow'],
    correctAnswer: 1,
    explanation:
      "DVC (Data Version Control) for datasets. How: dvc add data/train.csv stores hash in git (data/train.csv.dvc), actual data in S3. Git tracks metadata (<1KB), S3 stores data (100GB). Reproduce: dvc checkout pulls exact data version from S3. Benefits: (1) Version control like code (tags, branches), (2) Efficient (doesn't store duplicates), (3) Reproducible (any dataset version retrievable), (4) Works with git (same workflow). Git: Only for small files (<100MB). Large files slow git. Docker: For environment, not data. MLflow: Tracks dataset version (reference) but doesn't store data. Best: DVC for large datasets, git for small files/metadata.",
  },
  {
    id: 'deml-mc-5',
    question:
      'What is the primary purpose of Great Expectations in ML pipelines?',
    options: [
      'Train models faster',
      'Automated feature engineering',
      'Data validation and quality checks',
      'Model deployment',
    ],
    correctAnswer: 2,
    explanation:
      'Great Expectations: Data validation framework. Define expectations (tests) on data: expect_column_to_exist("age"), expect_column_values_to_be_between("age", 0, 120), expect_column_values_to_not_be_null("user_id"), expect_column_values_to_be_in_set("status", ["active", "inactive",]). Run on data ingestion: df_ge = ge.from_pandas(df); results = df_ge.validate(). If tests fail → block pipeline, alert team, use previous day\'s data. Benefits: (1) Catch data issues early (before training on bad data), (2) Automated checks (no manual inspection), (3) Documentation (expectations document data schema). Example: Fraud data usually 0.5% fraud rate. If 15% → expectation fails → investigate. Not for: Training speed, feature engineering (use Featuretools), deployment (use MLflow).',
  },
];
