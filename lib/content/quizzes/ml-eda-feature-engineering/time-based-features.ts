/**
 * Quiz questions for Time-Based Features section
 */

export const timebasedfeaturesQuiz = [
  {
    id: 'q1',
    question:
      'Explain why cyclical features like month or hour should be encoded using sine and cosine transformations rather than simple label encoding. Provide a specific example showing the problem with label encoding.',
    sampleAnswer:
      'Cyclical features have no true beginning or end - they wrap around. Label encoding breaks this circular relationship, creating artificial distance. PROBLEM WITH LABEL ENCODING: Month encoded as [1,2,3,...,12] makes December(12) and January(1) appear far apart (distance=11), but they are actually adjacent! Model treats December as furthest from January. Hour encoded as [0,1,...,23] makes 23:00 and 00:00 appear distant (distance=23) when they are 1 hour apart. REAL IMPACT: Predicting retail sales with months [1-12]. December has Christmas shopping spike, January has post-holiday dip. With label encoding, model cannot learn that pattern continues from 12→1. Predictions terrible for December-January transition. SINE/COSINE SOLUTION: Transform month m into: month_sin = sin(2π*m/12), month_cos = cos(2π*m/12). These create circular representation: December(12) maps to same point as "month 0" would, preserving closeness to January(1). Forms perfect circle in 2D space (sin, cos). MATHEMATICAL PROOF: Distance between December and January in sin/cos space is small! sin(2π*12/12)=0, cos(2π*12/12)=1. sin(2π*1/12)≈0.5, cos(2π*1/12)≈0.87. Euclidean distance ≈ 0.5 (close!). With label encoding: |12-1|=11 (far!). WHY BOTH SIN AND COS: Need both to uniquely identify each point on circle. Sin alone: both month 3 and month 9 have same sin value. Sin+Cos together: unique (x,y) coordinate for each month. APPLIES TO: Any cyclical feature - hour of day, day of week, day of year, angle/direction. IMPLEMENTATION: Always use for cyclical features in any model. Especially critical for linear models (trees more robust but still benefit).',
    keyPoints: [
      'Cyclical features wrap around (December→January are adjacent)',
      'Label encoding creates false distance (December=12, January=1 far apart)',
      'Sine/cosine transformation preserves circular nature',
      'Both sin and cos needed for unique identification',
      'Critical for months, hours, days of week',
      'Dramatically improves model performance on temporal patterns',
    ],
  },
  {
    id: 'q2',
    question:
      'Discuss lag features and rolling window features. How do they differ, when would you use each, and what are the risks of creating too many of them?',
    sampleAnswer:
      'Both capture temporal dependencies but serve different purposes. LAG FEATURES: Direct past values shifted forward. sales_lag_1 = yesterday sales, sales_lag_7 = last week sales. WHEN TO USE: (1) Strong autocorrelation - past values directly predict future. (2) Time series forecasting. (3) When specific past periods matter (yesterday, last week, last year). (4) Short-term dependencies. EXAMPLES: Stock prices (yesterday price predicts today), Daily sales (weekly patterns), Website traffic (hourly patterns). PROS: Simple, interpretable, captures direct relationships. CONS: Each lag is separate feature - many lags = many features. Missing values at start. ROLLING WINDOW FEATURES: Aggregated statistics over time windows. rolling_mean_7 = average of last 7 days, rolling_std_30 = volatility over 30 days. WHEN TO USE: (1) Smoothing noisy data - averages remove noise. (2) Capturing trends - moving averages show direction. (3) Volatility measurement - rolling std shows instability. (4) Context over periods - monthly average provides baseline. EXAMPLES: Sales rolling mean (smooth out daily noise), Stock price rolling std (measure volatility), Temperature rolling mean (smooth weather fluctuations). PROS: Smooth noise, single feature captures window, interpretable (moving average). CONS: Lags in response (average includes old data), overlap between windows, missing values at start. DIFFERENCES: Lag: specific point in past. Rolling: aggregated window. Lag: captures exact relationship. Rolling: captures average pattern. RISKS OF TOO MANY: (1) OVERFITTING: 100 lag features on 1000 samples = overfitting guaranteed. (2) MULTICOLLINEARITY: lag_1 and lag_2 highly correlated. rolling_7 and rolling_14 overlap significantly. (3) COMPUTATIONAL COST: Training slows down with many features. (4) INTERPRETABILITY LOSS: Which of 50 lags matter? Hard to explain. (5) DIMINISHING RETURNS: lag_1, lag_7, lag_30 usually sufficient. lag_1 through lag_365 mostly redundant. BEST PRACTICES: (1) Start with business-meaningful lags (daily, weekly, monthly, yearly). (2) Use domain knowledge (retail: 7-day lags for weekly patterns). (3) Check autocorrelation plot (ACF) - shows which lags matter. (4) Use rolling windows for noise reduction, lags for direct relationships. (5) Typically 5-10 temporal features sufficient unless specific reason for more. EXAMPLE: E-commerce sales. Keep: lag_1 (yesterday), lag_7 (last week), rolling_mean_7 (week trend), rolling_mean_30 (month trend). Skip: lag_2 through lag_6 (redundant with lag_1 and rolling_7).',
    keyPoints: [
      'Lag: exact past values, rolling: aggregated windows',
      'Lag for autocorrelation, rolling for trends/smoothing',
      'Too many leads to overfitting and multicollinearity',
      'Use domain knowledge to select meaningful periods',
      'Typically 5-10 temporal features sufficient',
      'Check ACF plot to identify important lags',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the concept of "time since event" features. Why are they powerful, and how do you handle them properly to avoid data leakage in production?',
    sampleAnswer:
      'Time-since features measure elapsed time from significant events: days_since_last_purchase, months_since_signup, hours_since_last_login. Powerful because they capture decay and recency effects. WHY POWERFUL: (1) RECENCY MATTERS: Recent events more relevant than old. Customer who purchased yesterday more likely to return than one from 2 years ago. (2) DECAY PATTERNS: Engagement decays over time. Predict churn: longer since last visit → higher churn probability. (3) LIFECYCLE STAGES: Time since signup indicates customer maturity. New customers (30 days) behave differently from veterans (2 years). (4) EVENT IMPACT DECAY: Effect of marketing campaign decays over time. days_since_campaign: 1=strong effect, 60=weak effect. EXAMPLES: E-COMMERCE: days_since_last_purchase (predict repeat purchase), months_since_first_purchase (customer lifetime), hours_since_cart_abandonment (predict conversion). SaaS: days_since_last_login (predict churn), weeks_since_paid (payment cycle), months_since_signup (maturity). FINANCE: days_since_last_transaction (predict dormancy), months_since_account_opening (lifecycle), days_since_last_payment (credit risk). DATA LEAKAGE RISK: WRONG APPROACH: Using days_until_churn to predict churn. This uses future information (when they will churn)! CORRECT: days_since_last_activity (past information only). PRODUCTION CONSIDERATIONS: (1) COMPUTATION AT PREDICTION TIME: Feature must be computable with only current and past data. days_since_signup: ✓ Can compute (current_date - signup_date). days_until_renewal: ✗ Cannot compute if renewal date unknown. (2) HANDLING MISSING EVENTS: What if customer never had event? Options: Use large value (999 days = never happened), Use separate flag (has_ever_purchased), Use median/mean as default. (3) CAPPING: Very long times provide no info (1000 days vs 2000 days same meaning). Cap at meaningful threshold: days_since_purchase capped at 365 (treat >365 same as 365). (4) UPDATING IN PRODUCTION: Feature changes over time! If trained on days_since_signup, must recompute daily in production as time passes. Event features update when events occur (purchase updates days_since_last_purchase). (5) TRAIN/TEST CONSISTENCY: Train: Compute time-since using data up to each training sample date. Test: Compute using data up to test sample date (simulates production). IMPLEMENTATION EXAMPLE: Training (2023 data): Sample from March 15: last purchase Feb 1 → days_since = 42. Production (June 2024): New sample June 15: last purchase May 30 → days_since = 16 (recomputed!). KEY: Always compute using information available at prediction time only.',
    keyPoints: [
      'Time-since captures event recency and decay effects',
      'Powerful for churn, conversion, engagement predictions',
      'Must use only past events (avoid future information leakage)',
      'Handle missing events with large values or flags',
      'Cap at meaningful thresholds (365 days)',
      'Recompute in production as time passes',
    ],
  },
];
