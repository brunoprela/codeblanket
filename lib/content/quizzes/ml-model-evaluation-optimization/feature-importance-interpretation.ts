import { QuizQuestion } from '../../../types';

export const featureImportanceInterpretationQuiz: QuizQuestion[] = [
  {
    id: 'feature-importance-interpretation-dq-1',
    question:
      'Explain the difference between built-in feature importance, permutation importance, and SHAP values. When would you use each method?',
    sampleAnswer: `These three methods measure feature importance from different perspectives:

**Built-in Feature Importance (Tree Models)**
- **How it works:** Measures how much each feature reduces impurity (Gini/entropy) when creating splits
- **Advantages:** Very fast, no additional computation, available during training
- **Disadvantages:** Only for tree-based models, biased toward high-cardinality features, training-set based (may not reflect test performance)
- **When to use:** Quick exploratory analysis, initial feature selection with tree models

**Permutation Importance**
- **How it works:** Shuffle one feature's values, measure performance drop. Importance = baseline_score - shuffled_score
- **Advantages:** Model-agnostic (works with ANY model), based on actual predictions, provides uncertainty estimates
- **Disadvantages:** Can underestimate importance of correlated features, computationally expensive (requires multiple evaluations)
- **When to use:** Validating built-in importance, comparing features across different model types, need prediction-based importance

**SHAP Values (SHapley Additive exPlanations)**
- **How it works:** Uses game theory (Shapley values) to fairly distribute prediction among features. Considers all feature combinations.
- **Advantages:** Theoretically sound (satisfies local accuracy, missingness, consistency), provides direction of effect (+/-), explains individual predictions, handles feature interactions
- **Disadvantages:** Computationally expensive, requires careful interpretation
- **When to use:** Explaining individual predictions to stakeholders, debugging specific cases, regulatory requirements (model interpretability), fairness analysis

**Comparison Example:**
Predicting house prices:
- Built-in importance: "square_feet" is 30% important (used frequently in tree splits)
- Permutation: "square_feet" importance = 0.15 (shuffling it drops R² from 0.85 to 0.70)
- SHAP: For house #123, square_feet=2000 contributes +$50K to prediction (shows direction and magnitude)

**When to Use:**
1. **Exploratory Analysis:** Start with built-in importance (fast)
2. **Model Validation:** Use permutation importance to verify
3. **Stakeholder Communication:** Use SHAP for explaining specific predictions
4. **Regulatory Compliance:** SHAP values for transparent decision-making

**Red Flag:** If methods strongly disagree, investigate:
- Correlated features
- Data leakage
- Model instability`,
    keyPoints: [
      'Built-in: Fast, training-based, tree-specific',
      'Permutation: Model-agnostic, prediction-based, computationally expensive',
      'SHAP: Individual predictions, theoretically sound, handles interactions',
      'Use multiple methods to validate findings',
      'Strong disagreement indicates potential issues',
    ],
  },
  {
    id: 'feature-importance-interpretation-dq-2',
    question:
      'You discover that "customer_id" has the highest feature importance in your churn prediction model. What does this indicate, and what steps would you take to address it?',
    sampleAnswer: `High importance for "customer_id" is a major red flag indicating **data leakage** or **overfitting to individual customers**. This is a critical issue that must be addressed before deployment.

**What This Indicates:**

1. **Data Leakage:**
   - Model memorized specific customer IDs rather than learning general patterns
   - ID might be correlated with target in training data but won't generalize
   - Example: Early customers (low IDs) churned more → model learns "low ID = churn"

2. **Overfitting:**
   - Model has essentially created a lookup table: customer_123 → will churn
   - Zero generalization to new customers
   - Test accuracy might be deceptively high if test customers were in training

3. **Invalid Feature:**
   - Customer ID should NEVER be predictive in properly designed systems
   - If it is, something is wrong with your data or problem framing

**Steps to Address:**

**1. Investigate Why It's Important:**
\`\`\`python
# Check correlation with target
churn_by_id = df.groupby('customer_id')['churned'].mean()
# Are certain IDs perfectly predictive?

# Check temporal relationship
# Are low IDs (early customers) different from high IDs (recent)?
\`\`\`

**2. Remove the Feature:**
\`\`\`python
# Drop customer_id from training
X_clean = X.drop('customer_id', axis=1)
# Retrain model
\`\`\`

**3. Check for Related Issues:**
- Are other identifier columns leaking? (account_number, email_hash, etc.)
- Is there temporal leakage? (using future information)
- Are there derived features based on customer_id?

**4. Verify Data Collection:**
- Ensure training/test split is by customer, not by rows
- Avoid data snooping: same customer shouldn't appear in both train and test

**5. Retrain and Compare:**
\`\`\`python
# Before: with customer_id
model_with_id: AUC = 0.95, customer_id importance = 0.40

# After: without customer_id
model_without_id: AUC = 0.78, top feature = 'usage_frequency'
\`\`\`

**6. Validate New Model:**
- Check that new top features make business sense
- Test on completely new customers
- Monitor production performance

**7. Root Cause Analysis:**
- Why did customer_id have signal?
- Example: Early customers had different onboarding → should create "onboarding_version" feature instead
- Example: IDs correlate with signup date → create "days_since_signup" feature

**Real-World Example:**
A retail churn model had high importance for "customer_id." Investigation revealed:
- Low IDs = customers from acquired company (high churn rate)
- High IDs = organic customers (low churn rate)
- Solution: Create "acquisition_source" feature instead of using ID

**Key Lessons:**
- High importance for ID columns = data leakage
- Always validate that important features make business sense
- Feature engineering should capture why IDs matter (temporal, cohort, etc.)
- Test on truly unseen customers to catch these issues

**Prevention:**
- Explicitly exclude ID columns from training
- Use domain knowledge to validate features
- Implement cross-validation by customer (not by row)
- Monitor feature importance in production`,
    keyPoints: [
      'High importance for customer_id indicates data leakage',
      'Model memorized customers rather than learning patterns',
      'Remove ID and investigate why it was predictive',
      'Create proper features (acquisition source, tenure) instead',
      'Validate on completely new customers',
      'Always check that important features make business sense',
    ],
  },
  {
    id: 'feature-importance-interpretation-dq-3',
    question:
      'How do SHAP values help explain individual predictions? Provide a concrete example of how you would use SHAP to explain a loan rejection to a customer.',
    sampleAnswer: `SHAP values decompose a prediction into contributions from each feature, showing exactly how much each feature pushed the prediction up or down. This is crucial for explaining individual decisions.

**How SHAP Works:**
- Base value: Average prediction across all loans
- Each feature adds or subtracts from this base value
- Final prediction = base value + sum of all SHAP values
- Satisfies local accuracy: sum of contributions equals actual prediction

**Concrete Example: Loan Rejection**

**Scenario:**
- Customer: Sarah, applied for $50,000 personal loan
- Model prediction: 0.32 probability of repayment (rejected, threshold = 0.70)
- Customer wants to know why she was rejected

**SHAP Analysis:**
\`\`\`python
# Base value (average): 0.60 (60% of loans are repaid)

Feature contributions:
• credit_score = 580        → -0.15 (decreases probability)
• debt_to_income = 0.48     → -0.08 (decreases)
• employment_length = 2     → -0.04 (decreases)
• annual_income = $45,000   → -0.02 (slight decrease)
• loan_purpose = 'debt_con' → +0.01 (slight increase)

Final prediction: 0.60 - 0.15 - 0.08 - 0.04 - 0.02 + 0.01 = 0.32
\`\`\`

**Explanation to Customer:**

"Dear Sarah,

Thank you for your loan application. Unfortunately, we cannot approve your request at this time. Here's a transparent explanation of our decision:

**Why You Were Declined:**
Our credit model predicted a 32% likelihood of successful repayment, below our 70% threshold. Here's what influenced this decision:

1. **Credit Score (580) - Most Significant Factor**
   - Impact: -15 percentage points
   - Why: Credit scores below 620 significantly increase risk
   - **How to improve:** 
     * Pay all bills on time for 6-12 months
     * Reduce credit card balances below 30% of limits
     * Expected improvement: +50-100 points in 6 months

2. **Debt-to-Income Ratio (48%) - Second Factor**
   - Impact: -8 percentage points
   - Why: You're spending 48% of income on debt payments (we prefer <36%)
   - **How to improve:**
     * Pay down $5,000 in existing debt
     * This would lower DTI to 38% (borderline acceptable)

3. **Employment Length (2 years)**
   - Impact: -4 percentage points
   - Why: Shorter employment history increases risk
   - **How to improve:**
     * Continue current employment (3+ years is ideal)
     * Alternative: Show stable income history from previous jobs

4. **Annual Income ($45,000)**
   - Impact: -2 percentage points
   - Minor factor, but slightly below average for $50,000 loan

**Path to Approval:**

Scenario 1: Quick approval (3-6 months)
- Improve credit score to 630+ (pay bills on time, reduce balances)
- Pay down $3,000 in debt (DTI to 42%)
- Expected approval probability: ~75%

Scenario 2: Strong approval (6-12 months)
- Improve credit score to 680+
- Pay down $8,000 in debt (DTI to 35%)
- Complete 3 years employment
- Expected approval probability: ~90%

**Alternative Options:**
- Apply for smaller amount ($30,000): 58% approval probability
- Add co-signer with better credit: Could increase probability significantly
- Secured loan (with collateral): Different criteria apply

Please contact us if you have questions about this decision or your improvement plan.

Sincerely,
Lending Team"

**Why This Works:**

1. **Transparent:** Customer understands exact reasons for rejection
2. **Actionable:** Concrete steps to improve (not just "improve your credit")
3. **Quantified:** Shows which factors matter most (-15 points vs -2 points)
4. **Fair:** Mathematical, not subjective; same rules for everyone
5. **Regulatory Compliant:** Meets Adverse Action notice requirements
6. **Empowering:** Customer has a roadmap, not just a rejection

**Technical Implementation:**
\`\`\`python
import shap

# Get SHAP values for Sarah's application
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sarah_features)

# Create visualization
shap.waterfall_plot(shap_values[0])
# Shows: base value → each feature contribution → final prediction

# Identify top negative contributors
negative_contributors = sorted(
    [(feat, val) for feat, val in zip(features, shap_values[0]) if val < 0],
    key=lambda x: x[1]
)

# Generate actionable recommendations
for feature, impact in negative_contributors[:3]:
    recommendation = get_improvement_plan(feature, sarah_data[feature])
    print(f"{feature}: {impact:.2f} → {recommendation}")
\`\`\`

**Real-World Impact:**
- Regulatory compliance (FCRA Adverse Action notices)
- Reduced customer complaints
- Clear improvement path increases reapplication rate
- Demonstrates fairness and transparency`,
    keyPoints: [
      'SHAP decomposes prediction into feature contributions',
      'Each feature shows direction (+/-) and magnitude',
      'Provides actionable feedback (which factors to improve)',
      'Quantifies impact of each factor',
      'Essential for regulatory compliance and customer trust',
      'Transforms black-box rejection into transparent, actionable guidance',
    ],
  },
];
