import { QuizQuestion } from '../../../types';

export const classificationMetricsQuiz: QuizQuestion[] = [
  {
    id: 'classification-metrics-dq-1',
    question:
      "Explain the precision-recall tradeoff using a concrete example (e.g., spam filter or medical diagnosis). Why can't you maximize both simultaneously, and how do you decide which to prioritize?",
    sampleAnswer: `The precision-recall tradeoff is fundamental to classification: improving one typically worsens the other. Understanding this tradeoff and choosing the right balance depends on the costs of different types of errors.

**The Tradeoff Explained:**

**Precision** = TP/(TP+FP) asks: "Of all positive predictions, how many were correct?"
**Recall** = TP/(TP+FN) asks: "Of all actual positives, how many did we catch?"

These compete because they respond oppositely to the prediction threshold.

**Concrete Example: Cancer Screening**

Imagine a model predicting cancer with probability scores 0-1. We need to choose a threshold:

**Scenario A: Low Threshold (e.g., 0.2)**
- Predict "cancer" if probability ≥ 0.2
- Many positive predictions → catches almost all cancer cases
- **High Recall** (90%+): Few missed cancers (low FN)
- **Low Precision** (40%): Many false alarms (high FP)
- Result: 90% of cancers detected, but 60% of predictions are false alarms

**Scenario B: High Threshold (e.g., 0.8)**
- Predict "cancer" only if probability ≥ 0.8
- Few positive predictions → only very confident cases
- **Low Recall** (60%): Miss many cancers (high FN)
- **High Precision** (95%): Very few false alarms (low FP)
- Result: Only 60% of cancers detected, but 95% of positive predictions are correct

**Why They Can't Both Be Maximized:**

The fundamental tension:
- To increase recall: Lower threshold → more positive predictions → catches more true positives BUT also more false positives → precision decreases
- To increase precision: Raise threshold → fewer positive predictions → fewer false positives BUT also miss true positives → recall decreases

Mathematical constraint: Given fixed TP, improving one requires trading off the other through FP vs FN.

**Decision Framework:**

**Prioritize Recall When:**
1. **False negatives are catastrophic**
   - Cancer screening: Missing cancer can be fatal
   - Airport security: Missing a threat is unacceptable
   - Fraud detection: Undetected fraud costs money

2. **Follow-up verification is cheap**
   - Initial screening can have false alarms
   - Experts verify positives in second stage
   - Example: Cheap blood test → expensive biopsy

3. **Cost structure: FN >> FP**
   - Cost of missing positive vastly exceeds cost of false alarm

**Prioritize Precision When:**
1. **False positives are very costly**
   - Spam filter: Can't risk blocking important emails
   - Product recommendations: Bad recommendations frustrate users
   - Credit denial: False rejections lose good customers

2. **No second-stage verification**
   - Decision is final based on model
   - No human in the loop to catch errors

3. **Cost structure: FP >> FN**
   - Cost of false alarm exceeds cost of missing positive

**Balance with F1-Score When:**
- Costs are roughly equal
- Need single metric for optimization
- Example: Fraud detection where both FP and FN matter

**Real-World Cancer Screening Decision:**

\`\`\`
Screening Stage (Prioritize Recall):
- Threshold: 0.2
- Recall: 95% (catch almost all cancers)
- Precision: 30% (many false alarms acceptable)
- Rationale: Can't risk missing cancer; false positives go to Stage 2

Diagnostic Stage (Balance with F1):
- Threshold: 0.6
- Recall: 80%
- Precision: 85%
- Rationale: More expensive tests, need balance

Treatment Decision (Prioritize Precision):
- Threshold: 0.9
- Recall: 60%
- Precision: 98%
- Rationale: Treatment has side effects; must be confident
\`\`\`

**Key Insight**: The tradeoff exists because every prediction threshold creates a different FP/FN balance. Your choice depends on the relative costs of these errors in your specific application, not on statistical elegance.`,
    keyPoints: [
      'Precision and recall tradeoff because threshold changes affect FP and FN oppositely',
      'Lower threshold → more positive predictions → higher recall, lower precision',
      'Higher threshold → fewer positive predictions → lower recall, higher precision',
      'Prioritize recall when false negatives are catastrophic (cancer, security)',
      'Prioritize precision when false positives are very costly (spam filter, recommendations)',
      'Use F1-score when costs are balanced',
      'Decision depends on business context and relative costs of FP vs FN',
    ],
  },
  {
    id: 'classification-metrics-dq-2',
    question:
      'Why is accuracy often a poor metric for imbalanced datasets? Provide a specific example demonstrating the "accuracy paradox" where a useless model achieves high accuracy.',
    sampleAnswer: `Accuracy is dangerously misleading for imbalanced datasets because it can reward models that simply exploit class imbalance rather than learning meaningful patterns.

**The Accuracy Paradox:**

A model can achieve high accuracy while being completely useless by always predicting the majority class.

**Concrete Example: Credit Card Fraud Detection**

Dataset characteristics:
- Total transactions: 100,000
- Fraudulent transactions: 100 (0.1%)
- Legitimate transactions: 99,900 (99.9%)

**Model A: "Naive" Classifier**
Strategy: Always predict "legitimate" (never predict fraud)

Performance:
- Predictions: All 100,000 labeled as "legitimate"
- Correct predictions: 99,900 (all legitimate cases)
- Incorrect predictions: 100 (missed all fraud)
- **Accuracy: 99.9%** ✓

Business value: **ZERO**
- Frauds detected: 0 out of 100 (0%)
- Useless for fraud prevention
- Might as well not have a model

**Model B: "Better" Classifier**
Strategy: Actual ML model that detects patterns

Performance:
- Predictions: 280 fraud, 99,720 legitimate
- True positives (fraud caught): 80
- False positives (false alarms): 200
- True negatives (correct legitimate): 99,700
- False negatives (missed fraud): 20
- **Accuracy: 99.78%** ✗ (Lower!)

Business value: **HIGH**
- Frauds detected: 80 out of 100 (80%)
- Prevents $80,000 in fraud (assuming $1000/fraud)
- False alarms: 200 (manageable review load)

**The Paradox:**
- Model A: 99.9% accurate, catches 0% of fraud → Useless
- Model B: 99.78% accurate, catches 80% of fraud → Valuable
- Lower accuracy is better!

**Why Accuracy Fails:**

1. **Dominated by Majority Class**
   - 99.9% of accuracy comes from majority class
   - Minority class performance is invisible
   - Can achieve 99.9% by ignoring minority class entirely

2. **Hides Critical Failures**
   - Missing all 100 frauds = only 0.1% accuracy drop
   - Seems minor, but 100% failure on what matters

3. **Inverted Incentive**
   - Optimizing for accuracy incentivizes ignoring minority class
   - Model learns: "Don't predict rare class, maximize accuracy"

**Mathematical Breakdown:**

For imbalanced data with positive class proportion p (where p << 0.5):

Naive model (always predict negative):
- Accuracy = 1 - p

Example accuracy for different imbalance ratios:
- 10% positive: Naive accuracy = 90%
- 1% positive: Naive accuracy = 99%
- 0.1% positive: Naive accuracy = 99.9%

The rarer the positive class, the higher the accuracy of the useless model!

**Better Metrics for Imbalanced Data:**

1. **Precision and Recall**
   \`\`\`
   Model A (Naive):
   - Precision: Undefined (no positive predictions)
   - Recall: 0% (caught nothing)
   - Reveals uselessness immediately
   
   Model B (Useful):
   - Precision: 80/280 = 28.6%
   - Recall: 80/100 = 80%
   - Shows it's actually detecting fraud
   \`\`\`

2. **F1-Score**
   \`\`\`
   Model A: F1 = 0 (precision and recall both terrible)
   Model B: F1 = 0.422 (meaningful detection)
   \`\`\`

3. **ROC-AUC**
   \`\`\`
   Model A: AUC = 0.5 (random guessing)
   Model B: AUC = 0.85 (good discrimination)
   \`\`\`

**Real-World Disaster Example:**

A company deployed a "99.5% accurate" fraud detector. After 6 months:
- Actual fraud detection rate: 5%
- Lost $5 million to undetected fraud
- Model was just predicting "not fraud" for almost everything
- Accuracy was high because fraud was rare
- Company learned expensive lesson: never use accuracy alone for imbalanced data

**When Accuracy Is Acceptable:**

Only use accuracy when:
1. Classes are balanced (roughly 40-60% split)
2. All errors are equally costly
3. You report it alongside precision/recall/F1

**Best Practice for Imbalanced Data:**

\`\`\`python
# DON'T do this:
score = accuracy_score(y_test, y_pred)
print(f"Model score: {score}")  # Misleading!

# DO this:
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
\`\`\`

**Key Takeaway**: For imbalanced data, accuracy is not just uninformative—it's actively misleading. It can make terrible models look good and good models look similar. Always use precision, recall, F1, and ROC-AUC instead.`,
    keyPoints: [
      'Accuracy is dominated by majority class in imbalanced datasets',
      'Naive "always predict majority" model achieves high accuracy while being useless',
      'Accuracy paradox: lower accuracy can indicate better model for imbalanced data',
      'Accuracy hides critical failures on minority class (the class that often matters most)',
      'For 1% positive class, predicting all negative gives 99% accuracy but 0% recall',
      'Use precision, recall, F1, and ROC-AUC for imbalanced datasets instead',
      'Never optimize for accuracy alone on imbalanced data—incentivizes ignoring minority class',
    ],
  },
  {
    id: 'classification-metrics-dq-3',
    question:
      'Compare and contrast ROC-AUC and Precision-Recall AUC. When is each more appropriate, and why does ROC-AUC become overly optimistic for highly imbalanced datasets?',
    sampleAnswer: `ROC-AUC and Precision-Recall AUC both evaluate classifier ranking quality but respond very differently to class imbalance, making them appropriate for different scenarios.

**Definitions:**

**ROC (Receiver Operating Characteristic) Curve:**
- X-axis: False Positive Rate (FPR) = FP/(FP+TN)
- Y-axis: True Positive Rate (TPR/Recall) = TP/(TP+FN)
- AUC: Area Under ROC Curve

**Precision-Recall Curve:**
- X-axis: Recall = TP/(TP+FN)
- Y-axis: Precision = TP/(TP+FP)
- AUC: Area Under PR Curve

**Key Mathematical Difference:**

ROC uses TN (true negatives) in FPR:
- FPR = FP/(FP+TN)

PR uses only positives (no TN):
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)

This seemingly small difference has huge implications for imbalanced data.

**Why ROC-AUC Becomes Overly Optimistic:**

**Example: Extreme Imbalance (0.1% positive class)**

Dataset:
- Positives: 100
- Negatives: 99,900
- Total: 100,000

**Mediocre Classifier Performance:**
- TP = 70 (caught 70% of positives)
- FP = 1,000 (1,000 false alarms)
- TN = 98,900
- FN = 30 (missed 30% of positives)

**ROC Metrics:**
\`\`\`
TPR (Recall) = TP/(TP+FN) = 70/100 = 0.70
FPR = FP/(FP+TN) = 1000/99900 = 0.01

ROC looks great:
- High TPR (70%)
- Low FPR (1%)
- ROC-AUC likely 0.90+ (seems excellent!)
\`\`\`

**Precision-Recall Metrics:**
\`\`\`
Recall = TP/(TP+FN) = 70/100 = 0.70
Precision = TP/(TP+FP) = 70/1070 = 0.065 (6.5%!)

PR reveals problems:
- Only 6.5% of positive predictions are correct
- 93.5% are false alarms
- Model is actually terrible for practical use!
\`\`\`

**Why the Difference?**

**ROC-AUC is "tricked" by large TN count:**
- FPR = 1000/99900 ≈ 1% looks small
- But 1,000 false positives is huge in practice!
- The massive number of true negatives (98,900) makes FPR look good
- ROC-AUC doesn't care that precision is 6.5%

**PR-AUC correctly identifies the problem:**
- Precision = 70/1070 = 6.5% shows most predictions are wrong
- No TN in calculation—can't hide behind large negative class
- Directly measures what matters: prediction quality

**Concrete Business Example:**

**Application: Rare Disease Detection (0.1% prevalence)**

Model predicts 1,000 people have the disease:
- 70 actually have it (TP)
- 930 don't have it (FP)

**ROC-AUC says:** "Great model! 90% AUC"
- High TPR, low FPR
- Looks very discriminative

**PR-AUC says:** "Poor model! 40% AUC"
- Only 7% precision
- 93% of flagged patients don't have the disease
- Massive waste of expensive follow-up tests

**Business Reality:**
- 930 healthy people undergo unnecessary expensive/invasive tests
- Huge cost and patient stress
- ROC-AUC hid this problem
- PR-AUC revealed it

**When to Use Each:**

**Use ROC-AUC When:**

1. **Balanced or moderately imbalanced datasets**
   - Class ratio roughly 40-60% or at worst 20-80%
   - TN count doesn't dominate

2. **FPR is meaningful metric**
   - When you care about false positive rate among negatives
   - Security screening: FPR = innocent people inconvenienced

3. **Comparing multiple models on same dataset**
   - Standard benchmark metric
   - Widely used and understood

4. **Both classes matter equally**
   - Want to understand tradeoffs across both classes

**Use Precision-Recall AUC When:**

1. **Highly imbalanced datasets (>95% one class)**
   - Positive class < 5%
   - ROC-AUC will be overly optimistic

2. **Positive class is focus**
   - Only care about detecting positives well
   - Don't care much about TN count
   - Example: Fraud detection—only positive predictions matter

3. **Cost of FP is high relative to TN**
   - Every false positive is expensive
   - True negatives are cheap/default state

4. **Business cares about precision**
   - "Of my positive predictions, how many are right?"
   - More intuitive than FPR for stakeholders

**Imbalance Rule of Thumb:**

\`\`\`
Positive class ratio:
  40-60%: Use ROC-AUC
  20-40%: Either works, but PR more conservative
  5-20%: Prefer PR-AUC
  < 5%: Must use PR-AUC (ROC will be misleading)
\`\`\`

**Example Comparison:**

**Fraud Detection (1% fraud rate):**

Model A:
- ROC-AUC: 0.92 (looks excellent)
- PR-AUC: 0.45 (reveals mediocrity)
- Reality: 92% of fraud predictions are false alarms

Model B:
- ROC-AUC: 0.89 (looks slightly worse)
- PR-AUC: 0.78 (actually much better)
- Reality: 78% of fraud predictions are correct

ROC-AUC ranks Model A > Model B (wrong!)
PR-AUC ranks Model B > Model A (correct!)

**Why This Matters in Production:**

If you deploy Model A based on ROC-AUC:
- Operations team drowns in false alarms
- Real frauds buried in noise
- Team stops trusting model
- Model gets turned off

If you use PR-AUC and deploy Model B:
- Fewer false alarms
- Higher precision means actionable alerts
- Team trusts model
- Model provides business value

**Best Practice:**

\`\`\`python
# Always report both for comparison
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
imbalance_ratio = sum(y_test) / len(y_test)

print(f"Positive class ratio: {imbalance_ratio:.1%}")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")

if imbalance_ratio < 0.05:
    print("⚠️  Highly imbalanced! Trust PR-AUC over ROC-AUC")
\`\`\`

**Key Insight:** ROC-AUC becomes optimistic for imbalanced data because it includes TN (which is huge for minority class problems) in FPR calculation. PR-AUC focuses only on positive class performance, revealing the true difficulty of the prediction task. For imbalanced data, PR-AUC is the honest metric.`,
    keyPoints: [
      'ROC-AUC uses FPR (includes TN), PR-AUC uses Precision (no TN)',
      'Large TN count in imbalanced data makes FPR look artificially small',
      'ROC-AUC can be 0.9+ while precision is only 10% for highly imbalanced data',
      'PR-AUC reveals true prediction quality by focusing only on positive class',
      'Use ROC-AUC for balanced data (40-60% ratio)',
      'Use PR-AUC for imbalanced data (<5% minority class)',
      'For fraud/rare disease detection, PR-AUC is the honest metric',
      'Always report both metrics and note class imbalance ratio',
    ],
  },
];
