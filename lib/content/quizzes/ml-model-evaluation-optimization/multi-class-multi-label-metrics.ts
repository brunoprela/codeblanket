import { QuizQuestion } from '../../../types';

export const multiClassMultiLabelMetricsQuiz: QuizQuestion[] = [
  {
    id: 'multi-class-multi-label-metrics-dq-1',
    question:
      'Explain the differences between macro-averaged, micro-averaged, and weighted-averaged F1 scores for multi-class classification. When should you use each one?',
    sampleAnswer: `The three averaging strategies for multi-class metrics handle class imbalance and importance differently, leading to dramatically different interpretations.

**Definitions:**

**Macro-Average (Simple Mean):**
\`\`\`
Calculate metric for each class independently
Average the results (all classes weighted equally)

F1_macro = (F1_class0 + F1_class1 + ... + F1_classN) / N
\`\`\`

**Weighted-Average (Frequency-Weighted Mean):**
\`\`\`
Calculate metric for each class independently
Average weighted by class frequency (support)

F1_weighted = Σ(F1_classi × support_i) / total_samples
\`\`\`

**Micro-Average (Global Aggregation):**
\`\`\`
Aggregate all TP, FP, FN across classes first
Then calculate metric from aggregated counts

TP_total = TP_class0 + TP_class1 + ... + TP_classN
FP_total = FP_class0 + FP_class1 + ... + FP_classN
FN_total = FN_class0 + FN_class1 + ... + FN_classN

F1_micro = 2 × TP_total / (2×TP_total + FP_total + FN_total)
\`\`\`

**Concrete Example:**

Customer Segmentation (3 classes):
- Bronze: 800 customers (80%), F1=0.95
- Silver: 150 customers (15%), F1=0.70
- Gold: 50 customers (5%), F1=0.40

**Macro-Average:**
\`\`\`
F1_macro = (0.95 + 0.70 + 0.40) / 3 = 0.68

Treats all classes equally
Poor Gold performance (5% of data) gets equal weight
Result: Heavily penalized by poor minority class
\`\`\`

**Weighted-Average:**
\`\`\`
F1_weighted = (0.95×800 + 0.70×150 + 0.40×50) / 1000
            = (760 + 105 + 20) / 1000
            = 0.885

Weighted by class size
Dominated by Bronze performance (80% of data)
Result: High score despite poor Gold performance
\`\`\`

**Micro-Average:**
\`\`\`
For multi-class: F1_micro = Accuracy = 0.89

Aggregates all predictions
Each prediction counts equally (regardless of class)
Result: Same as overall accuracy
\`\`\`

**When to Use Each:**

**Use Macro-Average When:**

1. **All classes are equally important**
   - Medical diagnosis: Each disease equally critical
   - Fraud types: Each fraud category matters
   - Want to ensure good performance on ALL classes

2. **Want to expose poor minority class performance**
   - Don't want majority class to hide issues
   - Example: 95% accuracy might hide 0% recall on rare class
   - Macro average will be low if any class performs poorly

3. **Balanced importance despite imbalance**
   - Customer segments: Even though Gold is 5%, it's most valuable
   - Rare diseases: Small proportion but high importance

**Use Weighted-Average When:**

1. **Class importance proportional to frequency**
   - More common classes should influence metric more
   - Reflects natural business priorities
   - Example: Bronze customers are 80% of revenue

2. **Need representative overall performance**
   - Want metric to reflect typical prediction scenario
   - More frequent classes = more predictions in practice
   - Balanced view considering data distribution

3. **Comparing models on same imbalanced dataset**
   - Fair comparison when class distribution matters
   - Standard for many competitions

**Use Micro-Average When:**

1. **Overall correctness is what matters**
   - Just want to know: "what % of predictions are correct?"
   - Equivalent to accuracy for multi-class
   - Simple, interpretable

2. **All predictions equally important**
   - Don't distinguish between classes
   - Example: Product categorization where any correct category is equally good

3. **Large class imbalance, care about total accuracy**
   - When majority class performance dominates business value

**Decision Framework:**

\`\`\`
Are minority classes important? (despite being rare)
  YES → Use Macro (ensures all classes perform well)
  NO → Use Weighted or Micro

Is class importance proportional to frequency?
  YES → Use Weighted (common classes weighted more)
  NO → Use Macro (all classes equal weight)

Just want overall accuracy?
  YES → Use Micro (= accuracy for multi-class)
  NO → Use Macro or Weighted
\`\`\`

**Real-World Example: Medical Diagnosis**

Diagnosing 3 conditions from symptoms:
- Healthy: 900 patients (90%), F1=0.98
- Flu: 80 patients (8%), F1=0.85
- Cancer: 20 patients (2%), F1=0.40

\`\`\`
Macro F1: (0.98 + 0.85 + 0.40)/3 = 0.74
→ Reveals poor cancer detection (critical!)

Weighted F1: 0.98×0.9 + 0.85×0.08 + 0.40×0.02 = 0.958
→ High score hides cancer detection failure (dangerous!)

Micro F1: 0.95 (accuracy)
→ Looks good but misses 60% of cancers

Decision: Use Macro F1 because missing cancer (2% of data) is
catastrophic, even though it's rare. Weighted F1 would hide this
critical failure by being dominated by the 90% healthy cases.
\`\`\`

**Key Principle**: Choose based on business priorities:
- Equal class importance → Macro
- Frequency-weighted importance → Weighted  
- Overall accuracy → Micro

For safety-critical applications (medical, security), always use Macro to ensure no class is neglected.`,
    keyPoints: [
      'Macro: Simple mean across classes, treats all classes equally',
      'Weighted: Frequency-weighted mean, common classes have more influence',
      'Micro: Aggregates all TP/FP/FN then calculates (equals accuracy for multi-class)',
      'Use Macro when minority classes are equally important (safety-critical)',
      'Use Weighted when importance proportional to frequency (business typical)',
      'Use Micro when overall correctness is the goal',
      'Macro exposes poor minority class performance; Weighted can hide it',
    ],
  },
  {
    id: 'multi-class-multi-label-metrics-dq-2',
    question:
      'Describe the key differences between multi-class and multi-label classification, and explain why they require different evaluation metrics. Provide concrete examples of each.',
    sampleAnswer: `Multi-class and multi-label classification are fundamentally different problem types that require different approaches to evaluation.

**Key Structural Differences:**

**Multi-class Classification:**
- Each sample belongs to EXACTLY ONE class
- Mutually exclusive categories
- Output: Single class label per sample
- Classes compete—picking one excludes all others

**Multi-label Classification:**
- Each sample can belong to MULTIPLE classes simultaneously
- Non-exclusive categories
- Output: Set of class labels per sample (can be empty, one, or many)
- Labels are independent—any combination possible

**Concrete Examples:**

**Multi-class: Digit Recognition (0-9)**
\`\`\`
Image → Model → Single digit

Sample 1: Image of "7" → Predict: 7 ✓
Sample 2: Image of "3" → Predict: 3 ✓
Sample 3: Image of "8" → Predict: 9 ✗

Cannot be multiple digits simultaneously
Must choose one and only one class
\`\`\`

**Multi-label: Movie Genres**
\`\`\`
Movie → Model → Set of genres

Sample 1: "Inception" → Predict: {Action, Sci-Fi, Thriller} ✓
Sample 2: "The Avengers" → Predict: {Action, Comedy, Sci-Fi} ✓
Sample 3: "Titanic" → Predict: {Drama, Romance} (missing "History") ✗

Can have multiple genres
Can have zero genres (though rare)
Labels are independent—any combination valid
\`\`\`

**Why Different Metrics Are Needed:**

**1. Multi-class Metrics Assume Mutual Exclusivity**

Multi-class confusion matrix is NxN where N = number of classes:
\`\`\`
      Pred 0  Pred 1  Pred 2
True 0   50      3       2
True 1    4     48       1
True 2    2      1      49

Each row sums to total samples in that class
Each sample appears once
\`\`\`

For multi-label, this doesn't work! A sample can be in multiple "true" rows.

**2. Multi-label Requires Overlap Metrics**

Consider movie with true labels {Action, Comedy}:
\`\`\`
Predicted: {Action, Sci-Fi}

Multi-class thinking: "Wrong! Should be Action+Comedy, not Action+Sci-Fi"
→ Treat as completely incorrect

Multi-label thinking: "Partial overlap"
→ Got 1/2 true labels correct (Action ✓)
→ Added 1 false label (Sci-Fi ✗)
→ Missed 1 true label (Comedy ✗)
→ 50% correct at label level
\`\`\`

**Multi-label Specific Metrics:**

**1. Hamming Loss**
\`\`\`
Fraction of labels incorrectly predicted

True:  [1, 1, 0, 0, 1]  (Action, Comedy, Sci-Fi)
Pred:  [1, 0, 1, 0, 1]  (Action, Sci-Fi)

Errors: Position 1 (missed Comedy), Position 2 (false Sci-Fi)
Hamming Loss = 2/5 = 0.40 (40% of labels wrong)

Why not use for multi-class: Would require treating each class
as separate binary problem, losing class relationships
\`\`\`

**2. Exact Match Ratio**
\`\`\`
Fraction of samples where ALL labels match exactly

Sample 1: True={A,B}, Pred={A,B} → Match ✓
Sample 2: True={A,C}, Pred={A,B,C} → No match (extra B) ✗
Sample 3: True={B}, Pred={B} → Match ✓

Exact Match = 2/3 = 0.67

Very strict metric—even one wrong label = failure
For multi-class, this is just accuracy (always exact or not)
\`\`\`

**3. Jaccard Score (Intersection over Union)**
\`\`\`
Overlap between true and predicted label sets

True: {Action, Comedy, Drama}
Pred: {Action, Comedy, Romance}

Intersection: {Action, Comedy} = 2 labels
Union: {Action, Comedy, Drama, Romance} = 4 labels
Jaccard = 2/4 = 0.50

Measures partial credit for overlap
Not applicable to multi-class (no notion of partial overlap)
\`\`\`

**Why Standard Multi-class Metrics Fail for Multi-label:**

**Accuracy** (Exact Match) is too strict:
\`\`\`
Movie with true genres {Action, Comedy, Drama}
Pred: {Action, Comedy} (missing Drama)

Multi-label Exact Match: 0 (all labels must match) ✗
But got 2/3 labels correct! Should get partial credit.

Hamming Loss: 0.20 (only 1/5 labels wrong) ✓
Jaccard: 0.67 (good overlap) ✓
→ Better reflect partial success
\`\`\`

**Precision/Recall** work differently:
\`\`\`
Multi-class:
  Precision = TP / (TP + FP) for each class separately
  One confusion matrix captures all interactions

Multi-label:
  Precision averaged across samples or labels
  Need separate binary evaluation per label
  Can have precision/recall per label AND overall
\`\`\`

**Practical Example: Medical Diagnosis**

**Multi-class Scenario: Disease Classification**
\`\`\`
Symptoms → Single disease diagnosis
Patient can have:
  - Flu OR
  - Common Cold OR
  - Pneumonia OR
  - Healthy

Mutually exclusive (for this simplified model)
Use: Confusion matrix, multi-class F1, Cohen's Kappa
\`\`\`

**Multi-label Scenario: Symptom Detection**
\`\`\`
Patient data → Multiple symptoms
Patient can have:
  - Fever + Cough + Fatigue + Headache (all simultaneously)
  - Or any combination
  - Or none

Not mutually exclusive
Use: Hamming loss, per-label F1, Jaccard score
\`\`\`

**Evaluation Comparison:**

Multi-class movie categorization (pick primary genre):
\`\`\`
True: Action
Pred: Sci-Fi

Evaluation: Wrong! 0% accuracy for this sample
→ Binary success/failure
\`\`\`

Multi-label movie tagging (all applicable genres):
\`\`\`
True: {Action, Sci-Fi, Adventure}
Pred: {Action, Sci-Fi}

Evaluation:
  - Hamming: 1/5 = 20% error (missed Adventure)
  - Jaccard: 2/3 = 67% overlap
  - Per-label: Action✓, Sci-Fi✓, Adventure✗, Comedy✓, Drama✓
→ Captures partial success
\`\`\`

**Key Takeaway:**

Multi-class assumes mutual exclusivity and uses metrics designed for single-label decisions. Multi-label allows multiple simultaneous labels and requires metrics that handle overlap, partial matches, and label-wise errors. Using multi-class metrics for multi-label problems (or vice versa) gives misleading or meaningless results. Always match the metric type to your problem structure.`,
    keyPoints: [
      'Multi-class: Each sample has ONE class (mutually exclusive)',
      'Multi-label: Each sample can have MULTIPLE classes (non-exclusive)',
      'Multi-class uses NxN confusion matrix and standard precision/recall',
      'Multi-label needs overlap metrics: Hamming loss, Jaccard, Exact Match',
      'Multi-class accuracy is binary (right/wrong); multi-label allows partial credit',
      'Examples: Multi-class = digit recognition; Multi-label = movie genres',
      'Must match evaluation approach to problem structure for meaningful results',
    ],
  },
  {
    id: 'multi-class-multi-label-metrics-dq-3',
    question:
      'For a highly imbalanced 10-class classification problem where one class represents 85% of data, explain why reporting only overall accuracy and macro-averaged F1 might give conflicting signals. How would you provide a complete evaluation?',
    sampleAnswer: `When dealing with severe class imbalance, different metrics can tell contradictory stories because they emphasize different aspects of performance. Understanding why requires examining what each metric actually measures.

**Scenario Setup:**

10-class classification with extreme imbalance:
- Class 0: 8,500 samples (85%) - "Normal/Majority"
- Classes 1-9: 1,500 samples total (15%) - Various minorities

**Model Performance:**
- Class 0: 98% F1 (excellent on majority)
- Classes 1-9: Variable performance (30%-80% F1)

**Why Metrics Conflict:**

**Overall Accuracy: 93%**
\`\`\`
Calculation:
  Correct predictions: 8,330 (class 0) + 800 (classes 1-9) = 9,130
  Total: 10,000
  Accuracy: 91.3%

Why it's high:
  - Dominated by 85% majority class
  - Class 0 performance (98%) heavily weighted
  - Minority class failures barely register
  
Signal: "Model is excellent! 93% accurate!"
\`\`\`

**Macro-Averaged F1: 0.62**
\`\`\`
Calculation:
  F1 scores: [0.98, 0.70, 0.45, 0.60, 0.30, 0.75, 0.50, 0.65, 0.40, 0.80]
  Macro F1: Sum / 10 = 6.13 / 10 = 0.613

Why it's low:
  - All classes weighted equally
  - Poor minority performance (0.30, 0.40, 0.45) drags down average
  - Class 0's 98% gets same weight as Class 4's 30%
  
Signal: "Model is mediocre! Only 62% average performance!"
\`\`\`

**The Conflict:**

Accuracy says: "93% - Ready for production!"
Macro F1 says: "62% - Needs improvement!"

Both are correct for what they measure:
- Accuracy: "What % of predictions are right?" → 93% (high)
- Macro F1: "Average per-class performance?" → 62% (mediocre)

**Why This Happens:**

**1. Different Weighting Schemes**

Accuracy weights by frequency:
\`\`\`
Accuracy ≈ 0.85×(Class 0 performance) + 0.15×(Minority performance)
         ≈ 0.85×0.98 + 0.15×0.53
         ≈ 0.833 + 0.080
         ≈ 0.913 (91.3%)

Dominated by majority class
\`\`\`

Macro F1 weights equally:
\`\`\`
Macro F1 = (F1_class0 + F1_class1 + ... + F1_class9) / 10
         = Each class contributes 10% to final score
         = Poor minorities significantly impact score
\`\`\`

**2. Different Questions Answered**

Accuracy: "If I make a random prediction, what's my chance of being right?"
→ In practice, 85% of predictions are Class 0, so high accuracy expected

Macro F1: "How well does model perform on each class on average?"
→ Exposes that 40-50% of classes have poor performance

**Complete Evaluation Framework:**

**1. Report Multiple Perspectives:**

\`\`\`python
print("Overall Metrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Macro F1: {macro_f1:.3f}")
print(f"  Weighted F1: {weighted_f1:.3f}")
print(f"  Micro F1: {micro_f1:.3f}")

print("\\nClass Distribution:")
for i in range(10):
    print(f"  Class {i}: {support[i]} samples ({support[i]/total*100:.1f}%)")

print("\\nPer-Class Performance:")
for i in range(10):
    print(f"  Class {i}: Precision={prec[i]:.3f}, Recall={rec[i]:.3f}, F1={f1[i]:.3f}")
\`\`\`

**2. Confusion Matrix Analysis:**

Examine which classes are confused:
\`\`\`
Class 4 (30% F1):
  - True: 150 samples
  - Predicted correctly: 45 (30% recall)
  - Misclassified as Class 0: 80 (53% of errors)
  - Misclassified as other: 25

→ Model biased toward predicting majority class
→ Need to address Class 4 specifically
\`\`\`

**3. Weighted F1 for Context:**

\`\`\`
Weighted F1: 0.89

Provides business-relevant view:
  - If predictions match real-world distribution
  - Reflects typical user experience
  - More representative than macro for deployment
\`\`\`

**4. Per-Class Deep Dive:**

Group classes by performance:
\`\`\`
Excellent (F1 > 0.80): Classes 0, 9 (8,700 samples, 87%)
Good (F1 0.60-0.80): Classes 1, 5, 7 (550 samples, 5.5%)
Poor (F1 < 0.60): Classes 2, 3, 4, 6, 8 (750 samples, 7.5%)

Issue: 7.5% of data (750 samples) has poor performance
→ May be acceptable if these are less critical
→ Unacceptable if these are high-value minorities
\`\`\`

**5. Business Context Integration:**

\`\`\`
If minority classes are:

High-value customers (Classes 1-9):
  → Macro F1 (0.62) is the honest metric
  → Can't afford poor performance on valuable segments
  → Need improvement despite 93% accuracy

Rare but unimportant cases:
  → Weighted F1 (0.89) is appropriate
  → 93% accuracy acceptable for business
  → Majority class performance is what matters
\`\`\`

**Recommended Complete Report:**

\`\`\`
Model Evaluation Summary
========================

Overall Performance:
  - Accuracy: 93.0% (High, but dominated by majority class)
  - Weighted F1: 89.3% (Representative of typical prediction)
  - Macro F1: 61.3% (Average across all classes - reveals struggles)
  - Micro F1: 93.0% (Same as accuracy for multi-class)

Class Distribution:
  - Majority (Class 0): 85% of data, F1=0.98 (Excellent)
  - Minorities (Classes 1-9): 15% of data, avg F1=0.47 (Poor)

Performance Breakdown:
  - 87% of samples: Excellent performance (F1 > 0.80)
  - 5.5% of samples: Good performance (F1 0.60-0.80)
  - 7.5% of samples: Poor performance (F1 < 0.60) ⚠️

Critical Issues:
  - Classes 2, 4, 8 have F1 < 0.50
  - High false negative rate for minorities
  - Model biased toward predicting Class 0

Recommendation:
  [Based on business context]
  
  If minorities are critical:
    → Reject model - Macro F1 too low
    → Consider class weights, resampling, or separate models
  
  If minorities are low-priority:
    → Accept model - Weighted F1 adequate
    → Monitor minority class performance in production
\`\`\`

**Key Principle:**

Never report a single metric for imbalanced multi-class problems. The conflict between accuracy and macro F1 is a feature, not a bug—it exposes the model's actual behavior. Report:

1. Accuracy (overall correctness)
2. Macro F1 (per-class average)
3. Weighted F1 (frequency-adjusted)
4. Per-class metrics (where problems hide)
5. Confusion matrix (how errors occur)

Then interpret based on business priorities and class importance.`,
    keyPoints: [
      'Accuracy dominated by majority class (85% weight), shows overall correctness',
      'Macro F1 treats all classes equally, exposes poor minority class performance',
      'Conflicting signals are informative: 93% accuracy + 62% macro F1 = imbalance issues',
      'Weighted F1 provides middle ground, representative of real prediction distribution',
      'Must report multiple metrics: accuracy, macro F1, weighted F1, per-class breakdown',
      'Confusion matrix reveals which minority classes struggle and why',
      'Complete evaluation requires business context: are minorities critical or not?',
      'Never rely on single metric for imbalanced multi-class problems',
    ],
  },
];
