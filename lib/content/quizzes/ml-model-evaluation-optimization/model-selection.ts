import { QuizQuestion } from '../../../types';

export const modelSelectionQuiz: QuizQuestion[] = [
  {
    id: 'model-selection-dq-1',
    question:
      'You trained 5 models and Model A has AUC=0.87, Model B has AUC=0.86. Should you automatically choose Model A? Explain what other factors you would consider before making a decision.',
    sampleAnswer: `No, don't automatically choose Model A based solely on a 0.01 AUC difference. Here's what to consider:

**1. Statistical Significance:**
- Is the 0.01 difference statistically significant or due to random variation?
- Use paired t-test on cross-validation scores
- If p-value > 0.05, the difference may not be meaningful

**2. Practical Significance:**
- Does 0.01 AUC translate to meaningful business impact?
- For fraud detection: 0.01 AUC might save $100K/year → significant
- For content recommendation: 0.01 AUC might be imperceptible → not significant

**3. Inference Latency:**
- Model A: Deep neural network (50ms per prediction)
- Model B: Logistic regression (0.5ms per prediction)
- For real-time applications, Model B might be required despite lower AUC

**4. Training Time:**
- Model A: 6 hours to train, Model B: 5 minutes
- If you retrain daily, Model B might be more practical

**5. Interpretability:**
- In regulated industries (finance, healthcare), explainability may be mandatory
- A simpler, interpretable Model B might be required over black-box Model A

**6. Model Complexity & Maintenance:**
- Model A: Complex ensemble requiring specialized libraries
- Model B: Simple model that any engineer can maintain
- Long-term maintenance cost matters

**7. Robustness:**
- Check performance variance: Model A might have higher std (less stable)
- Test on out-of-distribution data
- Model with lower mean but higher stability might be preferred

**Decision Framework:**
- If difference is not statistically significant → choose simpler model
- If difference is significant but small → weigh non-performance factors
- If Model A meets all constraints and difference is meaningful → choose Model A

**Conclusion:** Model selection requires balancing performance, latency, interpretability, maintenance, and business value—not just maximizing a single metric.`,
    keyPoints: [
      'Small performance differences may not be statistically significant',
      'Consider inference latency for production deployment',
      'Interpretability matters in regulated industries',
      'Training time affects retraining frequency',
      'Model complexity impacts long-term maintenance',
      'Practical significance != statistical significance',
    ],
  },
  {
    id: 'model-selection-dq-2',
    question:
      'Explain the concept of "overfitting to the test set" in model selection. How can this happen, and what strategies prevent it?',
    sampleAnswer: `Overfitting to the test set occurs when you make too many model selection decisions based on test set performance, causing the test set to essentially become a validation set. This leads to overly optimistic performance estimates.

**How It Happens:**

1. **Iterative Testing:**
   - Train Model A → test → AUC = 0.82
   - Try Model B → test → AUC = 0.84
   - Try Model C → test → AUC = 0.83
   - Choose Model B because it's best on test set
   - Problem: You've implicitly optimized for test set performance

2. **Multiple Comparisons:**
   - Testing 20 models on same test set
   - By chance, one might perform well even if it's not truly better
   - p-value correction needed (Bonferroni)

3. **Hyperparameter Tuning on Test Set:**
   - Tuning hyperparameters using test set feedback
   - Test set becomes validation set
   - No clean holdout for final evaluation

**Consequences:**
- Reported performance is optimistically biased
- Model will likely underperform in production
- You've "leaked" information from test set into model selection

**Prevention Strategies:**

**1. Three-Way Split:**
   - Training (60%): Train models
   - Validation (20%): Select models & tune hyperparameters
   - Test (20%): Final evaluation ONCE
   - Use test set only for final, single evaluation

**2. Nested Cross-Validation:**
   - Outer loop: Performance evaluation
   - Inner loop: Model selection & hyperparameter tuning
   - Provides unbiased performance estimate
   - More expensive but rigorous

**3. Limit Test Set Usage:**
   - Use test set sparingly (< 3 times)
   - For final model only
   - Track how many times you've used it

**4. Statistical Correction:**
   - Bonferroni correction: Divide α by number of comparisons
   - Example: Testing 20 models, use α = 0.05/20 = 0.0025

**5. Fresh Test Data:**
   - Periodically collect new test data
   - Validates that model still performs on unseen data

**Best Practice:**
- Use validation set (or CV) for all model selection decisions
- Reserve test set for final evaluation
- Report test set performance only once
- If you must iterate, use nested CV for unbiased estimates`,
    keyPoints: [
      'Testing multiple models on same test set leads to overfitting',
      'Multiple comparisons increase chance of false positives',
      'Use three-way split: train/validation/test',
      'Reserve test set for final evaluation only',
      'Nested CV provides unbiased estimates with repeated evaluation',
      'Statistical correction (Bonferroni) when testing many models',
    ],
  },
  {
    id: 'model-selection-dq-3',
    question:
      'Your company needs to deploy a model to detect fraudulent transactions in real-time. You have three candidates: (1) Logistic Regression (AUC=0.82, 1ms latency), (2) Gradient Boosting (AUC=0.88, 50ms latency), (3) Neural Network (AUC=0.90, 200ms latency). Which would you choose and why? Consider the full business context.',
    sampleAnswer: `I would choose **Gradient Boosting (AUC=0.88, 50ms latency)** as the optimal balance. Here's my reasoning:

**Requirements Analysis:**

**Performance:**
- Minimum AUC requirement likely ~0.85 for fraud detection
- Higher AUC = fewer false negatives (missed fraud) and false positives (declined legitimate transactions)
- Both Gradient Boosting (0.88) and Neural Network (0.90) meet requirements

**Latency:**
- Real-time fraud detection requires sub-100ms response
- User experience: >100ms delay noticeable, >500ms unacceptable
- This rules out the Neural Network (200ms) immediately

**Business Impact:**

**Option 1: Logistic Regression (0.82, 1ms)**
- Pros: Very fast, simple to maintain
- Cons: Lower AUC means:
  * More fraud slips through (higher loss)
  * More false positives (angry customers)
- Business cost: At 1M transactions/day, 0.06 AUC difference might mean $500K/year in additional fraud losses

**Option 2: Gradient Boosting (0.88, 50ms)** ⭐
- Pros:
  * Good performance (0.88 AUC)
  * Acceptable latency (50ms << 100ms threshold)
  * Feature importance for explainability
  * Proven in production at scale
- Cons:
  * More complex than logistic regression
  * Moderate training time (may require retraining infrastructure)
- Business impact: Optimal balance of performance and speed

**Option 3: Neural Network (0.90, 200ms)**
- Pros: Best performance
- Cons:
  * Too slow for real-time (200ms >> 100ms threshold)
  * User frustration with delays
  * May lose customers to competitors
- Business cost: Slow checkout may reduce conversion by 5% = huge revenue loss

**Additional Considerations:**

**Interpretability:**
- Fraud detection often requires explanation for declined transactions
- Gradient Boosting provides feature importance and SHAP values
- This helps customer service explain decisions
- Neural networks are harder to explain

**Adaptability:**
- Fraud patterns evolve quickly
- Need frequent retraining (daily or weekly)
- Gradient Boosting: reasonable training time
- Neural Network: longer training time may be problematic

**Infrastructure:**
- 50ms latency means can handle 20 requests/second per instance
- 200ms means only 5 requests/second
- Neural Network would require 4x more servers = higher cost

**Decision: Gradient Boosting**
- Meets performance requirements (0.88 > 0.85 threshold)
- Well within latency budget (50ms < 100ms)
- Good explainability for customer service
- Reasonable training time for frequent retraining
- Lower infrastructure cost than Neural Network

**Contingency Plan:**
- Monitor performance in production
- If fraud losses are unacceptable, optimize Neural Network
- Consider model distillation: train smaller NN to mimic large one
- Ensemble Logistic Regression + Gradient Boosting for speed-performance tradeoff`,
    keyPoints: [
      'Real-time requirements eliminate high-latency models',
      'Balance performance with latency constraints',
      'Consider business impact beyond metrics',
      'Interpretability matters for fraud detection',
      'Training time affects adaptability to new fraud patterns',
      'Infrastructure cost scales with latency',
      'Choose model that meets all constraints, not just best metric',
    ],
  },
];
