export const classificationMetricsQuiz = {
  title: 'Classification Metrics - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `You're building a fraud detection system for credit card transactions where only 0.1% of transactions are fraudulent. A naive model that predicts "not fraud" for everything achieves 99.9% accuracy. Explain why accuracy is misleading here and design a proper evaluation strategy including appropriate metrics, decision threshold selection, and business-aware evaluation.`,
      expectedAnswer: `Accuracy is misleading because it's dominated by majority class - predicting all negative gives 99.9% accuracy but catches 0% fraud! **Proper Evaluation**: 1) **Primary metric**: Recall (catch as many frauds as possible) and Precision (minimize false alarms), 2) **F1 score**: Balance both, but can weight toward recall (F2 score), 3) **PR-AUC**: Better than ROC-AUC for imbalanced data, 4) **Confusion matrix**: Track actual TP, FP, FN, TN counts. **Threshold Selection**: Don't use default 0.5! Plot precision-recall curve, choose threshold based on business constraints. Example: If each fraud costs $500 and each false alarm costs $5 in customer friction, find threshold where expected cost is minimized. **Business Metrics**: 1) Detection rate (recall) - want >95%, 2) False alarm rate - keep <1% of legitimate transactions, 3) Cost per transaction, 4) Customer friction. **Additional**: Use cost-sensitive learning, SMOTE for training, different thresholds for different transaction sizes.`,
      difficulty: 'advanced' as const,
      category: 'Imbalanced Data',
    },
    {
      id: 2,
      question: `Compare ROC-AUC and PR-AUC (Precision-Recall AUC) as evaluation metrics. When would you prefer one over the other? Provide specific examples where each is more appropriate, and explain the mathematical/practical reasoning.`,
      expectedAnswer: `**ROC-AUC**: Plots TPR (recall) vs FPR at various thresholds. Area under curve measures overall discrimination ability. **PR-AUC**: Plots Precision vs Recall. More informative for imbalanced data. **Key Difference**: ROC uses true negatives (FPR=FP/(FP+TN)), PR ignores them (Precision=TP/(TP+FP)). **When to use ROC-AUC**: 1) Balanced datasets (50/50 split), 2) When you care about both classes equally, 3) When true negative rate matters (disease screening where negative diagnosis needs accuracy), 4) Standard benchmark for comparison. **When to use PR-AUC**: 1) Highly imbalanced data (fraud: 0.1%, spam: 5%, etc.), 2) When positive class is focus (we care about catching positives), 3) When true negatives are trivial (millions of non-fraudulent transactions - not informative). **Example**: Spam detection (5% spam) - ROC-AUC might show 0.95 even with poor performance because true negatives are easy. PR-AUC better reflects actual precision/recall tradeoff. **Medical screening**: ROC-AUC appropriate because true negatives matter (confirming healthy patients).`,
      difficulty: 'intermediate' as const,
      category: 'Metrics',
    },
    {
      id: 3,
      question: `You've trained three different models for email spam classification: Model A (Precision=0.95, Recall=0.70), Model B (Precision=0.80, Recall=0.95), Model C (Precision=0.88, Recall=0.88). Which model would you deploy and why? How would your answer change for different applications (spam filtering, fraud detection, medical diagnosis)?`,
      expectedAnswer: `**Model A (High Precision, Low Recall)**: Rarely marks legitimate email as spam but misses 30% of spam. **Model B (Low Precision, High Recall)**: Catches 95% of spam but 20% false positives (legitimate emails to spam). **Model C (Balanced)**: Balanced approach. **For Spam Filtering, choose Model A**: False positives (legitimate → spam) are worse than false negatives (spam → inbox). Missing spam is annoying, losing important email is critical. Users can manually delete spam. **For Fraud Detection, choose Model B**: Missing fraud ($500-$5000 loss) worse than false alarms ($5 friction). Can review flagged transactions. Better to be cautious. **For Medical Diagnosis, choose Model B**: Missing disease (false negative) can be life-threatening. False positive leads to more testing (acceptable cost). Then use Model A as secondary screen after positive to reduce unnecessary treatments. **General Strategy**: 1) Quantify cost of FP vs FN, 2) Use cost-sensitive evaluation, 3) Consider two-stage approach (high recall filter + high precision confirmation), 4) Different thresholds for different risk levels.`,
      difficulty: 'advanced' as const,
      category: 'Model Selection',
    },
  ],
};
