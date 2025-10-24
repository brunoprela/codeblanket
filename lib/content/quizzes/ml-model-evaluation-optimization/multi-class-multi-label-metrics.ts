export const multiClassMultiLabelMetricsQuiz = {
  title: 'Multi-class & Multi-label Metrics - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `Explain the difference between macro-average, micro-average, and weighted-average for multi-class classification metrics. When would each be most appropriate? Provide a specific scenario for each and explain why that averaging method is best suited.`,
      expectedAnswer: `**Macro-average**: Calculate metric per class, then average (all classes equal weight). Use when: 1) All classes equally important (disease types - rare disease as important as common), 2) Want to highlight performance on minority classes. **Micro-average**: Aggregate all predictions globally, then calculate metric (all samples equal weight). Use when: 1) Overall accuracy matters most, 2) Large class imbalance is acceptable, 3) Frequent classes are more important (user behavior prediction - common actions matter most). **Weighted-average**: Calculate per class, average weighted by class frequency. Use when: 1) Balance between macro/micro, 2) Want to reflect class distribution but see per-class performance. **Scenarios**: 1) **Medical diagnosis (5 rare diseases, 1 common)**: Macro - missing rare disease as bad as common one, 2) **E-commerce categories (Electronics: 50%, Books: 30%, Other: 20%)**: Weighted - maintain business distribution, 3) **Content moderation (millions of safe, thousands of unsafe)**: Micro - overall accuracy on all posts matters. **Key**: Macro highlights weakness in rare classes, micro can hide it.`,
      difficulty: 'intermediate' as const,
      category: 'Multi-class',
    },
    {
      id: 2,
      question: `Design an evaluation strategy for a multi-label text classification system that tags news articles with multiple topics (Politics, Sports, Technology, etc.). What metrics would you use? How would you handle the fact that some articles have 1 tag while others have 5? What about tag co-occurrence patterns?`,
      expectedAnswer: `**Core Metrics**: 1) **Hamming Loss**: Fraction of wrong labels per sample - good for overall error rate, 2) **Exact Match Ratio**: % of samples with all labels correct - strict but important, 3) **Jaccard Score (IoU)**: Intersection over union - handles variable label counts naturally, 4) **F1 (samples average)**: Per-sample F1 averaged - rewards correct labels, penalizes missing/wrong. **Per-Label Analysis**: Track precision, recall, F1 for each tag independently to identify problematic tags. **Handling Variable Labels**: 1) **Jaccard** naturally handles this - normalizes by union, 2) Report distribution of label counts in predictions vs ground truth, 3) Stratify evaluation by #labels (separate metrics for 1-tag, 2-tag, 3+ tag articles). **Tag Co-occurrence**: 1) Build confusion matrix for tag pairs (e.g., Politics+Economy often together), 2) Track conditional probabilities P(tag_i | tag_j), 3) Evaluate on frequent vs rare combinations separately, 4) Check if model learns correlations (predict Economy→predict Politics too). **Calibration**: Plot predicted probability distributions per tag - are probabilities well-calibrated? **Business Metrics**: Reader engagement per tag, recommendation click-through rate.`,
      difficulty: 'advanced' as const,
      category: 'Multi-label',
    },
    {
      id: 3,
      question: `You're comparing two multi-class sentiment analysis models (Positive/Negative/Neutral) with these results: Model A (Macro F1=0.75, Micro F1=0.85), Model B (Macro F1=0.82, Micro F1=0.80). Explain what these numbers tell you about each model's strengths and weaknesses. Which would you choose for a product review system?`,
      expectedAnswer: `**Analysis**: Model A shows large gap (0.10) between macro and micro - this indicates imbalanced performance. High micro F1 (0.85) means overall accuracy is good, but low macro F1 (0.75) means poor performance on minority classes. Likely: great at majority class, poor at rare classes. Model B shows small gap (0.02) - more balanced across classes. Lower micro but higher macro means better on all classes. **Interpretation**: 1) Check class distribution - if Positive=60%, Neutral=30%, Negative=10%, Model A probably excels at Positive/Neutral but fails on Negative, 2) Model B more balanced - similar performance across all sentiments. **For Product Reviews, choose Model B**: 1) All sentiments matter - can't ignore negative reviews (most critical!), 2) Negative class likely minority but most actionable for business, 3) Want consistent performance for all users, 4) Macro F1=0.82 better than 0.75 shows Model B more robust. **Verification**: Check per-class confusion matrices, especially Negative class recall/precision. Model A might achieve high micro F1 by predicting Positive/Neutral well but missing most Negatives (business disaster). **Exception**: If negative reviews are 1% (spam/extreme cases), might prefer Model A for overall accuracy.`,
      difficulty: 'advanced' as const,
      category: 'Interpretation',
    },
  ],
};
