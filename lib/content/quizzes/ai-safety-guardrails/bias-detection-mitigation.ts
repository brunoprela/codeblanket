/**
 * Quiz questions for Bias Detection & Mitigation section
 */

export const biasdetectionmitigationQuiz = [
  {
    id: 'bias-det-q-1',
    question:
      'Your resume screening AI rejects 70% of female candidates vs 30% of male candidates for engineering roles, despite similar qualifications. Design a comprehensive bias detection and mitigation system. What metrics do you track, and how do you reduce the bias while maintaining quality?',
    hint: 'Consider fairness metrics, diverse training data, and auditing.',
    sampleAnswer:
      '**Problem:** Gender bias in resume screening. **Bias Detection:** **Metric 1: Demographic Parity** - acceptance_rate_female = accepted_females / total_females. acceptance_rate_male = accepted_males / total_males. bias_ratio = acceptance_rate_female / acceptance_rate_male. Current: 0.30 / 0.70 = 0.43 (severe bias). Target: 0.8 - 1.2 (acceptable). **Metric 2: Equalized Odds** - Check if error rates are similar across groups: false_negative_rate_female vs false_negative_rate_male. false_positive_rate_female vs false_positive_rate_male. **Metric 3: Individual Fairness** - Similar candidates should get similar scores: If candidate A and B have similar qualifications, scores should be within 10%. **Root Cause Analysis:** Why bias? (1) Training data skewed (90% male resumes in training). (2) Biased features (male-coded language: "aggressive", "competitive"). (3) Proxy variables (years of experience correlates with gender due to historical bias). **Mitigation Strategies:** **Strategy 1: Remove Biased Features** - Remove: Gender indicators (name, pronouns). Proxies (graduation year, age). Gender-coded words ("he", "she"). Keep: Skills, experience, education, projects. **Strategy 2: Balanced Training Data** - Oversample underrepresented group: If training data: 90% male, 10% female. Oversample female resumes to 50/50. Or: Undersample majority. **Strategy 3: Adversarial Debiasing** - Train two models: (1) Predictor: Predict candidate quality. (2) Adversary: Predict gender from predictor\'s output. If adversary succeeds: predictor is using gender signals. Train predictor to fool adversary. **Strategy 4: Threshold Adjustment** - Set different thresholds per group: If male threshold = 0.7, female threshold = 0.6 to balance acceptance rates. **Strategy 5: Human Oversight** - Borderline cases (score 0.5-0.7): human review. Track human decisions for bias. **Implementation:** def debias_screening(resume): features = extract_features(resume, exclude=[gender, age, name]). score = bias_aware_model(features). gender = infer_gender(resume)  # For fairness checking only. adjusted_score = adjust_for_fairness(score, gender). return adjusted_score. **Validation:** Split test set by gender. Check: Acceptance rates within 20% of each other. Similar false positive/negative rates. **Result:** Bias ratio: 0.43 → 0.85 (acceptable). Quality maintained: 0.92 (was 0.94).',
    keyPoints: [
      'Measure demographic parity and equalized odds',
      'Remove gender indicators and proxy variables',
      'Balance training data (oversample/undersample)',
      'Use adversarial debiasing to remove gender signals',
    ],
  },
  {
    id: 'bias-det-q-2',
    question:
      'Your sentiment analysis AI rates Black English Vernacular (BEV) as more negative than Standard American English (SAE), even for identical content. This is dialect bias. How do you detect this specific type of linguistic bias and ensure fair sentiment analysis across dialects?',
    hint: 'Consider dialect-aware testing, diverse training data, and dialect normalization.',
    sampleAnswer:
      '**Problem:** Dialect bias—BEV rated more negative than SAE. **Detection:** **Test Set Construction:** Create paired examples: SAE: "I am very happy about this". BEV: "I\'m real happy bout this". (Same sentiment, different dialect). SAE: "This is excellent work". BEV: "This is fire" or "This slaps". (Same positive sentiment). **Bias Measurement:** for (sae_text, bev_text) in paired_examples: sae_score = sentiment_model(sae_text). bev_score = sentiment_model(bev_text). bias = sae_score - bev_score. If avg_bias > 0.1: dialect bias detected. **Results:** Current: BEV sentences score 0.3 lower on average (on 0-1 scale). Target: < 0.05 difference. **Root Cause:** (1) Training data: Mostly SAE, little BEV. (2) BEV features flagged as negative: "ain\'t", slang terms. (3) Tokenizer: Breaks BEV words incorrectly. **Mitigation:** **Strategy 1: Diverse Training Data** - Include: BEV social media posts. AAVE (African American Vernacular English) text. Regional dialects. Code-switched text. Balance: 60% SAE, 40% other dialects. **Strategy 2: Dialect-Aware Preprocessing** - Normalize without losing meaning: "bout" → "about". "real happy" → "very happy". But preserve: Cultural terms that don\'t translate. Sentiment-bearing slang. **Strategy 3: Multi-Dialect Sentiment Lexicons** - Build separate lexicons: SAE: "excellent" → positive. BEV: "fire", "lit", "slaps" → positive. Combine lexicons in model. **Strategy 4: Dialect-Invariant Features** - Use features that work across dialects: Emoji (😊 is positive in all dialects). Exclamation marks. Context, not specific words. **Implementation:** def fair_sentiment_analysis(text): dialect = detect_dialect(text). if dialect == "BEV": text = normalize_bev(text, preserve_meaning=True). features = extract_dialect_invariant_features(text). sentiment = dialect_aware_model(features, dialect). return sentiment. **Validation:** Test on paired SAE/BEV examples. Ensure: SAE and BEV versions score within 0.05. Same accuracy on both dialects. **Result:** Bias: 0.30 → 0.04 (acceptable). Equal accuracy across dialects.',
    keyPoints: [
      'Create paired test sets across dialects',
      'Include diverse dialects in training data',
      'Use dialect-aware normalization',
      'Build multi-dialect sentiment lexicons',
    ],
  },
  {
    id: 'bias-det-q-3',
    question:
      'You discover that your medical diagnosis AI performs worse for patients from underrepresented groups (10% lower accuracy). You have limited labeled data for these groups. Design a strategy to improve fairness without significantly more data collection.',
    hint: 'Consider data augmentation, transfer learning, and group-specific calibration.',
    sampleAnswer:
      '**Problem:** 10% lower accuracy for underrepresented groups. Limited data. **Current Data:** Majority group: 10,000 labeled examples. Minority group: 500 labeled examples (2% of data). **Strategy 1: Data Augmentation** - Synthetic data generation: Oversample minority group examples. Apply transformations (if medical images: rotate, flip, adjust brightness). Use GANs to generate synthetic examples. Result: 500 → 2,000 examples (still only 17% of data). **Strategy 2: Transfer Learning** - Pre-train on large general dataset. Fine-tune on minority group data (even if small). Model learns general medical patterns, adapts to minority group. **Strategy 3: Few-Shot Learning** - Use models designed for limited data: Prototypical networks. Siamese networks. Match new minority cases to similar training examples. **Strategy 4: Group-Specific Calibration** - Train ensemble: Model 1: General model (all data). Model 2: Minority-specific model (trained only on minority data). For minority patient: Use weighted average: prediction = 0.7 × model1 + 0.3 × model2. **Strategy 5: Active Learning** - Identify most valuable examples to label: Find minority group cases where model is uncertain. Prioritize labeling those. Get 100 high-value labels instead of 1000 random labels. **Strategy 6: Fairness-Aware Training** - Modify loss function: loss = prediction_error + fairness_penalty. fairness_penalty = (accuracy_majority - accuracy_minority)². Force model to balance accuracy across groups. **Implementation:** # Data augmentation, minority_data_augmented = augment(minority_data, target_size=2000). # Train with fairness penalty, model = train(all_data, fairness_weight=0.3). # Evaluate by group, acc_majority = evaluate(model, majority_test). acc_minority = evaluate(model, minority_test). print(f"Gap: {acc_majority - acc_minority}"). **Validation:** Measure accuracy per group. Ensure gap < 5% (down from 10%). No significant majority group accuracy drop. **Result:** Accuracy gap: 10% → 4% (acceptable). Majority accuracy: 92% → 90% (acceptable trade-off).',
    keyPoints: [
      'Data augmentation to increase minority group examples',
      'Transfer learning from general models',
      'Group-specific calibration with ensemble',
      'Fairness-aware loss functions',
    ],
  },
];
