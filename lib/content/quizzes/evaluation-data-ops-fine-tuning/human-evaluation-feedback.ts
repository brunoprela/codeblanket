/**
 * Discussion questions for Human Evaluation & Feedback section
 */

export const humanEvaluationFeedbackQuiz = [
  {
    id: 'human-eval-q-1',
    question:
      'Design a human evaluation protocol for a customer service chatbot that needs to balance quality, cost, and speed. Your options are: (1) Expert evaluators ($50/hour, high quality, slow), (2) Crowdworkers ($10/hour, variable quality, fast), (3) End users (free, authentic, very noisy). How would you combine these approaches, and what evaluation rubric would you use?',
    hint: 'Consider using different evaluator types for different purposes: experts for edge cases and rubric validation, crowdworkers for volume, end users for in-context feedback. Think about a multi-dimensional rubric beyond just correctness.',
    sampleAnswer:
      "**Multi-Tier Evaluation Strategy:**\n\n**Tier 1 - Expert Evaluators (5% of samples, $50/hr):**\n- Use for: Edge cases, ambiguous queries, rubric calibration\n- Tasks: Detailed annotations, writing gold standards, reviewing disagreements\n- Sample 100 high-stakes conversations per week\n- Cost: ~$250/week\n\n**Tier 2 - Trained Crowdworkers (20% of samples, $10/hr):**\n- Use for: Routine quality checks, volume measurement\n- Tasks: Apply established rubric to standard conversations\n- Qualification test required (80% agreement with experts)\n- Triple annotation on random 10% for quality control\n- Sample 1000 conversations per week\n- Cost: ~$300/week\n\n**Tier 3 - End User Feedback (100% coverage, free):**\n- Use for: In-context ratings, issue detection\n- Tasks: Thumbs up/down, 5-star rating, optional comment\n- Real-time feedback after each conversation\n- ~5% response rate expected\n- Flag extreme negatives for expert review\n\n**Evaluation Rubric (1-5 scale on each dimension):**\n1. **Correctness**: Did the bot provide accurate information?\n2. **Completeness**: Did it address all parts of the query?\n3. **Tone**: Was it appropriately professional and empathetic?\n4. **Efficiency**: Did it resolve the issue without unnecessary back-and-forth?\n5. **Safety**: Did it avoid harmful, biased, or inappropriate content?\n\n**Weighted Score**: \`0.35×Correctness + 0.25×Completeness + 0.15×Tone + 0.15×Efficiency + 0.10×Safety\`\n\n**Quality Control:**\n- Gold standard set: 200 examples with expert consensus\n- All annotators must pass calibration (>0.75 agreement with gold)\n- Monthly re-calibration required\n- Inter-annotator agreement monitored (target: Cohen\'s Kappa > 0.60)\n- Disagree ments >2 points flagged for expert resolution\n\n**Decision Rules:**\n- Overall score ≥4.0: Good, no action\n- 3.0-3.9: Acceptable, monitor trend\n- <3.0 or any dimension <2.0: Investigate and improve\n\n**Cost-Quality Tradeoff:**\n- Total cost: ~$550/week (~$2,200/month)\n- Coverage: 5% expert-reviewed, 20% crowd-reviewed, 100% user-rated\n- This balances statistical significance (1000+ evaluations/week) with quality (expert validation) and authenticity (user feedback).",
    keyPoints: [
      'Multi-tier approach: experts for calibration, crowdworkers for volume, users for authenticity',
      'Multi-dimensional rubric beyond correctness: tone, completeness, efficiency, safety',
      'Gold standard sets and qualification tests ensure annotator quality',
      "Inter-annotator agreement monitoring (Cohen\'s Kappa, Krippendorff's Alpha)",
      'Cost-effective: Focus expensive expert time on high-value samples',
      'Continuous calibration: Regular re-training prevents annotator drift',
    ],
  },
  {
    id: 'human-eval-q-2',
    question:
      'You collect user feedback with a thumbs up/down button. After 10,000 interactions, you see: 300 thumbs up (3%), 100 thumbs down (1%), 9,600 no feedback (96%). Your model team says "96% neutral, 3:1 positive:negative ratio, looks great!" What is wrong with this interpretation, and how would you design a better feedback system?',
    hint: 'Consider response bias: who chooses to give feedback and when? Users are more likely to respond after extreme experiences. Also think about designing for different feedback modalities and contexts.',
    sampleAnswer:
      "**Problems with Current Interpretation:**\n\n**1. Response Bias:** The 96% silent majority likely has opinions, but we don't know them. Users give feedback when:\n- Extremely satisfied (thumbs up)\n- Extremely dissatisfied (thumbs down)\n- Middle experiences (the majority) get no signal\n\n**2. Context Blindness:** A thumbs down could mean:\n- Incorrect answer\n- Correct but slow answer\n- Correct but rude tone\n- Correct answer to wrong question (user error)\nWe can't diagnose or improve without context.\n\n**3. Survivor Bias:** Users who had bad experiences may leave and never give feedback.\n\n**4. Dangerous Conclusion:** Only 1% explicit dissatisfaction, but could be 20%+ actual dissatisfaction.\n\n**Better Feedback System Design:**\n\n**Multi-Modal Feedback:**\n```python\nclass FeedbackSystem:\n    def __init__(self):\n        self.strategies = [\n            ActiveSampling(),  # Explicitly ask subset of users\n            PassiveFeedback(),  # Always-available thumbs/stars\n            ContextualPrompts(),  # Smart prompts at key moments\n            BehavioralSignals(),  # Implicit feedback from actions\n        ]\n```\n\n**1. Active Sampling (20% of sessions):**\n- After conversation: \Rate this response: ⭐⭐⭐⭐⭐\\n- Stratified sampling: Ask 40% of first-time users, 10% of regular users\n- Incentive: \Your feedback improves the system\\n- Expected response rate: 30-40%\n- **Benefit:** Unbiased sample of all experiences, not just extremes\n\n**2. Passive Feedback (100% of sessions):**\n- Always-visible 👍👎 buttons\n- If clicked: \What went wrong? [Multiple choice]\\n  - Incorrect information\n  - Unhelpful response\n  - Took too long\n  - Rude/inappropriate\n  - Other: [text field]\n- **Benefit:** Captures extreme experiences with diagnostic context\n\n**3. Contextual Prompts (triggered intelligently):**\n- If session >10 messages: \Are you finding what you need?\\n- If user repeats query: \Was the previous answer helpful?\\n- After long response: \Was this response too long?\\n- **Benefit:** Catches issues in real-time, specific context\n\n**4. Behavioral Signals (automatic):**\n- Engagement: Did user read full response? (scroll depth)\n- Follow-up: Did they immediately ask clarifying questions?\n- Abandonment: Did they leave mid-conversation?\n- Return rate: Do they come back?\n- **Benefit:** Implicit feedback from 100% of users, no survey fatigue\n\n**Analysis Strategy:**\n```python\ndef analyze_feedback (data):\n    # Don't just count thumbs; model user satisfaction\n    \n    # Active sample gives unbiased baseline\n    baseline_satisfaction = data['active_sample',].mean()\n    # Expected: ~3.8/5.0 stars (not 96% positive!)\n    \n    # Passive feedback shows extremes\n    critical_issues = data['passive',].filter (thumb_down=True)\n    # Categorize and prioritize: safety > correctness > UX\n    \n    # Behavioral signals predict silent churn\n    churn_risk = predict_churn (data['behavioral',])\n    # Find: high initial usage, declining engagement\n    \n    # Combined health score\n    health = (\n        0.5 * baseline_satisfaction / 5.0 +  # Absolute quality\n        0.3 * (1 - churn_risk) +  # Retention\n        0.2 * critical_issue_resolution_rate  # Responsiveness\n    )\n    \n    return {\n        'health_score': health,\n        'baseline_sat': baseline_satisfaction,\n        'response_rate': data['active_sample',].response_rate(),\n        'critical_issues': critical_issues,\n        'churn_risk': churn_risk,\n    }\n```\n\n**Key Improvements:**\n1. **Unbiased Measurement:** Active sampling eliminates response bias\n2. **Diagnostic Depth:** Multi-choice follow-ups explain WHY\n3. **Silent Signals:** Behavioral data from everyone, not just responders\n4. **Smart Prompting:** Context-aware, doesn't annoy users\n5. **Actionable Insights:** Categorized issues → prioritized improvements",
    keyPoints: [
      'Response bias: Only extreme experiences generate feedback, silent majority is unknown',
      'Context blindness: Need to know WHY users are satisfied/dissatisfied',
      'Active sampling: Explicitly ask subset of users to get unbiased baseline',
      'Multi-modal: Combine active surveys, passive buttons, contextual prompts, behavioral signals',
      'Behavioral signals: Implicit feedback from all users (engagement, abandonment, return rate)',
      'Never assume silence = satisfaction; could be churn, apathy, or survey fatigue',
    ],
  },
  {
    id: 'human-eval-q-3',
    question:
      'Your human evaluators have inter-annotator agreement (Cohen\'s Kappa) of only 0.45 on a 5-point quality scale, indicating moderate disagreement. The task is "Rate the helpfulness of this AI response (1-5)." How would you diagnose the source of disagreement and improve agreement to Kappa > 0.70?',
    hint: 'Low agreement can stem from ambiguous guidelines, subjective criteria, insufficient training, or genuinely ambiguous examples. Consider rubric clarity, annotator calibration, and task decomposition.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Diagnose disagreement: confusion matrix, problematic examples, per-annotator bias',
      'Decompose subjective criteria into objective sub-dimensions (binary/3-point easier than 5-point)',
      'Extensive training: Gold standard set, calibration tests, review disagreements',
      'Reduce cardinality: 3-point scale easier to agree on than 5-point',
      'Continuous monitoring: Hidden gold standards detect annotator drift',
      'Low agreement often means bad rubric, not bad annotators—fix task definition first',
    ],
  },
];
