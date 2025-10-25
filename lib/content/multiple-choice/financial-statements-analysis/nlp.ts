export const nlpMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'Why use FinBERT instead of generic BERT for financial sentiment analysis?',
    options: [
      'FinBERT is faster',
      "FinBERT is pre-trained on financial text and understands domain-specific language; 'beat expectations' is positive in finance but generic BERT might misclassify",
      "Generic BERT doesn't work on financial data",
      'FinBERT is free while BERT costs money',
      'No difference in performance',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. FinBERT is fine-tuned on financial news, earnings calls, and analyst reports, learning domain-specific semantics. Examples: (1) "Beat earnings" = positive (FinBERT knows, generic BERT might not), (2) "Charges" in finance = bad (write-offs) vs generic text might be neutral, (3) "Exposure" can be good (market exposure) or bad (risk exposure) depending on context. Research shows FinBERT achieves 85%+ accuracy on financial sentiment vs 70% for generic BERT. Always use domain-specific models for specialized tasks.`,
  },

  {
    id: 2,
    question:
      'Earnings call transcript shows positive sentiment but stock drops 5%. What might explain this?',
    options: [
      'NLP model failed completely',
      'Market is irrational',
      "Guidance was weak despite positive tone; 'beat quarter but lowered outlook' is negative signal; or positive language can't overcome weak fundamentals",
      "Sentiment analysis doesn't work",
      'Stock always moves opposite to sentiment',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Common scenarios: (1) **"Beat-and-lower"**: Company beat Q3 but lowered Q4 guidance - positive tone but negative forward outlook, (2) **Quality of beat**: Beat by 1¢ vs expectations of 5¢ beat (disappointment despite "beat"), (3) **Management credibility**: If management historically overpromises, positive tone discounted, (4) **Sector rotation**: Good results but investors rotating out of sector. Key insight: Combine sentiment with (1) Guidance changes, (2) Beat magnitude, (3) Forward metrics. Positive sentiment + weak guidance = SELL signal despite "positive" call.`,
  },

  {
    id: 3,
    question:
      'MD&A section has Fog Index of 22 (very complex). What does this suggest?',
    options: [
      'High-quality, detailed disclosure',
      'Management is being transparent',
      'Potential obfuscation; research shows companies increase complexity when hiding problems or facing litigation; complexity ≠ thoroughness',
      'Normal for financial documents',
      'Better than simple language',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Fog Index >18 is "extremely difficult" to read. Research (Li 2008, Loughran & McDonald 2014) shows: (1) Companies facing litigation increase document complexity by 15-20%, (2) Firms with declining performance use more complex language, (3) High complexity correlates with lower future returns and higher restatement risk. Why? Management deliberately obfuscates bad news in dense prose. Normal MD&A: Fog Index 14-16. Fog Index 22 is red flag - either management is hiding something or lacks communication skills (both concerning). Action: Flag for deeper investigation.`,
  },

  {
    id: 4,
    question:
      "How can NLP detect 'forward-looking statements' vs historical facts in 10-K?",
    options: [
      'Cannot be detected automatically',
      "Use keyword matching for temporal markers: 'will', 'expect', 'anticipate', 'project', 'believe', 'plan' indicate forward-looking; past tense indicates historical",
      'All 10-K content is historical',
      'Manual review only',
      'NLP is not suitable for this task',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Forward-looking statement detection using NLP: (1) **Temporal markers**: "will achieve", "expect to grow", "plan to launch" = forward-looking, (2) **Modal verbs**: "should", "could", "may", "might" indicate uncertainty/future, (3) **Hedging**: "believe", "anticipate" = forward-looking, (4) **Tense**: Past tense = historical facts. Implementation: Train classifier on labeled data or use rule-based system. Why it matters: Forward-looking statements have legal safe harbor (can be wrong), while historical facts must be accurate. Investors should weight them differently. Companies often hide weak current performance by emphasizing optimistic forward-looking statements.`,
  },

  {
    id: 5,
    question:
      "Topic modeling (LDA) on 5 years of Apple MD&As reveals 'China' topic weight increased from 5% to 25%. What does this indicate?",
    options: [
      'Nothing significant - normal variation',
      'Increasing China exposure/dependency; suggests both opportunity (growth market) and risk (concentration, geopolitical); warrants deeper analysis',
      'Apple is moving headquarters to China',
      'Topic modeling is inaccurate',
      'MD&A became longer',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. 5x increase in China-topic weight (5% → 25% of MD&A discussion) indicates: (1) **Growing importance**: China now 25% of business/risk discussion, (2) **Increased exposure**: Revenue, supply chain, or market risk concentrated in China, (3) **Rising concern**: Management feels need to discuss more (regulatory risk, geopolitical tensions), (4) **Dependency risk**: If China represents 25% of discussion but 40% of revenue, company may be downplaying concentration. Action: (1) Verify actual China revenue %, (2) Check for supply chain concentration, (3) Assess geopolitical risk (tariffs, sanctions), (4) Compare to peers. This pattern seen before Apple-China tensions in 2019-2020, providing early warning signal.`,
  },
];
