export const nlpDiscussionQuestions = [
  {
    id: 1,
    question:
      'Design system to analyze 100 earnings call transcripts and correlate sentiment with stock price movement. What NLP techniques and validation approach?',
    answer: `**System Design**:

1. **Data Collection**: Scrape transcripts from SeekingAlpha, Motley Fool
2. **NLP Pipeline**:
   - FinBERT sentiment analysis per paragraph
   - Aggregate to overall call sentiment score (-1 to +1)
   - Extract financial metrics (revenue, guidance, margins)
   - Topic modeling (LDA) for themes

3. **Correlation Analysis**:
   - Compare sentiment to stock price change (T+1, T+5, T+30 days)
   - Control variables: market return, sector performance
   - Regression: Return = α + β₁(Sentiment) + β₂(Surprise) + ε

4. **Validation**:
   - Out-of-sample testing on 20% holdout
   - Check if positive sentiment → positive returns
   - Measure statistical significance (p-value <0.05)

**Expected Finding**: Positive sentiment calls correlate with +2-5% outperformance over 30 days, particularly when sentiment diverges from consensus expectations.`,
  },

  {
    id: 2,
    question:
      "How would you detect when management's tone in MD&A becomes more evasive or uncertain? What linguistic features indicate this?",
    answer: `**Evasive Language Detection**:

**Linguistic Red Flags**:
1. **Increased hedge words**: "possibly", "perhaps", "approximately", "certain circumstances"
2. **Passive voice increase**: "Mistakes were made" vs "We made mistakes"
3. **Vague attributions**: "Market conditions" instead of specific reasons
4. **Increased document length** (padding with jargon)
5. **Higher Fog Index** (deliberately complex writing)

**NLP Implementation**:
\`\`\`python
def detect_evasiveness(current_mda, prior_mda):
    hedge_words = ['approximately', 'substantially', 'potentially', 'certain']
    
    current_hedges = count_words(current_mda, hedge_words)
    prior_hedges = count_words(prior_mda, hedge_words)
    
    fog_current = calculate_fog_index(current_mda)
    fog_prior = calculate_fog_index(prior_mda)
    
    if current_hedges > prior_hedges * 1.3 and fog_current > fog_prior + 2:
        return "HIGH EVASIVENESS - Management obfuscating"
\`\`\`

**Action**: Flag for manual review, check if corresponds with deteriorating financials.`,
  },

  {
    id: 3,
    question:
      'You build sentiment model predicting +2% average return after positive earnings calls. But 30% of positive-sentiment calls result in negative returns. How do you improve?',
    answer: `**Model Improvement Strategies**:

1. **Feature Engineering**:
   - Add: Actual results vs guidance (earnings surprise %)
   - CEO vs CFO sentiment separately (CFO more reliable)
   - Q&A tone vs prepared remarks
   - Analyst question aggressiveness

2. **Context-Aware Sentiment**:
   - "Challenges" in growing company (okay) vs declining company (bad)
   - Positive sentiment with weak guidance (sell signal)

3. **Ensemble Approach**:
   - Combine sentiment + financial metrics + insider trading
   - Weight by management credibility history

4. **False Positive Analysis**:
   - Why did 30% of positive calls fail?
   - Likely: Overly optimistic management, sector headwinds, or "talking a good game" despite weak fundamentals

**Result**: Improved model with 75-80% accuracy by combining NLP sentiment with quantitative signals.`,
  },
];
