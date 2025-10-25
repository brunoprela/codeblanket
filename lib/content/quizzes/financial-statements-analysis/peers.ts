export const peersDiscussionQuestions = [
  {
    id: 1,
    question:
      'Three SaaS peers trade at EV/Revenue multiples of 8x, 10x, and 12x with revenue growth of 20%, 30%, and 40%. Your target company has 35% growth. Estimate fair valuation multiple and explain approach.',
    answer: `**Regression Approach**:

Growth vs Multiple:
- 20% growth → 8x multiple
- 30% growth → 10x multiple  
- 40% growth → 12x multiple

Linear relationship: 10% growth = +2x multiple

For 35% growth: 10x + (5% × 0.2) = **11x EV/Revenue**

**Justification**: Company growing 35% (between 30-40% peers) should trade between 10-12x. Linear interpolation gives 11x as fair value.

**Application**: If target has $500M revenue, fair EV = $5.5B. If trading at $4B, it's **27% undervalued** (\$4B vs $5.5B fair value).`,
  },

  {
    id: 2,
    question:
      'Compare P/E vs PEG ratio. Company A: P/E=25, growth=25%. Company B: P/E=15, growth=10%. Which is cheaper on PEG basis?',
    answer: `**PEG Calculations**:
- **Company A**: PEG = 25 / 25 = **1.0** (fairly valued)
- **Company B**: PEG = 15 / 10 = **1.5** (overvalued)

**Company A is cheaper** despite higher P/E because growth justifies valuation. Company B appears cheap at 15x P/E but expensive at 1.5 PEG.

**Conclusion**: Always adjust P/E for growth - high P/E with high growth can be better value than low P/E with low growth.`,
  },

  {
    id: 3,
    question:
      "All five sector peers trade at 15-17x P/E. Does this mean they're fairly valued?",
    answer: `**No** - relative valuation only shows pricing vs peers, NOT absolute value. If entire sector is overvalued (bubble), all peers can be expensive together.

**Example**: In 2000, all internet stocks traded at 50-100x P/E. Relative to each other they looked "fair," but absolutely they were in a bubble.

**Solution**: Use absolute metrics (DCF, FCF yield, historical averages) alongside relative valuation. If sector median is 17x P/E but historical average is 12x, entire sector may be 40% overvalued.`,
  },
];
