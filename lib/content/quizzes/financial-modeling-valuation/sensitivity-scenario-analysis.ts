export const sensitivityScenarioAnalysisQuiz = [
  {
    id: 'ssa-q-1',
    question: 'Your DCF model shows $10B valuation. Your VP asks: "What if terminal growth is 3.5% instead of 2.5%?" You recalculate and get $12.5B (+25%). The VP says: "Great, so valuation is $10-12.5B range." Critique this conclusion and provide proper sensitivity analysis framework.',
    sampleAnswer: 'Proper sensitivity analysis framework: The VP\'s conclusion is incomplete. Issues: (1) One-way sensitivity only—changed terminal growth but held WACC constant. If rates rise, WACC increases AND terminal growth decreases (correlated variables). (2) No probability weighting—treats $10B and $12.5B as equally likely. (3) Ignores other variables—revenue growth, margins, CapEx also uncertain. Proper approach: (1) Two-way sensitivity (terminal growth × WACC), (2) Three scenarios (bear/base/bull) with coherent assumptions, (3) Probability-weighted expected value, (4) Tornado chart to identify top 3-4 drivers. Conclusion: Valuation range is $8B-$14B with $10.5B expected value (60% weight on base case).',
    keyPoints: [
      'One-way sensitivity insufficient; need two-way (correlated variables like WACC and terminal growth)',
      'Valuation ranges require probability weighting; bear/base/bull scenarios with coherent assumptions',
      'Tornado chart identifies key drivers; focus sensitivity on top 3-4 impactful variables',
    ],
  },
  {
    id: 'ssa-q-2',
    question: 'Build three scenarios (bear/base/bull) for a SaaS company DCF. Base case: 20% revenue growth, 25% EBITDA margin, 3% terminal growth, 10% WACC = $5B valuation. What assumptions would you use for bear and bull cases? Why?',
    sampleAnswer: 'Coherent scenario framework: Bear case: 10% revenue growth (competition intensifies), 20% EBITDA margin (price pressure reduces margins), 2% terminal growth (market matures), 12% WACC (higher risk perception). Valuation: ~$3B (-40%). Bull case: 30% revenue growth (market leadership), 30% EBITDA margin (scale efficiencies), 3.5% terminal growth (secular tailwinds), 9% WACC (lower risk). Valuation: ~$7.5B (+50%). Key: Variables move together logically. Bear has low growth AND low margins AND high risk. Bull has high growth AND high margins AND low risk. Don\'t mix (e.g., bull growth with bear margins = incoherent).',
    keyPoints: [
      'Bear scenario: All negative factors together (low growth, low margins, high WACC, low terminal growth)',
      'Bull scenario: All positive factors together (high growth, high margins, low WACC, higher terminal growth)',
      'Scenarios must be internally consistent; don\'t mix bull growth with bear margins (incoherent)',
    ],
  },
  {
    id: 'ssa-q-3',
    question: 'Your tornado chart shows: (1) Revenue growth: $3B range, (2) EBITDA margin: $2.5B range, (3) WACC: $2B range, (4) Terminal growth: $1.8B range, (5) CapEx: $0.3B range. How do you use this to focus analysis and communicate risk?',
    sampleAnswer: 'Tornado chart application: Focus: Revenue growth and EBITDA margin are top drivers ($3B and $2.5B ranges). Spend 80% of diligence effort validating these assumptions (customer contracts, competitive positioning, cost structure). De-prioritize: CapEx has minimal impact ($0.3B range). Don\'t waste time debating 4% vs 5% CapEx assumption—it doesn\'t move the needle. Communication: "This valuation is most sensitive to revenue growth and margins. We have high confidence in these (signed contracts, historical margins). WACC and terminal growth are market-dependent (less controllable). CapEx assumption has negligible impact." Risk management: Hedge revenue risk (customer concentration, competitive threats), Improve margin predictability (long-term vendor contracts, cost structure optimization).',
    keyPoints: [
      'Focus diligence on top 2-3 tornado drivers (revenue growth, EBITDA margin); ignore low-impact variables',
      'Communication: "Valuation most sensitive to X and Y; we have high confidence because..." (evidence-based)',
      'Risk management: Hedge/de-risk top drivers (customer diversification, margin protection); ignore minor variables',
    ],
  },
];
