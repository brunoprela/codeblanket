import { MultipleChoiceQuestion } from '@/lib/types';

export const mockInterviewProblemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mip-mc-1',
    question:
      'In the coin flip game where you stop when ahead, what happens if the payoff changes to +$1 for heads, +$0 for tails (instead of +$1, -$1)?',
    options: [
      'Expected value is infinite - you should play forever',
      'Expected value is $1 - stop after first flip',
      'Expected value is $2 - same as original game',
      'Strategy changes - stop after accumulating at least $5',
    ],
    correctAnswer: 0,
    explanation:
      "With +$1 for heads, +$0 for tails, your balance never decreases! After k flips, expected balance = k/2 (k flips × 50% heads × $1). You never want to stop because E[next flip | current balance B] = B + $0.50 > B always. Expected value is infinite (or until you run out of time). This is the key difference from original: original has -$1 for tails creating stopping incentive, this version has no downside. In practice, you'd stop when marginal value of time equals marginal expected gain ($0.50/flip), but mathematically EV is unbounded. This demonstrates importance of payoff structure: even small changes (tails: -$1 → $0) completely alter optimal strategy.",
  },
  {
    id: 'mip-mc-2',
    question:
      'In the options arbitrage problem, put-call parity is C - P = S - K×e^(-rT). What does this relationship fundamentally represent?',
    options: [
      'The expected payoff of a portfolio under risk-neutral probability',
      'Two portfolios with identical payoffs at expiration must have identical costs today',
      'The relationship between implied and realized volatility',
      'The risk premium for holding options vs. stock',
    ],
    correctAnswer: 1,
    explanation:
      "Put-call parity is a no-arbitrage condition: Portfolio A (long call, short put) has payoff = S_T - K at expiration. Portfolio B (long stock, short K bonds) has payoff = S_T - K at expiration. Since payoffs are IDENTICAL in all states, the costs must be equal today: C - P = S - K×e^(-rT). If not equal, arbitrage exists. This doesn't require any assumptions about probabilities (option 1), volatility (option 3), or risk premia (option 4). It\'s purely about replication: two ways to create the same payoff must cost the same. Violations allow risk-free profit. This is why we can detect mispricing: C - P = $2 but S - K = $0 means prices are inconsistent with no-arbitrage.",
  },
  {
    id: 'mip-mc-3',
    question:
      'In the "first ace" problem with expected 10.6 draws, what happens if you remove all four aces before starting, then ask "when would the first ace have appeared?"',
    options: [
      "The question doesn't make sense - there are no aces to find",
      'Expected position is still 10.6 by original probability calculation',
      'Expected position is undefined - could be any of the 52 positions',
      'Use conditional probability: given no ace in drawn cards, update position estimate',
    ],
    correctAnswer: 1,
    explanation:
      "This is a subtle probability question about counterfactuals! Even though we removed the aces, we can still ask: \"where would the first ace have been in a random shuffle?\" The answer is still 10.6 by symmetry - before removing aces, each of the 52 positions is equally likely for any ace, so expected position of first ace is (52+1)/(4+1) = 10.6. Removing the aces doesn't change where they would have been in the original shuffle. Option 4 is wrong because we're not conditioning on observed cards - we're asking about the hypothetical shuffle. This tests understanding of: (a) probability is about information states, not physical presence, (b) expected values are calculated before observing outcomes. Similar to: \"I flipped a coin but didn't look - what's P(heads)?\" Answer: still 50%, even though outcome is determined.",
  },
  {
    id: 'mip-mc-4',
    question:
      'In dynamic delta hedging of short options, why must you rebalance more frequently as expiration approaches?',
    options: [
      'Theta (time decay) increases near expiration, requiring more frequent trades',
      'Gamma increases near expiration, causing delta to change more rapidly with stock moves',
      'Vega decreases near expiration, reducing importance of volatility hedging',
      'Regulatory requirements mandate daily rebalancing in final week',
    ],
    correctAnswer: 1,
    explanation:
      "Gamma is the key! Near expiration, especially for at-the-money options, gamma explodes. Gamma measures how fast delta changes: Δ_new = Δ_old + Γ×ΔS. High gamma means delta changes dramatically with small stock moves, requiring frequent rebalancing to maintain delta neutrality. Far from expiration: gamma is low, delta changes slowly, weekly rebalancing might suffice. Near expiration (last week): gamma is high, delta changes rapidly, may need hourly/continuous rebalancing. Example: ATM option 1 day to expiry has gamma ≈ 0.04, so $1 stock move changes delta by 0.04 (4% of position). With 1000 contracts, $1 move requires buying/selling 40 shares to maintain hedge. Theta (option 1) causes P&L but doesn't affect hedge frequency. Vega (option 3) decreases but that's separate from delta hedging. Regulation (option 4) is false.",
  },
  {
    id: 'mip-mc-5',
    question:
      'In an interview, you solve a problem correctly but the interviewer says "What if we change this assumption?" What is the BEST response?',
    options: [
      "Start solving immediately to show you're quick",
      'Ask clarifying questions about the new assumption, then outline approach before calculating',
      'Explain why the changed assumption might not be realistic',
      "Reference similar problems you've seen before",
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 demonstrates professional problem-solving! When assumptions change: (1) Clarify exactly what\'s different - "So now the coin is 60-40 instead of 50-50?" (2) Consider impact - "This means the game has positive expectation..." (3) Outline approach - "I\'ll use the same recursive equation but with different probabilities..." (4) Then calculate. Jumping straight to calculation (option 1) shows poor process. Questioning realism (option 3) can seem defensive, though mentioning it briefly is OK. Referencing other problems (option 4) is fine but secondary. Interviewers test adaptability with "what if" questions. They want to see: structured thinking under new constraints, ability to modify existing framework, recognition of what changes and what stays same. In real trading: assumptions change constantly (volatility spikes, liquidity dries up, correlations break), so adaptability is crucial.',
  },
];
