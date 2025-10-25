import { MultipleChoiceQuestion } from '@/lib/types';

export const tradingGamesSimulationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tgs-mc-1',
    question:
      'In market making, why does inventory risk become MORE severe as you approach expiration/end of game?',
    options: [
      'You have less time to gradually unwind the position through natural trading',
      'The true value becomes more volatile near the end',
      'Your bid-ask spread must widen due to regulatory requirements',
      'Other market participants become more aggressive',
    ],
    correctAnswer: 0,
    explanation:
      "Time horizon is the critical factor. With many rounds remaining, you can gradually reduce inventory through slightly biased quotes without taking significant losses. Example: Long 3 shares in Round 3 of 10? No problem - you have 7 rounds to slowly work out of the position by quoting ask slightly below fair value. Long 3 shares in Round 9 of 10? BIG problem - only 1 round left, must close aggressively (potentially at significant loss) or carry overnight risk. The position itself hasn't changed, but available time has. This mirrors real trading: intraday positions can be managed gradually; end-of-day positions require aggressive closing. Volatility (option 2) doesn't necessarily increase near end. Regulations (option 3) don't mandate wider spreads. Other participants (option 4) might be aggressive but that's not the primary driver of YOUR inventory risk.",
  },
  {
    id: 'tgs-mc-2',
    question:
      'In the 100-card game, you bid 33. Card drawn is 50. You lose $33. Your friend says "you should have bid higher!" What\'s the correct response?',
    options: [
      "You're right, I should bid higher next time to win more often",
      "One outcome doesn't determine optimal strategy - 33 maximizes expected value across all possible cards",
      'I should have bid 50 to make it exactly 50-50',
      "The game is rigged, I shouldn't play at all",
    ],
    correctAnswer: 1,
    explanation:
      "This tests understanding of expected value vs. single outcomes. Yes, bidding 34 or 50 would have won THIS particular round. But optimal strategy is based on expected value across ALL 100 possible cards, not one realized outcome. Bidding 33 loses the least on average (or wins the most, depending on game variant). This is like poker: folding AA preflop is wrong even if the flop would have been all Kings (and you would have lost). Decisions are evaluated on available information, not results. Option 1 (results-oriented thinking) is wrong - you don't change optimal strategy based on one outcome. Option 3 (bid 50) isn't optimal mathematically. Option 4 might be true (game could be unfavorable overall) but doesn't address the strategy question. The correct response acknowledges the variance of outcomes while maintaining strategy discipline. In trading: you'll have losing trades even with positive-EV strategies.",
  },
  {
    id: 'tgs-mc-3',
    question:
      'Why is half-Kelly often preferred over full Kelly in practice, despite having lower expected growth?',
    options: [
      'Because half-Kelly has higher median outcome than full Kelly',
      'Because full Kelly requires perfect knowledge of probabilities; half-Kelly is more robust to estimation errors',
      'Because half-Kelly guarantees you never lose money',
      'Because regulations prohibit betting more than 5% of capital',
    ],
    correctAnswer: 1,
    explanation:
      "Robustness to estimation errors is the key advantage of fractional Kelly. Example: You think p=0.55 (5% edge), so full Kelly says bet 10%. But what if true p=0.52 (only 2% edge)? Then full Kelly at 10% is OVERBETTING (true Kelly would be 4%). This leads to poor geometric growth and increased risk. Half-Kelly at 5% is much safer: even if edge is smaller than estimated, you're not drastically overbetting. Option 1 is false: full Kelly actually has higher median (it's the maximum of geometric growth rate). Option 3 is false: half-Kelly still has risk, just less than full Kelly. Option 4 is false: no such regulation exists. The key insight: in real trading, you never know true probabilities exactly. Estimation error is inevitable. Fractional Kelly (25-50%) provides insurance against estimation errors while still capturing most of the growth. Professional traders use this principle universally.",
  },
  {
    id: 'tgs-mc-4',
    question:
      'In the Envelope Game (one has $X, other has $2X), why doesn\'t the "always switch" argument work?',
    options: [
      'The conditional probabilities P($Y/2 | you have $Y) and P($2Y | you have $Y) are not both 0.5',
      'Switching costs money, which the argument ignores',
      'You should actually switch only 50% of the time, not always',
      'The argument is correct - you should always switch',
    ],
    correctAnswer: 0,
    explanation:
      "The error is in conditional probability. The \"always switch\" argument assumes: \"I have $Y, so the other envelope has $Y/2 or $2Y with equal probability.\" But this is wrong! Let\'s say X=$100 (smaller envelope). Two scenarios: (A) You picked $100: other has $200. P(A) = 50%. (B) You picked $200: other has $100. P(B) = 50%. If you observe Y=$100: You're definitely in scenario A, so other envelope has $200 (not $50 - that doesn't exist!). P($200|Y=$100) = 100%, not 50%. If you observe Y=$200: You're definitely in scenario B, so other envelope has $100 (not $400). P($100|Y=$200) = 100%, not 50%. The correct conditional probabilities are 100-0 or 0-100 depending on which envelope you picked, not 50-50 as the naive argument assumes. Expected gain from switching: 0.5(+$100) + 0.5(-$100) = $0. Option 2 is wrong (no switching cost stated). Option 3 misses the point (it's not about mixing strategies). Option 4 is the naive wrong answer.",
  },
  {
    id: 'tgs-mc-5',
    question:
      'You\'re playing a market making game and after 5 rounds, you\'re down $2.00. Interviewer asks "How do you feel about your strategy?" Best response?',
    options: [
      "Admit your strategy was wrong and describe how you'll change it",
      'Defend your strategy by explaining the expected value is positive despite current loss',
      'Ask to restart the game with a different strategy',
      'Blame bad luck and say you would have won with better card draws',
    ],
    correctAnswer: 1,
    explanation:
      "Professional traders evaluate strategies on process, not short-term results. Being down $2 after 5 rounds is normal variance! You could be playing optimally and still be down due to randomness. The correct response: \"I'm comfortable with my strategy. I've been managing inventory appropriately, adapting quotes based on trading patterns, and making decisions with positive expected value. Short-term variance is expected - I've seen similar downswings in optimal play during simulations. I'll continue executing the strategy and expect mean reversion over remaining rounds.\" This shows: (1) Process over results thinking, (2) Understanding of variance, (3) Confidence without arrogance, (4) Willingness to stick with sound strategy through drawdowns. Option 1 (changing strategy) shows weak conviction and results-oriented thinking. Option 3 (restart) shows poor adaptation. Option 4 (blame luck) is unprofessional. In real trading, you'll have losing days/weeks even with great strategies. Staying disciplined is critical.",
  },
];
