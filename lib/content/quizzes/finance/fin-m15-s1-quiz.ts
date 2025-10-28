export default {
  id: 'fin-m15-s1-quiz',
  title: 'Risk Management Fundamentals - Quiz',
  questions: [
    {
      id: 1,
      question:
        'Which of the following best describes the difference between expected loss and unexpected loss in risk management?',
      options: [
        'Expected loss is the mean of the loss distribution; unexpected loss is the standard deviation',
        'Expected loss is covered by pricing/reserves; unexpected loss requires capital',
        'Expected loss occurs frequently with small amounts; unexpected loss is rare but large',
        'All of the above are correct',
      ],
      correctAnswer: 3,
      explanation:
        'All three statements are correct and complementary. Expected loss is the mean of the loss distribution (statistical definition), should be covered by pricing and reserves (business practice), and typically consists of frequent small losses (empirical observation). Unexpected loss is the standard deviation (volatility around the mean), requires capital to absorb (regulatory requirement), and represents rare but potentially large losses (tail events). This distinction is fundamental to risk management: price for expected losses, hold capital for unexpected losses.',
    },
    {
      id: 2,
      question: 'A portfolio has 99% daily VaR of $10M. What does this mean?',
      options: [
        'The portfolio will lose exactly $10M on 1% of days',
        'There is a 1% chance of losing $10M or more in one day',
        'The average daily loss is $10M',
        'The maximum possible loss is $10M',
      ],
      correctAnswer: 1,
      explanation:
        'VaR at 99% confidence means there is a 1% probability (1 day in 100) of losing $10M or more. It is NOT the exact loss amount (losses could exceed $10M), NOT the average loss (expected loss is typically much smaller), and NOT the maximum possible loss (tail losses can be much larger). VaR is a threshold: losses exceed this amount 1% of the time. This is why VaR is criticized for not capturing tail risk—it tells you the threshold but not how bad losses could be beyond that threshold (which is why CVaR/Expected Shortfall is often preferred).',
    },
    {
      id: 3,
      question:
        'In the context of the three lines of defense, which statement is correct?',
      options: [
        'First line (business) owns risk; second line (risk management) monitors; third line (internal audit) provides independent assurance',
        'First line identifies risk; second line manages risk; third line reports risk',
        'All three lines report directly to the CEO to ensure independence',
        'The three lines should be combined for efficiency',
      ],
      correctAnswer: 0,
      explanation:
        'The three lines of defense model clearly delineates responsibility: First line (business units) OWNS and manages risk day-to-day—they make risk decisions and are accountable. Second line (risk management, compliance) provides oversight, sets policies, monitors, and challenges the first line. Third line (internal audit) provides independent assurance to the board that the first and second lines are functioning properly. Option B is incorrect because the first line manages (not just identifies) risk. Option C is wrong because the third line reports to the board/audit committee (not CEO) to maintain independence. Option D contradicts the entire purpose of segregation of duties. The model prevents conflicts of interest and ensures checks and balances.',
    },
    {
      id: 4,
      question:
        'Which risk metric is most appropriate for a non-normal return distribution with fat tails?',
      options: [
        'Standard deviation',
        'Beta',
        'VaR (parametric)',
        'CVaR (Expected Shortfall)',
      ],
      correctAnswer: 3,
      explanation:
        'CVaR (Conditional VaR, also called Expected Shortfall) is superior for fat-tailed distributions because it measures the average loss BEYOND the VaR threshold, capturing tail risk. Standard deviation and parametric VaR both assume normality and underestimate tail risk (option A and C are wrong). Beta measures systematic risk relative to a benchmark but does not capture tail behavior (option B is wrong). Fat tails mean extreme events are more likely than normal distribution predicts. CVaR asks: "Given we\'re in the worst 1%, what\'s the average loss?" This is more informative than VaR which only gives a threshold. For example, if 99% VaR = $10M, CVaR might be $15M (average loss in worst 1% of cases), revealing the tail is severe.',
    },
    {
      id: 5,
      question:
        'A firm\'s risk appetite statement specifies "maximum 99% VaR of $200M." Currently, firm VaR is $180M. What should the risk manager do?',
      options: [
        'Nothing—the firm is within risk appetite',
        'Alert management that firm is at 90% of risk capacity',
        'Recommend immediate position reduction',
        'Increase the VaR limit to $250M to allow growth',
      ],
      correctAnswer: 1,
      explanation:
        "Being at 90% of risk capacity ($180M of $200M limit) warrants alerting management even though technically within limits. Good risk management is proactive: at 90% utilization, there's little buffer for market moves or new opportunities. Option A (do nothing) is passive and could lead to unintended breaches from normal market volatility. Option C (immediate reduction) is overreaction—firm is within appetite. Option D (increase limit) misunderstands governance—limits are set by board based on capital, not adjusted opportunistically. Best practice: Alert management at 80-90% utilization, discuss whether to slow growth, take risk off, or increase limits (with proper board approval and capital backing). Risk appetite is not a target to maximize—it's a guardrail.",
    },
  ],
} as const;
