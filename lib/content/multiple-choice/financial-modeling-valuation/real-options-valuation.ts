import { MultipleChoiceQuestion } from '@/lib/types';

export const realOptionsValuationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'rov-mc-1',
    question:
      'In real options valuation, why does higher volatility INCREASE project value (unlike traditional NPV)?',
    options: [
      'Higher volatility reduces risk',
      'Volatility creates more upside scenarios to exploit',
      'Options allow capturing upside while limiting downside',
      'Higher volatility is incorrectly measured',
    ],
    correctAnswer: 2,
    explanation:
      'Option 3 is correct: Options capture upside while limiting downside. Traditional NPV: Higher volatility = higher risk = lower value (discount rate increases). Real options: Higher volatility = more extreme scenarios. With flexibility: Exercise in good scenarios (capture upside), Abandon in bad scenarios (avoid downside). Asymmetry: unlimited upside, capped downside → volatility increases expected payoff. Example: Oil field can abandon if prices crash (downside $0) but keeps operating if prices spike (upside unbounded) → volatility helps!',
  },
  {
    id: 'rov-mc-2',
    question:
      'A project has traditional NPV of -$10M. Which real option is most likely to make the project valuable?',
    options: [
      'Option to expand (growth option)',
      'Option to delay investment',
      'Option to abandon',
      'Option to switch inputs',
    ],
    correctAnswer: 1,
    explanation:
      "Option 2 is correct: Option to delay. If traditional NPV = -$10M today, don't invest now. But if you can WAIT (delay option), uncertainty may resolve favorably: If market improves, invest later (positive NPV). If market worsens, never invest (lose only sunk costs, not full $10M). Option to delay is call option—right but not obligation to invest. Waiting is valuable when: High uncertainty, NPV near zero ($10M is close to breakeven), Information will arrive soon (1-2 years). Other options less relevant: Expansion requires initial investment (but project is NPV-negative). Abandonment requires operating first (but shouldn't start). Switching is valuable mid-operation (but haven't started).",
  },
  {
    id: 'rov-mc-3',
    question:
      'In Black-Scholes real options model, what does the strike price (K) represent?',
    options: [
      'PV of project cash flows',
      'Investment cost required',
      'Current market value',
      'Expected NPV',
    ],
    correctAnswer: 1,
    explanation:
      "Option 2 is correct: Investment cost. Black-Scholes mapping: S = PV of project cash flows (asset value), K = Investment cost (what you pay to exercise option), T = Time until decision must be made, σ = Volatility of project returns, r = Risk-free rate. Call option analogy: Financial call: Right to buy stock at strike K. Real option: Right to invest K to get project worth S. If S > K at expiration (PV of cash flows exceeds investment), exercise (invest). If S < K, don't exercise (abandon). Option value = max(S - K, 0) at expiration, discounted for time and risk.",
  },
  {
    id: 'rov-mc-4',
    question:
      'A pharmaceutical company values drug development as staged investment (Phase I, II, III). This is an example of which real option type?',
    options: [
      'Growth option',
      'Abandonment option',
      'Switching option',
      'Compound option',
    ],
    correctAnswer: 3,
    explanation:
      'Option 4 is correct: Compound option (option on option). Drug development stages: Phase I ($10M): Option to proceed to Phase II. Phase II ($50M): Option to proceed to Phase III. Phase III ($200M): Option to launch product. Each stage is option to buy next stage option → compound option. Valuation: Work backwards from Phase III (launch option value), Discount to Phase II (Phase II is option to get Phase III option), Discount to Phase I (Phase I is option to get Phase II option). Abandonment embedded: Can stop after any phase if results poor (asymmetry). Compound options common in: Staged R&D, Real estate development (permits → site prep → construction), Venture capital (Series A → B → C rounds).',
  },
  {
    id: 'rov-mc-5',
    question:
      'Traditional DCF values a tech platform at $500M (assuming launch 1 product). Real options DCF values at $800M (includes launching 3 more products if first succeeds). What drives the $300M difference?',
    options: [
      'Abandonment option',
      'Growth option',
      'Timing option',
      'Switching option',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: Growth option. Platform enables future products (option to expand). Traditional DCF: Values only 1 product (ignores future opportunities). Real options DCF: Values 1 product + option to launch 3 more if successful. $300M = value of growth options (future expansion opportunities). Characteristics: Platform investments have high growth option value, Initial product may be NPV-negative (loss leader), But opens doors to lucrative follow-ons (option value). Examples: AWS (started for Amazon, now serves enterprises), iPhone (platform for App Store), Drug platforms (one molecule leads to derivatives). Key: Always value platforms with real options (traditional DCF severely understates value).',
  },
];
