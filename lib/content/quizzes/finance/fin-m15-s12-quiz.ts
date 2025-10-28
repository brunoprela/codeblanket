export default {
  id: 'fin-m15-s12-quiz',
  title: 'Position Limits and Risk Limits - Quiz',
  questions: [
    {
      id: 1,
      question:
        "A trader's VaR limit is $10M. Current VaR is $9.8M (98% utilized). Should the risk manager be concerned?",
      options: [
        'No—trader is within limit',
        'Yes—at 98% utilization, any small market move could cause breach',
        'No—trader is skilled at maximizing risk capacity',
        'Yes—but only if limit has been breached before',
      ],
      correctAnswer: 1,
      explanation:
        'At 98% utilization, the trader has only $200K buffer. Normal market volatility could easily cause a breach without any new trades. Best practice is to alert at 80-90% utilization and restrict new risk-increasing trades at 90-95%. Option A is dangerously complacent—technically compliant but practically vulnerable. Option C misses the point—limits are guardrails, not targets to maximize. Option D is wrong—high utilization is concerning regardless of history. The problem: VaR is calculated EOD but positions change intraday. A trader at 98% in morning could easily breach by noon from market moves. Better risk management: (1) Alert at 80%, (2) Warning at 90%, (3) Soft block at 95%, (4) Hard block at 100%. Limits need buffers for: market volatility, calculation uncertainty, intraday fluctuations. Running at 98% = no buffer = high probability of unintentional breach.',
    },
    {
      id: 2,
      question: 'What is the difference between hard limits and soft limits?',
      options: [
        'Hard limits are higher than soft limits',
        'Hard limits cannot be breached (system enforced); soft limits can be breached with approval',
        'Hard limits apply to senior traders; soft limits to junior traders',
        'There is no meaningful difference',
      ],
      correctAnswer: 1,
      explanation:
        "Hard limits are system-enforced and absolutely cannot be breached—trades are blocked automatically. Soft limits can be breached with appropriate approval (desk head, risk manager, CRO depending on severity). Hard limits for: Regulatory limits (violating = illegal), Credit lines (can't trade beyond funding), Risk capacity (exceeds firm capital). Soft limits for: Internal risk allocations (judgment needed), VaR budgets (context matters), Concentration limits (may have good reason). Option A is wrong—not about magnitude but enforceability. Option C is wrong—both types apply to all traders based on limit type, not seniority. Option D is wrong—the distinction is critical. Example: Regulatory position limit = hard (no override possible). Trader VaR limit = soft (can request override with justification). The override process provides flexibility while maintaining control: request → justify → approve → track → review performance.",
    },
    {
      id: 3,
      question:
        'A kill switch is triggered, blocking all trading. What should happen next?',
      options: [
        'Resume trading immediately—kill switches cause more harm than good',
        'Investigate the trigger, assess if genuine issue, restore controls, then resume with management approval',
        'Keep trading blocked for 24 hours as punishment',
        'Fire the trader who triggered it',
      ],
      correctAnswer: 1,
      explanation:
        'Kill switch triggers require disciplined investigation and restoration process: (1) Confirm what triggered it (loss threshold? abnormal volume? system error?), (2) Investigate if genuine issue (rogue algo? fat finger? pricing error?), (3) Fix root cause if issue found, (4) Test that controls working, (5) Management approval to resume (CRO or desk head), (6) Document incident and lessons. Option A is dangerous—kill switches exist for reason. Option C is punitive without logic—resume when safe, not on arbitrary timeline. Option D is unfair—trigger might have prevented disaster (should reward, not punish). The Knight Capital lesson: no kill switch → $440M loss. Compare to firms with kill switches: losses stopped at $5-10M (still bad, but 98% better). Post-trigger: most resume within 30 minutes if false alarm, 2-4 hours if genuine issue requiring fixes. Monthly testing ensures kill switch works when needed.',
    },
    {
      id: 4,
      question:
        'Firm-wide VaR limit is $200M. Sum of desk VaR limits is $250M. Is this an error?',
      options: [
        'Yes—sum of desk limits must equal firm limit',
        "No—diversification means desk VaRs don't add up to firm VaR",
        'Yes—desk limits should sum to less than firm limit (need buffer)',
        'Both B and C are partially correct',
      ],
      correctAnswer: 3,
      explanation:
        "This is intentionally designed. Desk VaRs don't add up to firm VaR due to diversification—if equity desk has $100M VaR and bond desk has $80M VaR, firm VaR might only be $150M (not $180M) if they're negatively correlated. So allocating $250M limit to desks (sum) while firm limit is $200M makes sense because actual firm VaR will be less due to diversification. However, the $50M gap ($250M - $200M) should not all be allocated—need to reserve 10-15% as buffer for: correlation changes (diversification disappears in crisis), limit breaches by one desk, new opportunities. Option A is naive—ignores diversification. Option B is correct but incomplete. Option C is correct but incomplete. Best practice: Allocate 85-90% of firm limit to desks (accounting for diversification), reserve 10-15% unallocated buffer. This ensures firm limit not breached even if one desk at 100% and correlations increase.",
    },
    {
      id: 5,
      question:
        'A trader repeatedly requests limit overrides (10 times in one month). What does this indicate?',
      options: [
        'The trader is highly skilled and finding good opportunities',
        'The limit is too tight and should be increased',
        'The trader is not respecting risk limits and/or the limit is misaligned with strategy',
        'Override process is working well',
      ],
      correctAnswer: 2,
      explanation:
        "Frequent override requests indicate a problem: Either (1) Trader doesn't respect limits and pushes boundaries constantly, or (2) Limit is genuinely wrong for this strategy (too tight). Both need investigation. Review: Are override trades profitable (if yes, limit may be too tight; if no, trader has bad judgment). Option A is naive—good traders stay within limits most of the time. Option B might be true but needs analysis—don't automatically increase limits. Option D is wrong—overrides should be rare (5% of trades), not 10+ per month. Acceptable override frequency: 1-2 per quarter for normal performance, 5-10 for exceptional opportunity. Monthly = problem. Action: (1) If trades profitable: increase limit, (2) If trades unprofitable: reduce limit + warning, (3) If mix: have conversation about strategy vs. limits alignment. Track override performance—if consistent underperformance, reject future overrides and potentially reduce base limit. Limits should be constraints, not negotiating positions.",
    },
  ],
} as const;
