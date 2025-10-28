export default {
  id: 'fin-m15-s7-quiz',
  title: 'Operational Risk - Quiz',
  questions: [
    {
      id: 1,
      question:
        'Knight Capital lost $440M in 45 minutes due to a software glitch. This is an example of which type of operational risk?',
      options: [
        'People risk (rogue trader)',
        'Process risk (failed reconciliation)',
        'Systems risk (IT failure)',
        'External event risk (cyber attack)',
      ],
      correctAnswer: 2,
      explanation:
        'Knight Capital is a classic Systems risk example—IT failure causing massive losses. Old trading code was accidentally deployed, sending millions of erroneous orders. Option A (people risk) would be intentional misconduct like Jerome Kerviel. Option B (process risk) would be control failures like missing reconciliation. Option D (external event) would be attack from outside. Knight was internal systems failure: deployment error + lack of kill switch + no anomaly detection. The incident prompted regulatory focus on: (1) Change management controls, (2) Automated kill switches, (3) Pre-production testing requirements, (4) Real-time order monitoring. It also showed operational risk can be catastrophic—$440M loss in 45 minutes nearly bankrupted a major firm. Lesson: Systems risk needs same rigor as market risk.',
    },
    {
      id: 2,
      question:
        'Why did Basel regulators abandon AMA (Advanced Measurement Approach) for operational risk capital in favor of simpler SMA (Standardized Measurement Approach)?',
      options: [
        'AMA was too expensive to implement',
        'AMA models were too complex, inconsistent across firms, and easily gamed without improving accuracy',
        'All banks preferred the standardized approach',
        'AMA violated accounting principles',
      ],
      correctAnswer: 1,
      explanation:
        "Basel abandoned AMA because despite enormous complexity and cost ($50M+ implementation), it delivered: (1) Wildly inconsistent results across firms (same bank, different models → 2.5x capital difference), (2) Gaming opportunities (firms optimized for low capital), (3) No evidence of better accuracy than simple formulas, (4) Lack of comparability (regulators couldn't compare banks). Option A is partially true but not the main reason. Option C is backwards—banks wanted AMA hoping for lower capital. Option D is irrelevant. The AMA retreat is rare—regulators rarely abandon complex approaches for simple ones. It shows operational risk is fundamentally difficult to model quantitatively (fat tails, changing risk landscape, sparse data). SMA uses income as proxy (more business = more op risk) plus internal loss multiplier. Simple, robust, harder to game. Lesson: Sometimes simple rules beat complex models.",
    },
    {
      id: 3,
      question:
        'A bank uses Loss Distribution Approach (LDA) for operational risk, modeling frequency with Poisson (λ=5 events/year) and severity with Lognormal (μ=13, σ=2). What does this approach assume?',
      options: [
        'All operational losses are equal in size',
        'Operational losses occur independently (frequency and severity are independent)',
        'The bank will have exactly 5 operational events per year',
        'Operational risk has decreased over time',
      ],
      correctAnswer: 1,
      explanation:
        "LDA assumes independence: (1) Frequency and severity are independent—number of events doesn't affect size of each event, (2) Events are independent of each other. These assumptions are often violated in practice—major operational failures tend to be correlated (one systems failure can cascade). Option A is wrong—lognormal severity explicitly models varying loss sizes. Option C misunderstands Poisson—λ=5 is average, actual events vary (could be 0, could be 12). Option D has no basis. The independence assumption is problematic: In crisis, multiple operational risks can materialize together (IT failures + staffing issues + process breakdowns). This is why firms supplement LDA with scenario analysis. Despite flaws, LDA is useful for: (1) Separating frequency and severity, (2) Monte Carlo simulation for tail risk, (3) Better than ignoring operational risk.",
    },
    {
      id: 4,
      question:
        'Which of the following is the MOST effective operational risk management technique?',
      options: [
        'Insurance to transfer all operational risk',
        'Prevention through strong controls, segregation of duties, and culture',
        'Detection through monitoring and audits',
        'Capital to absorb losses when they occur',
      ],
      correctAnswer: 1,
      explanation:
        "Prevention is most effective—stop losses before they happen. Strong controls (maker-checker, segregation of duties, automated limits), combined with risk-aware culture (employees speak up about issues), prevent >90% of operational losses. Option A (insurance) is limited—only covers certain types, has deductibles, claims are slow, and doesn't cover reputation damage. Option C (detection) is secondary—catches problems but after they've started. Option D (capital) is last resort—you've already lost money, just absorbing it. The hierarchy: Prevent > Detect > Mitigate > Absorb. Example: Jerome Kerviel (Societe Generale $7B loss) was prevented by: better controls (segregated duties), detected earlier by: monitoring, mitigated by: position limits, absorbed with: capital. Prevention would have saved $7B vs. using capital. Yet prevention is \"boring\" and underfunded compared to capital models.",
    },
    {
      id: 5,
      question:
        'A firm experiences 10 operational loss events per year averaging $500K each (expected annual loss = $5M). For capital purposes, 99.9% Operational VaR is $50M. What is the Unexpected Loss requiring capital?',
      options: ['$5M', '$45M', '$50M', '$55M'],
      correctAnswer: 1,
      explanation:
        "Unexpected Loss = VaR - Expected Loss = $50M - $5M = $45M. Capital should cover unexpected losses, not expected. Expected Loss ($5M) should be covered by pricing and provisions (business expense). Unexpected Loss ($45M) represents volatility—years when losses far exceed average—and requires capital buffer. Option A ($5M) is expected loss. Option C ($50M) is total VaR. Option D ($55M) would be VaR + EL (double-counting). This distinction is crucial: Banks don't need capital for expected losses (that's why insurance has premiums), but need capital for unexpected losses (the 1-in-1000 year event). For operational risk, Unexpected Loss is typically 3-10x Expected Loss due to extreme tail (rare but huge events like $7B rogue trader). This is why operational risk capital can be $10B+ for large banks despite expected losses of $1-2B.",
    },
  ],
} as const;
