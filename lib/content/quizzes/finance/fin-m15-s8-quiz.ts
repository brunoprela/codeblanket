export default {
  id: 'fin-m15-s8-quiz',
  title: 'Liquidity Risk - Quiz',
  questions: [
    {
      id: 1,
      question:
        'What is the key difference between funding liquidity risk and market liquidity risk?',
      options: [
        'Funding liquidity is short-term; market liquidity is long-term',
        'Funding liquidity is whether you can pay obligations; market liquidity is whether you can sell assets quickly',
        'Funding liquidity applies to banks; market liquidity applies to hedge funds',
        'There is no difference—they are the same risk',
      ],
      correctAnswer: 1,
      explanation:
        'Funding liquidity risk is the ability to meet cash obligations when due (can you pay the bills?). Market liquidity risk is the ability to sell assets quickly without large price discount (can you convert assets to cash?). These interact in death spirals: Funding stress → must sell assets → market illiquidity causes fire-sale losses → capital erodes → more funding stress → death. Bear Stearns and Lehman died from this spiral in days. Option A is wrong—both are immediate concerns. Option C is wrong—both risks affect all financial institutions. Option D is wrong—they are distinct but correlated. LCR (Liquidity Coverage Ratio) addresses funding risk: hold enough liquid assets for 30 days. NSFR (Net Stable Funding Ratio) addresses structural funding: match long-term assets with long-term funding.',
    },
    {
      id: 2,
      question:
        'A bank has LCR of 110% (requirement is 100%). The risk manager says "We\'re fine." What is the concern with this reasoning?',
      options: [
        'LCR only covers 30 days—what if stress lasts longer?',
        '110% is below the recommended buffer of 120%',
        "LCR doesn't account for intraday liquidity needs",
        'All of the above',
      ],
      correctAnswer: 3,
      explanation:
        "All three concerns are valid. LCR tests 30-day survival, but funding crises can last months (2008 lasted over a year)—what happens on day 31? Even at 110%, there's minimal buffer—market stress or unexpected outflow could breach. LCR is end-of-day measure but intraday liquidity matters (payment systems, margin calls due at noon). Option A: COVID showed crises can last >30 days. Option B: Best practice is 120-150% LCR to maintain buffer. Option C: 2008 showed intraday liquidity crunch (Lehman couldn't make morning payments). At 110%, bank is technically compliant but practically vulnerable. Better practice: 150% LCR + term funding for core assets + intraday liquidity monitoring + contingent liquidity sources (Fed discount window access ready). Lesson: Regulatory minimum is not a target—maintain buffer.",
    },
    {
      id: 3,
      question:
        'Why does NSFR (Net Stable Funding Ratio) assign only 50% "available stable funding" credit to wholesale funding, even if it has 1-year maturity?',
      options: [
        'Wholesale funding has 50% default risk',
        'Wholesale funding can disappear in stress (not truly stable), even if contractually 1-year',
        'The 50% factor is arbitrary with no rationale',
        'Retail funding is always more expensive than wholesale',
      ],
      correctAnswer: 1,
      explanation:
        'Wholesale funding from institutional counterparties can disappear even before contractual maturity because institutions run at first sign of trouble. In 2008, repo markets (wholesale funding) froze overnight despite being "secured." Prime brokerage clients fled within days. Regulators learned: legal maturity ≠ effective maturity in stress. Retail deposits are more stable (FDIC insurance keeps them). Hence NSFR assigns: Retail deposits 90-95% ASF credit, Wholesale funding 50% ASF credit (recognizing it will run), Equity 100% ASF (can\'t run). Option A is wrong—default risk is separate. Option C is wrong—50% reflects empirical behavior in 2008. Option D is irrelevant to stability. The 50% factor forces banks to not rely too heavily on wholesale funding—must diversify to retail deposits and term debt. Lesson: Contractual maturity is legal fiction; effective maturity in stress is what matters.',
    },
    {
      id: 4,
      question:
        "A bank's contingency funding plan (CFP) has three stages: Green (normal), Amber (stressed), Red (crisis). At what point should the bank trigger Amber status?",
      options: [
        'Only when a limit is breached',
        'Proactively at early warning signs (CDS spread widening, deposit outflows accelerating)',
        'Never—wait for regulator to declare emergency',
        'Only if LCR falls below 100%',
      ],
      correctAnswer: 1,
      explanation:
        'CFPs should trigger proactively at early warning indicators, not wait for actual breach or crisis. Early warning signs include: CDS spreads widening 50bp+, wholesale funding costs rising, deposit outflows accelerating (>5% in 3 days), negative media coverage, counterparty nervousness. Triggering Amber early allows: preventive actions (extend funding, stop asset growth, prepare asset sales), time to execute orderly deleveraging, communication with stakeholders. Option A (wait for breach) is reactive—too late. Option C (wait for regulator) is passive—regulator shouldn\'t know before you. Option D (wait for LCR breach) is too late—need to act at 120-110% to prevent falling below 100%. Bear Stearns failed because it didn\'t activate CFP early enough—went from "fine" to dead in 3 days. Lesson: Activation thresholds should be early (amber at 80% of normal, red at 90% of limits), not at failure point.',
    },
    {
      id: 5,
      question:
        'Which asset is assigned the highest liquidity value (lowest haircut) under LCR?',
      options: [
        'AAA-rated corporate bonds',
        'Agency MBS (Fannie Mae)',
        'Central bank reserves',
        'Gold',
      ],
      correctAnswer: 2,
      explanation:
        'Central bank reserves are Level 1 HQLA with 0% haircut—they are cash at the central bank, instantly available, no price risk. LCR hierarchy: Level 1 (0% haircut): Cash, central bank reserves, sovereign debt. Level 2A (15% haircut): Agency debt, AAA corporates. Level 2B (50% haircut): Lower-rated corporates, equities. Option A (AAA corporates) is Level 2A = 15% haircut. Option B (Agency MBS) is Level 2A = 25% haircut. Option D (gold) is not HQLA under Basel (ironic given historical role as liquid asset). The 0% haircut for central bank reserves reflects: (1) No price risk, (2) No credit risk, (3) Instant access, (4) Unlimited in stress (can always deposit more). This is why banks hold large reserves at Fed/ECB—costs opportunity (low yield) but provides liquidity buffer. Post-2008: Bank reserves at Fed increased from $10B to $3T+, partly for LCR compliance.',
    },
  ],
} as const;
