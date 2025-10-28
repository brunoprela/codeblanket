export default {
  id: 'fin-m15-s4-quiz',
  title: 'Stress Testing and Scenario Analysis - Quiz',
  questions: [
    {
      id: 1,
      question: 'What is the primary purpose of reverse stress testing?',
      options: [
        'To test how the portfolio performs if the 2008 crisis happens again',
        'To identify scenarios that would cause the firm to fail, then assess their plausibility',
        'To test the resilience of IT systems under load',
        'To reverse engineer competitor strategies',
      ],
      correctAnswer: 1,
      explanation:
        'Reverse stress testing starts with the outcome (firm failure—breaching capital requirements, illiquidity, etc.) and works backward to find scenarios that would cause it. This forces firms to identify their vulnerabilities. Unlike forward stress testing (apply scenario → measure impact), reverse stress testing asks: "What would kill us?" Then assess if those scenarios are plausible. Option A describes historical scenario testing. Option C describes system testing. Option D is unrelated. Reverse stress testing is powerful because: (1) Reveals unknown vulnerabilities, (2) Required by regulators post-2008, (3) Challenges assumption of invincibility. Example: Firm discovers "Corporate defaults >10% + repo market freeze" would cause failure, then assess: how likely is this? Can we prevent it?',
    },
    {
      id: 2,
      question:
        'A firm runs 100 historical stress scenarios. All show losses <$50M. Should the firm conclude its stress VaR is $50M?',
      options: [
        'Yes—the worst historical scenario defines stress VaR',
        'No—historical scenarios may not cover worst plausible future events',
        'Yes, but only if the scenarios cover 30+ years of history',
        'No—stress VaR should always be 3x normal VaR regardless of scenarios',
      ],
      correctAnswer: 1,
      explanation:
        "Historical scenarios are limited by what has actually happened—they don't cover all plausible future events. The worst hasn't happened yet. Using 2005-2007 scenarios would show small losses, but this doesn't mean 2008 couldn't happen. Option A is dangerously complacent. Option C doesn't solve the problem—even 30 years might miss tail events (e.g., no pandemic in 30 years, then COVID). Option D is arbitrary—why 3x? Best practice: Combine historical scenarios with hypothetical scenarios that are plausible but haven't occurred (e.g., \"What if rates rise 400bp like 1980s?\"). Stress testing should make you uncomfortable—if all scenarios show manageable losses, you're not stressing enough. Regulators look for \"severely adverse\" scenarios, not just bad days from history.",
    },
    {
      id: 3,
      question:
        'CCAR/DFAST stress testing requires US banks to project losses under the Fed\'s "severely adverse" scenario. What is the main challenge?',
      options: [
        'The scenario is kept secret until submission',
        'Banks must estimate losses 9 quarters forward with complex interdependencies',
        'The scenario changes daily making planning impossible',
        'Small banks are exempt creating unfair advantage',
      ],
      correctAnswer: 1,
      explanation:
        'CCAR requires projecting 9 quarters (2.25 years) of losses under stress, modeling complex interactions: credit losses, market losses, pre-provision revenue, loan loss provisions, RWA changes, capital ratios. This is extremely difficult because: (1) Long horizon with compounding uncertainty, (2) Second-order effects (stress causes borrowers to default → losses → reduced lending → more economic stress), (3) Management actions must be realistic. Option A is wrong—scenarios are published in advance (February, submissions April). Option C is wrong—scenario is annual (stable). Option D is wrong—only banks >$100B assets must participate (smaller banks face different requirements). The challenge is methodological: building models that predict stress losses 2 years out with interconnected risks. Firms spend 500-1000 person-hours on CCAR annually.',
    },
    {
      id: 4,
      question:
        'A portfolio stress test shows: "If equities fall 30%, portfolio loses $500M." The risk manager says: "This is fine, we have $1B capital." What is the problem with this reasoning?',
      options: [
        'There is no problem—$1B capital exceeds $500M loss',
        'The scenario ignores other risks that would occur simultaneously with equity crash',
        'Equity could fall more than 30%',
        'Capital requirements increase in stress, so $1B might not be enough',
      ],
      correctAnswer: 1,
      explanation:
        'Single-factor stress tests are misleading because risks don\'t occur in isolation. When equities fall 30% (major crash), you\'d also see: credit spreads widen, volatility spike, illiquidity, correlations go to 1, flight to quality. The total loss could be $800M+, not $500M. Option A accepts the flawed reasoning. Option C is true but not the main issue. Option D is true but secondary. The key error is univariate thinking. Best practice: Multivariate stress scenarios that reflect realistic combinations. In 2008, firms that only tested "credit spreads +200bp" or "equity -20%" separately were blindsided when BOTH happened simultaneously (plus funding crisis, plus illiquidity). Correlation matters: in stress, everything moves together. Stress tests must reflect this or they underestimate risk by 50-75%.',
    },
    {
      id: 5,
      question:
        'What is the difference between scenario analysis and sensitivity analysis?',
      options: [
        'Scenario analysis tests extreme events; sensitivity analysis tests small moves',
        'Scenario analysis is multivariate (multiple risk factors); sensitivity analysis is univariate (one factor)',
        'Scenario analysis is qualitative; sensitivity analysis is quantitative',
        'There is no difference—the terms are interchangeable',
      ],
      correctAnswer: 1,
      explanation:
        'Sensitivity analysis examines one risk factor at a time (e.g., "rates +100bp, all else constant") showing individual sensitivities. Scenario analysis combines multiple factors (e.g., "rates +200bp AND equity -20% AND spreads +150bp") reflecting realistic combinations. Option A is partially true but not the key distinction—both can test extreme or small moves. Option C is wrong—both are quantitative. Option D is wrong—they serve different purposes. Use sensitivity analysis to understand drivers (which risk factor matters most?). Use scenario analysis to understand realistic outcomes (risk factors move together). Example: Sensitivity shows "rates +100bp costs $10M." But scenario "recession" includes rates -200bp, spreads +300bp, equity -30% → net impact might be +$5M (spreads hurt more than rates help). Scenarios capture interactions that sensitivities miss.',
    },
  ],
} as const;
