export default {
  id: 'fin-m15-s14-quiz',
  title: 'Risk Reporting and Dashboards - Quiz',
  questions: [
    {
      id: 1,
      question: 'A board risk report should focus on which of the following?',
      options: [
        'Position-level detail for all holdings',
        'High-level metrics: Firm VaR vs limit, stress test results, top risks, limit breaches',
        'Technical details of risk models and methodologies',
        'Daily P&L fluctuations',
      ],
      correctAnswer: 1,
      explanation:
        "Board reports should be high-level (1-2 pages + appendix): Firm-wide VaR trend, utilization vs limits, stress test outcomes, top 10 risks, material limit breaches, capital adequacy. Board wants to answer: \"Is firm within risk appetite? Any issues needing board attention?\" Option A is wrong—board doesn't need position-level (that's for desk heads). Option C is wrong—boards aren't modelers (include model summary in appendix for interested directors). Option D is wrong—daily P&L volatility is noise; board cares about trends and sustainability. Report frequency: Quarterly (monthly for large firms). Format: Executive summary with clear visuals (gauges, trend charts), drill-down appendix. Key: Actionable exceptions, not exhaustive data dumps. If everything's fine, say so in one page. If there's a problem, explain clearly: what, why, action being taken.",
    },
    {
      id: 2,
      question:
        'What is the main advantage of exception-based reporting over regular comprehensive reports?',
      options: [
        'Exception reports are shorter and faster to produce',
        'Exception reports surface items requiring attention, avoiding "needle in haystack" problem',
        'Exception reports are required by regulators',
        'Exception reports eliminate the need for detailed analysis',
      ],
      correctAnswer: 1,
      explanation:
        'Exception reporting shows only items needing attention (breaches, large moves, outliers), making problems immediately visible. Comprehensive report: 100 pages, user scans for 30 minutes to find 3 issues. Exception report: 1 page listing those 3 issues, user sees them in 10 seconds. Option A is side benefit, not main advantage. Option C is wrong—regulators typically want comprehensive reports. Option D is wrong—still need analysis, just focused on exceptions. Best practice: Daily exception report (1-2 pages: active breaches, large losses, unusual events) + weekly comprehensive report (full detail for those who want it). Exception reporting philosophy: "No news is good news." If nothing exceptional, report should say "No exceptions today" (30 seconds to read). If 5 exceptions, list them with context and action. This prevents alert fatigue—users trust that if they see something, it matters. Comprehensive reports create noise; exception reports create signal.',
    },
    {
      id: 3,
      question:
        'A heat map showing desk × risk metric with color coding (green/yellow/red) is best used for:',
      options: [
        'Precise numerical analysis',
        'Scanning many items quickly to identify problems',
        'Showing time series trends',
        'Detailed drill-down investigation',
      ],
      correctAnswer: 1,
      explanation:
        'Heat maps excel at pattern recognition—scan 50 desks × 5 metrics (250 cells) in 10 seconds to spot red cells requiring attention. Human visual system is excellent at detecting color outliers. Option A is wrong—heat maps sacrifice precision for scanability (can\'t read exact numbers). Option C is wrong—that\'s trend charts. Option D is wrong—heat maps are entry point; clicking cell shows detail. Use case: Risk manager arrives morning, views heat map, instantly sees "Equity Desk is red on VaR" → clicks → investigates. Without heat map, would scan table of 250 numbers (slow, error-prone). Design principle: Color for urgency (red = needs attention NOW), size for importance, position for hierarchy (most important top-left). Bad heat maps: too many colors (confusing), wrong colors (red/green colorblind issues—use orange/blue), too fine granularity (50 shades of green).',
    },
    {
      id: 4,
      question: 'How should risk reports balance timeliness versus accuracy?',
      options: [
        'Always prioritize accuracy—report when perfect',
        'Always prioritize timeliness—report immediately even if rough',
        'Layer reports: real-time approximate + EOD accurate',
        'Split the difference with moderate delay and moderate accuracy',
      ],
      correctAnswer: 2,
      explanation:
        "Layered reporting provides best of both: Real-time (seconds): 95% accurate, enables immediate decisions (pre-trade checks, alerts). Intraday (hourly): 98% accurate, monitoring. EOD (next morning): 99.9% accurate, official reporting. Each layer serves different purpose—real-time for operations, EOD for compliance. Option A misses the real-time need. Option B sacrifices reliability. Option D is mediocre at everything. Example: Real-time dashboard shows P&L ≈ +$5M (parametric, incremental VaR). EOD report shows P&L = +$4.8M (full revaluation, all adjustments). 4% difference is acceptable—real-time gave directionality (+$5M ballpark), EOD gave precision. User knows real-time is approximate and confirms with EOD. This is how professional firms operate—can't sacrifice speed OR accuracy, so layer both.",
    },
    {
      id: 5,
      question:
        'Regulatory risk reports (e.g., Basel Pillar 3) differ from internal management reports in what key way?',
      options: [
        'Regulatory reports are more detailed',
        'Regulatory reports use standardized formats mandated by regulation; management reports are customized for decisions',
        'Regulatory reports are produced monthly; management reports daily',
        'Regulatory reports focus on historical data; management reports focus on forward-looking',
      ],
      correctAnswer: 1,
      explanation:
        "Regulatory reports must follow standardized templates (Basel Pillar 3 disclosure, CCAR submissions, etc.) to enable comparison across firms. Management reports are customized for decision-making—show what managers need to see, formatted for action. Option A is sometimes true but not the key distinction. Option C varies—both can be daily or quarterly. Option D is partially true but not the defining difference. Regulatory reporting: Fixed format (can't change table structure), prescribed calculations (must use Basel formula even if internal model differs), public disclosure (Pillar 3), pass/fail criteria (meet capital ratios). Management reporting: Flexible format (optimize for decision-making), best-available calculations (use best models), private (competitive info), continuous improvement (optimize metrics). Tension: Firms must maintain both systems—regulatory for compliance, internal for actual risk management. Often very different numbers!",
    },
  ],
} as const;
