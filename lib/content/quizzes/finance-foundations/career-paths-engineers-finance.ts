export const careerPathsEngineersFinanceQuiz = [
  {
    id: 'cpef-q-1',
    question:
      "You're deciding between three job offers: (1) Two Sigma - Quant Researcher ($350K total, PhD required, research-focused, 60 hr weeks), (2) Stripe - Senior Engineer ($400K with equity, full-stack web, 50 hr weeks), (3) Goldman Sachs - Strat ($280K, trading desk support, 70 hr weeks). Analyze: compensation (cash vs equity, bonus structure, vesting), career growth (skills developed, exit opportunities), work-life balance, risk factors, and long-term wealth potential. Which would you choose for a 10-year career horizon and why? Consider: what if Stripe IPO fails? What if Two Sigma strategies underperform? What if banking bonus pool shrinks?",
    sampleAnswer: `Comprehensive Career Offer Analysis:

**1. Total Compensation Analysis (10-Year Horizon)**

Two Sigma - Quant Researcher:
- Year 1: $350K ($200K base + $150K bonus)
- Typical progression: +15-20% annually if performing
- Year 5: ~$600K-700K
- Year 10: $900K-1.2M (if you become senior/principal)
- 10-year total: ~$6-7M (assumes good performance)
- Structure: 60% base, 40% bonus (tied to strategy P&L)
- Risk: Bonus can drop to $50K in bad years, or termination if strategies fail
- Vesting: Cash comp (no equity), immediate
- Tax: Ordinary income (37% federal + state)

Stripe - Senior Engineer:
- Year 1: $400K ($240K base + $160K RSUs)
- Typical progression: +10-15% annually
- Year 5: ~$550K-650K
- Year 10: $750K-900K (Staff/Principal level)
- 10-year total: ~$5.5-6.5M (assumes moderate growth)
- BUT: If IPO successful (Stripe valued at $50B → $100B), early equity could be worth $2-10M extra
- Structure: 60% cash, 40% equity (4-year vesting, monthly)
- Risk: Equity worth $0 if company fails (10-20% chance for late-stage startup)
- Tax: Long-term capital gains on equity (20% if held >1 year post-vest)

Goldman Sachs - Strat:
- Year 1: $280K ($150K base + $130K bonus)
- Typical progression: +20-30% when promoted (every 2-3 years)
- Year 5 (VP): ~$500K-600K
- Year 10 (Director/MD): $800K-1.5M
- 10-year total: ~$5.5-7M (assumes promotions on schedule)
- Structure: 50% base, 50% bonus (varies 0.5x-2x with firm performance)
- Risk: Bonus highly variable (2008 crisis: bonuses cut 50-80%)
- Vesting: Deferred comp for MD+ (RSUs, stock options)
- Tax: Ordinary income (37% + state)

**Present Value Analysis** (8% discount rate):
- Two Sigma: $6.5M → PV = $4.3M
- Stripe: $6M + $5M equity (50% prob) → PV = $4.1M
- Goldman: $6M → PV = $4.0M

Two Sigma wins on pure compensation (highest PV).

**2. Career Growth & Skills Development**

Two Sigma:
- Skills: ML/AI (cutting-edge), statistical modeling, research methodology, Python/C++
- Deep expertise: Time series forecasting, alternative data, portfolio optimization
- Transferable: Google Brain, Meta AI, DeepMind (research roles)
- Limited: Narrow focus (trading strategies), hard to pivot to non-finance

Stripe:
- Skills: Full-stack engineering, distributed systems, API design, product development
- Broad expertise: Payments, compliance, scaling (millions of TPS), microservices
- Transferable: Any tech company (FAANG, startups), CTO path
- Growth: Product management, eng leadership, startup founder

Goldman Sachs:
- Skills: Derivatives pricing, Excel/VBA, Python, financial modeling, client communication
- Mixed expertise: Finance knowledge (valuable) + legacy tech (Excel, VBA less valuable)
- Transferable: Other banks, hedge funds, corporate development, fintech (domain expertise)
- Limited: Tech skills lag modern stack (not learning React, Kubernetes, etc.)

**Skill Transferability Ranking**:
1. Stripe (highest - modern tech skills valuable everywhere)
2. Two Sigma (moderate - ML/AI valuable but finance-specific)
3. Goldman (moderate - finance expertise but legacy tech)

**3. Exit Opportunities (After 5 Years)**

Two Sigma → Where can you go?
- Other quant funds: Citadel, Renaissance, DE Shaw (lateral or up)
- Tech AI labs: Google Brain, Meta AI ($400K-600K, less pressure)
- Startups: ML-focused fintech, quant trading platform (CTO, equity)
- Downside: Hard to exit finance (skills are trading-specific)

Stripe → Where can you go?
- FAANG: L6/E6 level ($500K-700K total comp)
- Other fintech: Robinhood, Coinbase (Staff+, $600K-1M)
- Startups: VP Eng, CTO (lower cash, big equity upside)
- Big tech infra: Uber, Airbnb payments engineering
- Upside: Broadest options (tech skills universal)

Goldman → Where can you go?
- Buy-side: Hedge funds, asset managers ($400K-800K)
- Other banks: JPM, Morgan Stanley (lateral)
- Fintech: Plaid, Chime (senior eng, $350K-600K)
- Corporate: Tech companies (corp dev, strategy)
- Downside: Tech skills gap (need to upskill for modern stack)

**Exit Opportunity Ranking**:
1. Stripe (most options)
2. Goldman (good within finance)
3. Two Sigma (narrow but high-paying)

**4. Work-Life Balance**

Two Sigma:
- Hours: 60/week average (spikes to 70-80 during research deadlines)
- Flexibility: Moderate (research has deadlines, but you control schedule)
- Stress: High (performance pressure, strategies must work)
- Vacation: 3-4 weeks, but guilty taking it
- On-call: No (research role)
- Rating: ⭐⭐ (2/5)

Stripe:
- Hours: 50/week average (spikes to 60 during launches)
- Flexibility: High (remote options, async culture)
- Stress: Moderate (product deadlines, but no P&L pressure)
- Vacation: Unlimited (actually take 4-5 weeks)
- On-call: Yes (1 week per month for production systems)
- Rating: ⭐⭐⭐⭐ (4/5)

Goldman Sachs:
- Hours: 70/week average (can spike to 80-90 during deals)
- Flexibility: Low (need to be on desk, traders need you)
- Stress: High (traders yelling, urgent requests, market pressure)
- Vacation: 3-4 weeks, hard to take (market doesn't stop)
- On-call: Effectively always (traders call at 6am if something breaks)
- Rating: ⭐⭐ (2/5)

**Work-Life Balance Ranking**:
1. Stripe (best balance)
2. Two Sigma (moderate)
3. Goldman Sachs (worst)

**5. Risk Factors & Downside Scenarios**

Two Sigma Risks:
- Strategy underperformance: Bonus drops to $50K (from $150K)
- Termination: "Up or out" culture - if strategies don't work, you're gone
- Market disruption: If quant strategies stop working (market regime change)
- Mitigation: Diversify strategies, build strong track record, save aggressively

Stripe Risks:
- IPO fails: Equity worth $0 (lose $160K/year x 4 years = $640K)
- Valuation down: If Stripe IPO at $30B (vs $50B), equity worth 60% less
- Competition: Adyen, PayPal gaining share
- Mitigation: Negotiate higher base ($270K vs $240K), join post-IPO

Goldman Sachs Risks:
- Bonus pool shrinks: Bad market year → bonus cut 50-70%
- Layoffs: 2008-style crisis → 10-30% staff reduction
- Automation: Strat roles being replaced by self-service tools
- Mitigation: Build broad skillset, network internally, save for down years

**Risk-Adjusted Returns**:
- Two Sigma: High volatility (big bonus swings), high expected value
- Stripe: Moderate volatility (equity risk), high upside potential
- Goldman: High volatility (bonus cuts), moderate expected value

**6. Long-Term Wealth Potential (20-Year Horizon)**

Scenario: Save 50% of income, invest at 8% annually

Two Sigma Path:
- Years 1-10: Save $3.5M → Grows to $5.0M by year 20
- Years 11-20: Assume $1.2M avg comp, save $600K/yr → $14.7M
- Total at Year 20: $19.7M
- Millionaire by: Year 3

Stripe Path:
- Years 1-10: Save $3.0M cash + $2.5M equity → $7.8M by year 20
- Years 11-20: Assume $900K avg comp, save $450K/yr → $11.0M
- Total at Year 20: $18.8M (if IPO successful)
- If IPO fails: $14.3M (no equity value)
- Millionaire by: Year 3

Goldman Path:
- Years 1-10: Save $3.0M → Grows to $4.3M by year 20
- Years 11-20: Assume $1.1M avg comp, save $550K/yr → $13.5M
- Total at Year 20: $17.8M
- Millionaire by: Year 4

**Wealth Ranking** (Expected Value):
1. Two Sigma: $19.7M
2. Stripe: $18.8M (with IPO), $14.3M (without)
3. Goldman Sachs: $17.8M

**7. MY RECOMMENDATION: Stripe**

Despite lower pure cash comp, I'd choose Stripe for:

**Primary Reasons**:
1. **Skill Development**: Modern tech stack (React, Kubernetes, distributed systems) vs specialized finance skills
2. **Exit Options**: Can move anywhere in tech vs locked into finance
3. **Work-Life Balance**: 50 hrs vs 60-70 hrs = 10-20 extra hours/week for life, side projects, family
4. **Equity Upside**: $5M+ potential if Stripe continues growing (vs pure cash at Two Sigma)
5. **Product Impact**: Building products used by millions vs internal trading strategies

**Secondary Reasons**:
6. **Culture**: Stripe known for excellent eng culture, remote flexibility, written communication
7. **Learning**: Exposure to payments, compliance, global infrastructure
8. **Entrepreneurship**: Better prep for starting own company (product + business skills)

**When to Choose Two Sigma Instead**:
- You LOVE research and can't imagine doing product engineering
- Maximum wealth accumulation is only goal (pure comp optimization)
- You have PhD and want to use it
- You're okay with narrow specialization

**When to Choose Goldman Instead**:
- You want finance domain expertise for later buy-side move
- Investment banking prestige/network is valuable to you
- You're okay grinding for 5-10 years for MD promotion
- You want to learn from best traders/investors

**The 10-Year Plan**:
Years 1-5 Stripe: Learn payments, scale, build equity ($2-3M vested)
Years 6-10 Stripe: Staff/Principal, leadership ($4-5M total comp)
Year 11+: Either (A) Continue Stripe as Principal/VP, or (B) Start fintech company with deep payments knowledge + $10M net worth cushion

**Final Answer**: Stripe - for skills, balance, optionality, and long-term wealth potential (not just next 10 years, but 20-30 year career).`,
    keyPoints: [
      'Compensation: Two Sigma highest ($6.5M/10yr), Stripe $6M + equity upside, Goldman $6M but volatile bonus',
      'Skills: Stripe best (modern tech, transferable), Two Sigma specialized (ML for trading), Goldman legacy (Excel/VBA)',
      'Work-life: Stripe 50hrs (⭐⭐⭐⭐), Two Sigma 60hrs (⭐⭐), Goldman 70hrs (⭐⭐)',
      'Exit options: Stripe broadest (any tech), Two Sigma narrow (AI/quant), Goldman finance-focused',
      'Risk-adjusted: Stripe wins on skills + balance + equity upside despite lower pure cash comp',
    ],
  },
  {
    id: 'cpef-q-2',
    question:
      'Design a self-study plan to break into quantitative finance as a software engineer (no finance background, no PhD). Timeline: 6 months. Cover: (1) foundational knowledge (finance, statistics, ML), (2) practical projects (trading strategies, backtesting), (3) portfolio building (GitHub, blog, competitions), (4) networking strategy, (5) application tactics. What books/courses/resources would you use? What projects would demonstrate competence? How would you position yourself to tier-2 quant funds?',
    sampleAnswer: `Answer to be completed (6-month self-study plan covering: Month 1-2 foundations with books "Python for Finance" + fast.ai ML course + Coursera statistics, Month 3-4 build 3 projects (momentum strategy backtest, pairs trading with cointegration, sentiment analysis trading bot) all on GitHub with detailed README, Month 5 compete in Kaggle Jane Street competition + write technical blog posts explaining strategies, Month 6 network via QuantConnect forum + LinkedIn + informational interviews, application tactics include targeting tier-2 funds like Wolverine/Optiver/IMC that hire without PhD, emphasize software engineering expertise + demonstrable quant skills via projects, prepare for interview with "Heard on the Street" probability questions + LeetCode + explaining your backtests, realistic timeline is 6-12 months to first offer at tier-2 fund starting $180-250K, then build track record to move to tier-1).`,
    keyPoints: [
      'Foundations (2 months): "Python for Finance" book, fast.ai ML, statistics (Coursera), portfolio theory basics',
      'Projects (2 months): Momentum backtest, pairs trading, sentiment analysis bot - all on GitHub with detailed documentation',
      'Portfolio (1 month): Kaggle competitions (Jane Street), technical blog posts, QuantConnect live strategies',
      'Networking (ongoing): LinkedIn (connect with quant devs), QuantConnect forums, informational interviews, conferences',
      'Application (1 month): Target tier-2 funds (Optiver, IMC, Wolverine), emphasize SWE skills + quant projects, realistic $200-300K starting comp',
    ],
  },
  {
    id: 'cpef-q-3',
    question:
      "You're a senior engineer at FAANG ($500K total comp, L6 level, 45 hr weeks) considering a move to finance. Compare trade-offs for: (1) Jane Street (market maker, $600K+, 55 hrs, OCaml), (2) BlackRock (asset manager, $400K, 45 hrs, Python/risk systems), (3) Robinhood (fintech, $550K with equity, 50 hrs, React/Python). Analyze: compensation change (including equity risk), skill development, career trajectory, reversibility (can you return to tech?), and regret minimization. Which minimizes regret over 20-year career?",
    sampleAnswer: `Answer to be completed (comprehensive analysis covering: Jane Street offers $600-800K all-cash with proprietary OCaml that's hard to transfer back to tech but ultimate trading/systems challenge, BlackRock $400K is pay cut but best work-life balance and transferable Python/risk skills, Robinhood $550K with equity upside similar to FAANG culture/stack but fintech risk, regret minimization framework suggests staying FAANG for most engineers unless you specifically want trading challenge (Jane Street) or fintech product building (Robinhood), reversibility analysis shows Robinhood easiest to return to FAANG (same React/Python stack), Jane Street hardest (specialized OCaml/trading), BlackRock moderate (risk systems niche but valuable), ultimate recommendation depends on personal priorities but for risk-averse choose BlackRock for minimal disruption, for maximum comp choose Jane Street, for best of both worlds stay FAANG and do quant trading as side project).`,
    keyPoints: [
      'Compensation: Jane Street $600-800K cash (20% up), BlackRock $400K (20% down), Robinhood $550K + equity (lateral)',
      'Skills: Jane Street specialized (OCaml, trading), BlackRock transferable (Python, risk), Robinhood FAANG-like (React, Python)',
      'Reversibility: Robinhood easiest to return (same stack), Jane Street hardest (specialized), BlackRock moderate',
      'Regret minimization: Stay FAANG unless specific passion for trading (Jane Street) or fintech products (Robinhood)',
      'Work-life: BlackRock best (45hrs), Robinhood similar (50hrs), Jane Street more intense (55hrs + trading pressure)',
    ],
  },
];
