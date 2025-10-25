/**
 * Quiz questions for SLIs, SLOs, and SLAs section
 */

export const slisQuiz = [
  {
    id: 'q1',
    question:
      'Explain the relationship between SLIs, SLOs, SLAs, and Error Budgets. How do these concepts work together to drive engineering decisions?',
    sampleAnswer:
      'These concepts form a hierarchy that enables data-driven reliability management. **SLI (Service Level Indicator)**: What you measure. Quantitative measure of service behavior. Example: "Percentage of HTTP requests that succeed" or "Percentage of requests served in < 100ms." Must be: Measurable, user-centric, actionable. **SLO (Service Level Objective)**: Target for SLI. Internal goal without consequences. Example: "99.9% of requests succeed over 30-day window" or "95% of requests complete in < 100ms." Should be: Achievable, stricter than SLA, drives eng priorities. **SLA (Service Level Agreement)**: Customer promise with consequences. Contract with financial penalties if breached. Example: "99.9% monthly uptime. If < 99.9%, 10% service credit. If < 99%, 30% credit." Should be: Less strict than SLO (buffer), legally binding, measurable. **Error Budget**: Allowed unreliability. Formula: 100% - SLO%. Example: SLO 99.9% → Error budget 0.1% → 43.2 minutes downtime/month allowed. **How They Work Together**: (1) **Measurement**: SLI tells you current performance (99.95% success rate this month). (2) **Target**: SLO tells you goal (99.9% success rate). (3) **Decision Framework**: Error budget tells you if you can take risks. Budget remaining > 0: Ship features, experiment, take calculated risks. Budget near 0: Slow releases, focus on reliability. Budget exhausted: Feature freeze, only reliability work. (4) **Customer Protection**: SLA protects customers (with buffer from SLO). **Example Flow**: SLO: 99.9% availability (43.2 min budget/month). Week 1: 10 min outage → 33.2 min remaining → OK, continue. Week 2: 20 min outage → 13.2 min remaining → WARN, slow down releases. Week 3: 15 min outage → Budget exhausted! → FREEZE, only reliability. SLA: 99.9%, still met (have 0.05% buffer from SLO 99.95% actual target). **Value**: Quantifies reliability, enables objective decisions, balances innovation vs stability, shared language between eng and business.',
    keyPoints: [
      'SLI = what you measure, SLO = target, SLA = customer promise',
      'Error budget = 100% - SLO% = allowed unreliability',
      'Budget drives decisions: remaining → ship features, exhausted → freeze',
      'SLO should be stricter than SLA (buffer for safety)',
      'Enables data-driven balance between innovation and stability',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is "100% uptime" not a good SLO? What are the costs of increasing reliability from 99% to 99.9% to 99.99%?',
    sampleAnswer:
      '100% uptime is impossible, prohibitively expensive, and prevents innovation. **Why 100% is Impossible**: (1) Hardware fails (hard drives, network cards, power supplies). (2) Software has bugs (no software is perfect). (3) Dependencies fail (cloud providers, third parties). (4) Natural disasters, BGP leaks, solar flares. (5) Planned maintenance requires downtime. Even Google, Amazon, Facebook have outages. **Why 100% Prevents Innovation**: (1) Innovation requires risk (new features might have bugs). (2) Error budget of 0% means you can never deploy. (3) Slow deployment velocity (extreme testing, long rollout periods). (4) Fear of change freezes the product. **The Cost of Nines**: **99% Availability** (3.65 days/year downtime): Cost: Moderate. Architecture: Single region, basic monitoring, manual failover. Effort: Regular engineering team. **99.9% Availability** (8.76 hours/year): Cost: Significant (+3-5x). Architecture: Multi-AZ deployment, automated monitoring, automated failover, 24/7 on-call. Effort: Dedicated reliability engineering. **99.99% Availability** (52 minutes/year): Cost: Exponential (+10x from 99.9%). Architecture: Multi-region, active-active, chaos engineering, advanced monitoring. Effort: Large SRE team, expensive infrastructure (triple redundancy). **99.999% Availability** (5.26 minutes/year): Cost: Astronomical (+100x from 99.9%). Architecture: Global distribution, custom hardware, massive redundancy. Effort: Only for life-critical systems (911, air traffic control, pacemakers). Only huge companies can afford (Google, Amazon). **Practical Example**: E-commerce site: 99% might be fine (tolerate some downtime). Payment processing: 99.9% necessary (revenue-critical). Emergency services: 99.999% required (lives at stake). **Recommendation**: Start with achievable SLO (99% or 99.5%), measure actual performance, iterate up based on business need and user feedback. Most services should target 99.9%, not higher.',
    keyPoints: [
      '100% impossible (hardware fails, bugs exist, dependencies fail)',
      "100% prevents innovation (can't deploy with 0% error budget)",
      'Cost increases exponentially: 99% → 99.9% (5x), 99.9% → 99.99% (10x)',
      '99.99%+ requires: multi-region, massive redundancy, large SRE team',
      'Most services should target 99.9%, not higher',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you set SLOs for a new service that has no historical data? Walk through the process from measurement to setting initial targets.',
    sampleAnswer:
      'Setting SLOs for a new service requires measurement first, then iterative refinement based on data and user feedback. **Step 1: Identify User Journeys** (Week 0): Determine critical user-facing operations. Example API service: User login, Load dashboard, Submit form, Fetch data. **Step 2: Choose SLIs** (Week 0): For each operation, define measurable indicators: Availability SLI: % of HTTP 2xx/3xx responses. Latency SLI: % of requests < 100ms, % requests < 500ms. Freshness SLI: % of data < 5 minutes old. **Step 3: Instrument and Collect Baseline** (Weeks 1-4): Deploy service to staging/beta with full instrumentation. Don\'t set alerts yet, just collect data. Measure: Current availability (e.g., 99.7%), Current p50 latency (e.g., 45ms), Current p99 latency (e.g., 350ms). Track over 4 weeks to see patterns (weekday vs weekend, peak vs off-peak). **Step 4: Talk to Stakeholders** (Week 4): Product: What user experience is acceptable? "Occasional errors OK, but < 1% failure rate." Users: What\'s their tolerance? Survey beta users: "5-second page loads unacceptable, 1-second is fine." Business: What\'s the cost/ benefit ? "Each 9 costs more, justify based on churn." ** Step 5: Set Conservative Initial SLOs ** (Week 4): Start with achievable targets slightly worse than baseline.If baseline availability is 99.7 % → Set initial SLO: 99.0 % (achievable with buffer).If baseline p99 latency is 350ms → Set initial SLO: 500ms (achievable).Rationale: Better to exceed than miss.Builds confidence.Can tighten later. ** Step 6: Deploy to Production ** (Weeks 5 - 8): Monitor against SLOs.Track error budget consumption.Don\'t page on SLO violations yet (still learning). **Step 7: Iterative Refinement** (Months 2-6): After 30 days, review: Did we meet SLO easily? Tighten (99.0% → 99.5%). Did we miss SLO? Loosen or improve service. Get user feedback: Are they happy? Complaints about speed → tighten latency SLO. Adjust every quarter based on data. **Step 8: Set Up SLO-Based Alerting** (Month 3): Alert on error budget burn rate, not raw SLI. Fast burn: 2% of monthly budget in 1 hour → Page. Medium burn: 5% of budget in 6 hours → Ticket. **Example Timeline**: Month 0: Measure, collect data. Month 1: Set SLO 99%, monitor. Month 3: Achieve 99.8%, tighten to 99.5%. Month 6: Achieve 99.6%, tighten to 99.9%. Month 12: Stable at 99.9%, set SLA 99.9% for customers.',
    keyPoints: [
      'Step 1: Measure baseline performance for 4 weeks',
      'Step 2: Talk to stakeholders (product, users, business)',
      'Step 3: Set conservative initial SLOs (easier than baseline)',
      'Step 4: Monitor and iterate quarterly based on data',
      'Step 5: Tighten SLOs as service matures and improves',
    ],
  },
];
