/**
 * SLIs, SLOs, and SLAs Section
 */

export const slisSection = {
  id: 'slis-slos-slas',
  title: 'SLIs, SLOs, and SLAs',
  content: `Service Level Indicators (SLIs), Service Level Objectives (SLOs), and Service Level Agreements (SLAs) form the foundation of Site Reliability Engineering (SRE). They provide a quantitative framework for measuring and maintaining service reliability. Understanding these concepts is essential for building production systems and acing system design interviews.

## Overview

### **The Hierarchy**

\`\`\`
SLA (Agreement)
 └─ SLO (Objective)
     └─ SLI (Indicator)
         └─ Metrics (Measurements)
\`\`\`

**Quick Definitions**:
- **SLI**: What you measure (e.g., latency, error rate)
- **SLO**: Target value for SLI (e.g., 99.9% uptime)
- **SLA**: Business contract with consequences (e.g., refund if < 99.9%)

---

## Service Level Indicators (SLIs)

### **What is an SLI?**

**SLI** = A carefully defined quantitative measure of a service's behavior

**Characteristics**:
- **Quantitative**: A number, not "fast" or "reliable"
- **User-Centric**: Measures user experience
- **Measurable**: Can be calculated from metrics
- **Actionable**: Can be improved

### **Good SLIs**

**Request Latency**:
\`\`\`
SLI: Proportion of requests served in < 100ms
Example: 95% of requests served in < 100ms
\`\`\`

**Availability**:
\`\`\`
SLI: Proportion of successful requests
Example: 99.9% of requests succeed
\`\`\`

**Throughput**:
\`\`\`
SLI: Requests served per second
Example: 10,000 requests/second
\`\`\`

**Durability**:
\`\`\`
SLI: Proportion of data retained
Example: 99.999999999% of objects retained (S3's 11 nines)
\`\`\`

### **SLI Specification**

**Format**: \`(Good Events / Total Events) × 100%\`

**Example: Latency SLI**
\`\`\`
SLI: Percentage of API requests served in < 100ms

Measurement:
  Good Events: Requests with latency < 100ms
  Total Events: All requests
  
Calculation:
  If 950 out of 1000 requests < 100ms
  SLI = (950 / 1000) × 100% = 95%
\`\`\`

**Example: Availability SLI**
\`\`\`
SLI: Percentage of successful HTTP requests

Measurement:
  Good Events: HTTP 2xx, 3xx responses
  Bad Events: HTTP 5xx responses
  Total Events: All HTTP requests
  
Calculation:
  If 9950 success, 50 errors
  SLI = (9950 / 10000) × 100% = 99.5%
\`\`\`

### **Common SLI Types**

**1. Request/Response SLIs**
- Availability: % successful requests
- Latency: % requests below threshold
- Throughput: Requests/second

**2. Data Processing SLIs**
- Freshness: % data processed within time window
- Completeness: % of data successfully processed
- Correctness: % of data correctly processed

**3. Storage SLIs**
- Durability: % of data retained
- Availability: % of read/write operations successful
- Latency: % operations completed in < X ms

---

## Service Level Objectives (SLOs)

### **What is an SLO?**

**SLO** = Target value or range for an SLI

**Format**: \`SLI ≥ Target% over Time Window\`

**Example**:
\`\`\`
SLO: 99.9% of API requests succeed over 30-day window
     └─ SLI: Request success rate
     └─ Target: 99.9%
     └─ Window: 30 days
\`\`\`

### **Why SLOs Matter**

**Without SLOs**:
- "Make it more reliable" (how reliable?)
- Unclear when to prioritize reliability vs features
- No data-driven decisions
- Can't measure improvement

**With SLOs**:
- "Achieve 99.9% uptime" (clear target)
- When SLO met → features; when at risk → reliability
- Data-driven prioritization
- Track progress quantitatively

### **Setting SLOs**

**Process**:

1. **Measure Current Performance**
   \`\`\`
   Past 30 days: 99.5% availability
   \`\`\`

2. **Talk to Users**
   - What level of reliability do they need?
   - What's acceptable downtime?

3. **Consider Costs**
   - 99% → 99.9% = moderate cost
   - 99.9% → 99.99% = exponential cost

4. **Start Conservatively**
   \`\`\`
   If current: 99.5%
   Set SLO: 99.0% (achievable)
   Later increase: 99.5% → 99.9%
   \`\`\`

5. **Iterate**
   - Review quarterly
   - Adjust based on data

### **SLO Examples**

**API Service**:
\`\`\`
SLO 1: 99.9% of requests succeed (30-day window)
SLO 2: 95% of requests complete in < 100ms (30-day window)
SLO 3: 99% of requests complete in < 500ms (30-day window)
\`\`\`

**Video Streaming**:
\`\`\`
SLO 1: 99.95% of video starts within 2 seconds
SLO 2: 99.9% of streaming sessions have < 1% rebuffer
SLO 3: 99% of playback time is at requested quality
\`\`\`

**Data Pipeline**:
\`\`\`
SLO 1: 99.9% of data processed within 1 hour
SLO 2: 99.99% of data is processed correctly
SLO 3: 99.9% of pipeline runs succeed
\`\`\`

### **Multiple SLOs**

**Why Multiple?**
- Single metric doesn't capture full experience
- Balance trade-offs
- Different aspects of reliability

**Example**:
\`\`\`
Service: E-commerce Checkout

SLO 1: Availability 99.9%
  → Ensures service is up

SLO 2: Latency p99 < 500ms
  → Ensures service is fast

SLO 3: Error rate < 0.1%
  → Ensures service is correct

All three needed for good experience!
\`\`\`

---

## Error Budgets

### **What is an Error Budget?**

**Error Budget** = Amount of unreliability allowed by SLO

**Formula**: \`Error Budget = 100% - SLO%\`

**Example**:
\`\`\`
SLO: 99.9% availability
Error Budget: 100% - 99.9% = 0.1%

Over 30 days:
  Total minutes: 30 × 24 × 60 = 43,200 minutes
  Allowed downtime: 43,200 × 0.1% = 43.2 minutes
  
Error Budget: 43.2 minutes of downtime per month
\`\`\`

### **Common Error Budgets**

| SLO | Error Budget | Downtime/Month | Downtime/Year |
|-----|-------------|----------------|---------------|
| 90% | 10% | 3 days | 36.5 days |
| 99% | 1% | 7.2 hours | 3.65 days |
| 99.9% | 0.1% | 43.2 minutes | 8.76 hours |
| 99.99% | 0.01% | 4.32 minutes | 52.6 minutes |
| 99.999% | 0.001% | 26 seconds | 5.26 minutes |

### **Using Error Budgets**

**Decision Framework**:

**Budget Remaining > 0** (SLO met):
- ✅ Ship new features
- ✅ Experiment
- ✅ Take calculated risks
- ✅ Aggressive deployments

**Budget Near 0** (at risk):
- ⚠️ Slow down feature development
- ⚠️ Focus on reliability
- ⚠️ More testing
- ⚠️ Cautious deployments

**Budget Exhausted** (SLO breached):
- 🚫 Feature freeze
- 🚫 Only reliability work
- 🚫 No risky changes
- 🚫 Focus on incident prevention

**Benefits**:
- Quantitative decision-making
- Balance innovation and stability
- Shared language between dev and ops

### **Error Budget Example**

\`\`\`
Month of January:

SLO: 99.9% availability
Error Budget: 43.2 minutes

Week 1: 10 min downtime (incident)
  Remaining: 33.2 minutes (77% left)
  Status: OK, continue normal pace

Week 2: 20 min downtime (bad deploy)
  Remaining: 13.2 minutes (31% left)
  Status: ⚠️ At risk, slow down releases

Week 3: 15 min downtime (database issue)
  Remaining: -1.8 minutes (exhausted!)
  Status: 🚫 Feature freeze, reliability only
\`\`\`

---

## Service Level Agreements (SLAs)

### **What is an SLA?**

**SLA** = Business contract with financial consequences

**Components**:
1. **SLO Target**: The promised level (e.g., 99.9%)
2. **Consequences**: What happens if breached (e.g., refund)
3. **Measurement**: How it's calculated
4. **Exclusions**: What doesn't count (planned maintenance)

### **SLA vs SLO**

**SLO** (Internal):
- Internal target
- No financial penalties
- More aggressive (e.g., 99.95%)
- Guides engineering decisions

**SLA** (External):
- Customer-facing promise
- Financial consequences
- More conservative (e.g., 99.9%)
- Legal/business document

**Relationship**:
\`\`\`
SLA: 99.9% (customer promise)
SLO: 99.95% (internal target)
Buffer: 0.05% (safety margin)

Why buffer? If SLO breached, still have buffer before SLA breached
\`\`\`

### **SLA Example (AWS EC2)**

\`\`\`
Service: AWS EC2
SLA: 99.99% monthly uptime percentage

If Uptime < 99.99%:
  Service Credit: 10%
  
If Uptime < 99.0%:
  Service Credit: 30%

Calculation:
  Uptime % = (Total Minutes - Downtime) / Total Minutes × 100%
  
Exclusions:
  - Scheduled maintenance (with 5-day notice)
  - Factors outside AWS control
  - Your actions that cause outage
\`\`\`

### **SLA Example (Stripe)**

\`\`\`
Service: Stripe API
SLA: 99.99% uptime

If API Uptime < 99.99%:
  - 95-99.99%: No credit
  - 90-95%: 10% credit
  - < 90%: 25% credit

Maximum Credit: 25% of monthly fees

Measurement:
  - 5-minute intervals
  - Failed if error rate > 5%
  - Excludes maintenance windows
\`\`\`

---

## Measuring SLIs

### **Where to Measure**

**Client-Side** (Best):
- Measures actual user experience
- Includes network latency
- True end-to-end

**Example**:
\`\`\`
User's browser → measures page load time
Real user monitoring (RUM)
\`\`\`

**Server-Side** (Common):
- Easier to measure
- More reliable
- Doesn't include client factors

**Example**:
\`\`\`
Server response time
Excludes: network latency, client rendering
\`\`\`

**Load Balancer** (Practical):
- Centralized measurement
- Consistent
- Close to user

**Recommendation**: Start with server-side, add client-side when possible

### **Time Windows**

**Rolling Window** (Recommended):
\`\`\`
SLO: 99.9% over rolling 30-day window
\`\`\`
- At any point, look back 30 days
- Continuous measurement
- Smooth out spikes

**Calendar Window**:
\`\`\`
SLO: 99.9% per calendar month
\`\`\`
- Reset on 1st of month
- Can "bank" good days early
- Gaming possible

**Best Practice**: Use rolling window

---

## Alerting on SLOs

### **Error Budget Burn Rate**

**Problem**: Waiting until budget exhausted is too late

**Solution**: Alert on burn rate

**Example**:
\`\`\`
SLO: 99.9% (43.2 min budget/month)

Fast Burn (Page immediately):
  If burning 2% of monthly budget in 1 hour
  → At this rate, budget exhausted in 2 days
  → ALERT: Critical

Medium Burn (Ticket):
  If burning 5% of monthly budget in 6 hours
  → At this rate, budget exhausted in 5 days
  → ALERT: Warning

Slow Burn (Dashboard):
  If burning 10% of monthly budget in 3 days
  → At this rate, budget exhausted in 30 days
  → Monitor closely
\`\`\`

### **Multi-Window Alerts**

Alert on short and long windows

**Example**:
\`\`\`
Alert 1 (Fast burn, high urgency):
  Condition: Error budget 2% consumed in 1 hour
  Action: Page on-call immediately

Alert 2 (Medium burn, medium urgency):
  Condition: Error budget 5% consumed in 6 hours
  Action: Create ticket

Alert 3 (Slow burn, low urgency):
  Condition: Error budget 50% consumed in 7 days
  Action: Dashboard notification
\`\`\`

---

## Best Practices

### **Do's**
✅ Start with user-facing SLIs
✅ Set achievable SLOs (don't aim for perfection)
✅ Use error budgets for decision-making
✅ Alert on error budget burn rate
✅ Make SLOs stricter than SLAs
✅ Review and adjust SLOs quarterly
✅ Document SLI calculations clearly
✅ Involve stakeholders in setting SLOs

### **Don'ts**
❌ Too many nines (99.999% very expensive)
❌ SLOs without measurement
❌ SLIs that don't reflect user experience
❌ Ignoring error budgets
❌ Same SLO and SLA (no buffer)
❌ Static SLOs (never adjusted)
❌ Alert on SLI directly (use error budget burn)

---

## The Cost of Nines

### **Reliability vs Cost**

**Increasing Nines**:
- 90% → 99%: Moderate effort
- 99% → 99.9%: Significant effort
- 99.9% → 99.99%: Exponential effort
- 99.99% → 99.999%: Extreme effort and cost

**Why?**

**99% Availability** (3.65 days/year downtime):
- Single region
- Basic monitoring
- Manual failover
- Business hours support

**99.9% Availability** (8.76 hours/year):
- Multi-AZ deployment
- Automated monitoring
- Automated failover
- 24/7 on-call

**99.99% Availability** (52 minutes/year):
- Multi-region deployment
- Advanced monitoring
- Chaos engineering
- Multiple on-call engineers
- Expensive infrastructure

**99.999% Availability** (5.26 minutes/year):
- Global distribution
- Active-active redundancy
- Significant investment in reliability
- Large on-call teams
- Very expensive

### **When to Choose**

**99%**: Internal tools, non-critical services
**99.9%**: Most production services, e-commerce
**99.99%**: Payment processing, banking
**99.999%**: Emergency services, life-critical systems

**Key Insight**: Don't over-engineer. 99.9% is sufficient for most services.

---

## Interview Tips

### **Key Concepts**

1. **SLI**: What you measure (latency, availability)
2. **SLO**: Target for SLI (99.9% uptime)
3. **SLA**: Contract with consequences
4. **Error Budget**: Allowed unreliability (100% - SLO)
5. **Burn Rate**: How fast error budget is consumed

### **Common Questions**

**Q: What's the difference between SLO and SLA?**
A: SLO is internal target (99.95%), SLA is customer promise with financial penalties (99.9%). SLO should be stricter to provide buffer.

**Q: How do you use error budgets?**
A: If budget remaining, ship features. If budget low, focus on reliability. If exhausted, feature freeze.

**Q: Why not aim for 100% uptime?**
A: Exponentially expensive, prevents innovation, unnecessary for most services, and actually impossible to achieve.

**Q: How would you set SLOs for a new service?**
A: Measure current performance for 2-4 weeks, talk to users about needs, start with achievable target (e.g., 99%), iterate based on data.

---

## Real-World Examples

### **Google**
- **Target**: 99.99% for Search
- **Practice**: Error budgets drive decisions
- **Result**: Balance between innovation and reliability

### **Netflix**
- **SLO**: 99.99% availability for streaming
- **Approach**: Chaos engineering to maintain SLO
- **Result**: Maintain SLO during 10x traffic spikes

### **AWS**
- **SLA**: 99.99% for EC2
- **SLO**: Internal SLO likely 99.999%
- **Buffer**: Provides safety margin

---

## Summary

SLIs, SLOs, and SLAs provide a framework for reliability:

1. **SLI**: What you measure (availability, latency)
2. **SLO**: Target value (99.9% uptime)
3. **SLA**: Business contract (99.9% or refund)
4. **Error Budget**: Allowed unreliability (100% - SLO)
5. **Burn Rate**: Alert when consuming budget too fast

Key insights:
- Start with achievable SLOs, iterate
- Use error budgets for decision-making
- SLO should be stricter than SLA
- Don't aim for perfection (too expensive)
- Alert on burn rate, not SLI directly

This framework enables data-driven decisions about reliability vs innovation.`,
};
