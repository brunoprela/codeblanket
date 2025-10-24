/**
 * Incident Management Section
 */

export const incidentManagementSection = {
  id: 'incident-management',
  title: 'Incident Management',
  content: `Despite best efforts, incidents happen. Services go down, data gets corrupted, and customers are affected. How your organization responds to incidents determines whether they become minor blips or major disasters. Effective incident management minimizes impact, accelerates recovery, and turns failures into learning opportunities.

## What is an Incident?

**Incident**: An unplanned interruption or reduction in quality of service

**Characteristics**:
- User-impacting (not just internal)
- Requires immediate response
- Violates SLOs or SLAs
- Needs coordination

**Examples**:
- Service outage (can't access website)
- Data loss (customer data deleted)
- Performance degradation (slow responses)
- Security breach (unauthorized access)

**Not Incidents**:
- Planned maintenance
- Minor bugs (no user impact)
- Internal service issues (if not user-facing)

---

## Incident Severity

### **Severity Levels**

**SEV-1 (Critical)**:
- **Impact**: Complete outage or major data loss
- **Users Affected**: Most or all users
- **Revenue Impact**: Significant ($10K+/hour)
- **Response**: Immediate, all-hands
- **Examples**:
  - Entire service down
  - Data breach affecting customers
  - Payment processing failed

**SEV-2 (High)**:
- **Impact**: Significant degradation
- **Users Affected**: Subset of users
- **Revenue Impact**: Moderate
- **Response**: Within 15 minutes
- **Examples**:
  - Slow page loads (5s+)
  - Login failures for some users
  - Search not working

**SEV-3 (Medium)**:
- **Impact**: Minor degradation
- **Users Affected**: Small subset
- **Revenue Impact**: Low
- **Response**: Within 1 hour
- **Examples**:
  - Non-critical feature broken
  - Elevated error rates
  - Slow admin dashboard

**SEV-4 (Low)**:
- **Impact**: Cosmetic issues
- **Users Affected**: Minimal
- **Revenue Impact**: None
- **Response**: Next business day
- **Examples**:
  - Typos in UI
  - Minor visual glitches
  - Non-critical alerts

### **Severity Examples**

| Scenario | Severity | Reason |
|----------|----------|--------|
| AWS us-east-1 down | SEV-1 | Service completely down |
| Checkout 50% success rate | SEV-1 | Revenue-critical broken |
| Search returns no results | SEV-2 | Major feature broken |
| Profile image upload slow | SEV-3 | Minor feature degraded |
| Typo on settings page | SEV-4 | Cosmetic issue |

---

## Incident Lifecycle

### **1. Detection** (MTTD - Mean Time To Detect)

**Methods**:
- **Automated Monitoring**: Alerts fire
- **User Reports**: Support tickets
- **Manual Discovery**: Engineer notices
- **Social Media**: Twitter complaints

**Best**: Automated monitoring detects before users notice

**Goal**: MTTD < 5 minutes

### **2. Response** (Initial)

**Immediate Actions**:
1. **Acknowledge**: Someone takes ownership
2. **Assess**: Determine severity
3. **Communicate**: Notify stakeholders
4. **Mobilize**: Page additional people if needed

**Timeline**:
- SEV-1: < 5 minutes
- SEV-2: < 15 minutes
- SEV-3: < 1 hour

### **3. Investigation** (MTTI - Mean Time To Investigate)

**Process**:
1. **Gather Info**: Metrics, logs, traces
2. **Form Hypothesis**: What's causing it?
3. **Test Hypothesis**: Validate with data
4. **Identify Root Cause**: Find source

**Common Pitfalls**:
- Jumping to conclusions
- Fixing symptoms not root cause
- Not documenting investigation

### **4. Mitigation** (MTTR - Mean Time To Recovery)

**Strategies**:

**Rollback** (Fastest):
\`\`\`
Bad deployment → Rollback to previous version → Fixed
Timeline: 5-15 minutes
\`\`\`

**Hotfix**:
\`\`\`
Bug in production → Deploy fix → Fixed
Timeline: 30-60 minutes
\`\`\`

**Failover**:
\`\`\`
Primary database down → Failover to replica → Fixed
Timeline: 5-10 minutes
\`\`\`

**Scale Up**:
\`\`\`
Too much traffic → Scale to 3x instances → Fixed
Timeline: 5-10 minutes (if auto-scaling)
\`\`\`

**Goal**: MTTR < 30 minutes (SEV-1)

### **5. Resolution**

**Verify**:
- ✅ Metrics back to normal
- ✅ Error rates down
- ✅ Users can access service
- ✅ No ongoing impact

**Communicate**:
- Update status page
- Notify customers
- Internal all-clear

### **6. Post-Incident** (Post-Mortem)

Within 48 hours:
1. Write post-mortem
2. Identify action items
3. Assign owners
4. Track to completion

---

## Incident Roles

### **Incident Commander (IC)**

**Responsibilities**:
- Overall coordination
- Decision-making
- Communication with stakeholders
- Declaring incident resolved

**Skills**:
- Stay calm under pressure
- Clear communication
- Decisive
- Technical knowledge (helpful but not required)

**Not Responsible For**:
- Fixing the issue (delegates to engineers)
- Coding (focuses on coordination)

### **On-Call Engineer**

**Responsibilities**:
- Initial response
- Investigation
- Implementing fixes
- Technical decisions

**Skills**:
- Deep system knowledge
- Debugging
- Quick thinking

### **Communications Lead**

**Responsibilities**:
- Update status page
- Customer communications
- Internal updates
- Social media monitoring

**Skills**:
- Clear writing
- Empathy
- Quick turnaround

### **Subject Matter Expert (SME)**

**Responsibilities**:
- Provide domain expertise
- Assist investigation
- Validate hypotheses

**When Needed**:
- Complex technical issues
- Multiple system dependencies

### **Scribe**

**Responsibilities**:
- Document timeline
- Record decisions
- Track action items
- Capture investigation notes

**Why Important**:
- Post-mortem data
- Learning opportunity
- Compliance/audit trail

---

## Incident Communication

### **Internal Communication**

**Incident Channel** (Slack):
\`\`\`
#incident-2024-01-15

10:23 - Alert: API error rate spiked to 15%
10:24 - @john acknowledged, investigating
10:26 - Hypothesis: Database connection pool exhausted
10:28 - Confirmed: Pool at 100%, queries queuing
10:30 - Action: Increasing pool size from 50 to 200
10:32 - Deployed config change
10:35 - Metrics improving, error rate down to 2%
10:40 - All clear, monitoring for 30 minutes
11:10 - Incident resolved
\`\`\`

**Updates Frequency**:
- SEV-1: Every 15 minutes
- SEV-2: Every 30 minutes
- SEV-3: Hourly

### **External Communication**

**Status Page** (Example):
\`\`\`
🔴 MAJOR OUTAGE
Last updated: 10:45 AM PST

We are investigating issues with our API. Users may experience
errors when logging in or accessing their dashboards.

Timeline:
- 10:23 AM: Identified issue
- 10:30 AM: Implementing fix
- 10:40 AM: Monitoring recovery

Next update in 15 minutes.
\`\`\`

**After Resolution**:
\`\`\`
✅ RESOLVED
Last updated: 11:10 AM PST

The issue affecting API access has been resolved. All systems
are operating normally.

Summary:
- Duration: 47 minutes
- Root cause: Database connection pool exhaustion
- Impact: Users experienced intermittent errors
- Fix: Increased connection pool size

We apologize for the disruption. A detailed post-mortem will
be published within 48 hours.
\`\`\`

### **Email to Customers**

\`\`\`
Subject: Service Disruption - Resolved

Dear Customer,

On January 15, 2024, from 10:23 AM to 11:10 AM PST, you may
have experienced errors when accessing our service.

What happened:
Our API experienced high error rates due to database connection
pool exhaustion during a traffic spike.

Impact:
Approximately 15% of requests resulted in errors. You may have
seen error messages or timeouts.

Resolution:
We increased the connection pool size and validated recovery.
The service is now operating normally.

Prevention:
We are implementing auto-scaling for database connections and
improved monitoring to prevent similar issues.

We sincerely apologize for the disruption.

- The Engineering Team
\`\`\`

---

## Incident Response Tools

### **War Room**

**Physical or Virtual** (Zoom/Slack):
- Incident Commander
- Engineers
- Stakeholders
- Screen sharing
- Quick decisions

### **Incident Management Platforms**

**PagerDuty**:
- Alert routing
- On-call schedules
- Incident declaration
- Communication hub
- Post-mortem templates

**Opsgenie** (Atlassian):
- Similar to PagerDuty
- Jira integration
- Alert grouping
- Escalation policies

**incident.io**:
- Slack-native
- Auto-creates channels
- Role assignment
- Timeline tracking
- Post-mortem generation

### **Status Pages**

**Statuspage.io** (Atlassian):
- Public status page
- Incident updates
- Subscribe for notifications
- Historical uptime

**Atlassian Statuspage**:
- Component-level status
- Maintenance windows
- Email/SMS notifications

### **Runbooks**

**Where**:
- Wiki (Confluence, Notion)
- Git repository (Markdown)
- Dedicated platform (Runbook.io)

**Content**:
- Common incident scenarios
- Step-by-step resolution
- Escalation paths
- Useful queries and commands

---

## Post-Mortems (Post-Incident Reviews)

### **Purpose**

- **Learn** from failures
- **Prevent** recurrence
- **Share** knowledge
- **Improve** processes

**Not For**:
- Blame
- Punishment
- Finger-pointing

### **Blameless Post-Mortems**

**Principle**: Assume everyone acted in good faith with information available at the time

**Instead of**:
❌ "Bob deployed bad code"

**Say**:
✅ "The deployment lacked sufficient testing, and monitoring didn't catch the issue early"

**Focus**: Systems and processes, not individuals

### **Post-Mortem Template**

\`\`\`markdown
# Post-Mortem: API Outage - January 15, 2024

## Summary
On January 15, 2024, from 10:23-11:10 AM PST, our API experienced
elevated error rates (15%), impacting approximately 10,000 users.
The root cause was database connection pool exhaustion.

## Impact
- Duration: 47 minutes
- Users Affected: ~10,000 (20% of active users)
- Revenue Impact: Estimated $5,000 in lost transactions
- Services Affected: Login, Dashboard, Checkout

## Root Cause
During a traffic spike (3x normal), the database connection pool
(fixed at 50 connections) was exhausted. New requests queued and
eventually timed out.

## Timeline (PST)
- 10:15: Traffic spike begins (unknown cause)
- 10:23: Alert fires: API error rate > 5%
- 10:24: On-call engineer acknowledges
- 10:26: Hypothesis: Database connection issue
- 10:28: Confirmed: Connection pool at 100%
- 10:30: Decision: Increase pool size to 200
- 10:32: Config deployed
- 10:35: Error rate improving
- 10:40: Metrics normal, monitoring
- 11:10: Incident declared resolved

## What Went Well
✅ Alert fired within 3 minutes
✅ On-call responded immediately
✅ Root cause identified in 5 minutes
✅ Fix implemented in 10 minutes
✅ Clear communication throughout

## What Went Wrong
❌ Connection pool size was static
❌ No auto-scaling for connections
❌ Monitoring didn't alert on connection pool usage
❌ No load testing for traffic spikes
❌ Runbook incomplete

## Action Items
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| Implement dynamic connection pool sizing | @alice | Jan 22 | P0 |
| Add connection pool utilization metrics | @bob | Jan 20 | P0 |
| Alert when pool > 80% | @bob | Jan 20 | P0 |
| Load test for 5x traffic | @charlie | Jan 29 | P1 |
| Update runbook with connection pool issues | @david | Jan 18 | P2 |
| Implement auto-scaling triggers | @alice | Feb 5 | P1 |

## Lessons Learned
1. Static resource pools are dangerous during traffic spikes
2. Monitor saturation metrics (connection pool usage)
3. Load testing should include spike scenarios
4. Auto-scaling should apply to all resources, not just instances

## Prevention
- Dynamic connection pool sizing based on load
- Alerts on resource saturation
- Regular load testing including spike scenarios
- Circuit breakers to prevent cascade if pool exhausted again
\`\`\`

### **Post-Mortem Meeting**

**Attendees**:
- Incident responders
- Engineering leads
- Product managers
- Anyone interested (open to all)

**Agenda**:
1. Review incident (10 min)
2. Discuss timeline (15 min)
3. What went well (10 min)
4. What went wrong (15 min)
5. Action items (10 min)

**Duration**: 60 minutes max

**Follow-Up**: Track action items to completion

---

## Metrics

### **MTTX Metrics**

**MTTD** (Mean Time To Detect):
\`\`\`
Time from incident start to detection
Target: < 5 minutes
\`\`\`

**MTTR** (Mean Time To Recovery):
\`\`\`
Time from detection to resolution
Target: < 30 minutes (SEV-1)
\`\`\`

**MTTA** (Mean Time To Acknowledge):
\`\`\`
Time from alert to acknowledgment
Target: < 5 minutes
\`\`\`

**MTTI** (Mean Time To Investigate):
\`\`\`
Time from acknowledgment to root cause
Target: < 15 minutes
\`\`\`

### **Incident Frequency**

\`\`\`
SEV-1 incidents per month
Target: < 1 per month
\`\`\`

### **Action Item Completion**

\`\`\`
% of post-mortem action items completed on time
Target: > 90%
\`\`\`

---

## Best Practices

### **Do's**
✅ Declare incidents early (false positive OK)
✅ Clear roles and responsibilities
✅ Blameless post-mortems
✅ Document everything (timeline, decisions)
✅ Communicate frequently
✅ Follow action items to completion
✅ Practice with game days
✅ Learn from near-misses

### **Don'ts**
❌ Blame individuals
❌ Skip post-mortems
❌ Ignore action items
❌ Undercommunicate
❌ Hero culture (one person fixes everything)
❌ Repeat same incidents
❌ Hide incidents

---

## Interview Tips

### **Key Concepts**

1. **Incident**: Unplanned user-impacting event
2. **Severity**: SEV-1 (critical) to SEV-4 (low)
3. **Incident Commander**: Coordinates response
4. **MTTR**: Mean Time To Recovery
5. **Blameless Post-Mortem**: Focus on systems, not people

### **Common Questions**

**Q: Walk through how you would respond to a production outage.**
A: Acknowledge alert, assess severity, declare incident, mobilize team, investigate root cause, implement fix, verify recovery, communicate throughout, conduct post-mortem.

**Q: What's the role of an Incident Commander?**
A: Coordinate response, make decisions, communicate with stakeholders, delegate technical work. They don't fix the issue—they organize the response.

**Q: Why blameless post-mortems?**
A: Blame culture prevents learning. People hide mistakes. Blameless culture encourages transparency, identifies systemic issues, and prevents recurrence.

**Q: How do you prevent the same incident from happening twice?**
A: Post-mortem with action items, assign owners and deadlines, track to completion, follow up on implementation.

---

## Real-World Examples

### **AWS**
- **Post-Mortems**: Public summaries for major outages
- **Correction of Errors**: Detailed technical analysis
- **Lesson**: Transparency builds trust

### **Google**
- **SRE Book**: Chapters on incident management
- **Blameless Culture**: Pioneer of blameless post-mortems
- **Lesson**: Learning organizations iterate faster

### **PagerDuty**
- **Dogfooding**: Use their own product
- **MTTR**: Reduced from 4 hours to 30 minutes
- **Lesson**: Good tools and processes matter

---

## Summary

Effective incident management minimizes impact and enables learning:

1. **Severity**: SEV-1 (critical) to SEV-4 (low)
2. **Roles**: Incident Commander, On-Call, Communications Lead
3. **Lifecycle**: Detect → Respond → Investigate → Mitigate → Resolve → Post-Mortem
4. **Communication**: Internal (Slack), External (Status Page, Email)
5. **Metrics**: MTTR, MTTD, Incident Frequency
6. **Post-Mortems**: Blameless, actionable, learning-focused
7. **Prevention**: Action items tracked to completion

Key insight: Incidents are learning opportunities. Organizations that handle incidents well—quick response, clear communication, blameless culture—turn failures into improvements. Don't waste a good incident!`,
};
