/**
 * Alerting Strategies Section
 */

export const alertingStrategiesSection = {
  id: 'alerting-strategies',
  title: 'Alerting Strategies',
  content: `Alerting is the bridge between observability and action. Good alerts wake you up for real problems. Bad alerts train you to ignore them (alert fatigue). This section covers how to design effective alerting systems that maximize signal and minimize noise.

## What is Alerting?

**Alerting** is the practice of automatically notifying on-call engineers when something requires human attention.

**Purpose**:
- Detect problems before users do
- Reduce Mean Time To Detection (MTTD)
- Reduce Mean Time To Resolution (MTTR)
- Ensure SLO compliance
- Enable proactive response

**Not for**:
- FYI information (use dashboards)
- Debugging (use logs)
- Metrics collection (use monitoring)

**Golden Rule**: Every alert must be actionable and require immediate human intervention

---

## Alert Fatigue

### **The Problem**

**Alert Fatigue**: When engineers receive so many alerts that they ignore them

**Consequences**:
- Real alerts missed in noise
- Delayed incident response
- Team burnout
- "Boy who cried wolf" syndrome

**Statistics**:
- Average: 15-20 alerts/day per engineer
- High-fatigue teams: 50+ alerts/day
- Result: Critical alerts missed for hours

### **Causes**1. **Too Many Alerts**
   - Alert on everything "just in case"
   - Low thresholds
   - No aggregation

2. **Non-Actionable Alerts**
   - "FYI" notifications
   - Can't do anything about it
   - Requires no immediate action

3. **Duplicate Alerts**
   - Same issue, multiple alerts
   - Cascading failures
   - No grouping

4. **False Positives**
   - Alert fires but no real problem
   - Static thresholds on dynamic metrics
   - Flapping alerts

### **Solution**

✅ **Alert on symptoms, not causes**
✅ **Every alert = page = immediate action**
✅ **Aggregate related alerts**
✅ **Use anomaly detection**
✅ **Tune thresholds based on data**
✅ **Regular alert audits**

---

## Alerting Philosophy

### **Google SRE Principles**

**1. Pages Should Be Rare**
- Only for urgent, user-impacting issues
- Target: < 2 pages/week per engineer

**2. Every Page Should Be Actionable**
- Clear what's wrong
- Clear what to do
- Has runbook

**3. Alerts Should Have Appropriate Urgency**
- Urgent: Page immediately
- Important: Ticket for working hours
- Info: Dashboard only

**4. Alerts Should Be Novel**
- No repeated alerts for same issue
- Suppress duplicates

### **Alert Tiers**

**P0 (Critical)**: Immediate page
- Service down
- Data loss risk
- Security breach
- Revenue impact

**P1 (High)**: Page if during business hours
- Degraded performance
- Error rate elevated
- Approaching resource limits

**P2 (Medium)**: Create ticket
- Non-critical service impact
- Potential future issue
- Maintenance required

**P3 (Low)**: Dashboard/log only
- Informational
- Metrics trending
- Non-urgent updates

---

## Alert on Symptoms, Not Causes

### **Symptom-Based Alerts** ✅

Alert on what users experience

**Example**:
\`\`\`
ALERT: API latency p99 > 1s
What: Users experiencing slow responses
Impact: Poor user experience
Action: Investigate and fix
\`\`\`

### **Cause-Based Alerts** ❌

Alert on infrastructure metrics

**Example**:
\`\`\`
ALERT: CPU > 80%
What: High CPU usage
Impact: Maybe? Maybe not?
Action: Unclear - is this a problem?
\`\`\`

**Why Symptom > Cause**:
- High CPU might be fine (batch job)
- Low CPU can still mean user issues (blocked on I/O)
- Users care about their experience, not your CPU

**Exception**: Alert on causes that predict symptoms
- Disk 95% full → Will cause service failure
- Memory leak detected → Will cause crash

---

## What to Alert On

### **Service Level Objectives (SLOs)**

**Best Practice**: Alert when at risk of breaching SLO

**Example**:
\`\`\`
SLO: 99.9% availability (43.2 minutes/month downtime budget)

Alert: Error budget 50% consumed in 1 week
Action: Slow down releases, focus on stability
\`\`\`

**Why**: Proactive - fix before SLO breach

### **The Four Golden Signals** (Google SRE)

**1. Latency**
\`\`\`
ALERT: API latency p99 > 1s for 5 minutes
\`\`\`

**2. Traffic**
\`\`\`
ALERT: Request rate dropped 50% (possible outage)
ALERT: Request rate spiked 10x (possible DDoS)
\`\`\`

**3. Errors**
\`\`\`
ALERT: Error rate > 1% for 5 minutes
\`\`\`

**4. Saturation**
\`\`\`
ALERT: Database connections > 90% of pool
ALERT: Disk usage > 90%
\`\`\`

### **RED Method** (for services)

**Rate**: Unusual request rate changes
**Errors**: Error rate above threshold
**Duration**: Latency above threshold

### **USE Method** (for resources)

**Utilization**: > 80% sustained
**Saturation**: Queue depth increasing
**Errors**: Hardware errors

---

## Alert Thresholds

### **Static Thresholds**

Simple, fixed values

**Example**:
\`\`\`
ALERT: error_rate > 1%
ALERT: latency_p99 > 1s
ALERT: cpu_usage > 80%
\`\`\`

**Pros**: Simple, easy to understand
**Cons**: Doesn't adapt to traffic patterns

### **Dynamic Thresholds** (Anomaly Detection)

Thresholds based on historical patterns

**Example**:
\`\`\`
Normal: 100-200 req/s during business hours
Current: 500 req/s
ALERT: Request rate 3x higher than normal
\`\`\`

**Pros**: Adapts to patterns, fewer false positives
**Cons**: Complex, requires ML/statistics

**When to Use**:
- Metrics with daily/weekly patterns
- Gradual degradations
- Noisy metrics

### **Rate of Change**

Alert on rapid changes

**Example**:
\`\`\`
ALERT: Error rate increased 10x in 5 minutes
(Even if absolute rate is low)
\`\`\`

**Use Cases**:
- New deployments
- Sudden traffic changes
- Cascading failures

### **Multivariate Alerts**

Combine multiple conditions

**Example**:
\`\`\`
ALERT:
  error_rate > 5% AND
  latency_p99 > 2s AND
  duration > 5 minutes
\`\`\`

**Benefit**: Reduce false positives

---

## Alert Duration and Frequency

### **Duration**

How long condition must be true before alerting

**Too Short** (1 minute):
- Many false positives
- Transient spikes
- Alert fatigue

**Too Long** (30 minutes):
- Slow to detect
- Users already impacted
- Delayed response

**Best Practice**: 3-5 minutes for most alerts
- Filters transient blips
- Fast enough to catch real issues

**Example**:
\`\`\`
# Bad: Alerts on single data point
ALERT: cpu_usage > 80%

# Good: Sustained over time
ALERT: cpu_usage > 80% for 5 minutes
\`\`\`

### **Re-Alert Frequency**

How often to re-send alert if condition persists

**Options**:
1. **Once**: Alert once, manual close
2. **Recurring**: Every N minutes while firing
3. **Escalating**: Increase urgency over time

**Best Practice**: Alert once + auto-resolve
- Reduces noise
- Engineers acknowledge and investigate
- Auto-resolve when condition clears

---

## Alert Grouping and Deduplication

### **Problem**

Single issue causes 100 alerts:
\`\`\`
Database down → 50 services alert → 200 alerts fire
\`\`\`

### **Solutions**

**1. Alert Grouping**
Group related alerts together
\`\`\`
Alert Group: Database Connectivity
├─ Service A: Cannot connect to DB
├─ Service B: Cannot connect to DB
├─ Service C: Cannot connect to DB
Summary: 47 services affected
\`\`\`

**2. Alert Deduplication**
Only alert once for same issue
\`\`\`
✓ First occurrence: Send alert
✗ Repeat occurrences: Suppress
\`\`\`

**3. Alert Suppression**
Suppress downstream alerts when upstream fails
\`\`\`
IF load_balancer_down:
  SUPPRESS all_backend_service_alerts
\`\`\`

**4. Alert Dependencies**
Model service dependencies
\`\`\`
Database → API → Frontend

IF database_down:
  SUPPRESS api_alerts, frontend_alerts
  ALERT: Root cause: Database
\`\`\`

---

## Alert Context

### **What to Include**

**1. What\'s Wrong**
\`\`\`
ERROR: API latency p99 > 1s
Current: 2.3s
Threshold: 1s
\`\`\`

**2. Impact**
\`\`\`
Affected: 1000 users/minute
Service: checkout-api
Region: us-east-1
\`\`\`

**3. When**
\`\`\`
Started: 2024-01-15 10:23 UTC (3 minutes ago)
Duration: 3m 42s
\`\`\`

**4. Why (if known)**
\`\`\`
Possible cause: Database query slow
Related: Database CPU 95%
\`\`\`

**5. What to Do**
\`\`\`
Runbook: https://wiki/runbooks/api-latency
Dashboard: https://grafana/dashboard/api
Logs: https://kibana/logs?trace=abc-123
\`\`\`

**6. Useful Links**
- Grafana dashboard
- Log query
- Recent deployments
- Runbook

### **Example Alert**

\`\`\`
🚨 CRITICAL: API Error Rate High

What: checkout-api error rate > 5%
Current: 12% (baseline: 0.5%)
Impact: 200 users/min experiencing errors
Duration: 5 minutes
Region: us-east-1

Recent Changes:
  ✓ Deployed v2.5.0 8 minutes ago

Links:
  📊 Dashboard: https://grafana/...
  📝 Logs: https://kibana/...
  📚 Runbook: https://wiki/...
  🔄 Rollback: kubectl rollout undo...

Action: Investigate or rollback deployment
\`\`\`

---

## Alert Channels

### **Channel Selection**

**PagerDuty / Opsgenie** (Urgent):
- Critical alerts
- On-call rotation
- Escalation policies

**Slack / Teams** (Important):
- Team notifications
- Non-urgent alerts
- Status updates

**Email** (Low Priority):
- Daily summaries
- Weekly reports
- Audit logs

**SMS** (Critical):
- When PagerDuty fails
- Backup channel

**Phone Call** (Critical):
- Escalation after N minutes
- Major incidents

### **Routing Rules**

\`\`\`
IF severity == "critical":
  → PagerDuty → On-call engineer

IF severity == "warning":
  → Slack #alerts channel

IF time == "business_hours" AND severity == "medium":
  → Create ticket

IF time == "off_hours" AND severity == "medium":
  → Queue for tomorrow
\`\`\`

---

## On-Call and Escalation

### **On-Call Rotation**

**Best Practices**:
- 1 week rotations (balance coverage and burden)
- Clear handoff process
- Compensation (time off or pay)
- Maximum 2-3 pages per night
- Backup on-call for escalation

### **Escalation Policies**

**Example**:
\`\`\`
Alert fires:
├─ 0 min: Page primary on-call
├─ 5 min: No ack → Page backup on-call
├─ 10 min: No ack → Page manager
└─ 15 min: No ack → Page CTO
\`\`\`

### **Follow-the-Sun**

For global teams:
- APAC team: 8am-8pm Asia
- EMEA team: 8am-8pm Europe
- Americas team: 8am-8pm Americas

**Benefit**: Reduce night pages

---

## Alert Tuning

### **Metrics to Track**

**Alert Volume**:
- Total alerts per day
- Alerts per engineer
- Alert by severity

**Alert Quality**:
- True positives vs false positives
- Time to acknowledge
- Time to resolve
- Alert-to-incident ratio

**Target Metrics**:
- < 5 pages/week per engineer
- > 80% alerts are actionable
- < 10% false positive rate
- < 5 minute MTTA (time to acknowledge)

### **Regular Alert Review**

**Weekly**:
- Review all fired alerts
- Identify false positives
- Tune thresholds
- Update runbooks

**Monthly**:
- Alert effectiveness analysis
- Remove noisy alerts
- Add missing alerts
- Review on-call load

**Questions to Ask**:
- Was action taken for this alert?
- Could user have noticed before alert?
- Was alert clear and actionable?
- Did runbook help?

---

## Runbooks

### **What is a Runbook?**

Step-by-step guide for responding to specific alerts

**Structure**:
1. **Alert Description**: What this alert means
2. **Impact**: Who/what is affected
3. **Diagnosis**: How to investigate
4. **Resolution**: How to fix
5. **Prevention**: Long-term fix

### **Example Runbook**

\`\`\`markdown
# API High Latency Runbook

## Alert
API latency p99 > 1s for 5+ minutes

## Impact
- Users experience slow page loads
- Possible checkout failures
- Revenue impact

## Diagnosis

### 1. Check Dashboard
- [API Dashboard](https://grafana/...)
- Look for latency spike timing
- Check request rate (traffic spike?)

### 2. Check Traces
- [Jaeger](https://jaeger/...)
- Find slow trace examples
- Identify bottleneck span

### 3. Check Database
- Database CPU/connections
- Slow query log
- Lock contention

## Common Causes

### Database Slow Query
**Symptoms**: DB span > 500ms in traces
**Fix**: 
  1. Identify slow query
  2. Add index (if missing)
  3. Optimize query
  4. Deploy fix

### Traffic Spike
**Symptoms**: Request rate 10x normal
**Fix**:
  1. Scale up: \`kubectl scale deployment api --replicas=20\`
  2. Investigate traffic source
  3. Consider rate limiting

### Memory Leak
**Symptoms**: Memory increasing, GC pauses
**Fix**:
  1. Restart affected pods: \`kubectl rollout restart deployment api\`
  2. Investigate memory leak
  3. Deploy fix

## Escalation
If unresolved in 15 minutes:
- Escalate to @backend-leads
- Consider rollback

## Prevention
- Add database query performance tests
- Set up auto-scaling
- Add rate limiting
\`\`\`

---

## Advanced Alerting Patterns

### **Composite Alerts**

Alert on combinations of metrics

**Example**:
\`\`\`
ALERT: High Error Rate AND Recent Deployment
Condition:
  error_rate > 5% AND
  deployment_time < 10 minutes ago
Action: Likely bad deployment, consider rollback
\`\`\`

### **Forecasting Alerts**

Predict future problems

**Example**:
\`\`\`
ALERT: Disk will be full in 4 hours
Current: 85% full
Growth rate: 5%/hour
Projected: 100% at 2pm today
\`\`\`

### **Ratio Alerts**

Alert on ratios, not absolutes

**Example**:
\`\`\`
ALERT: Error rate > 1% (instead of errors > 100/min)
Why: Adapts to traffic
\`\`\`

---

## Best Practices

### **Do's**
✅ Alert on symptoms (user impact)
✅ Include context and runbooks
✅ Set appropriate thresholds (3-5 min duration)
✅ Group and deduplicate alerts
✅ Regular alert review and tuning
✅ Track alert quality metrics
✅ Escalation policies
✅ Make alerts actionable

### **Don'ts**
❌ Alert on everything "just in case"
❌ Non-actionable alerts
❌ Alerts without runbooks
❌ Ignore alert fatigue
❌ Static thresholds on dynamic metrics
❌ Duplicate alerts for same issue
❌ Alert on causes instead of symptoms

---

## Interview Tips

### **Key Concepts**1. **Alert Fatigue**: Too many alerts → ignored alerts
2. **Symptom vs Cause**: Alert on user impact, not infrastructure
3. **Actionable**: Every alert must require human action
4. **Four Golden Signals**: Latency, traffic, errors, saturation
5. **SLO-Based**: Alert when at risk of breaching SLO

### **Common Questions**

**Q: How would you design alerts for a new service?**
A: Start with Four Golden Signals (latency, errors, traffic, saturation), use SLO-based alerting, include context and runbooks, tune based on feedback.

**Q: How do you prevent alert fatigue?**
A: Alert only on actionable issues, group related alerts, tune thresholds, regular alert review, use anomaly detection.

**Q: When should you page vs create a ticket?**
A: Page for urgent, user-impacting issues requiring immediate action. Ticket for important but not urgent issues that can wait until business hours.

---

## Real-World Examples

### **Google**
- **Target**: < 2 pages/week per engineer
- **Practice**: Aggressive alert tuning
- **Result**: Highly actionable, low-noise alerts

### **Netflix**
- **Practice**: Alert on customer impact (streaming failures)
- **Tool**: Internal "Resilience Engineering" team
- **Result**: 99.99% uptime

### **PagerDuty** (dogfooding)
- **Practice**: Use own product for alerting
- **Metric**: 95% of alerts are actionable
- **Result**: Fast incident response

---

## Summary

Effective alerting is critical for system reliability:

1. **Alert on Symptoms**: User impact, not infrastructure
2. **Make Actionable**: Every alert requires immediate action
3. **Prevent Fatigue**: < 5 pages/week per engineer
4. **Add Context**: Runbooks, dashboards, logs
5. **Group and Deduplicate**: Single issue shouldn't cause 100 alerts
6. **Tune Regularly**: Review and improve based on data

Good alerting enables fast incident response. Bad alerting causes burnout and missed critical issues. Treat alert design with the same rigor as production code.`,
};
