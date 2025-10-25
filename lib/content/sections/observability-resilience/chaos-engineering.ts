/**
 * Chaos Engineering Section
 */

export const chaosEngineeringSection = {
  id: 'chaos-engineering',
  title: 'Chaos Engineering',
  content: `"The best time to find out how your system behaves under failure is not during an actual outage." Chaos Engineering is the discipline of experimenting on distributed systems to build confidence in their ability to withstand turbulent conditions. Rather than waiting for failures to happen in production, chaos engineering proactively injects failures to find weaknesses before they cause outages.

## What is Chaos Engineering?

**Chaos Engineering** is the practice of intentionally introducing failures into systems to test their resilience.

**Definition** (Netflix):
> "Chaos Engineering is the discipline of experimenting on a system in order to build confidence in the system's capability to withstand turbulent conditions in production."

**Core Idea**:
- Systems will fail (guaranteed)
- Better to find failures in controlled experiments
- Than to discover them during 2am outages

**Not About**:
- Breaking things randomly
- Creating chaos for chaos' sake
- Testing in production without safeguards

**About**:
- Scientific experiments
- Learning system behavior
- Building confidence in resilience

---

## Why Chaos Engineering?

### **The Problem**

**Traditional Testing**:
- Unit tests: Test individual functions
- Integration tests: Test component interactions
- Load tests: Test performance

**What\'s Missing?**: Real-world failure scenarios
- Network partitions
- Hardware failures
- Cascading failures
- Unexpected interactions

**Example Failures Missed by Traditional Testing**:
- Service A fails → Service B exhausts connections → System crashes
- Network latency spikes → Timeouts cascade → Everything fails
- Database replica lag → Read-after-write inconsistency → Data corruption

### **Benefits**

**1. Find Weaknesses Before Users Do**
\`\`\`
Chaos Experiment: Kill random instances
Discovery: No health checks configured!
Fix: Add health checks + auto-healing
Result: Actual instance failures don't affect users
\`\`\`

**2. Validate Resilience Mechanisms**
\`\`\`
Chaos Experiment: Increase latency to database
Expected: Circuit breaker opens, fallback to cache
Actual: Timeouts cascade, service crashes
Fix: Add circuit breaker configuration
\`\`\`

**3. Practice Incident Response**
- Team gets practice handling failures
- Runbooks are tested and improved
- On-call engineers gain confidence

**4. Build Organizational Confidence**
- Prove system can handle failures
- Reduce fear of deployments
- Enable faster iteration

---

## Principles of Chaos Engineering

### **1. Build a Hypothesis Around Steady State**

Define normal behavior as measurable output

**Example**:
\`\`\`
Hypothesis: "Even if 10% of instances fail, the system will:
  - Maintain 99.9% success rate
  - Keep p99 latency < 500ms
  - Serve 10,000 req/s throughput"
\`\`\`

### **2. Vary Real-World Events**

Simulate realistic failures

**Examples**:
- Instance termination
- Network latency increase
- Dependency failures
- Resource exhaustion
- Time drift

### **3. Run Experiments in Production**

Production is different from staging:
- Real traffic patterns
- Real data volumes
- Real dependencies
- Real failure modes

**Safeguards**:
- Start small (1% of traffic)
- Have kill switch
- Monitor closely
- Business hours only (initially)

### **4. Automate Experiments**

Manual chaos doesn't scale

**Automated Chaos**:
- Run continuously
- Part of deployment pipeline
- Catch regressions automatically

### **5. Minimize Blast Radius**

Start small, expand gradually

\`\`\`
Phase 1: Single instance in canary environment
Phase 2: 1% of production traffic
Phase 3: 10% of production traffic
Phase 4: Full production
\`\`\`

---

## Types of Chaos Experiments

### **1. Instance Termination**

Kill random instances

**Purpose**: Test auto-scaling and health checks

**Experiment**:
\`\`\`
1. Identify: Service with 10 instances
2. Hypothesis: System handles loss of 1 instance
3. Action: Randomly terminate 1 instance
4. Observe: Does traffic route away? Is instance replaced?
5. Verify: Success rate and latency unchanged
\`\`\`

**Common Findings**:
- Health checks not configured
- No auto-scaling
- Stateful instances (bad!)
- Sessions lost

### **2. Network Latency**

Inject artificial latency

**Purpose**: Test timeouts and circuit breakers

**Experiment**:
\`\`\`
1. Identify: API calls to payment service
2. Hypothesis: Circuit breaker opens at 3s latency
3. Action: Add 5s latency to payment service
4. Observe: Do circuit breakers activate?
5. Verify: Graceful degradation, no cascading failures
\`\`\`

**Common Findings**:
- No timeouts configured
- Timeouts too long (30s)
- No circuit breakers
- Blocking calls freeze everything

### **3. Network Partition**

Simulate network splits

**Purpose**: Test partition tolerance (CAP theorem)

**Experiment**:
\`\`\`
1. Identify: Multi-region deployment
2. Hypothesis: Service remains available in each region
3. Action: Block network between regions
4. Observe: Does each region function independently?
5. Verify: Eventually consistent after heal
\`\`\`

**Common Findings**:
- Services require cross-region communication
- Split-brain scenarios
- Data inconsistency after partition heals

### **4. Resource Exhaustion**

Consume CPU, memory, disk

**Purpose**: Test resource limits and graceful degradation

**CPU Exhaustion**:
\`\`\`
Action: Consume 100% CPU on instance
Expected: Load balancer routes traffic away
Actual: Instance keeps receiving traffic, all requests slow
Fix: Health checks should monitor CPU
\`\`\`

**Memory Exhaustion**:
\`\`\`
Action: Consume all available memory
Expected: OOM killer restarts container
Actual: Container crashes, not restarted
Fix: Configure restart policies
\`\`\`

**Disk Full**:
\`\`\`
Action: Fill disk to 100%
Expected: Service handles gracefully
Actual: Service crashes on write
Fix: Monitor disk usage, add alerts
\`\`\`

### **5. Dependency Failures**

Simulate downstream service failures

**Purpose**: Test fallback mechanisms

**Database Down**:
\`\`\`
Action: Stop database
Expected: Read from cache, write to queue
Actual: All requests fail
Fix: Add caching layer
\`\`\`

**External API Down**:
\`\`\`
Action: Return errors from payment API
Expected: Retry with exponential backoff
Actual: Infinite retries overwhelm system
Fix: Implement circuit breaker
\`\`\`

### **6. Time Travel**

Simulate clock drift or time zone changes

**Purpose**: Test time-dependent logic

**Experiment**:
\`\`\`
Action: Set instance clock forward 24 hours
Expected: System handles gracefully
Actual: Certificates expire, tokens invalid
Fix: NTP synchronization, handle drift
\`\`\`

### **7. DNS Failures**

Simulate DNS resolution issues

**Purpose**: Test DNS caching and failover

**Experiment**:
\`\`\`
Action: Block DNS queries
Expected: Service uses cached DNS entries
Actual: All requests fail immediately
Fix: Implement DNS caching
\`\`\`

---

## Chaos Engineering Tools

### **Chaos Monkey** (Netflix)

Randomly terminates instances

**Features**:
- Runs during business hours
- Terminates instances randomly
- Respects opt-out labels
- Integrates with AWS, GCP

**Configuration**:
\`\`\`yaml
enabled: true
schedule: "0 9-17 * * MON-FRI" # Business hours only
minTimeBetweenKills: 300 # 5 minutes minimum
terminationProbability: 0.1 # 10% chance
\`\`\`

### **Gremlin**

Commercial chaos engineering platform

**Features**:
- CPU, memory, disk attacks
- Network attacks (latency, packet loss)
- State attacks (shutdown, time travel)
- Kubernetes integration
- Blast radius control

**Example Attack**:
\`\`\`
Attack: CPU
Target: service=api, environment=production
Intensity: 100% CPU
Duration: 5 minutes
Blast Radius: 10% of instances
\`\`\`

### **Chaos Toolkit**

Open-source chaos engineering toolkit

**Experiment Definition**:
\`\`\`yaml
version: 1.0.0
title: "API can handle database latency"
description: "Verify circuit breaker opens on high DB latency"

steady-state-hypothesis:
  title: "API is healthy"
  probes:
    - type: probe
      name: "success-rate"
      provider:
        type: http
        url: "http://api/health"
        expected_status: 200

method:
  - type: action
    name: "inject-latency"
    provider:
      type: python
      module: chaosgcp
      func: inject_latency
      arguments:
        service: "database"
        latency_ms: 3000
        duration_seconds: 300

rollbacks:
  - type: action
    name: "remove-latency"
    provider:
      type: python
      module: chaosgcp
      func: remove_latency
\`\`\`

### **Litmus** (Kubernetes)

Chaos engineering for Kubernetes

**Features**:
- Pod deletion
- Container kill
- Network chaos
- I/O chaos
- CRD-based experiments

**Example**:
\`\`\`yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: api-chaos
spec:
  appinfo:
    appns: production
    applabel: "app=api"
  experiments:
    - name: pod-delete
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: "60"
            - name: CHAOS_INTERVAL
              value: "10"
\`\`\`

### **Pumba** (Docker)

Network and stress testing for Docker

**Example**:
\`\`\`bash
# Add 3s latency to container
pumba netem --duration 5m delay --time 3000 my-container

# Packet loss
pumba netem --duration 5m loss --percent 30 my-container

# Kill container
pumba kill --signal SIGKILL my-container
\`\`\`

### **Chaos Mesh** (Kubernetes)

Cloud-native chaos engineering platform

**Features**:
- Network chaos (partition, latency, corruption)
- Pod chaos (kill, failure)
- Stress chaos (CPU, memory)
- I/O chaos
- Time chaos

---

## Implementing Chaos Engineering

### **Phase 1: Foundation**

**Week 1-2**: Set up monitoring
- Dashboards for key metrics
- Alerts for SLO violations
- Baseline performance data

**Week 3-4**: Document steady state
- Define normal behavior
- Establish success criteria
- Create runbooks

### **Phase 2: First Experiments**

**Start Simple**:
\`\`\`
Experiment 1: Terminate single instance
Environment: Staging
Blast Radius: 1 instance
Observer: Entire team watching
\`\`\`

**Iterate**:
1. Run experiment
2. Observe results
3. Fix issues
4. Document learnings
5. Repeat

### **Phase 3: Production**

**Gradual Rollout**:
\`\`\`
Week 1: Canary environment (1% traffic)
Week 2: Single AZ (10% traffic)
Week 3: Multiple AZs (50% traffic)
Week 4: Full production
\`\`\`

### **Phase 4: Automation**

**Continuous Chaos**:
- Automated experiments
- Part of CI/CD pipeline
- Chaos as a service
- Self-healing validation

---

## Game Days

### **What is a Game Day?**

**Game Day**: Scheduled chaos engineering exercise with entire team

**Purpose**:
- Practice incident response
- Test runbooks
- Build team confidence
- Find system weaknesses

### **Planning**

**2 Weeks Before**:
- Choose scenarios
- Notify team
- Prepare runbooks
- Set up monitoring

**1 Week Before**:
- Review scenarios
- Assign roles
- Test tools
- Dry run

**Day Of**:
- Brief team
- Run experiments
- Document observations
- Debrief

### **Example Game Day**

**Scenario**: Database Failure

**9:00 AM**: Kickoff meeting
- Review objectives
- Assign roles (incident commander, engineers, observers)

**9:30 AM**: Inject failure
- Primary database goes down
- Team doesn't know what failure is

**9:35 AM**: Team responds
- Detect issue
- Follow runbook
- Failover to replica

**10:00 AM**: Restore
- Validate system health
- Confirm data consistency

**10:30 AM**: Debrief
- What went well?
- What went wrong?
- Action items

### **Common Scenarios**

1. **Database Failure**: Primary DB down
2. **Region Outage**: Entire AWS region unavailable
3. **DDoS Attack**: 10x normal traffic
4. **Deployment Failure**: Bad code pushed to production
5. **Dependency Failure**: Payment gateway down

---

## Measuring Success

### **Metrics**

**Before Chaos Engineering**:
- Mean Time To Recovery (MTTR): 4 hours
- Outage frequency: 2 per month
- Incident severity: High

**After 6 Months**:
- MTTR: 30 minutes (8x improvement)
- Outage frequency: 0.5 per month (4x improvement)
- Incident severity: Low

**Experiment Metrics**:
- Experiments run per week
- Issues found per experiment
- Time to fix issues
- Team confidence scores

---

## Best Practices

### **Do's**
✅ Start in non-production environments
✅ Get organizational buy-in
✅ Define hypothesis before experiment
✅ Have rollback plan
✅ Monitor closely
✅ Document everything
✅ Start with small blast radius
✅ Gradually expand scope
✅ Automate experiments
✅ Make it continuous

### **Don'ts**
❌ Surprise experiments (notify team)
❌ Run without monitoring
❌ Skip hypothesis
❌ Ignore findings
❌ Chaos for chaos' sake
❌ Run during peak traffic (initially)
❌ Skip rollback plan
❌ Blame individuals for failures found

---

## Interview Tips

### **Key Concepts**

1. **Chaos Engineering**: Deliberately inject failures to test resilience
2. **Hypothesis**: Define expected behavior before experiment
3. **Blast Radius**: Start small, expand gradually
4. **Steady State**: Measurable normal system behavior
5. **Production**: Run in prod for realistic results

### **Common Questions**

**Q: What is chaos engineering?**
A: Deliberately introducing failures into systems to test resilience, find weaknesses before they cause outages, and build confidence in system behavior.

**Q: Why run chaos experiments in production?**
A: Production has unique characteristics (traffic patterns, data, dependencies) that can't be replicated in staging. Use safeguards like small blast radius and monitoring.

**Q: How do you convince management to do chaos engineering?**
A: Frame as risk reduction (find issues before customers do), show cost of outages, start small in non-prod, demonstrate value with metrics.

**Q: What\'s the difference between chaos engineering and testing?**
A: Testing validates known scenarios. Chaos engineering discovers unknown failure modes and emergent behaviors in complex systems.

---

## Real-World Examples

### **Netflix**
- **Chaos Monkey**: Started it all (2011)
- **Simian Army**: Suite of chaos tools
- **Scale**: Runs continuously in production
- **Result**: Survived AWS outages gracefully

### **Amazon**
- **Game Days**: Monthly practice
- **Region Failures**: Test multi-region failover
- **Result**: Improved MTTR by 10x

### **Google**
- **DiRT** (Disaster Recovery Testing)
- **Annual Event**: Company-wide chaos
- **Result**: Found critical issues before production impact

---

## Summary

Chaos Engineering builds resilient systems through controlled experiments:

1. **Purpose**: Find weaknesses before customers do
2. **Approach**: Scientific experiments with hypotheses
3. **Scope**: Start small (staging, 1% traffic), expand gradually
4. **Types**: Instance failures, network issues, resource exhaustion
5. **Tools**: Chaos Monkey, Gremlin, Chaos Toolkit, Litmus
6. **Practice**: Game Days for team preparation
7. **Automation**: Continuous chaos in CI/CD pipeline

Key insight: The goal isn't breaking things—it's learning how your system behaves under failure and improving resilience. Start simple, document findings, fix issues, and iterate. Chaos Engineering transforms "I hope our system is resilient" into "I know our system is resilient because we proved it."`,
};
