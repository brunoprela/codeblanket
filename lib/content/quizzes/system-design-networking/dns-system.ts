/**
 * Quiz questions for DNS (Domain Name System) section
 */

export const dnssystemQuiz = [
  {
    id: 'dns-global-design',
    question:
      "You're designing a globally distributed web application that serves users in North America, Europe, and Asia. Explain how you would use DNS to optimize latency and provide automatic failover. Include specific DNS features, record types, and discuss trade-offs between different approaches.",
    sampleAnswer: `**Architecture Overview**:

I would implement a multi-layered DNS strategy using GeoDNS, health checking, and automatic failover to optimize global performance and reliability.

**1. DNS Provider Selection**

*Choice: AWS Route 53*
- GeoDNS (geolocation routing)
- Health checking with automatic failover
- Low latency (Anycast network)
- 100% SLA
- Integration with AWS services

*Alternative: Multi-provider setup*
- Primary: Route53
- Secondary: Cloudflare
- Provides redundancy if one DNS provider has outage

**2. Global Infrastructure Setup**

Deploy application in 3 regions:

\`\`\`
US-East (Virginia):
  - Primary: 1.2.3.4
  - Secondary: 1.2.3.5

EU-West (Ireland):
  - Primary: 5.6.7.8
  - Secondary: 5.6.7.9

AP-Southeast (Singapore):
  - Primary: 9.10.11.12
  - Secondary: 9.10.11.13
\`\`\`

**3. DNS Record Structure**

*Root domain record (with geolocation)*:
\`\`\`
# North America
example.com  IN  A  1.2.3.4  (US-East primary)
  - Geolocation: North America
  - TTL: 300 seconds (5 minutes)
  - Health check: HTTPS /health endpoint

# Fallback if US-East fails
example.com  IN  A  1.2.3.5  (US-East secondary)
  - Health check: HTTPS /health endpoint

# Europe
example.com  IN  A  5.6.7.8  (EU-West primary)
  - Geolocation: Europe
  - TTL: 300 seconds

# Asia
example.com  IN  A  9.10.11.12  (AP-Southeast primary)
  - Geolocation: Asia-Pacific
  - TTL: 300 seconds

# Global default (if no geo match)
example.com  IN  A  1.2.3.4  (US-East, largest capacity)
\`\`\`

**4. Health Checking Configuration**

\`\`\`javascript
// Route53 Health Check
{
  type: 'HTTPS',
  resourcePath: '/health',
  port: 443,
  requestInterval: 30, // seconds
  failureThreshold: 3, // 3 consecutive failures
  measureLatency: true,
  regions: ['us-east-1', 'eu-west-1', 'ap-southeast-1',]
}

// Health endpoint response
{
  "status": "healthy",
  "timestamp": 1640000000,
  "checks": {
    "database": "ok",
    "cache": "ok",
    "api": "ok"
  }
}
\`\`\`

**5. Failover Logic**

*Scenario: US-East primary fails*

\`\`\`
Time 0:00 - US-East health check fails (1st failure)
Time 0:30 - US-East health check fails (2nd failure)
Time 1:00 - US-East health check fails (3rd failure)
→ Route53 marks unhealthy
→ DNS switches to US-East secondary (1.2.3.5)

Time 1:00 - 6:00 - Users gradually switch (TTL = 5 minutes)
Time 6:00 - All users on secondary

If secondary also fails:
→ Route53 switches to nearest healthy region (EU or Asia)
\`\`\`

**6. TTL Strategy**

*Normal operations: 300 seconds (5 minutes)*
- Short enough for reasonably fast failover
- Long enough to reduce DNS query load
- 95% of queries answered from cache

*During planned maintenance: 60 seconds*
- Lower TTL 1 hour before maintenance
- Allows quick traffic shifting
- Raise back to 300 seconds after

*Trade-off analysis*:

| TTL | Failover Time | DNS Queries/day (1M users) | Benefit |
|-----|---------------|----------------------------|---------|
| 60s | 1-3 minutes | 1.44 billion | Fast failover |
| 300s | 5-10 minutes | 288 million | Balanced |
| 3600s | 1-2 hours | 24 million | Low query load |

**Chosen**: 300 seconds - optimal balance.

**7. Latency Optimization**

*GeoDNS routing*:
- User in New York → US-East (latency: 20ms)
- User in London → EU-West (latency: 15ms)
- User in Tokyo → AP-Southeast (latency: 10ms)

*Without GeoDNS*:
- All users → US-East
- User in London → US-East (latency: 150ms)
- User in Tokyo → US-East (latency: 250ms)

**Expected improvement**: 80-90% latency reduction for international users.

*Latency-based routing (alternative)*:
- Route53 measures actual latency from user to each region
- Routes to lowest-latency endpoint
- More accurate than pure geo-routing
- Trade-off: Requires more complex setup

**8. Monitoring & Alerting**

\`\`\`javascript
// CloudWatch Alarms
{
  metric: 'HealthCheckStatus',
  threshold: 1, // Unhealthy
  evaluationPeriods: 1,
  action: [
    'SNS notification to on-call',
    'PagerDuty alert',
    'Slack webhook'
  ]
}

// DNS query metrics
{
  metric: 'QueryCount',
  dimension: 'Region',
  period: '5 minutes',
  statistic: 'Sum'
}

// Health check latency
{
  metric: 'HealthCheckLatency',
  threshold: 1000, // ms
  action: 'Alert if region degraded'
}
\`\`\`

**9. Cost Analysis**

*Route53 costs*:
- Hosted zone: $0.50/month
- Standard queries: $0.40 per million
- Geo queries: $0.70 per million
- Health checks: $0.50 per endpoint per month

*Calculation (1M users, 300s TTL)*:
- Queries per day: 288M
- Queries per month: 8.6B
- Cost: 8,600 × $0.70 = $6,020/month

*Optimization*:
- Increase TTL to 600s → Halves cost to $3,010/month
- Trade-off: Slower failover (10-15 minutes)

**10. Advanced Features**

*Weighted routing for canary deployments*:
\`\`\`
example.com:
  - 95% → Stable version (1.2.3.4)
  - 5% → Canary version (1.2.3.100)

Monitor canary error rates
If errors < 0.1% → Increase to 50%, then 100%
If errors > 1% → Roll back to 0%
\`\`\`

*Traffic flow with complex routing*:
\`\`\`
Start → Geolocation check
  ├─ North America → Latency-based (US-East vs US-West)
  ├─ Europe → Latency-based (EU-West vs EU-Central)
  └─ Asia → Latency-based (AP-Southeast vs AP-Northeast)

Each region → Weighted routing (canary)
  ├─ 95% → Primary
  └─ 5% → Canary

Each endpoint → Failover routing
  ├─ Primary
  └─ Secondary
\`\`\`

**Trade-offs Discussion**:

**1. GeoDNS vs Latency-Based Routing**

*GeoDNS (chosen)*:
- Pro: Simpler, predictable
- Con: Less accurate (user in London routed to EU might have better latency to US-East if EU degraded)

*Latency-based*:
- Pro: More accurate, routes to actually-fastest endpoint
- Con: More complex, requires continuous latency measurement

**2. Active-Active vs Active-Passive Failover**

*Active-Active (chosen)*:
- All regions serve traffic simultaneously
- Pro: Better resource utilization, no "cold" servers
- Con: More complex, need cross-region data consistency

*Active-Passive*:
- One region primary, others standby
- Pro: Simpler, clear primary
- Con: Wasted capacity in standby regions

**3. Short TTL vs Long TTL**

*Short TTL (300s, chosen)*:
- Pro: Fast failover, quick to adapt to changes
- Con: More DNS queries (higher cost, load)

*Long TTL (3600s)*:
- Pro: 95% fewer DNS queries, lower cost
- Con: Slow failover (1-2 hours)

**4. Single Provider vs Multi-Provider**

*Single (Route53, chosen for simplicity)*:
- Pro: Simpler configuration, single pane of glass
- Con: DNS provider becomes single point of failure

*Multi (Route53 + Cloudflare)*:
- Pro: Redundancy, no single point of failure
- Con: Complex synchronization, higher cost

**Expected Results**:

- **Latency reduction**: 80-90% for international users
- **Availability**: 99.99% (with multi-region failover)
- **Failover time**: 5-10 minutes (with TTL=300s)
- **Cost**: $6,000/month for DNS (1M users)
- **DNS query load**: 288M queries/day (95% cached)

This design provides excellent global performance with automatic failover, balancing complexity, cost, and reliability.`,
    keyPoints: [
      'Use GeoDNS to route users to nearest region (80-90% latency reduction)',
      'Implement health checks with automatic failover to secondary servers',
      'Set TTL to 300 seconds for balance between failover speed and DNS load',
      'Deploy in 3 regions (US, EU, Asia) with primary/secondary in each',
      'Route53 provides GeoDNS, health checking, and automatic failover',
      'Monitor health check status and DNS query patterns for early issue detection',
      'Consider multi-provider DNS (Route53 + Cloudflare) for ultimate reliability',
      'Trade-off: Short TTL (fast failover) vs Long TTL (lower query cost)',
    ],
  },
  {
    id: 'dns-propagation',
    question:
      'Your company is migrating from on-premises servers to AWS. You need to update DNS records to point to new AWS load balancers, but you have 10 million users globally and cannot afford downtime. Design a migration strategy that minimizes risk. Explain how DNS propagation works, potential issues, and your testing approach.',
    sampleAnswer: `**Migration Context**:
- Current: example.com → On-premises (1.2.3.4)
- Target: example.com → AWS ALB (54.210.100.200)
- Users: 10 million globally
- Requirement: Zero downtime

**Understanding DNS Propagation**:

DNS changes don't happen instantly because of caching at multiple levels:

\`\`\`
Browser Cache (2-30 minutes)
    ↓
OS Cache (varies, often 1 hour)
    ↓
Router/ISP Resolver (respects TTL, usually)
    ↓
Intermediate Caches (CDNs, proxies)
    ↓
Authoritative DNS (source of truth)
\`\`\`

**Propagation timeline**:
- Minimum: Your TTL value (e.g., 5 minutes)
- Typical: 1-4 hours (most users)
- Maximum: 24-48 hours (worst case, some resolvers ignore TTL)

**Migration Strategy**:

**Phase 1: Preparation (Day -7 to -1)**

*Step 1.1: Audit current DNS (Day -7)*
\`\`\`bash
# Check current records
dig example.com +short
# Output: 1.2.3.4

# Check TTL
dig example.com +noall +answer
# example.com. 3600 IN A 1.2.3.4
#              ^^^^ Current TTL: 1 hour
\`\`\`

*Step 1.2: Lower TTL (Day -7)*
\`\`\`
Old: example.com. 3600 IN A 1.2.3.4
New: example.com. 300 IN A 1.2.3.4
     (5 minutes)
\`\`\`

**Why**: When we actually change the IP, users will pick up the change in 5 minutes instead of 1 hour.

**Critical**: Wait for old TTL to expire before proceeding (wait 1 hour).

*Step 1.3: Set up AWS infrastructure (Day -7 to -1)*
\`\`\`
AWS ALB: 54.210.100.200
    ↓
Target Group
    ↓
EC2 Instances (or ECS/Lambda)
\`\`\`

*Step 1.4: Deploy application to AWS (Day -3)*
- Deploy code to AWS
- Set up databases, caches, etc.
- DO NOT change DNS yet
- Application running but not receiving prod traffic

*Step 1.5: Test AWS environment (Day -3)*
\`\`\`bash
# Test by direct IP
curl -H "Host: example.com" http://54.210.100.200

# Test by overriding DNS locally
# Add to /etc/hosts:
54.210.100.200 example.com

# Run full test suite
npm run test:integration
\`\`\`

**Phase 2: Migration (Day 0)**

*Step 2.1: Enable AWS read replicas (if using database)*
\`\`\`
On-prem DB → AWS RDS (read replica)
\`\`\`
- Set up real-time replication
- AWS can serve read traffic immediately
- Writes still go to on-prem (for easy rollback)

*Step 2.2: Implement parallel running*
\`\`\`
# Update application to log to both systems
logger.log("Old", event);
logger.logToAWS("New", event);

# Compare outputs (dark launching)
# Ensure AWS behaves identically
\`\`\`

*Step 2.3: Create new DNS record (Hour 0)*
\`\`\`
example.com. 300 IN A 54.210.100.200 (AWS ALB)
\`\`\`

**Change propagates over next 5-30 minutes**:
- Minute 0-5: Early adopters switch to AWS
- Minute 5-10: 50% on AWS, 50% on on-prem
- Minute 10-30: 95% on AWS, 5% on on-prem
- Hour 1-24: Stragglers gradually switch

*Step 2.4: Monitor both environments (Hour 0-2)*

\`\`\`javascript
// CloudWatch Dashboard
Metrics to watch:
  - Old server:
    * Request count (should decline)
    * Error rate (should stay low)
    * Latency (should stay stable)
  
  - New AWS:
    * Request count (should increase)
    * Error rate (MUST stay low)
    * Latency (MUST be similar to old)
    * ALB healthy host count
    * Target response time
\`\`\`

**Thresholds for rollback**:
- Error rate > 0.5% → Roll back
- Latency > 2x normal → Roll back
- Healthy hosts < 50% → Roll back

**Phase 3: Verification & Cleanup (Day 0-7)**

*Step 3.1: Verify propagation (Hour 2)*
\`\`\`bash
# Check from multiple locations
curl https://www.whatsmydns.net/api/dns?server=8.8.8.8&query=example.com

# Expected: Most resolvers return new IP

# Test from different ISPs
dig @8.8.8.8 example.com  # Google DNS
dig @1.1.1.1 example.com  # Cloudflare DNS
dig @208.67.222.222 example.com  # OpenDNS
\`\`\`

*Step 3.2: Monitor traffic split (Hour 0-24)*
\`\`\`
Hour 0: AWS 10%, On-prem 90%
Hour 1: AWS 50%, On-prem 50%
Hour 2: AWS 80%, On-prem 20%
Hour 6: AWS 95%, On-prem 5%
Hour 24: AWS 99%, On-prem 1%
Day 7: AWS 99.9%, On-prem 0.1%
\`\`\`

*Step 3.3: Keep old environment running (Day 0-7)*
- Don't decommission on-prem servers immediately
- Some users may have long-cached old IP
- Wait 7 days before shutting down

*Step 3.4: Database cutover (Day 3, if applicable)*
Once AWS is stable and handling most traffic:
\`\`\`
1. Stop writes to on-prem DB (maintenance mode)
2. Final sync to AWS RDS
3. Promote AWS RDS to primary
4. Resume writes to AWS
5. Downtime: 5-10 minutes
\`\`\`

**Alternative: Gradual weighted cutover**

Instead of immediate switch, use weighted routing:

\`\`\`
Day 0, Hour 0: 
  example.com (weight 95) → On-prem
  example.com (weight 5) → AWS
  (5% of traffic to AWS)

Hour 2 (if stable):
  weight 80 → On-prem
  weight 20 → AWS
  (20% to AWS)

Hour 6 (if stable):
  weight 20 → On-prem
  weight 80 → AWS

Hour 12:
  weight 100 → AWS
  (full cutover)
\`\`\`

**Requires**: DNS provider supporting weighted routing (Route53, Cloudflare)

**Benefit**: More gradual, easier to spot issues early

**Testing Strategy**:

**1. Pre-migration testing**

*Load testing on AWS*:
\`\`\`bash
# Simulate production load
artillery run loadtest.yml \\
  --target https://54.210.100.200 \\
  --output results.json

# Test 10x peak load
k6 run --vus 10000 --duration 30m loadtest.js
\`\`\`

*Chaos testing*:
\`\`\`
- Kill random EC2 instances (verify ALB failover)
- Increase latency artificially (verify timeout handling)
- Simulate database failure (verify replica failover)
\`\`\`

**2. During migration testing**

*Synthetic monitoring*:
\`\`\`javascript
// Pingdom, DataDog, or custom
const monitor = {
  url: 'https://example.com/health',
  interval: 60, // seconds
  locations: ['US-East', 'US-West', 'EU', 'Asia',],
  expectedStatus: 200,
  expectedLatency: '<500ms'
};

// Alert if any location fails 3 consecutive checks
\`\`\`

*Real User Monitoring (RUM)*:
\`\`\`javascript
// Client-side beacon
window.performance.timing;
// Send to analytics:
// - DNS lookup time
// - TCP connect time
// - Time to first byte
// - Full page load

// Compare before/after migration
\`\`\`

**3. Post-migration testing**

*Smoke tests*:
\`\`\`javascript
// Critical user journeys
describe('Post-migration smoke tests', () => {
  test('User can login', async () => {
    await login('test@example.com', 'password');
    expect(page.url()).toBe('https://example.com/dashboard');
  });
  
  test('User can make purchase', async () => {
    await addToCart('product-123');
    await checkout();
    expect(await getOrderStatus()).toBe('confirmed');
  });
});
\`\`\`

*A/B comparison*:
\`\`\`
Compare metrics between:
- Before migration (Day -1)
- After migration (Day +1)

Metrics:
- Error rate (should be equal)
- Latency (should be similar or better)
- Conversion rate (should be similar)
- User complaints (should not increase)
\`\`\`

**Rollback Plan**:

**If issues detected within first 2 hours**:
\`\`\`
1. Change DNS back to on-prem:
   example.com. 300 IN A 1.2.3.4

2. Wait 5-10 minutes for propagation

3. Most users back on stable on-prem

4. Investigate issues on AWS offline

5. Fix and try again tomorrow
\`\`\`

**If issues detected after 6+ hours**:
- More users on AWS, harder to roll back
- Consider fixing forward instead
- Use weighted routing to reduce AWS traffic
- Fix issues under reduced load

**Potential Issues & Mitigations**:

**Issue 1: Some users stuck on old IP**

*Cause*: Resolvers ignoring TTL, long OS cache

*Detection*: Support tickets "site not working"

*Mitigation*:
- Keep old servers running for 7 days
- Show banner: "Having issues? Clear your DNS cache"
- Instructions: ipconfig /flushdns (Windows), sudo dscacheutil -flushcache (Mac)

**Issue 2: Session loss during migration**

*Cause*: Sessions stored locally on old servers

*Mitigation*:
- Migrate to shared session store (Redis) before DNS change
- Use sticky sessions at load balancer
- Accept some session loss (users re-login)

**Issue 3: Database replication lag**

*Cause*: Writes to on-prem, reads from AWS replica

*Mitigation*:
- Monitor replication lag (<1 second acceptable)
- If lag grows, slow down writes or pause migration

**Issue 4: Monitoring blind spot**

*Cause*: Monitoring on old servers, didn't set up on AWS

*Mitigation*:
- Set up CloudWatch, DataDog on AWS BEFORE migration
- Parallel monitoring during cutover
- Don't rely solely on old monitoring

**Expected Timeline & Risk**:

| Phase | Duration | Risk | Traffic on AWS |
|-------|----------|------|----------------|
| Prep | 7 days | Low | 0% |
| DNS change | 5 min | Medium | 0→10% |
| Propagation | 30 min | Medium | 10→80% |
| Stabilization | 6 hours | Low | 80→95% |
| Cleanup | 7 days | Very Low | 95→99.9% |

**Success Criteria**:

✅ Error rate unchanged (<0.1%)  
✅ Latency improved or equal (p99 <500ms)  
✅ Zero data loss  
✅ 99.9% of users migrated within 24 hours  
✅ Rollback plan tested and ready  
✅ All monitoring in place  

**Cost of Gradual Migration**:

- Running both environments for 7 days
- Typical: Double infrastructure cost for 1 week
- ~$10k-50k depending on scale
- **Worth it** to ensure zero downtime

This comprehensive migration strategy ensures zero-downtime migration with multiple safety nets and clear rollback procedures.`,
    keyPoints: [
      'Lower TTL to 300s one week before migration, wait for old TTL to expire',
      'Deploy and test AWS environment thoroughly without changing DNS',
      'Change DNS record and monitor both environments for 2-6 hours',
      'Use weighted routing for gradual cutover (5% → 20% → 80% → 100%)',
      'Keep old environment running for 7 days (some resolvers cache longer)',
      'Implement comprehensive monitoring: error rate, latency, traffic split',
      'Have clear rollback criteria (<0.5% error rate, <2x latency)',
      'Test from multiple locations and ISPs to verify propagation',
    ],
  },
  {
    id: 'dns-ddos',
    question:
      "Explain how DNS amplification DDoS attacks work and how you would protect a high-traffic website's DNS infrastructure from such attacks. Include both preventive measures and mitigation strategies, considering cost and complexity trade-offs.",
    sampleAnswer: `**DNS Amplification Attack Explained**:

DNS amplification is a type of DDoS attack that exploits DNS to overwhelm a target with traffic.

**How it works**:

\`\`\`
Step 1: Attacker sends DNS query with SPOOFED source IP
  ↓
Attacker → DNS Server
Query: "Give me ALL records for example.com"
Source IP: VICTIM'S IP (spoofed)
Request size: ~60 bytes

Step 2: DNS server responds to victim
  ↓
DNS Server → Victim
Response: 4000 bytes of DNS data
(Amplification factor: 70x)

Step 3: Attacker sends millions of such queries
  ↓
Result: Victim receives Gbps of unwanted DNS responses
\`\`\`

**Amplification factor**: Response is 10-100x larger than request

**Why it's effective**:
- UDP allows IP spoofing (no handshake to verify source)
- DNS responses can be large (especially with DNSSEC)
- Open DNS resolvers on internet can be exploited
- Botnet can send millions of requests

**Example attack**:
\`\`\`
Botnet: 10,000 compromised devices
Each sends: 1000 queries/second
Total: 10M queries/second
Amplification: 50x
Result: 500M DNS responses/second sent to victim
        = ~4 Tbps of traffic
\`\`\`

**Protection Strategy**:

**Layer 1: DNS Provider Selection**

*Choose DDoS-resistant DNS provider*:

**Option A: Cloudflare DNS**
- Free tier includes unlimited DDoS protection
- Global Anycast network (200+ locations)
- Absorbs attacks across distributed network
- Cost: $0-200/month
- **Recommended for most websites**

**Option B: AWS Route53**
- Built-in DDoS protection (Shield Standard included)
- Anycast network
- Cost: Based on queries (~$0.40-0.70 per million)
- **Good for AWS-heavy infrastructure**

**Option C: NS1**
- Advanced DDoS protection
- Traffic steering and filtering
- Cost: $500-5000/month
- **For enterprise with complex needs**

*Why this helps*:
- Distributed network absorbs attack traffic
- No single point of failure
- Attack traffic distributed across 100+ servers

**Layer 2: Anycast Architecture**

\`\`\`
Same IP announced from multiple locations:

1.1.1.1 (Cloudflare DNS) announced from:
  - New York
  - London
  - Tokyo
  - ... 200+ locations

Attack traffic to 1.1.1.1:
  - Routed to nearest Cloudflare location
  - Distributed across 200+ servers
  - Each server handles small portion
\`\`\`

**Impact**:
- 100 Gbps attack spread across 200 servers = 500 Mbps per server (manageable)
- Without Anycast: 100 Gbps hits single server (overwhelmed)

**Layer 3: Rate Limiting**

*Per-IP rate limiting*:

\`\`\`
Rate limit DNS queries from single IP:
  - Normal user: 10-50 queries/minute
  - Attacker: 10,000+ queries/minute

Implementation:
  If queries from IP > 100/minute:
    - Drop additional queries
    - Return REFUSED status
    - Temporary ban (1 hour)
\`\`\`

*Example (Cloudflare)*:
\`\`\`
# Cloudflare firewall rule
(dns.qry.name eq "example.com" and dns.qry.rate > 100) 
then action: block
\`\`\`

**Layer 4: Response Rate Limiting (RRL)**

Limit identical responses to same IP:

\`\`\`
If DNS server sending same response to IP repeatedly:
  - First response: Send normally
  - 2nd-5th response (within 1 second): Send normally
  - 6+ responses: DROP or send truncated response

Purpose: Prevent being used as amplifier
\`\`\`

**Implementation (BIND DNS)**:
\`\`\`
rate-limit {
    responses-per-second 5;
    window 1;
    slip 2;
};
\`\`\`

**Layer 5: Query Filtering**

*Block suspicious queries*:

\`\`\`
Block:
  - Queries for "ANY" record type (used in amplification)
  - Excessively large queries
  - Queries from known malicious IPs
  - Queries to non-existent domains (NXDOMAIN flooding)
\`\`\`

*Example (Cloudflare)*:
\`\`\`javascript
// Block ANY queries
if (dns.qry.type == "ANY") {
  return REFUSED;
}

// Block suspicious query names
if (dns.qry.name.contains("amplification")) {
  return REFUSED;
}
\`\`\`

**Layer 6: Hide Authoritative Servers**

*Problem*: Attacker directly targets your authoritative DNS servers

*Solution*: Don't publish authoritative server IPs publicly

\`\`\`
Bad:
  Attacker can find: ns1.example.com → 1.2.3.4
  Directly attack 1.2.3.4

Good:
  Use DNS provider's hidden master
  Provider's Anycast network fronts requests
  Your servers only answer to provider
\`\`\`

**Layer 7: Monitoring & Alerting**

*Metrics to track*:

\`\`\`javascript
// Query rate
queries_per_second {
  threshold: 100000, // Alert if >100k QPS
  window: '1 minute'
}

// NXDOMAIN rate (non-existent domains)
nxdomain_rate {
  threshold: 0.1, // Alert if >10% NXDOMAIN
  window: '5 minutes'
}

// Response size
response_size_bytes {
  threshold: 4096, // Alert if responses consistently large
  window: '5 minutes'
}

// Geographic anomaly
queries_by_country {
  // Alert if sudden spike from unusual country
  threshold: '10x normal'
}
\`\`\`

*Alert channels*:
- PagerDuty (critical)
- Slack (warning)
- Email (info)

**Layer 8: Incident Response Plan**

*When under attack*:

**Step 1: Detect (within 1 minute)**
- Automated monitoring triggers alert
- Query rate > 10x normal
- Response: On-call engineer paged

**Step 2: Analyze (within 5 minutes)**
\`\`\`bash
# Check query sources
dig +stats @dns-server

# Check for amplification patterns
tcpdump -i eth0 'udp port 53' | grep 'ANY'

# Identify attack type:
# - Amplification (ANY queries)
# - Flood (high volume of legitimate-looking queries)
# - NXDOMAIN flood (random subdomains)
\`\`\`

**Step 3: Mitigate (within 15 minutes)**

*If using Cloudflare*:
\`\`\`
1. Enable "I'm Under Attack" mode
   - Challenge requests before resolving
   - Drops most attack traffic

2. Block attacker IPs/ASNs
   - Identify source networks
   - Add firewall rules

3. Enable DNSSEC (if not already)
   - Adds validation step
   - Slightly increases response size (trade-off)
\`\`\`

*If using Route53*:
\`\`\`
1. Enable AWS Shield Advanced ($3000/month)
   - DDoS Response Team
   - Real-time attack visibility
   - Cost protection (refund DDoS-related charges)

2. Use AWS WAF to filter
   - Rate-based rules
   - Geo-blocking if attack from specific region
\`\`\`

**Step 4: Communicate (within 30 minutes)**
- Status page update
- Social media notification
- Email to enterprise customers

**Step 5: Post-incident review (within 48 hours)**
- Attack timeline
- What worked / didn't work
- Improvements needed
- Update runbook

**Defense-in-Depth Summary**:

\`\`\`
Layer 1: DDoS-resistant DNS provider (Cloudflare/Route53)
    ↓
Layer 2: Anycast (distributes attack geographically)
    ↓
Layer 3: Rate limiting (per-IP, per-subnet)
    ↓
Layer 4: Response Rate Limiting (prevent being amplifier)
    ↓
Layer 5: Query filtering (block ANY, suspicious queries)
    ↓
Layer 6: Hide authoritative servers
    ↓
Layer 7: Monitoring & alerting
    ↓
Layer 8: Incident response plan
\`\`\`

**Cost-Benefit Analysis**:

| Solution | Cost/month | Protection Level | Complexity |
|----------|------------|------------------|------------|
| Cloudflare Free | $0 | High | Low |
| Cloudflare Pro | $20 | Very High | Low |
| Route53 + Shield Standard | $50-500 | High | Medium |
| Route53 + Shield Advanced | $3000+ | Very High | Medium |
| Custom DNS + DDoS protection | $5000+ | Custom | High |

**Recommendation by scale**:

**Small-Medium (< 1M users)**:
- Cloudflare Free or Pro
- Cost: $0-20/month
- Protection: Sufficient for most attacks

**Medium-Large (1M-10M users)**:
- Cloudflare Business or Route53
- Cost: $200-500/month
- Add: WAF, rate limiting rules

**Enterprise (10M+ users)**:
- Multiple DNS providers (Cloudflare + Route53)
- AWS Shield Advanced
- Dedicated DDoS team
- Cost: $5000-20000/month
- Protection: Can handle 10+ Tbps attacks

**Advanced Techniques**:

**1. DNS Cookies (RFC 7873)**

Prevent query flooding with client cookies:

\`\`\`
Client → Server: Query + Cookie
Server validates cookie
Invalid cookie → Drop query

Result: Attacker can't spoof + cookie = much harder attack
\`\`\`

**2. DNSSEC**

While primarily for authentication, DNSSEC helps with DDoS:
- Larger responses (can hurt if you're the amplifier)
- But validates responses (prevents some attack types)
- **Trade-off**: Complexity vs security

**3. BGP Blackholing**

For massive attacks:
\`\`\`
Advertise route to /dev/null
Attack traffic dropped at ISP level
Before it reaches your network

Downside: Also drops legitimate traffic
Use only as last resort
\`\`\`

**4. Scrubbing Centers**

Route traffic through DDoS scrubbing service:
\`\`\`
Internet → Scrubbing Center → Your Servers
                ↓
          (Filters attack traffic)
\`\`\`

Providers: Cloudflare, Akamai, Arbor Networks

**Real-World Examples**:

**Dyn Attack (2016)**:
- Mirai botnet targeted Dyn (major DNS provider)
- 1.2 Tbps attack
- Knocked out Twitter, Reddit, GitHub
- Lesson: Use multiple DNS providers

**Cloudflare (2020)**:
- 17.2 million requests/second
- Attack absorbed with no customer impact
- Demonstrates value of Anycast + scale

**Key Takeaways**:

1. Use DDoS-resistant DNS provider with Anycast (Cloudflare, Route53)
2. Implement multi-layered defense (rate limiting, RRL, filtering)
3. Monitor continuously for anomalies
4. Have incident response plan ready
5. Consider redundant DNS providers for mission-critical sites
6. Cost ranges from $0 (Cloudflare Free) to $3000+ (Shield Advanced) depending on scale
7. Prevention is cheaper than mitigation - set up defenses before attack

**Bottom Line**: 
For most websites, **Cloudflare** (free or $20/month) provides excellent DNS DDoS protection with minimal complexity. For AWS-heavy or enterprise deployments, **Route53 + Shield** offers integrated protection. The key is to act proactively before an attack, not reactively during one.`,
    keyPoints: [
      'DNS amplification exploits open resolvers to send massive responses to victim (70x amplification)',
      'Use DDoS-resistant DNS provider with Anycast (Cloudflare, Route53) to distribute attack',
      'Implement rate limiting per-IP and Response Rate Limiting (RRL) to prevent abuse',
      'Filter suspicious queries (ANY records, excessive query rates, malicious IPs)',
      "Hide authoritative DNS servers behind provider's network",
      'Monitor query rate, NXDOMAIN rate, and geographic anomalies for early detection',
      'Have incident response plan ready: detect, analyze, mitigate, communicate',
      'For most sites, Cloudflare ($0-20/month) provides excellent protection with low complexity',
    ],
  },
];
