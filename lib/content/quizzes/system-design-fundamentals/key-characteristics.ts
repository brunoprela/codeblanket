/**
 * Quiz questions for Key Characteristics of Distributed Systems section
 */

export const keycharacteristicsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between horizontal and vertical scaling, giving real-world examples of when you would choose each approach.',
    sampleAnswer:
      'HORIZONTAL SCALING (Scale Out): Adding more machines to handle increased load. VERTICAL SCALING (Scale Up): Adding more resources (CPU, RAM, disk) to existing machines. WHEN TO USE HORIZONTAL: (1) Need to scale beyond single machine limits (Instagram: 10K servers, not possible vertically). (2) Better fault tolerance (one server fails, others continue). (3) Cost-effective at massive scale (commodity hardware cheaper than supercomputers). (4) Stateless services (web servers, API servers) - easy to add more. (5) Example: Netflix adds 100 servers during peak viewing hours, removes during off-peak (auto-scaling). Drawbacks: More complex (need load balancer, service discovery, distributed state management). Data consistency challenges. WHEN TO USE VERTICAL: (1) Simpler architecture (no distributed systems complexity). (2) Data-intensive workloads (large RAM helps - Redis, databases). (3) Legacy applications not designed for distribution. (4) Small-medium scale (easier to manage single powerful machine). (5) Example: Startup database initially on one server, upgrade from 16GB to 128GB RAM as traffic grows. Drawbacks: Hardware limits (can only get so big), single point of failure, expensive (exponential cost increases), downtime during upgrades. REAL-WORLD HYBRID: Most companies use BOTH: Horizontally scale stateless services (web/API servers), Vertically scale databases initially, then shard (horizontal), Example: Reddit scales web servers horizontally (100s of instances), but PostgreSQL primary vertically (large instance), with horizontal read replicas. INTERVIEW TIP: Mention both approaches, explain trade-offs, show you understand when each makes sense.',
    keyPoints: [
      'Horizontal: Add more machines (better scaling, fault tolerance, complexity)',
      'Vertical: Add more power to existing machines (simpler, hardware limits)',
      'Horizontal best for stateless services, massive scale',
      'Vertical for data-intensive, legacy apps, early stages',
      'Most systems use hybrid approach',
    ],
  },
  {
    id: 'q2',
    question:
      "How do you achieve 99.99% availability for a critical service like a payment processing system? Walk through the architecture and explain each component's role.",
    sampleAnswer:
      'ACHIEVING 99.99% AVAILABILITY (52 min downtime/year): ARCHITECTURE COMPONENTS: (1) MULTI-REGION DEPLOYMENT: Deploy in 3+ AWS regions (US-East, US-West, EU). Active-active: All regions serve traffic simultaneously. If one region fails (disaster, AWS outage), others continue. Achieves: No single region failure causes outtime. (2) REDUNDANT LOAD BALANCERS: Multiple load balancers (AWS ELB across AZs). Health checks every 30 seconds. If one LB fails, DNS routes to others. Achieves: No single LB failure causes outage. (3) AUTO-SCALING WEB/API SERVERS: Minimum 6 servers across 3 availability zones (2 per AZ). Auto-scaling: Adds servers if load increases or servers fail. Health checks: Remove unhealthy servers automatically. Achieves: Handle traffic spikes, tolerate server failures. (4) DATABASE HIGH AVAILABILITY: Primary-replica setup with automatic failover. Synchronous replication to standby (hot standby ready to take over). If primary fails, standby promoted within seconds. Multi-region replication for disaster recovery. Achieves: No database failure causes downtime. (5) CACHING LAYER (Redis Cluster): Redis cluster (3+ nodes) for session/data caching. If one node fails, requests hash to other nodes. Achieves: Reduce database load, tolerate cache failures. (6) MESSAGE QUEUE (Kafka): Multi-broker Kafka cluster (5+ brokers). Replication factor 3 for each partition. If broker fails, consumers connect to replicas. Achieves: No message loss, async processing continues. (7) MONITORING & ALERTING: Health checks on every component (every 30 sec). Automated recovery: Restart failed services, add capacity. Alerting: Page on-call if automated recovery fails. Runbooks: Step-by-step recovery procedures. Achieves: Detect and fix issues quickly (reduce MTTR). (8) GRACEFUL DEGRADATION: If payment gateway down: Queue payments, process when available. If recommendation engine down: Skip recommendations, show static content. Prioritize: Core payment flow > ancillary features. Achieves: Partial functionality better than complete outage. (9) CHAOS ENGINEERING: Regularly test failures (Chaos Monkey style). Simulate: Server failures, region outages, network partitions. Validates: System actually tolerates failures. Achieves: Confidence in fault tolerance. CALCULATION: Single server availability: 99% (3.65 days/year down). Two independent servers: 1 - (0.01 × 0.01) = 99.99%. Three regions: Even higher (one region can be down). With all components: 99.99%+ achievable. COST: Higher than 99.9%, but worth it for payments (lost transactions cost more).',
    keyPoints: [
      'Multi-region active-active deployment',
      'Redundancy at every layer: LB, servers, DB, cache',
      'Automated health checks and failover',
      'Monitoring, alerting, and graceful degradation',
      'Regular chaos engineering to test fault tolerance',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the CAP theorem and give real-world examples of systems that prioritize Consistency vs Availability. How do you decide which to prioritize?',
    sampleAnswer:
      'CAP THEOREM: In distributed systems with network partitions, you can have at most 2 of 3: (C) Consistency: All nodes see same data simultaneously. (A) Availability: System always accepts requests. (P) Partition Tolerance: System works despite network failures. Since network partitions are inevitable in distributed systems, P is required. So choice is: CP vs AP. CP SYSTEMS (Choose Consistency over Availability): EXAMPLES: (1) Banking systems: Bank account balance must be consistent. Cannot show different balances on web vs mobile. During network partition: Reject writes to prevent inconsistency. User sees "Service temporarily unavailable" vs wrong balance. Why: Showing wrong balance or double-charging is unacceptable. (2) Inventory management (e-commerce): Cannot oversell items. During partition: Reject orders rather than oversell. Why: Customer satisfaction (avoiding order cancellations) > temporary unavailability. (3) Google Spanner: Strong consistency for critical data. Uses: Atomic clocks + consensus protocol (Paxos). Trade-off: Higher latency, less availability during partitions. TECHNOLOGIES: Traditional SQL (PostgreSQL, MySQL with single master), Spanner, HBase, MongoDB (with majority writes). AP SYSTEMS (Choose Availability over Consistency): EXAMPLES: (1) Social media (Facebook, Twitter): Feed can be slightly out of date. Like counts don\'t need to be exact immediately. During partition: Accept writes everywhere, reconcile later. User always sees something (maybe stale). Why: Downtime worse than seeing slightly old data. (2) DNS: Must always resolve domain names. Stale DNS record better than no resolution. Eventually consistent: Changes propagate slowly (TTL). Why: Entire internet depends on DNS availability. (3) Shopping cart (Amazon): Can add items even if inventory count slightly off. Resolve at checkout (strong consistency). Why: Browsing experience > exact real-time inventory. TECHNOLOGIES: Cassandra, DynamoDB, Riak, Couchbase (eventual consistency). HOW TO DECIDE: Ask: "What\'s worse: showing wrong data or being down?" CHOOSE CONSISTENCY (CP) when: Incorrect data causes financial loss (payments, inventory). Legal/compliance requirements (healthcare, finance). User expectation of accuracy (bank balance, stock prices). CHOOSE AVAILABILITY (AP) when: Downtime costs more than temporary inconsistency. User experience requires always-on (social media, DNS). Eventual consistency acceptable (likes, views, comments). HYBRID APPROACH: Many systems use BOTH: Strong consistency for critical paths (checkout, payments). Eventual consistency for non-critical (product views, recommendations). Example: Amazon: Shopping cart: AP (add items, always available). Checkout: CP (verify inventory, consistent payment). INTERVIEW TIP: Don\'t just say "use strong consistency." Explain trade-offs, show you understand CAP, justify choice based on business requirements.',
    keyPoints: [
      'CAP: Can have at most 2 of 3 during network partition',
      'CP systems (banking, payments): Consistency over availability',
      'AP systems (social media, DNS): Availability over consistency',
      'Decision based on business impact of incorrect data vs downtime',
      'Hybrid approach: Strong consistency for critical, eventual for non-critical',
    ],
  },
];
