/**
 * Quiz questions for Lease section
 */

export const leaseQuiz = [
  {
    id: 'q1',
    question:
      'Explain the key difference between a lock and a lease, and provide a scenario where using a lease prevents a deadlock that a traditional lock would cause.',
    sampleAnswer:
      'The fundamental difference is time-boundedness: locks are held indefinitely until explicitly released, leases expire automatically after a fixed duration. Deadlock scenario with locks: Node A acquires lock on Resource X. Node A crashes while holding the lock (never releases it). Resource X is locked forever—system deadlocked. No automatic recovery without manual intervention. Lease prevents deadlock: Node A acquires lease on Resource X for 30 seconds. Node A crashes at T=10s (doesn\'t renew lease).Lease expires at T=30s (automatically).At T=31s, Node B can acquire lease on Resource X.System automatically recovers from failure! Real- world example: Distributed coordination service (like Google Chubby).Service A holds lock to coordinate cluster operations.Service A has hardware failure, can\'t release lock. Traditional lock: Cluster stuck, requires operator to manually break lock. Lease: After lease expiration (say, 1 minute), lock automatically released. Service B takes over coordination. Trade-off: Brief unavailability during lease expiration period (the "dead time" between A crashing and lease expiring). However, this is vastly better than indefinite deadlock. Production systems (ZooKeeper, etcd, Chubby) all use leases for this reason—automatic failure recovery is essential for availability at scale.',
    keyPoints: [
      'Lock: Indefinite until released, no automatic expiry, crash causes deadlock',
      'Lease: Time-bound, expires automatically, crash recovers after expiration',
      'Deadlock prevention: Crashed node holding lock blocks forever vs lease expires',
      'Trade-off: Brief unavailability (lease expiration time) vs indefinite deadlock',
      'Production use: ZooKeeper, etcd, Chubby all use leases for automatic recovery',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a lease-based leader election system. How would you choose the lease duration, renewal frequency, and handle clock skew?',
    sampleAnswer:
      'Lease-based leader election design considerations: Basic Protocol: (1) Nodes compete for lease on special "leader" resource. (2) Winner holds lease for fixed duration (e.g., 30s). (3) Leader renews lease periodically (e.g., every 10s = 33% of duration). (4) Leader acts as leader only while holding valid lease. (5) If leader crashes, lease expires, another node acquires it. Lease Duration Selection: Too short (5s): Frequent renewal overhead, risk of false failures (transient network issue). Too long (5 minutes): Slow failover, system unavailable for minutes after leader crash. Sweet spot: 30-60s. Balance: Fast enough failover (acceptable downtime) vs tolerating transient network issues. Consider network reliability: unreliable network needs longer lease to avoid false positives. Renewal Frequency: Rule: Renew at 33-50% of lease duration. Example: 30s lease, renew every 10-15s. This gives 2-3 renewal attempts before expiry. If first renewal fails (network blip), second attempt succeeds. Safety margin accounts for network latency and processing delays. Clock Skew Handling: Problem: Node\'s clock differs from coordinator\'s clock. Solution: Use coordinator\'s time for expiry.Leader doesn\'t check its own clock to determine "am I still leader?" Leader tracks: "I acquired lease at coordinator-time T, expires at T+30s." Leader renews before coordinator-time T+30s. Additional safety: Renew conservatively at 33% (10s) rather than 90% (27s). Gives large safety margin for clock drift. Require NTP synchronization (clocks within 100ms). Monitor clock skew, alert if exceeds threshold. Implementation: etcd example: default lease TTL = 60s, keepalive every 20s (33%). Kubernetes control plane: lease duration = 15s, renew every 10s (67%, shorter duration compensates). Production: Test failure scenarios (kill leader, partition network) to validate failover time meets SLA.',
    keyPoints: [
      'Lease duration: 30-60s (balance failover speed vs false positives from transient issues)',
      'Renewal frequency: 33-50% of duration (2-3 attempts before expiry, safety margin)',
      "Clock skew: Use coordinator's time for expiry, renew conservatively, NTP sync required",
      'Safety margin: Renew early (33%) gives buffer for network latency and clock drift',
      'Examples: etcd (60s lease, 20s keepalive), Kubernetes (15s lease, 10s renew)',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how leases prevent split-brain in a leader election system and what happens if a node is partitioned from the coordinator but continues to think it holds the lease.',
    sampleAnswer:
      "Leases prevent split-brain by ensuring at most one active leader at any time through time-bounded permissions. Split-brain prevention mechanism: Leader holds lease from coordinator (e.g., 30s duration). To remain leader, must renew lease with coordinator before expiry. Network partition occurs: Leader partitioned from coordinator (can't reach coordinator to renew). Lease expires at T = 30s (coordinator's clock). Leader must stop acting as leader at T=30s (even if still running). Other partition: Another node contacts coordinator, acquires new lease at T=31s. Result: At most one active leader at any given time. There may be no leader during [T=0 to T=31], but never two leaders. Partitioned node scenario: Node A is leader with lease expiring at T=30s. At T=15s, network partition: A can't reach coordinator.A tries to renew at T = 20s, T = 25s (both fail—network partition).At T = 30s: A's lease expires. Critical: A must immediately stop acting as leader (stop accepting writes, making decisions). At T=31s: Node B (in other partition) acquires lease, becomes leader. At T=35s: Partition heals, A sees B is leader. What if A doesn't stop at T = 30s (buggy implementation) ? A continues to think it's leader, tries to operate. But: Resources implement fencing tokens. A's operations include old token (e.g., token = 5).B has new token (token = 6).Resources reject A's operations (old token < current token). Result: Even with buggy leader, fencing prevents corruption. Defense in depth: (1) Lease expiry prevents split-brain. (2) Fencing tokens prevent stale leader damage if lease enforcement fails. (3) Quorum-based operations require majority (additional layer). Production: Monitor lease renewal failures, alert if leader can't renew (potential partition).",
    keyPoints: [
      'Split-brain prevention: Leader must renew lease, stops acting after expiry, no concurrent leaders',
      "Network partition: Leader can't renew, lease expires, must stop immediately",
      'At most one leader: May be no leader during transition, but never two',
      'Fencing tokens: Even if buggy leader continues, resources reject old tokens',
      'Defense in depth: Lease expiry + fencing tokens + quorum prevent all failure modes',
    ],
  },
];
