/**
 * Quiz questions for Heartbeat section
 */

export const heartbeatQuiz = [
  {
    id: 'q1',
    question:
      'Compare the all-to-all heartbeat pattern with the leader-based heartbeat pattern. In what scenarios would you choose one over the other, and what are the scalability implications?',
    sampleAnswer:
      "All-to-all and leader-based patterns represent fundamentally different trade-offs. All-to-all: Every node sends heartbeat to every other node. Complexity: O(n²) messages per interval. Example: 10 nodes, every 1s = 10×9 = 90 messages/sec. 100 nodes = 9,900 messages/sec. Pros: (1) No single point of failure—each node independently monitors others. (2) Fast, distributed failure detection—any node can detect any failure. (3) No dependency on coordinator availability. Cons: (1) Network overhead grows quadratically—doesn't scale beyond ~100 nodes. (2) Each node must maintain state for all other nodes (memory overhead). (3) Message storm during network issues.Leader - based: Followers send heartbeats to leader, leader monitors all.Complexity: O(n) messages per interval.Example: 100 nodes, every 1s = 100 messages / sec.Pros: (1) Linear scaling—suitable for large clusters(1000s of nodes). (2) Centralized failure detection—consistent view of cluster health. (3) Lower network overhead.Cons: (1) Single point of failure—leader crash blocks failure detection temporarily. (2) Leader can become bottleneck at very large scale. (3) Dependency on leader availability.When to choose: All - to - all for critical small clusters(5 - 50 nodes) where failure detection is mission - critical and you can't tolerate any single point of failure. Example: Primary-secondary database replication with 3 nodes. Leader-based for larger clusters (50-1000 nodes) where centralized coordination is acceptable and network efficiency matters. Example: Raft consensus with 5-node cluster—followers heartbeat to leader, leader detects failures and triggers elections. Hybrid approach: Gossip protocol (Cassandra, Consul)—each node gossips with random subset (fanout=3), information spreads epidemically. O(n log n) propagation, scales to thousands of nodes.",
    keyPoints: [
      'All-to-all: O(n²) messages, no single point of failure, scales to ~100 nodes max',
      'Leader-based: O(n) messages, centralized, scales to 1000s of nodes, leader is bottleneck',
      'Choose all-to-all: Critical small clusters (3-50 nodes) where SPOF unacceptable',
      'Choose leader-based: Larger clusters (50-1000+) where efficiency matters',
      'Hybrid: Gossip protocol (O(n log n)) combines benefits, scales to thousands (Cassandra)',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a heartbeat system that minimizes false positives (declaring live nodes as dead). What timing parameters would you use and how would you handle network jitter?',
    sampleAnswer:
      "Minimizing false positives requires careful timing and adaptive strategies. Basic parameters: Heartbeat interval: 1 second (how often to send). Timeout threshold: 5 seconds (when to declare dead). Reasoning: Threshold = 3-10× interval gives buffer for transient delays. 5s allows 5 consecutive heartbeats to miss—unlikely all fail unless node truly dead. Multiple missed heartbeats strategy: Don't declare dead after single miss—require 3 - 5 consecutive misses.Track consecutive_misses counter.On heartbeat received: Reset counter to 0. On expected heartbeat not received: Increment counter.If counter reaches threshold(5): Declare dead.Example: Heartbeat expected every 1s.Misses at T = 1s, 2s, 3s, 4s, 5s(5 consecutive).Declare dead at T = 5s.If heartbeat arrives at T = 4s, reset counter, node remains alive.Adaptive timeout for network jitter: Problem: Fixed 5s timeout good for stable network, but slow(100ms latency) network has many false positives.Solution: Use running statistics (mean and standard deviation) of heartbeat arrival intervals.mean_interval = 1.0s, stddev = 0.2s (low jitter network).timeout = mean + 4×stddev = 1.0 + 4×0.2 = 1.8s (tighter).mean_interval = 1.0s, stddev = 0.5s (high jitter).timeout = mean + 4×stddev = 1.0 + 4×0.5 = 3.0s (looser).Phi Accrual Failure Detector: Even more sophisticated—calculate suspicion level (phi) based on how many standard deviations away from expected arrival time.Phi > 8 = very suspicious (declare dead).Adapts to varying network conditions automatically.Additional techniques: (1) Exponential backoff after declaring dead (reduce false positive impact). (2) Require quorum agreement—multiple nodes must agree node is dead(Consul SWIM protocol). (3) Indirect pings—if A can't reach B, ask C to ping B (distinguishes node failure from network issue). (4) Bidirectional heartbeats—both send, increases reliability. Production example: Cassandra gossip—1s interval, phi accrual with threshold=8, adapts to network jitter automatically. Kubernetes liveness probes—can configure failureThreshold=3 (require 3 consecutive failures) with periodSeconds=10.",
    keyPoints: [
      'Multiple missed heartbeats: Require 3-5 consecutive misses, not single miss',
      'Adaptive timeout: Use mean + 4×stddev of arrival intervals to handle jitter',
      'Phi accrual: Calculate suspicion level based on statistical distribution',
      'Quorum agreement: Multiple nodes must agree (Consul SWIM indirect pings)',
      'Production: Cassandra (phi=8, adapts), Kubernetes (failureThreshold configurable)',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how heartbeats are used differently in three contexts: failure detection, leader election, and lease renewal. How do the timing requirements differ?',
    sampleAnswer:
      "Heartbeats serve different purposes with different timing requirements. 1. Failure Detection (e.g., cluster membership): Purpose: Detect when node has crashed or become unreachable, remove from cluster. Timing: Moderate—balance detection speed vs false positives. Typical: 1-5s heartbeat interval, 5-30s timeout. Example: Cassandra gossip—1s interval, phi accrual threshold=8 (~10s effective timeout). Requirements: (1) False positives are costly (unnecessary failover, data migration). (2) But slow detection means routing requests to dead node (poor user experience). (3) Adapt to network conditions (use adaptive timeouts). 2. Leader Election (e.g., Raft): Purpose: Maintain leadership, trigger election when leader fails. Timing: Fast—quick election improves availability during failures. Typical: 150ms-1s heartbeat interval, 1-5s election timeout (randomized). Example: etcd/Raft—leader sends heartbeat every 150ms, followers have 1.5s election timeout. Requirements: (1) Fast detection critical—the longer the wait, the longer system has no leader. (2) Randomized timeouts prevent split votes (nodes don't all timeout simultaneously).(3) Must be faster than client request timeouts (clients shouldn't timeout waiting for unavailable leader). (4) False positive less costly here—unnecessary election is ok (just inefficient). 3. Lease Renewal (e.g., distributed locks): Purpose: Maintain exclusive access to resource, automatically release on failure. Timing: Depends on acceptable downtime vs renewal overhead. Typical: 30-60s lease duration, renew every 10-20s (33-50% of duration). Example: ZooKeeper session—30s timeout, client sends heartbeat every 10s. Requirements: (1) Lease duration = max time resource unavailable after holder crashes. (2) Renewal frequency must account for network latency + processing time (renew early). (3) Longer lease duration = less overhead but slower recovery. (4) False positive (failing to renew) means resource briefly unavailable—acceptable. Summary of differences: Failure detection: 1-5s interval, moderate timeout (balance false positives and detection speed). Leader election: 150ms-1s interval, fast timeout (availability), randomized. Lease renewal: 10-20s interval, long lease (30-60s), prioritize low overhead. Key insight: Timing should match the consequence of false positives and cost of slow detection for each use case.",
    keyPoints: [
      'Failure detection: 1-5s interval, moderate timeout, balance false positives vs detection speed',
      'Leader election: Fast (150ms-1s), quick timeout (1-5s), randomized, availability critical',
      'Lease renewal: Longer (10-20s interval, 30-60s lease), low overhead, automatic recovery',
      'Timing matches consequences: False positives costly for failure detection, less for election',
      'Examples: Cassandra (1s gossip), etcd (150ms leader), ZooKeeper (10s keepalive)',
    ],
  },
];
