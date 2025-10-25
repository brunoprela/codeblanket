/**
 * Quiz questions for Gossip Protocol section
 */

export const gossipprotocolQuiz = [
  {
    id: 'q1',
    question:
      'Explain how gossip protocol achieves O(log N) propagation time and why this makes it suitable for large-scale distributed systems. Provide a concrete example with a 1000-node cluster.',
    sampleAnswer:
      "Gossip protocol achieves logarithmic propagation through epidemic spreading where each informed node tells multiple others, causing exponential growth in informed nodes. Mechanism: Each node tells fanout peers (typically 3). Each round doubles (approximately) the number of informed nodes. Round 0: 1 node knows. Round 1: 1 node tells 3 → 4 nodes know. Round 2: 4 nodes each tell 3 → 13 nodes know. Round 3: 13 nodes each tell 3 → 40 nodes know. Round 4: ~121 nodes. Round 5: ~364 nodes. Round 6: ~1000 nodes. With fanout=3, takes log₃(1000) ≈ 6.3 rounds to reach all 1000 nodes. At 1 second per round, full propagation in ~7 seconds. Mathematical analysis: After k rounds, approximately 3^k nodes informed (with fanout=3). To reach N nodes: 3^k = N → k = log₃(N) = log(N)/log(3). Why suitable for large-scale: Scalability: Adding nodes doesn't dramatically increase propagation time. 1000 nodes: 7 rounds. 10,000 nodes: 9 rounds. 100,000 nodes: 11 rounds.No bottleneck: Unlike broadcast (single sender bottleneck), gossip distributes load—all informed nodes participate in spreading.Fault tolerance: If some nodes fail during propagation, others continue spreading.No single point of failure.Network efficiency: Total messages = O(N log N) vs broadcast O(N) from single source.But gossip has no bottleneck and tolerates failures.Example - Cassandra with 1000 nodes: Gossip interval: 1 second, fanout: 3. Schema change propagated: Round 1-7: All nodes receive schema.If 10 nodes crashed during propagation: Information still spreads via remaining nodes(981 healthy nodes more than enough).Compare to centralized broadcast: Master sends to all 1000 nodes.Master\'s network interface saturated (bottleneck). If master fails during broadcast, propagation stops. Gossip superior for large, unreliable clusters.",
    keyPoints: [
      'Exponential growth: Each round ~triples informed nodes (fanout=3), not linear',
      'O(log N) rounds: 1000 nodes in ~7 rounds, 100,000 nodes in ~11 rounds',
      'No bottleneck: All informed nodes participate, distributes load across cluster',
      "Fault tolerant: Node failures during propagation don't stop spreading",
      'Example: Cassandra 1000 nodes, 1s interval, 7 seconds full propagation',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare push gossip, pull gossip, and push-pull gossip. In what scenarios would each be most appropriate, and how do they differ in convergence guarantees?',
    sampleAnswer:
      'Three gossip variants trade off initial speed, eventual convergence, and network efficiency. Push Gossip (Rumor Mongering): Informed nodes actively push updates to random peers. Node A has update → pushes to B, C. B and C push to others. Characteristics: (1) Fast initial spread—information spreads exponentially in early rounds. (2) Redundant in later rounds—nodes keep pushing even after most know. (3) May not reach 100% (some nodes never randomly selected). Convergence: Fast initial (90% in log N rounds), but may leave stragglers. Use case: Broadcasting events where speed matters more than guaranteed delivery. Example: Alerting cluster of emergency (acceptable if 2-3% miss alert initially). Pull Gossip: Uninformed nodes pull updates from random peers. Node A doesn\'t know → asks B "what\'s your state?" Gets update from B.Characteristics: (1) Slower initial spread—relies on uninformed nodes asking. (2) Continues until convergence—uninformed nodes keep pulling. (3) Guaranteed eventual convergence (all nodes eventually pull).Convergence: Slower start, but ensures 100 % eventually.Use case: Anti - entropy, ensuring all nodes catch up.Example: New node joining cluster pulls full state from random established nodes.Push - Pull Hybrid: Nodes both push their state AND pull peer\'s state in single exchange. Node A → B: "Here\'s my state, send me yours." Both update their state. Characteristics: (1) Fast initial spread (push benefit). (2) Guaranteed convergence (pull benefit). (3) Most efficient—single round-trip does both. (4) Higher per-message cost (more data exchanged). Convergence: Fast initial (exponential) AND guaranteed eventual (complete). Use case: Production systems needing both speed and completeness. Example: Cassandra, Consul—fast propagation of updates, guaranteed eventual consistency. Scenario recommendations: Emergency broadcast (push): Fire alert to cluster, prioritize speed, ok if 2% miss initially. Anti-entropy repair (pull): Background process ensuring all data synchronized eventually. General purpose (push-pull): Membership updates, configuration changes—need fast propagation AND guaranteed delivery. Trade-off summary: Push = speed. Pull = completeness. Push-pull = both (most common in production).',
    keyPoints: [
      'Push: Fast initial spread, redundant later, may leave stragglers (90% coverage quickly)',
      'Pull: Slower start, guaranteed eventual convergence (100% eventually)',
      'Push-Pull: Fast AND complete, single exchange does both (most efficient)',
      'Use push: Emergency broadcasts prioritizing speed over completeness',
      'Use push-pull: Production systems (Cassandra, Consul) needing both speed and guarantee',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a gossip-based failure detection system for a 500-node cluster. What gossip parameters would you choose, how would you optimize for network overhead, and how would you detect network partitions?',
    sampleAnswer:
      'Designing gossip for 500-node cluster requires balancing detection speed, network overhead, and partition handling. Core parameters: Fanout: 3 (each node gossips with 3 random nodes per round). Rationale: log₃(500) ≈ 6 rounds for full propagation. Higher fanout (5) = faster but more network traffic. Lower fanout (2) = slower propagation. Sweet spot: 3. Gossip interval: 1 second. Rationale: Fast enough for reasonable failure detection (detectable in ~10 seconds), not so fast as to overwhelm network. Total traffic: 500 nodes × 3 messages × 1/sec = 1500 messages/sec (manageable). State exchange: Each gossip message includes: Node states (alive/dead/suspected), heartbeat counters, timestamps. Use digests first (hashes of state): If digests match, no need to exchange full state (optimization). Only exchange full state if digests differ. Network overhead optimization: (1) Compress messages (gzip/snappy). 10KB state → 2KB compressed. (2) Incremental updates: Only send changed node states since last gossip, not full cluster state. (3) Hierarchical gossip: Gossip more frequently within rack (lower latency), less frequently across racks. (4) Adaptive fanout: Reduce fanout if network congestion detected, increase if network idle. Failure detection: Combine gossip with SWIM protocol: Direct ping: Every second, ping random node M. If no response: Indirect ping: Ask K other nodes (K=3) to ping M. If any indirect ping succeeds: M alive, network issue between me and M. If all fail: M likely dead, mark as suspected. Gossip suspicion: Spread suspicion via gossip. If quorum agrees M is suspected: Mark M as dead. Network partition detection: Symptom: Sudden loss of connectivity to multiple nodes simultaneously (not gradual individual failures). Detection: If I lose contact with >10% of cluster within 5 seconds: Likely partition, not individual failures. Check reachability pattern: If I can reach {A,B,C} but not {D,E,F,G}, and A can\'t reach them either: Partition confirmed.Action: (1) Use quorum: My partition has 300 nodes (majority of 500) → continue operating. Other partition has 200 nodes (minority) → read - only or halt. (2) Gossip partition info: Nodes in majority partition gossip "partition detected, we have majority."(3) Alert operators: Network partition detected, minority nodes unavailable. (4) When partition heals: Gossip resumes, states merge, anti - entropy synchronizes diverged data.Production monitoring: (1) Message rate: Should be ~1500 / sec(500×3).Spikes indicate gossip storm. (2) State size: Growing state means nodes not being removed (memory leak). (3) Convergence time: Track how long for update to reach all nodes (should be < 10 seconds). (4) Partition events: Alert on detected partitions (investigate network issues).',
    keyPoints: [
      'Parameters: Fanout=3, interval=1s, total 1500 msgs/sec for 500 nodes',
      'Optimizations: Compression, digests, incremental updates, hierarchical gossip',
      'Failure detection: SWIM protocol (direct + indirect ping) + gossip to spread suspicion',
      'Partition detection: Sudden loss of >10% nodes, check reachability patterns, use quorum',
      'Monitoring: Message rate, state size, convergence time, partition events',
    ],
  },
];
