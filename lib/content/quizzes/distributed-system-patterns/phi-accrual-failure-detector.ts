/**
 * Quiz questions for Phi Accrual Failure Detector section
 */

export const phiaccrualfailuredetectorQuiz = [
  {
    id: 'q1',
    question:
      'Explain how Phi Accrual Failure Detector adapts to variable network conditions better than fixed-timeout failure detection. Provide an example with two different network scenarios.',
    sampleAnswer: `Phi accrual adapts by learning from historical heartbeat patterns, unlike fixed timeout which treats all networks identically. Fixed timeout problem: Set timeout=3 seconds. Fast network (mean=1s, stddev=0.1s): Heartbeats arrive 0.9s, 1.0s, 1.1s consistently. 3-second timeout works well (very safe). Slow network (mean=1s, stddev=0.5s): Heartbeats arrive 0.5s, 1.2s, 1.5s, 0.8s, 1.6s (high jitter). Occasionally 2.5s delay due to network congestion. 3-second timeout causes false positive (declares dead at 2.5s delay, but node alive). Must increase timeout to 5s to avoid false positives. But now fast network suffers (unnecessarily long detection). Phi accrual solution: Fast network: Learns mean=1s, stddev=0.1s. 1.5s delay = (1.5-1.0)/0.1 = 5 standard deviations away. Phi value very high (~8+) → Declare dead quickly. 1.2s delay = (1.2-1.0)/0.1 = 2 stddevs → Phi ~3 → Still alive. Adapts tightly to consistent pattern. Slow network: Learns mean=1s, stddev=0.5s. 1.5s delay = (1.5-1.0)/0.5 = 1 stddev away. Phi value low (~1) → Still alive (expected in this network). 2.5s delay = (2.5-1.0)/0.5 = 3 stddevs → Phi ~5 → Suspicious but possibly alive. Only very long delay (4+ seconds) triggers phi>8. Adapts loosely to variable pattern. Concrete example: Production deployment across AWS regions. US-East region: Low latency, consistent heartbeats (stddev=50ms). Fixed 3s timeout works. Cross-Atlantic (US-EU): Higher latency, variable (stddev=300ms). Fixed 3s timeout = many false positives. With phi accrual: Both regions use phi>8 threshold. US-East: Phi>8 at ~1.5s delay (very confident). EU: Phi>8 at ~3.5s delay (equally confident, but accounts for higher variance). Same threshold (phi=8), different timeouts (automatic adaptation). Result: Single failure detector configuration works across all network conditions.`,
    keyPoints: [
      "Fixed timeout: Same timeout for all networks, doesn't adapt to jitter / latency",
      'Phi accrual: Learns mean and stddev from history, calculates suspicion based on deviation',
      'Fast network (low stddev): Short delay triggers high phi, quick detection',
      'Slow network (high stddev): Same delay triggers low phi, tolerates jitter',
      'Production: Single phi threshold (e.g., 8) adapts automatically to different regions',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through the calculation of phi value when a heartbeat arrives 2 seconds late. Assume historical heartbeats show mean interval=1s and stddev=0.2s. Interpret the resulting phi value.',
    sampleAnswer: `Calculating phi value for late heartbeat demonstrates how statistical deviation maps to suspicion. Given: Historical intervals: mean=1s, stddev=0.2s (consistent network). Current situation: Expected heartbeat at T=0, actually arrives at T=2 (2 seconds late). Step 1: Calculate time since last heartbeat. time_since_last = 2 seconds. Step 2: Calculate z-score (how many standard deviations away from mean). z_score = (time_since_last - mean) / stddev. z_score = (2.0 - 1.0) / 0.2 = 5. Interpretation: 2-second interval is 5 standard deviations away from expected 1-second interval. Step 3: Calculate probability node is alive using normal distribution. P(alive) = 1 - CDF(z_score). CDF(5) ≈ 0.9999997 (area under normal curve up to 5 stddevs). P(alive) = 1 - 0.9999997 = 0.0000003 (0.00003%). Step 4: Calculate phi value. phi = -log₁₀(P(alive)). phi = -log₁₀(0.0000003) ≈ 6.5. Interpretation: Phi = 6.5 means we're 99.999997% confident the node is dead.Is this sufficient to declare dead? Depends on threshold: If threshold=8(Cassandra default): phi=6.5 < 8 → Don't declare dead yet (wait longer). If threshold=5: phi=6.5 > 5 → Declare dead. Contextual meaning: Given this network's consistency (stddev = 0.2s is very low), a 2-second delay is extremely unusual.For comparison, different network: If stddev=0.5s (higher jitter): z_score = (2.0 - 1.0) / 0.5 = 2 stddevs.CDF(2) ≈ 0.977, P(alive) ≈ 0.023(2.3 %).phi = -log₁₀(0.023) ≈ 1.6.Much lower phi! Same 2 - second delay, but less suspicious in higher - jitter network.Production implications: Network with consistent heartbeats (low stddev) → High phi for deviations → Fast failure detection.Network with variable heartbeats (high stddev) → Low phi for same deviation → Tolerates jitter, fewer false positives.Phi accrual automatically balances detection speed vs false positive rate based on observed network behavior.`,
    keyPoints: [
      'Z-score: (2.0 - 1.0) / 0.2 = 5 standard deviations away from expected',
      'Probability alive: ~0.00003% (extremely unlikely with 5 stddev deviation)',
      'Phi value: ~6.5 (high suspicion, but below typical threshold of 8)',
      'Context matters: Same delay has low phi in high-jitter network (stddev=0.5 → phi=1.6)',
      'Adaptive: Consistent network triggers high phi quickly, variable network more tolerant',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the cold start problem in phi accrual failure detection and describe three strategies to handle insufficient historical data when a node first joins a cluster.',
    sampleAnswer: `Cold start problem: New node has no historical heartbeat data, cannot calculate accurate mean/stddev, thus cannot compute meaningful phi values. Risk: Either declare all peers dead (phi=infinity due to no history) or declare all alive (phi=0), both wrong. Strategy 1: Default parameters until warm-up. Approach: Use reasonable defaults based on expected network characteristics. Example: mean_interval = 1.0s, stddev_interval = 0.5s (moderate jitter assumed). Minimum samples: Require 10-20 heartbeat samples before trusting phi values. Until then: Use default fixed timeout (e.g., 5 seconds) for failure detection. After collecting 10-20 samples: Switch to phi accrual with learned parameters. Pros: Safe (conservative defaults prevent false positives during learning). Cons: Less optimal during warm-up (may be too conservative or too aggressive). Production example: Cassandra requires minimum samples before enabling phi accrual. Strategy 2: Bootstrap from network profile. Approach: Characterize network environment beforehand (datacenter vs WAN, cloud vs on-prem). Provide network-specific defaults. LAN datacenter: mean=1s, stddev=0.1s (low jitter). Cross-region cloud: mean=1s, stddev=0.3s (moderate jitter). WAN: mean=1s, stddev=0.8s (high jitter). New node: Uses appropriate profile for its environment. Quickly adapts as real data arrives. Pros: Better initial parameters than blind guess. Cons: Requires knowing network type, maintaining profiles. Production example: Akka cluster allows configuring acceptable-heartbeat-pause based on deployment. Strategy 3: Learn from peers. Approach: New node queries established nodes for their learned parameters. Peer nodes share: mean_interval=1.05s, stddev=0.22s (from their history). New node: Bootstraps with peer\`s parameters as initial estimate.Refines with own observations over time.Pros: Instantly has realistic parameters for current network.Cons: Assumes network conditions similar across nodes (usually true).Potential for propagating bad data if peer had anomalous history.Implementation: New node joins cluster via gossip.Gossip message includes sender's failure detector statistics. Hybrid approach (recommended): Start with default or network profile (Strategy 1/2). Query peers for their statistics (Strategy 3). Use peer statistics if available, otherwise use defaults. Collect own samples (10-20). Gradually transition from peer/default stats to own learned statistics. Weight: 100% peer stats initially, decrease by 10% per sample, 100% own stats after 10 samples. This combines benefits: Safe defaults + quick bootstrap from peers + accurate learning. Production monitoring: Track "warmup state" metric (how many samples collected). Alert if node stuck in warmup (unable to collect samples = network issues). Monitor phi values during warmup (shouldn't be extreme).`,
    keyPoints: [
      'Cold start: No historical data, cannot calculate accurate phi, risk of false positives/negatives',
      'Strategy 1: Default parameters (mean=1s, stddev=0.5s) until 10-20 samples collected',
      'Strategy 2: Network profiles (LAN vs WAN defaults) based on deployment environment',
      'Strategy 3: Bootstrap from peer statistics (query established nodes for their learned params)',
      'Hybrid: Combine defaults, peer stats, gradual learning (recommended production approach)',
    ],
  },
];
