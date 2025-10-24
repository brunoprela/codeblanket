/**
 * Multiple choice questions for Phi Accrual Failure Detector section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const phiaccrualfailuredetectorMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'What does a phi value of 8 in Phi Accrual Failure Detector roughly indicate?',
      options: [
        'The node is 8 seconds behind in sending heartbeats',
        'There is about a 99.999999% confidence the node is dead',
        'The node has missed 8 consecutive heartbeats',
        'The system has detected 8 failed nodes',
      ],
      correctAnswer: 1,
      explanation:
        "A phi value of 8 indicates approximately 99.999999% confidence (or 10^-8 probability) that the node is dead. Phi is calculated as phi = -log₁₀(P(alive)), where P(alive) is the probability the node is still alive given the time since the last heartbeat. For phi=8: P(alive) = 10^-8 = 0.00000001 (one in 100 million chance alive). This extremely low probability makes phi=8 a strong signal for declaring a node dead, which is why Cassandra uses it as the default threshold. The phi value is based on statistical deviation from expected heartbeat arrival times—if heartbeats usually arrive every 1±0.1 seconds and it's been 3 seconds (many standard deviations away), phi climbs quickly. Option 1 is incorrect—phi is based on statistical deviation, not simple time difference. Option 3 is incorrect—phi considers distribution, not just count of misses. Option 4 is incorrect—phi is per-node calculation, not a cluster-wide count.",
    },
    {
      id: 'mc2',
      question:
        'How does Phi Accrual Failure Detector adapt to networks with high jitter compared to fixed timeout approaches?',
      options: [
        'It uses a longer fixed timeout for all networks',
        'It learns the mean and standard deviation of heartbeat intervals and adjusts suspicion based on observed variability',
        'It switches to TCP from UDP for more reliable delivery',
        'It increases the heartbeat frequency',
      ],
      correctAnswer: 1,
      explanation:
        "Phi accrual adapts to network conditions by learning from historical heartbeat arrival patterns, specifically calculating mean and standard deviation of intervals. For consistent networks (low jitter): Mean=1s, stddev=0.1s. A 1.5s delay is (1.5-1)/0.1 = 5 standard deviations away → High phi (~8+) → Quickly declare suspicious. For variable networks (high jitter): Mean=1s, stddev=0.5s. A 1.5s delay is (1.5-1)/0.5 = 1 standard deviation → Low phi (~1) → Still considered normal. Same delay, different suspicion levels based on network characteristics. This automatic adaptation is the key advantage over fixed timeouts. With a fixed 3-second timeout: Works for low-jitter (1±0.1s) networks. Causes false positives for high-jitter (1±0.5s) networks (occasional 2.5s delays trigger timeouts incorrectly). Phi accrual automatically adjusts—high jitter networks need larger deviations before high phi. No manual tuning required. Options 1, 3, and 4 don't explain the adaptive mechanism. The key is statistical learning from observed patterns.",
    },
    {
      id: 'mc3',
      question:
        'What is the "cold start" problem in phi accrual failure detection?',
      options: [
        "The detector doesn't work in cold environments",
        'There is insufficient historical data to calculate accurate phi values when a node first joins',
        'The system takes too long to start up',
        'Nodes are initially marked as dead before being recognized as alive',
      ],
      correctAnswer: 1,
      explanation:
        "The cold start problem refers to the lack of historical heartbeat data when a node first joins, making phi calculation impossible or unreliable. Phi accrual requires statistics: mean interval, standard deviation. When a node first joins: Has received 0-5 heartbeats. Cannot accurately calculate mean/stddev. Phi calculation would be meaningless or produce extreme values. Without proper handling, this causes problems: Might declare all peers dead (phi=infinity due to no history). Might declare all peers alive (phi=0 as fallback). Neither is correct. Solutions: Use default parameters until minimum samples collected (e.g., mean=1s, stddev=0.5s for first 10-20 heartbeats). Bootstrap from peer nodes (query established nodes for their learned parameters). Use fixed timeout initially, switch to phi accrual after warm-up. Example: Cassandra requires minimum samples before enabling phi-based detection. During warm-up (first minute), may use conservative defaults or rely on other nodes' views. After collecting 20 heartbeats, switch to learned parameters. Option 1 is a literal misinterpretation. Option 3 describes startup time, not the statistical issue. Option 4 describes a potential symptom but not the root cause.",
    },
    {
      id: 'mc4',
      question:
        'Why might you choose a higher phi threshold (e.g., 12) instead of the default 8 for a failure detector?',
      options: [
        'To detect failures more quickly',
        'To reduce false positives in an unreliable network',
        'To use less memory',
        'To support more nodes in the cluster',
      ],
      correctAnswer: 1,
      explanation:
        'A higher phi threshold reduces false positives by requiring higher confidence before declaring a node dead, which is useful in unreliable networks or when false failure detection is very costly. Phi threshold comparison: phi > 8: ~99.999999% confidence node is dead (default). phi > 12: ~99.9999999999% confidence node is dead (even more certain). Higher threshold means: More missed heartbeats or longer delays required before declaring dead. More tolerant of transient network issues, packet loss, delays. Slower failure detection (wait longer for higher confidence). Fewer false positives (incorrectly marking live nodes as dead). Use higher threshold when: Network is unreliable (frequent packet loss, high jitter). False positives are very costly (e.g., triggering expensive leader re-election, data rebalancing). Brief delays in failure detection are acceptable. Use lower threshold (e.g., 5) when: Network is reliable. Fast failure detection is critical. False positives are less costly than slow detection. Example: Cross-datacenter links (high latency, jitter) might use phi > 12 to avoid false positives from transient WAN issues. Local datacenter (low latency, reliable) might use phi > 5 for fast detection. Option 1 is opposite—higher threshold means slower detection. Options 3 and 4 are unrelated to threshold choice.',
    },
    {
      id: 'mc5',
      question:
        'In Phi Accrual Failure Detector, what does the "accrual" part of the name refer to?',
      options: [
        'It accumulates heartbeats in a buffer',
        'Suspicion gradually accrues over time rather than being a binary state',
        'It adds up the number of nodes that have failed',
        'It collects statistics from multiple sources',
      ],
      correctAnswer: 1,
      explanation:
        '"Accrual" refers to the continuous, gradually increasing nature of the phi value (suspicion level) over time, as opposed to binary alive/dead decisions. Traditional detectors: Node is either alive or dead (binary). Timeout expires → Instantly dead. Phi accrual: Suspicion gradually accrues as time passes without heartbeat. T=0s: Last heartbeat received, phi ≈ 0 (not suspicious). T=1s: Expected heartbeat time, phi ≈ 1-2 (slightly suspicious). T=2s: Still no heartbeat, phi ≈ 4-5 (more suspicious). T=3s: phi ≈ 8+ (very suspicious, likely dead). Suspicion continuously increases (accrues) as delay grows. This continuous scale provides flexibility: Different operations can use different thresholds. Read operations: phi > 5 (try another replica quickly, don\'t wait long). Leader election: phi > 12 (very confident before expensive operation). Single failure detector serves multiple use cases with different confidence requirements. The "accrual" emphasizes this gradual accumulation of suspicion rather than instant judgment. Option 1 misinterprets "accrual" as buffering. Option 3 confuses it with counting failures. Option 4 relates to data collection but misses the key concept of gradually increasing suspicion.',
    },
  ];
