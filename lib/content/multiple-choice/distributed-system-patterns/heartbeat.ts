/**
 * Multiple choice questions for Heartbeat section
 * Multiple choice questions for Gossip Protocol section
 * Multiple choice questions for Phi Accrual Failure Detector section
 * Multiple choice questions for Split-Brain Resolution section
 * Multiple choice questions for Hinted Handoff section
 * Multiple choice questions for Read Repair section
 * Multiple choice questions for Anti-Entropy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const heartbeatMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In a distributed system with 100 nodes using an all-to-all heartbeat pattern, how many heartbeat messages are sent per interval?',
    options: [
      '100 messages',
      '200 messages',
      '9,900 messages',
      '10,000 messages',
    ],
    correctAnswer: 2,
    explanation:
      "In an all-to-all heartbeat pattern, every node sends a heartbeat to every other node, resulting in N×(N-1) messages per interval. For 100 nodes: 100 × 99 = 9,900 messages per interval. Each node sends 99 heartbeats (to all others except itself). This O(n²) message complexity is why all-to-all doesn't scale beyond small clusters (~50-100 nodes). For 1000 nodes, it would be 999,000 messages! In contrast, leader-based heartbeat (all send to leader) would be O(n) = 100 messages, and gossip protocol with fanout=3 would be 3×100 = 300 messages. The benefit of all-to-all is fast, distributed failure detection with no single point of failure—any node can detect any failure immediately. The cost is network overhead. Option 1 (100) would be leader-based. Option 2 (200) would be bidirectional single pair. Option 4 (10,000) would include self-heartbeats, which aren't needed.",
  },
  {
    id: 'mc2',
    question:
      'Why are randomized election timeouts important in Raft leader election based on heartbeat timeouts?',
    options: [
      'To reduce network bandwidth usage',
      "To prevent split votes by ensuring nodes don't all become candidates simultaneously",
      'To make the system more unpredictable for security',
      'To balance load across all nodes equally',
    ],
    correctAnswer: 1,
    explanation:
      "Randomized election timeouts prevent split votes by ensuring nodes don't all transition to candidate state and start elections simultaneously. If all nodes had the same timeout (e.g., 5 seconds), when the leader fails, all would timeout together, all become candidates, all request votes, and the votes would be split evenly—no one gets majority, requiring another election round. With randomization (e.g., 1.5-3 seconds), one node times out first, becomes candidate, requests votes, and likely wins before others timeout. Example: Nodes A, B, C have timeouts 1.5s, 2.1s, 2.8s. Leader fails at T=0. A times out at T=1.5s, becomes candidate, requests votes. B and C still followers, vote for A. A wins with majority (3/3 votes) before B times out. Fast election, no split votes. Without randomization, elections could take multiple rounds as nodes keep splitting votes. Raft paper specifies randomization is critical for liveness (eventual progress). Options 1, 3, and 4 are incorrect—randomization is specifically about preventing simultaneous candidacy.",
  },
  {
    id: 'mc3',
    question:
      'What is the recommended relationship between heartbeat interval and timeout threshold to minimize false positives?',
    options: [
      'Timeout should equal heartbeat interval',
      'Timeout should be 0.5x the heartbeat interval',
      'Timeout should be 3-10x the heartbeat interval',
      'Timeout should be 100x the heartbeat interval',
    ],
    correctAnswer: 2,
    explanation:
      'Timeout should typically be 3-10x the heartbeat interval to allow multiple missed heartbeats before declaring failure, reducing false positives from transient network issues. Example: Heartbeat every 1 second, timeout 5 seconds. This allows 5 consecutive heartbeats to miss before failure declared, making it unlikely a healthy node is incorrectly flagged. If timeout = 1× interval (option 1), a single dropped packet triggers false failure (too sensitive). If timeout = 0.5× interval (option 2), physically impossible to succeed (timeout before first heartbeat arrives). If timeout = 100× interval (option 4), failure takes 100 seconds to detect (too slow for most systems). The multiplier choice depends on network reliability: Stable network (low latency, rare packet loss): 3× multiplier okay (faster detection). Unreliable network (high latency, frequent packet loss): 10× multiplier needed (avoid false positives). Production examples: Raft uses ~10× (leader heartbeat 150ms, follower timeout 1500ms). Cassandra uses ~10× (gossip 1s, failure detection ~10s with phi accrual).',
  },
  {
    id: 'mc4',
    question:
      'In the context of heartbeats for leader election vs heartbeats for failure detection, which typically requires faster heartbeat intervals?',
    options: [
      'Failure detection requires faster intervals',
      'Leader election requires faster intervals',
      'Both require the same interval',
      'Neither uses heartbeats',
    ],
    correctAnswer: 1,
    explanation:
      "Leader election typically requires faster heartbeat intervals because the time without a leader directly impacts system availability—the faster a new leader is elected, the shorter the unavailability window. In contrast, failure detection for cluster membership can tolerate slightly longer intervals since it's about eventually removing dead nodes, not immediate system availability. Example intervals: Raft leader heartbeat (leader election context): 150ms interval, 1500ms timeout. Fast to ensure quick re-election if leader fails (system operational again in 1.5s). Cassandra gossip (failure detection context): 1s interval, ~10s timeout (with phi accrual). Slower because removing a node from cluster isn't time-critical (requests can be retried or routed elsewhere). The reasoning: Leader failure = system can't make progress (writes blocked) = need fast election. Node failure = system continues with remaining nodes = slower detection okay. Trade-off: Faster heartbeats = more network/CPU overhead. Leader election justifies this cost due to availability impact. Failure detection prefers lower overhead with acceptable (slightly slower) detection.",
  },
  {
    id: 'mc5',
    question:
      'What is the purpose of using indirect pings in the SWIM failure detection protocol?',
    options: [
      'To encrypt heartbeat messages for security',
      'To distinguish between node failure and network partition between specific nodes',
      'To reduce the number of heartbeat messages needed',
      'To automatically repair failed nodes',
    ],
    correctAnswer: 1,
    explanation:
      "Indirect pings in SWIM (Scalable Weakly-consistent Infection-style Process Group Membership) distinguish between true node failure and network issues between specific nodes. Process: Node A pings Node B directly. If no response, A doesn't immediately declare B dead. Instead, A asks K other nodes (e.g., C, D, E) to ping B indirectly. If any indirect ping succeeds (e.g., C reaches B), B is alive—there's a network issue between A and B specifically. If all indirect pings fail, B is likely dead (all nodes can't reach it). Example: A-B link broken (asymmetric network failure) but B alive. Direct ping A→B fails. Indirect pings C→B, D→B, E→B succeed. Conclusion: B is alive, network issue between A and B only. Without indirect pings, A would incorrectly declare B dead, causing false failure detection. SWIM's use of indirect pings significantly reduces false positives in systems with complex network topologies or transient link failures. Options 1, 3, and 4 are incorrect—indirect pings are specifically about accurate failure detection.",
  },
];
