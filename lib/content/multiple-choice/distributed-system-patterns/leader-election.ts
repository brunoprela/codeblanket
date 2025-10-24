/**
 * Multiple choice questions for Leader Election section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const leaderelectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In the Raft consensus algorithm, what happens when a follower does not receive heartbeats from the leader within the election timeout period?',
    options: [
      'The follower immediately declares itself as the new leader',
      'The follower becomes a candidate and requests votes from other nodes',
      'The follower continues waiting and doubles its election timeout',
      'The follower contacts the previous leader to check if it is still alive',
    ],
    correctAnswer: 1,
    explanation:
      'In Raft, when a follower election timeout expires without receiving heartbeats from the leader, it transitions to the candidate state and initiates a new election by requesting votes from other nodes. It increments its term number, votes for itself, and sends RequestVote RPCs to all other nodes. If a candidate receives votes from a majority, it becomes the new leader. This randomized timeout mechanism (typically 150-300ms with jitter) prevents split votes and ensures quick leader election. Option 1 is incorrect because immediately declaring oneself leader would cause split-brain. Option 3 is incorrect because Raft does not double timeouts—it uses randomized timeouts to prevent synchronized elections. Option 4 is incorrect because the follower does not contact the previous leader; it assumes the leader is dead and initiates an election.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary purpose of using fencing tokens in a leader election system?',
    options: [
      'To encrypt communication between the leader and followers',
      'To prevent stale leaders from performing operations after a new leader has been elected',
      'To prioritize certain nodes to always become the leader',
      'To reduce the network overhead of heartbeat messages',
    ],
    correctAnswer: 1,
    explanation:
      'Fencing tokens are monotonically increasing numbers assigned to each elected leader. Their primary purpose is to prevent stale leaders (who may not know they have been replaced) from corrupting data. When a new leader is elected, it receives a higher token number. Resources (databases, file systems) track the highest token seen and reject operations from leaders with lower tokens. Example: Old leader has token=5, network partition occurs, new leader elected with token=6. When both try to write, resources accept token=6 and reject token=5, preventing the stale leader from causing damage even if quorum-based election fails. This provides defense-in-depth. Option 1 is incorrect—fencing tokens are for ordering/validation, not encryption. Option 3 is incorrect—tokens ensure safety, not priority. Option 4 is incorrect—tokens are carried in operations, not related to heartbeat overhead.',
  },
  {
    id: 'mc3',
    question:
      'In a 5-node cluster using quorum-based leader election, the network partitions into {Node A, Node B} and {Node C, Node D, Node E}. What happens?',
    options: [
      'Both partitions can elect leaders since they can still communicate internally',
      'Node A remains leader since it was the leader before the partition',
      'Only the {C, D, E} partition can elect a leader as it has the majority',
      'The cluster enters read-only mode and no elections occur',
    ],
    correctAnswer: 2,
    explanation:
      'With quorum-based leader election, only a partition containing a majority (more than half) of nodes can elect a leader. In a 5-node cluster, quorum is 3 nodes (⌈5/2⌉ = 3). The {C, D, E} partition has 3 nodes (≥ quorum), so it can elect a leader. The {A, B} partition has only 2 nodes (< quorum), so it cannot elect a leader and becomes unavailable. This is the fundamental mechanism that prevents split-brain: only one partition can have a majority, ensuring at most one active leader. This trades availability (minority partition is down) for consistency (no split-brain). Option 1 is incorrect because majority is required, not just internal communication. Option 2 is incorrect because Node A must give up leadership if it cannot reach a majority. Option 4 is incorrect because one partition (the majority) continues operating normally.',
  },
  {
    id: 'mc4',
    question:
      'Why is the Bully algorithm generally not recommended for large distributed systems (>100 nodes)?',
    options: [
      'It does not guarantee that a leader will be elected',
      'It has O(n²) message complexity in the worst case',
      'It requires all nodes to have synchronized clocks',
      'It cannot handle network partitions',
    ],
    correctAnswer: 1,
    explanation:
      'The Bully algorithm has O(n²) message complexity in the worst case. When the node with the lowest ID detects leader failure, it sends election messages to all higher-ID nodes. Each of those nodes then sends election messages to their higher-ID nodes, creating a cascade. In a cluster of 100 nodes, this could result in roughly 5,000 election messages (100×99/2). For 1,000 nodes, it is nearly 500,000 messages, causing a message storm that can overwhelm the network and slow down the election significantly. This makes it unsuitable for large clusters. In contrast, Raft uses O(n) messages—candidates send RequestVote to all nodes, and voting completes in one round. Option 1 is incorrect—Bully does guarantee election (node with highest ID wins). Option 3 is incorrect—Bully does not require clock sync, just node IDs. Option 4 is incorrect—while Bully can struggle with partitions, the main scalability issue is message complexity.',
  },
  {
    id: 'mc5',
    question:
      'In ZooKeeper-based leader election using ephemeral sequential nodes, how does a node determine if it is the leader?',
    options: [
      'It receives a "leader" message from ZooKeeper',
      'It checks if its node has the lowest sequence number among all nodes in the election directory',
      'It waits for a majority of nodes to vote for it',
      'It sends heartbeats and becomes leader if no one responds',
    ],
    correctAnswer: 1,
    explanation:
      'In ZooKeeper-based leader election, each candidate creates an ephemeral sequential node under a common election path (e.g., /election/n_0000000001, /election/n_0000000002, etc.). To determine leadership, a node simply checks if its sequence number is the lowest among all nodes in the election directory. If yes, it is the leader. If no, it watches the node with the next-lowest sequence number and waits. When the watched node disappears (node crash or connection loss), it checks again. This approach is simple, fair (based on order of joining), and avoids thundering herd (each node watches only one specific node). The ephemeral property ensures that if a leader crashes or loses connection to ZooKeeper, its node is automatically deleted, triggering the next node to become leader. This is much simpler than implementing Raft or Paxos yourself. Options 1, 3, and 4 describe mechanisms not used by ZooKeeper sequential node approach.',
  },
];
