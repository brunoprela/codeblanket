/**
 * Multiple choice questions for Lease section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const leaseMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the key difference between a lock and a lease in distributed systems?',
    options: [
      'Locks are faster than leases',
      'Leases automatically expire after a time period; locks do not',
      'Locks can only be held by one node; leases can be held by multiple nodes',
      'Leases require network communication; locks do not',
    ],
    correctAnswer: 1,
    explanation:
      "The key difference is that leases are time-bound and automatically expire, while locks are held indefinitely until explicitly released. This automatic expiration is crucial for distributed systems fault tolerance. With a traditional lock, if a node crashes while holding a lock, that resource is locked forever (deadlock) unless manually intervened. With a lease, if a node crashes while holding it, the lease expires automatically after its duration (e.g., 30 seconds), and another node can acquire it. Example: Node A holds a lock on resource X and crashes. Without automatic expiration, X is locked forever. With a lease of 30s, X becomes available at most 30s after A crashes. This trades brief unavailability (the expiration time) for automatic recovery. Option 1 is incorrect—performance isn't the key distinction. Option 3 is incorrect—both typically provide exclusive access. Option 4 is incorrect—both may require network communication in distributed systems.",
  },
  {
    id: 'mc2',
    question:
      'If a node holds a 30-second lease and wants to maintain it, when should it typically renew the lease?',
    options: [
      'After 29 seconds (just before expiration)',
      'After 10-15 seconds (33-50% of the lease duration)',
      'Immediately after acquiring it',
      'Never, leases renew automatically',
    ],
    correctAnswer: 1,
    explanation:
      "Best practice is to renew at 33-50% of the lease duration, which for a 30-second lease means renewing every 10-15 seconds. This provides multiple renewal attempts before expiration, accounting for network delays, coordinator processing time, and potential transient failures. Example: Renew at 10s, 20s. If the first renewal at 10s fails (network blip), there's a second attempt at 20s, and even a third could be tried at 25s before the 30s expiration. Renewing too late (option 1 at 29s) is dangerous—a single network delay causes the lease to expire even though the node is healthy, leading to false failures and unnecessary leadership changes. Renewing immediately (option 3) wastes resources. Leases don't auto-renew (option 4)—that's the point; failure to renew indicates the node is dead or partitioned. Modern systems like etcd use ~33% (renew at 10s for 30s lease), Kubernetes uses ~67% for shorter leases (renew at 10s for 15s lease), accepting slightly more overhead for a shorter lease duration.",
  },
  {
    id: 'mc3',
    question: 'How do leases prevent split-brain in a leader election system?',
    options: [
      'By encrypting all communication between nodes',
      'By ensuring only one node can hold the leader lease at a time, with automatic expiration',
      'By requiring all nodes to agree on the leader through voting',
      'By using the fastest network connection as the leader',
    ],
    correctAnswer: 1,
    explanation:
      "Leases prevent split-brain by ensuring at most one node holds the leader lease at any given time, with automatic expiration preventing stale leaders from continuing to operate. When a network partition occurs, the leader in the minority partition cannot renew its lease (can't reach the coordinator or majority of nodes). The lease expires automatically, and the node must stop acting as leader. Meanwhile, the majority partition can elect a new leader with a new lease. The time-boundedness is crucial: there may be a brief period with no leader (during lease expiration), but there's never two leaders simultaneously. Example: Leader A with 30s lease, network partition at T=10s. A tries to renew at T=20s, fails (partitioned). A's lease expires at T=30s, A stops acting as leader. Majority partition elects B at T=31s with new lease. Even if A's network recovers at T=35s, A won't act as leader (lease expired). Fencing tokens add additional safety (option 1's mechanism of rejecting stale leaders). Option 2 describes quorum-based election (complementary, not the lease mechanism itself). Option 4 is nonsensical.",
  },
  {
    id: 'mc4',
    question:
      'What is a major challenge when using leases in systems with clock skew between nodes?',
    options: [
      'Leases become permanently locked',
      "A node may think its lease has expired when it hasn't (or vice versa) based on its clock",
      'Lease duration must be longer than one hour',
      'Multiple nodes can hold the same lease simultaneously',
    ],
    correctAnswer: 1,
    explanation:
      "Clock skew can cause a node to incorrectly determine when its lease expires, leading to premature termination or overstaying. Example: Coordinator grants lease to Node A expiring at 12:00:30 (coordinator time). Node A's clock is 10 seconds fast (shows 12:00:10 when coordinator shows 12:00:00). Node A thinks lease expires at 12:00:30 (its time) = 12:00:20 (coordinator time). Node A stops acting as leader at coordinator time 12:00:20, even though lease valid until 12:00:30. System has no leader for 10 seconds unnecessarily. Reverse problem: Node A's clock slow, thinks lease still valid when it's expired, continues operating (stale leader). Solutions: (1) Use coordinator's time for expiry, not local time. (2) Renew conservatively at 33% of duration (large margin for clock skew). (3) Require NTP synchronization (keep clocks within 100ms). (4) Monitor clock skew, alert if excessive. Option 1 is incorrect—leases still expire. Option 3 is incorrect—duration isn't constrained by clock skew. Option 4 describes split-brain, which leases prevent through time-boundedness, not a consequence of clock skew.",
  },
  {
    id: 'mc5',
    question:
      'In lease-based distributed locking, what happens if a node is partitioned from the coordinator and its lease expires?',
    options: [
      'The node continues to operate as if it still holds the lease',
      'The node must immediately stop acting on the lease and cannot perform any operations requiring it',
      'The node can request a new lease from a different coordinator',
      'The partition is resolved automatically',
    ],
    correctAnswer: 1,
    explanation:
      "When a node is partitioned and cannot renew its lease, it must immediately stop acting on the lease the moment it expires. This is critical for correctness—the node cannot perform operations requiring the lease (e.g., writes if it's a leader, modifications to a locked resource). The node should proactively check: \"Can I reach the coordinator? Is my lease still valid?\" If either answer is no, stop immediately. Example: Node A holds lease on partition X, expires at T=30s. At T=15s, network partition prevents A from reaching coordinator. A tries to renew at T=20s, fails. A tries again at T=25s, fails. At T=30s, lease expires. A must stop processing requests for partition X immediately, even though A is still running and could serve requests (data is in memory). Continuing would risk split-brain—another node may acquire the lease at T=31s and start processing. Both operating simultaneously would cause data inconsistency. Option 1 would cause split-brain. Option 3 is incorrect—there's typically one coordinator (or majority quorum). Option 4 is incorrect—partitions don't auto-resolve; the system must handle them correctly.",
  },
];
