/**
 * Quiz questions for Distributed Transactions section
 */

export const distributedTransactionsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the blocking problem in Two-Phase Commit (2PC). Why is this a critical issue?',
    sampleAnswer:
      "2PC blocking problem: Coordinator fails after PREPARE phase (after participants voted COMMIT), before sending COMMIT/ABORT decision. Participants are stuck in 'prepared' state - holding locks, waiting for decision they'll never receive. They can't commit (coordinator might have decided abort), can't abort (coordinator might have decided commit). Resources locked indefinitely! Impact: (1) Locked data unavailable to other transactions → system unusable. (2) Cascading failures - blocked transactions block others. (3) Manual intervention required (DBA must decide commit/abort). (4) No automatic recovery. Example: Transfer $100 between accounts. Both accounts prepared, locks held. Coordinator crashes. Accounts locked, can't complete transfer, can't start new transfers. System degraded until manual fix. Solutions: (1) Timeouts (participants abort after timeout, but risky - coordinator might decide commit). (2) 3PC (adds CanCommit phase, non-blocking but has other issues). (3) Avoid 2PC entirely (Saga pattern, eventual consistency). This is why distributed transactions are avoided when possible - they're brittle and block on failures.",
    keyPoints: [
      'Coordinator fails after PREPARE, before COMMIT/ABORT decision',
      'Participants stuck holding locks, waiting forever',
      'Manual intervention required to resolve',
      'System degraded until resolved',
      'Avoid 2PC when possible - use Saga or eventual consistency',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Saga pattern (choreography vs orchestration) for distributed transactions. When would you use each?',
    sampleAnswer:
      'Saga: Long-running transactions via local transactions + compensating transactions. Choreography (event-driven): Services react to events, no central controller. OrderService creates order → publishes OrderCreated → InventoryService reserves → publishes InventoryReserved → PaymentService charges → publishes PaymentCharged. On failure: Publish failure event → compensating events flow backwards. Benefits: (1) Decoupled services. (2) No single point of failure. (3) Scales well. Drawbacks: (1) Hard to understand flow (scattered across services). (2) Difficult to debug. (3) Cyclic dependencies possible. Orchestration (coordinator): Central orchestrator calls services sequentially. OrderOrchestrator: call Inventory.reserve() → call Payment.charge() → call Shipment.create(). On failure: Orchestrator calls compensating actions in reverse. Benefits: (1) Centralized logic (easy to understand). (2) Easy to debug. (3) Explicit flow. Drawbacks: (1) Orchestrator = single point of failure. (2) Orchestrator = coupling point. (3) Orchestrator can bottleneck. Use choreography when: Microservices, event-driven architecture, decoupling critical. Use orchestration when: Need visibility, complex logic, team prefers centralized control. Hybrid possible: Orchestrate high-level, choreograph within bounded contexts.',
    keyPoints: [
      'Choreography: Event-driven, decoupled, hard to understand',
      'Orchestration: Central coordinator, easy to understand, coupling',
      'Both use compensating transactions for rollback',
      'Choreography for: Decoupling, scalability',
      'Orchestration for: Visibility, complex logic',
    ],
  },
  {
    id: 'q3',
    question:
      'Why does Google Spanner use specialized hardware (TrueTime)? What problem does it solve?',
    sampleAnswer:
      "Spanner provides global strong consistency despite being distributed worldwide. Challenge: Ordering events across continents with unreliable clocks. Clock skew = 100+ms typically. Can't use local timestamps for global ordering. TrueTime: API that returns time interval [earliest, latest] with bounded uncertainty (< 10 ms). Uses GPS + atomic clocks in every datacenter. Solves problem: (1) Transaction gets timestamp. (2) Wait out uncertainty before commit (if uncertainty = 7ms, wait 7ms). (3) Guarantees: If T1 commits before T2 starts, T1.timestamp < T2.timestamp. (4) External consistency (stronger than serializability). This enables: (1) Globally consistent snapshots (read at timestamp T sees all commits < T). (2) Strong consistency reads across continents. (3) Distributed transactions without 2PC blocking. Cost: (1) Requires specialized hardware (GPS + atomic clocks). (2) Higher latency (wait out uncertainty). (3) Google-only tech (though Cloud Spanner available as service). Why important: First practical global strongly-consistent database. Proves it's possible with right infrastructure. Inspired CockroachDB, YugabyteDB (without specialized hardware, using hybrid logical clocks).",
    keyPoints: [
      'TrueTime: Global clock with bounded uncertainty (<10ms)',
      'Wait out uncertainty before commit',
      'Enables external consistency (global ordering)',
      'Requires specialized hardware (GPS + atomic clocks)',
      'Enables global strong consistency (breakthrough)',
    ],
  },
];
