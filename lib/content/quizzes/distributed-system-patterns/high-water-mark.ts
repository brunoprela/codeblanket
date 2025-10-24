/**
 * Quiz questions for High-Water Mark section
 */

export const highwatermarkQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between Log End Offset (LEO) and High-Water Mark (HWM) in Kafka, and why this distinction is critical for consistency.',
    sampleAnswer:
      "LEO and HWM serve different purposes in maintaining consistency across replicas. Log End Offset (LEO): The next offset to be written—highest offset + 1. Each replica has its own LEO, indicating how much data it has received. Example: Replica A has messages [0,1,2,3], LEO=4 (next write goes to offset 4). High-Water Mark (HWM): The last offset that has been replicated to all in-sync replicas (ISR) and is safe to read. Calculated as min(LEO of all ISR members). Only the leader knows the authoritative HWM. Why distinction matters: Uncommitted data exists between HWM and LEO. Example: Leader has LEO=10, Follower1 LEO=9, Follower2 LEO=8, ISR={Leader, Follower1, Follower2}. HWM=min(10,9,8)=8. Messages 0-8 are committed and readable. Messages 9 are uncommitted—not safe to expose to consumers. If leader crashes, message 9 may be lost (not replicated to majority). Consistency guarantee: Consumers only read up to HWM, never seeing data that could disappear. When leader fails and new leader elected, both will have all messages up to HWM (by definition of HWM). New leader may discard messages beyond HWM (uncommitted). This prevents phantom reads where consumer sees data that later vanishes. Without HWM distinction, consumers could read leader's LEO(all data), including uncommitted messages, violating consistency.",
    keyPoints: [
      'LEO: Next offset to write (highest offset + 1), per-replica value indicating data received',
      'HWM: Last committed offset (min LEO of ISR), safe to read, only leader knows authoritatively',
      'Uncommitted data: Between HWM and LEO, not replicated to all ISR, may be lost',
      'Consistency: Consumers read only up to HWM, preventing phantom reads',
      'Leader failover: New leader truncates to HWM, ensuring consistency across failovers',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through what happens to the High-Water Mark during a leader failure and election in a 3-replica Kafka partition. Why must replicas truncate their logs to the HWM?',
    sampleAnswer:
      "Leader failure scenario demonstrates why HWM truncation is critical for consistency. Before failure: Leader (Broker1): Log [0,1,2,3,4,5], LEO=6, HWM=4. Follower1 (Broker2, in ISR): Log [0,1,2,3,4], LEO=5, HWM=4 (from leader). Follower2 (Broker3, in ISR): Log [0,1,2,3,4], LEO=5, HWM=4. HWM=4 means offsets 0-4 are committed (replicated to all ISR). Offsets 5 are uncommitted (only on leader). Failure: Broker1 (leader) crashes. Kafka controller detects failure, triggers election. Election: Broker2 elected as new leader (has most up-to-date log among available replicas). Truncation phase (critical): Broker2 checks its HWM=4 from before crash. Broker2 truncates its log to HWM: deletes uncommitted offset 5 (wasn't in HWM).Broker3 also truncates to HWM=4(deletes offset 5).New leader state: Broker2: Log [0, 1, 2, 3, 4], LEO=5, HWM=4(initially).Broker3: Log [0, 1, 2, 3, 4], LEO=5, HWM=4. When Broker1 recovers: Broker1 sees Broker2 is leader.Broker1 also truncates to HWM=4(deletes offsets 5).Broker1 becomes follower, syncs from Broker2.Why truncation is necessary: Without truncation, Broker1 would keep offset 5, Broker2/ 3 don't have it. Inconsistency! Consumers never saw offset 5 (it was beyond HWM). By truncating, all replicas agree on log [0,1,2,3,4]. New writes start at offset 5 (may be different data than old uncommitted offset 5). Result: Consistent state across all replicas, no data consumer saw was lost.",
    keyPoints: [
      'Before failure: Leader has uncommitted data (beyond HWM), followers behind LEO',
      'Election: New leader elected from ISR members with most up-to-date log',
      'Truncation: All replicas truncate logs to HWM, discarding uncommitted data',
      'Consistency: All replicas agree on committed data (up to HWM), no divergence',
      'Recovery: Old leader truncates when it comes back, syncs from new leader',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how In-Sync Replicas (ISR) relate to HWM calculation, and what happens to HWM advancement when a replica falls out of ISR.',
    sampleAnswer:
      "ISR and HWM are tightly coupled—ISR membership directly determines HWM value. ISR Definition: Set of replicas that are caught up with the leader (within replica.lag.time.max.ms, default 10s). Only ISR members participate in HWM calculation. HWM Calculation: HWM = min(LEO of all replicas in ISR). Example: Leader (LEO=100), Follower1 (LEO=99, in ISR), Follower2 (LEO=98, in ISR). HWM = min(100, 99, 98) = 98. Follower falls out of ISR: Suppose Follower2 becomes slow (network issue, overloaded). Follower2 falls behind: LEO=80 (20 offsets behind). If not caught up within replica.lag.time.max.ms (10s), removed from ISR. ISR shrinks: ISR = {Leader, Follower1}. HWM recalculation: HWM = min(100, 99) = 99. HWM advanced! Follower2 no longer holds back HWM. Impact: Write availability improved—producer (acks=all) only waits for Leader + Follower1. Durability slightly reduced—only 2 replicas instead of 3. Under-replicated partition warning triggered. Follower returns to ISR: Follower2 catches up (network fixed), LEO=99. Once within lag threshold, rejoins ISR. ISR = {Leader, Follower1, Follower2}. HWM recalculation: HWM = min(100, 99, 99) = 99 (unchanged in this case, but now considers Follower2 again). Why this design: Slow replicas don't block writes indefinitely(availability).Leader election still possible from ISR members(consistency).Durability maintained as long as majority of replicas in ISR.Production monitoring: Alert on under - replicated partitions(ISR < replication factor).",
    keyPoints: [
      'ISR: Replicas caught up with leader, participate in HWM calculation',
      'HWM = min(LEO of ISR members): Slow replica in ISR holds back HWM',
      'Falling out of ISR: Replica too slow removed from ISR, HWM advances without it',
      'Improved availability: Writes faster (wait for fewer replicas), reduced durability temporarily',
      'Rejoining ISR: Replica catches up, added back to ISR, participates in HWM again',
    ],
  },
];
