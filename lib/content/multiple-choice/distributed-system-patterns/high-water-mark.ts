/**
 * Multiple choice questions for High-Water Mark section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const highwatermarkMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In Apache Kafka, what is the difference between the Log End Offset (LEO) and the High-Water Mark (HWM)?',
    options: [
      'They are the same thing with different names',
      'LEO is the next offset to write; HWM is the last offset replicated to all in-sync replicas',
      'LEO is managed by consumers; HWM is managed by producers',
      'HWM is always greater than LEO',
    ],
    correctAnswer: 1,
    explanation:
      "LEO (Log End Offset) is the next offset to be written—it's the highest offset plus 1, representing how much data a replica has received. HWM (High-Water Mark) is the highest offset that has been replicated to all in-sync replicas (ISR) and is safe to read. The key difference is that data exists between HWM and LEO that is uncommitted—it's on some replicas but not yet replicated to all ISR members. For example, if Leader has LEO=10, Follower1 LEO=9, Follower2 LEO=9, then HWM=min(10,9,9)=9. Messages at offset 9 are uncommitted and not visible to consumers. This prevents consumers from seeing data that might disappear if the leader fails. HWM ≤ LEO always (never greater). Option 1 is incorrect—they're distinct concepts. Option 3 is incorrect—both are managed by brokers. Option 4 is incorrect—HWM is always ≤ LEO.",
  },
  {
    id: 'mc2',
    question:
      'When a Kafka leader fails and a new leader is elected, what must the new leader and other replicas do with their logs?',
    options: [
      'Keep all data unchanged and continue appending',
      'Truncate their logs to the High-Water Mark to ensure consistency',
      'Delete all data and rebuild from scratch',
      "Merge their logs with the previous leader's log",
    ],
    correctAnswer: 1,
    explanation:
      "When a new leader is elected, all replicas (including the new leader and the old leader when it recovers) must truncate their logs to the High-Water Mark from before the failure. This ensures consistency across all replicas by discarding uncommitted data that wasn't replicated to all ISR members. Example: Before failure, Leader had offsets 0-5, HWM=4, Follower had offsets 0-4. Leader fails. New leader (former follower) elected. New leader truncates to HWM=4 (nothing to truncate). When old leader recovers, it truncates offsets 5 (beyond HWM=4). All replicas now agree on offsets 0-4. This data is consistent because it was replicated to all ISR members. Offset 5 is lost, but consumers never saw it (it was beyond HWM). This trade-off (losing uncommitted data) is necessary to maintain consistency and prevent divergence. Option 1 would cause inconsistency. Option 3 is wasteful—most data is correct. Option 4 is complex and unnecessary.",
  },
  {
    id: 'mc3',
    question:
      'Why does Kafka only allow consumers to read up to the High-Water Mark, not the Log End Offset?',
    options: [
      'To improve read performance by caching data at the HWM',
      'To prevent consumers from seeing data that might be lost if the leader fails',
      'To enforce message ordering guarantees',
      'To allow producers to modify messages before they become visible',
    ],
    correctAnswer: 1,
    explanation:
      "Consumers are restricted to reading up to the HWM to prevent phantom reads—seeing data that later disappears. Data beyond HWM is uncommitted (not yet replicated to all in-sync replicas) and could be lost if the leader fails before replication completes. Example: Leader has messages up to offset 10 (LEO=11), but HWM=8 (only offsets 0-8 replicated to all ISR). Consumer reads and sees message at offset 9. Leader crashes. New leader elected (from ISR), which only has offsets 0-8. Offset 9 lost (wasn't committed). Consumer saw data that no longer exists! To prevent this, Kafka only allows reads up to HWM. Once HWM advances to include an offset (meaning it's replicated), consumers can read it, and it's guaranteed durable even if the leader fails. This trade-off (slightly stale reads) ensures consumers never see data that could vanish. Option 1 is incorrect—caching is separate. Option 3 is incorrect—ordering is maintained differently. Option 4 is incorrect—messages are immutable once written.",
  },
  {
    id: 'mc4',
    question:
      'What happens to the High-Water Mark when a replica falls out of the In-Sync Replica (ISR) set in Kafka?',
    options: [
      'The HWM stops advancing until the replica rejoins the ISR',
      'The HWM is recalculated without the slow replica, potentially advancing',
      'The HWM resets to 0 for safety',
      'The HWM remains unchanged indefinitely',
    ],
    correctAnswer: 1,
    explanation:
      "When a replica falls out of the ISR (In-Sync Replica set) because it's too slow, the HWM is recalculated using only the remaining ISR members, which often allows the HWM to advance faster. HWM = min(LEO of all ISR members). If a slow replica is holding back the HWM, removing it from ISR allows HWM to catch up to the faster replicas. Example: Leader LEO=100, Follower1 LEO=99, Follower2 LEO=80 (slow). Initially: HWM = min(100, 99, 80) = 80 (slow replica holds HWM back). Follower2 falls out of ISR (too far behind). Now: HWM = min(100, 99) = 99 (HWM advances!). Producers using acks=all now only wait for Leader and Follower1, improving throughput. The trade-off is reduced durability (only 2 replicas instead of 3) until Follower2 catches up and rejoins ISR. This design prioritizes availability (continue making progress) over temporarily losing one replica's redundancy. Option 1 is incorrect—HWM doesn't stop. Option 3 is incorrect—resetting would be catastrophic. Option 4 is incorrect—HWM is continuously recalculated.",
  },
  {
    id: 'mc5',
    question:
      'In Kafka producer configurations, what does acks=all guarantee in relation to the High-Water Mark?',
    options: [
      'The message is immediately visible to consumers',
      'The message is written to disk on the leader',
      'The message is replicated to all in-sync replicas and the HWM advances past it',
      'The message is sent to all partitions',
    ],
    correctAnswer: 2,
    explanation:
      "acks=all (or acks=-1) guarantees that the message is replicated to all in-sync replicas before the producer receives acknowledgment. This means the message is durable and the HWM will advance to include it (making it visible to consumers). The process: Producer sends message to leader. Leader writes locally and replicates to all ISR members. All ISR members acknowledge. HWM advances to include the new message. Leader acknowledges to producer. Only then does the producer receive confirmation. This provides the strongest durability guarantee—the message survives even if the leader fails immediately after acknowledging, because it's on all ISR members. In contrast, acks=1 only waits for the leader's write (fast but less durable), and acks=0 doesn't wait at all (fastest but no durability guarantee). Option 1 is close but imprecise—it's visible after HWM advances. Option 2 is insufficient—needs replication, not just leader write. Option 4 is incorrect—acks doesn't relate to partitions.",
  },
];
