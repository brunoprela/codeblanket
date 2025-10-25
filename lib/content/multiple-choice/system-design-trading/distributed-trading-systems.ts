import { MultipleChoiceQuestion } from '@/lib/types';

export const distributedTradingSystemsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dts-mc-1',
      question:
        'In the CAP theorem, what does a trading firm typically prioritize for their Order Management System?',
      options: [
        'Consistency and Partition tolerance (CP)',
        'Availability and Partition tolerance (AP)',
        'Consistency and Availability (CA)',
        'Only Partition tolerance',
      ],
      correctAnswer: 1,
      explanation:
        'Trading firms choose AP (Availability + Partition tolerance) for OMS: better to keep accepting orders with potentially stale position data than stop trading entirely. Lost trading opportunity costs $thousands per minute. Risk of stale data mitigated with pre-trade buffers (stop at 90% of limit). Network partitions will happen (must have P), and availability generates revenue. CA impossible (cannot avoid partitions). CP would stop trading during partitions (unacceptable).',
    },
    {
      id: 'dts-mc-2',
      question:
        'Why is PTP (Precision Time Protocol) preferred over NTP for high-frequency trading?',
      options: [
        'PTP is cheaper to implement than NTP',
        'PTP provides <1μs accuracy vs NTP 1-50ms accuracy',
        'PTP works across the internet, NTP only in LANs',
        'PTP is required by SEC regulations',
      ],
      correctAnswer: 1,
      explanation:
        'PTP provides sub-microsecond accuracy (<1μs) vs NTP millisecond accuracy (1-50ms). Critical for HFT where order priority determined by timestamp—1μs difference can mean winning vs losing order. PTP uses hardware-timestamped NICs ($500-2K each) and GPS-synchronized grandmaster clocks. More expensive than NTP (not cheaper). PTP typically used within datacenter (LAN), not internet. Not regulatory requirement, just competitive necessity for HFT.',
    },
    {
      id: 'dts-mc-3',
      question:
        'What is the primary purpose of using quorum in distributed trading systems?',
      options: [
        'To improve latency by parallel processing',
        'To prevent split-brain scenarios during network partitions',
        'To reduce hardware costs through redundancy',
        'To comply with regulatory requirements',
      ],
      correctAnswer: 1,
      explanation:
        "Quorum prevents split-brain: requires majority (N/2 + 1) agreement before accepting orders. During partition, only majority partition can make progress—minority safely rejects orders. Example: 3 regions partitioned into 2 vs 1. Majority (2) continues trading, minority (1) stops (safe). Without quorum, both sides think they're in charge → conflicting orders → data corruption. Not for latency (adds latency via cross-region coordination), not for cost reduction, not regulatory requirement. Pure safety mechanism.",
    },
    {
      id: 'dts-mc-4',
      question:
        'In a multi-region trading system, why would you use synchronous replication for positions but asynchronous for historical orders?',
      options: [
        'Synchronous is faster for small data',
        'Positions are critical for risk decisions, historical orders are not',
        'Asynchronous replication is more reliable',
        'Regulatory requirements mandate synchronous for positions',
      ],
      correctAnswer: 1,
      explanation:
        "Positions critical for risk decisions: must be accurate across regions to prevent limit breaches. Synchronous replication ensures all regions see same position before accepting new order. Cost: 80-150ms latency (acceptable for order submission). Historical orders: Used for analytics, not time-critical. Asynchronous replication (no blocking) avoids latency. Eventual consistency acceptable—historical data doesn't affect live trading decisions. Not about speed (synchronous slower), not about reliability (both reliable), not regulatory (though audit trail required).",
    },
    {
      id: 'dts-mc-5',
      question:
        'What is the minimum network latency between Chicago and New York due to physical distance?',
      options: [
        '1 millisecond',
        '4 milliseconds',
        '6 milliseconds',
        '10 milliseconds',
      ],
      correctAnswer: 2,
      explanation:
        'Physical limit: Chicago to NY = 1,200 km. Light in fiber optic = 200,000 km/s (2/3 speed in vacuum due to refractive index). Time = 1,200 / 200,000 = 0.006 seconds = 6ms one-way. With microwave (straight line, faster than fiber): ~4ms. Actual latency higher due to routing, switches, processing. Cannot beat physics—no software optimization can reduce below 6ms via fiber. This is why co-location matters: same datacenter = <1ms vs cross-country = 6ms+.',
    },
  ];
