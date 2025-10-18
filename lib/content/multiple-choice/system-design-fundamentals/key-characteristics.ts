/**
 * Multiple choice questions for Key Characteristics of Distributed Systems section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const keycharacteristicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of horizontal scaling over vertical scaling?',
    options: [
      'It is always cheaper',
      'It requires no code changes',
      'It can scale indefinitely and provides better fault tolerance',
      'It is simpler to implement',
    ],
    correctAnswer: 2,
    explanation:
      'Horizontal scaling (adding more machines) can scale indefinitely (no hardware limits like vertical) and provides fault tolerance (if one machine fails, others continue). Vertical scaling (bigger machines) hits hardware limits and has single point of failure. Trade-off: Horizontal is more complex (need load balancing, distributed state) but scales better. Not always cheaper initially (complexity costs), but more cost-effective at massive scale.',
  },
  {
    id: 'mc2',
    question:
      'A system has 99.99% availability. How much downtime per year is this?',
    options: [
      '8.7 hours',
      '52 minutes',
      '5.26 minutes',
      '0 minutes (perfect uptime)',
    ],
    correctAnswer: 1,
    explanation:
      '99.99% (four nines) = 52 minutes downtime per year. Calculation: 365 days × 24 hrs × 60 min = 525,600 min/year. 0.01% = 52.56 minutes. For reference: 99.9% = 8.7 hours, 99.999% = 5.26 minutes. Each additional nine requires exponentially more effort: redundancy, multi-region, automated failover, etc. Most SaaS apps target 99.9-99.99%.',
  },
  {
    id: 'mc3',
    question:
      'According to the CAP theorem, which two properties can you have during a network partition in a distributed system?',
    options: [
      'Consistency and Availability',
      'Consistency and Partition Tolerance, OR Availability and Partition Tolerance',
      'All three: Consistency, Availability, and Partition Tolerance',
      'Only Partition Tolerance',
    ],
    correctAnswer: 1,
    explanation:
      'During network partition, you must choose: CP (Consistency + Partition Tolerance): System rejects writes to stay consistent. Used for banking, payments. AP (Availability + Partition Tolerance): System accepts writes everywhere, becomes eventually consistent. Used for social media, DNS. Cannot have all three during partition. However, when no partition, can have both C and A.',
  },
  {
    id: 'mc4',
    question: 'What is the difference between latency and throughput?',
    options: [
      'They are the same thing',
      'Latency is time per operation; throughput is operations per time',
      'Latency is for databases; throughput is for networks',
      'Throughput is always higher than latency',
    ],
    correctAnswer: 1,
    explanation:
      'Latency: Time to complete one operation (e.g., 100ms per request). Throughput: Number of operations per time (e.g., 10,000 requests/second). Example: Video buffering: Low latency = quick start time. High throughput = smooth playback (many frames/sec). Can have: High latency + high throughput (batch processing), Low latency + low throughput (single-threaded operations). Often trade-off: Optimizing for one may hurt the other.',
  },
  {
    id: 'mc5',
    question: 'Which strategy does NOT help achieve high availability?',
    options: [
      'Deploying across multiple regions',
      'Using load balancers with health checks',
      'Storing all data on a single high-powered server',
      'Implementing automatic failover',
    ],
    correctAnswer: 2,
    explanation:
      'Single server is a single point of failure (SPOF) - if it fails, entire system goes down. Even if it\'s "high-powered," hardware failures happen. High availability requires: (1) Redundancy (multiple servers, datacenters, regions). (2) No single points of failure. (3) Automatic detection and failover. (4) Load balancing across instances. Single server might work for 99% availability but cannot achieve 99.99%+.',
  },
];
