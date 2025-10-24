/**
 * Multiple choice questions for Design WhatsApp section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const whatsappMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'WhatsApp handles 100 billion messages per day. What is the average messages per second the system must process?',
    options: [
      '100,000 messages/sec',
      '500,000 messages/sec',
      '1.16 million messages/sec',
      '10 million messages/sec',
    ],
    correctAnswer: 2,
    explanation:
      "100 billion messages / day ÷ 86,400 seconds/day = 1,157,407 messages/sec average. Peak (e.g., New Year's Eve globally): 3-5 million messages/sec. This scale requires: (1) Erlang for concurrency (handles millions of connections per server). (2) Cassandra for write throughput (millions of writes/sec). (3) Message batching and queuing (Kafka/Redis). (4) Global distribution (multiple data centers to spread load). For comparison: Twitter peak ~500K tweets/sec, Messenger ~500K messages/sec. WhatsApp's scale is 2-3x larger due to global penetration and reliance (primary communication for 2B users).",
  },
  {
    id: 'mc2',
    question:
      'Why does WhatsApp use Erlang instead of Java/Node.js for its server infrastructure?',
    options: [
      'Erlang is faster at computation',
      'Erlang lightweight processes enable 1M+ concurrent connections per server; Java threads limited to ~10K',
      'Erlang is easier to learn',
      'Erlang has better libraries',
    ],
    correctAnswer: 1,
    explanation:
      "ERLANG CONCURRENCY: Each connection = lightweight Erlang process (~1 KB overhead). 1 server with 256 GB RAM can handle 2M+ connections. Processes isolated (one crash doesn't affect others). JAVA/NODE.JS: Each connection = OS thread (~1 MB overhead). Limited to ~10K threads per server (memory exhausted). Context switching overhead. PRODUCTION IMPACT: WhatsApp: 1 server handles 2M connections. Java: Need 200 servers for same load. COST: 200x reduction in servers. This is why WhatsApp operated with minimal infrastructure (50 engineers, low costs). OTHER ERLANG BENEFITS: Hot code swapping (deploy without downtime), built-in fault tolerance (supervisor trees), designed for telecom (99.999% uptime). KEY INSIGHT: For I/O-bound workloads (networking, messaging), Erlang's concurrency model is superior to thread-based systems.",
  },
  {
    id: 'mc3',
    question:
      'WhatsApp implements end-to-end encryption. What does this mean for WhatsApp servers?',
    options: [
      'Servers can read messages but promise not to',
      'Servers cannot decrypt messages - only sender and recipient devices have keys',
      'Servers encrypt messages in transit only',
      'Encryption is optional',
    ],
    correctAnswer: 1,
    explanation:
      "END-TO-END ENCRYPTION: Message encrypted on Alice's device with Bob's public key. WhatsApp server receives encrypted blob (cannot decrypt - doesn't have Bob's private key). Server routes encrypted blob to Bob's device. Bob's device decrypts with private key. ZERO-KNOWLEDGE: WhatsApp has no access to message plaintext. Even if government subpoenas WhatsApp for messages, WhatsApp cannot provide (they don't have decryption keys). CONTRAST: Transport encryption (HTTPS): Encrypted client → server, decrypted at server, then re-encrypted server → recipient. Server sees plaintext. Most systems use transport encryption only. E2E encryption is stronger but more complex (key management, device recovery). This is why Facebook (owns WhatsApp) lobbied against E2E requirements - they can't read messages for ads/moderation.",
  },
  {
    id: 'mc4',
    question:
      'Alice sends a message to a group with 256 members. In naive implementation, how many times must the message be encrypted?',
    options: [
      'Once (same message to all)',
      '256 times (once per recipient with their public key)',
      '2 times (encrypt once, sign once)',
      '0 times (groups are not encrypted)',
    ],
    correctAnswer: 1,
    explanation:
      "E2E ENCRYPTION FOR GROUPS: Each member has unique key pair. Alice must encrypt message with each member's public key. 256 members = 256 encryption operations. PROBLEM: High CPU usage for large groups. OPTIMIZATION (Sender Keys): Generate random group session key. Encrypt message once with group key (symmetric encryption - fast). Encrypt group key 256 times (once per member with their public key). Share encrypted group keys + encrypted message. RESULT: 1 AES encryption (fast) + 256 RSA encryptions of small key (faster than encrypting full message 256 times). TRADE-OFF: Group keys means if one device compromised, all group messages readable. WhatsApp uses sender keys for efficiency. Signal uses pairwise encryption (stronger security, slower). This is classic security vs performance trade-off.",
  },
  {
    id: 'mc5',
    question:
      'WhatsApp stores messages on servers for 30 days, then deletes them. Why?',
    options: [
      'Legal requirement',
      'Storage cost savings: 30 days = 3 TB, permanent = 36 PB (10,000x difference)',
      'Messages become irrelevant after 30 days',
      'Technical limitation of Cassandra',
    ],
    correctAnswer: 1,
    explanation:
      'STORAGE CALCULATION: 100B messages/day × 1 KB = 100 TB/day. 30 days: 100 TB × 30 = 3 PB. Forever: 100 TB × 365 × 10 years = 365 PB. COST: 3 PB @ $0.023/GB/month = $70K/month (manageable). 365 PB = $8.4M/month (unsustainable). WHATSAPP PHILOSOPHY: Messages stored primarily on devices (client-side storage). Server storage for two purposes: (1) Offline delivery (until recipient connects), (2) New device setup (recent chat history). After 30 days, messages already delivered to device. If user switches phones, only loses messages >30 days old (acceptable). ALTERNATIVE (Telegram): Stores messages forever (cloud-based). Requires massive infrastructure (20x storage cost). Different product philosophy. KEY INSIGHT: Architecture decisions driven by cost at scale. 30-day retention balances functionality with affordability.',
  },
];
