/**
 * Multiple choice questions for DNS (Domain Name System) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dnssystemMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'dns-query-type',
    question:
      "In a typical DNS resolution, what type of query does a user's device make to the recursive resolver?",
    options: [
      'Iterative query',
      'Recursive query',
      'Authoritative query',
      'Cached query',
    ],
    correctAnswer: 1,
    explanation:
      'Users make recursive queries to their DNS resolver (like 8.8.8.8). The resolver then does all the work of querying root, TLD, and authoritative servers (using iterative queries between servers), and returns the final answer to the user. This is why it\'s called a "recursive" resolver.',
  },
  {
    id: 'dns-cname-limitation',
    question:
      "Why can't you use a CNAME record at the root/apex of a domain (e.g., example.com)?",
    options: [
      'CNAME records are too slow for root domains',
      'It conflicts with required SOA and NS records at the zone apex',
      'Root domains can only use AAAA records',
      'CNAME records are deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'RFC standards prohibit CNAME at the zone apex because every domain must have SOA and NS records, and CNAME means "this is an alias for another name, don\'t look for other records here." Having both would be contradictory. Solutions include using A records or ALIAS records (AWS Route53 proprietary feature).',
  },
  {
    id: 'dns-ttl-tradeoff',
    question:
      "You're planning to migrate your application to new servers. What DNS TTL strategy should you use?",
    options: [
      'Increase TTL to 24 hours before migration for stability',
      'Keep TTL unchanged and migrate immediately',
      'Lower TTL to 60 seconds before migration, wait for old TTL to expire, then migrate',
      'Set TTL to 0 to disable caching',
    ],
    correctAnswer: 2,
    explanation:
      'You should lower TTL before migration (e.g., to 60 seconds), wait for the old TTL period to expire so all caches refresh, then perform the migration. This ensures users switch to new servers quickly. After migration stabilizes, raise TTL back to reduce DNS query load. TTL=0 is often ignored by resolvers.',
  },
  {
    id: 'dns-security',
    question: 'What does DNSSEC primarily protect against?',
    options: [
      'DNS query eavesdropping by ISPs',
      'DNS cache poisoning and spoofing attacks',
      'DDoS attacks on DNS servers',
      'Slow DNS resolution',
    ],
    correctAnswer: 1,
    explanation:
      "DNSSEC uses cryptographic signatures to verify that DNS responses are authentic and haven't been tampered with, protecting against cache poisoning and spoofing. It does NOT encrypt queries (that's DNS over HTTPS/TLS), doesn't prevent DDoS, and doesn't improve speed. It ensures authenticity and integrity.",
  },
  {
    id: 'dns-load-balancing',
    question:
      'What is a major limitation of using round-robin DNS for load balancing?',
    options: [
      'It can only balance between 2 servers',
      'It requires expensive hardware load balancers',
      "It doesn't perform health checks, so traffic goes to failed servers",
      'It only works with IPv6',
    ],
    correctAnswer: 2,
    explanation:
      "Round-robin DNS returns multiple IP addresses, but it has no health checking. If one server fails, DNS still returns its IP, and some users will try to connect to the dead server. Modern solutions like AWS Route53 add health checks, but basic round-robin DNS doesn't have this. It works with unlimited servers and both IPv4/IPv6.",
  },
];
