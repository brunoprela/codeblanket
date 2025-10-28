/**
 * Multiple choice questions for Networking Basics
 */

import { MultipleChoiceQuestion } from '../../../types';

export const networkingBasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'net-mc-1',
    question: 'How many usable IP addresses are in a /24 CIDR block?',
    options: ['254', '255', '256', '512'],
    correctAnswer: 0,
    explanation:
      'A /24 CIDR block has 256 total IP addresses (2^8). However, 2 are reserved: network address (.0) and broadcast address (.255), leaving 254 usable IPs for hosts (.1 through .254).',
    difficulty: 'easy',
    topic: 'CIDR',
  },
  {
    id: 'net-mc-2',
    question: 'Security Group vs NACL: Which is stateful?',
    options: ['Security Group only', 'NACL only', 'Both', 'Neither'],
    correctAnswer: 0,
    explanation:
      'Security Groups are stateful - if you allow inbound traffic, the return traffic is automatically allowed. NACLs are stateless - you must explicitly allow both inbound AND outbound rules for bidirectional traffic.',
    difficulty: 'medium',
    topic: 'AWS Networking',
  },
  {
    id: 'net-mc-3',
    question: 'Which command shows listening TCP ports with process names?',
    options: ['netstat -tlnp', 'ping -p', 'ifconfig -l', 'route -n'],
    correctAnswer: 0,
    explanation:
      'netstat -tlnp shows TCP (-t) listening (-l) ports in numeric format (-n) with process info (-p). Modern alternative: ss -tlnp (faster and more feature-rich).',
    difficulty: 'easy',
    topic: 'Networking Tools',
  },
  {
    id: 'net-mc-4',
    question:
      'In AWS VPC subnet 10.0.1.0/24, what is the first usable IP address?',
    options: ['10.0.1.0', '10.0.1.1', '10.0.1.2', '10.0.1.4'],
    correctAnswer: 3,
    explanation:
      'AWS reserves the first 4 IPs and last IP in each subnet: .0 (network), .1 (VPC router), .2 (DNS), .3 (future use), .255 (broadcast). The first usable IP is 10.0.1.4.',
    difficulty: 'advanced',
    topic: 'AWS VPC',
  },
  {
    id: 'net-mc-5',
    question: 'Which DNS record type maps a domain name to an IPv4 address?',
    options: ['CNAME', 'MX', 'A', 'TXT'],
    correctAnswer: 2,
    explanation:
      'A (Address) record maps domain to IPv4 address. AAAA for IPv6. CNAME creates alias to another domain. MX specifies mail servers. TXT stores text information.',
    difficulty: 'easy',
    topic: 'DNS',
  },
];
