export const networkingBasicsDiscussion = [
  {
    id: 1,
    question:
      "Production application can't connect to RDS database. Security group allows port 3306 from application SG. What could be wrong and how do you debug?",
    answer:
      '**Troubleshooting Steps:** 1) Verify security group attached to RDS, 2) Check NACL rules (stateless!), 3) Verify route tables, 4) Test with telnet/nc from app instance, 5) Check RDS endpoint DNS resolution, 6) Verify VPC peering if cross-VPC. Common issues: NACL blocking return traffic, wrong security group attached, DNS resolution failure.',
  },
  {
    id: 2,
    question:
      'Design network architecture for 3-tier app (web, app, database) across 3 AZs in AWS VPC with public access to web tier only.',
    answer:
      '**Architecture:** VPC 10.0.0.0/16, 9 subnets: Public (10.0.1-3.0/24) for ALB, Private-App (10.0.11-13.0/24) for app servers, Private-DB (10.0.21-23.0/24) for RDS. Internet Gateway for public subnets. NAT Gateways in each AZ for private subnet internet access. Security Groups: ALB (80/443 from 0.0.0.0/0), App (8080 from ALB SG), DB (3306 from App SG). NACLs for additional subnet-level protection.',
  },
  {
    id: 3,
    question:
      'Application experiencing intermittent connection timeouts. How do you diagnose network issues?',
    answer:
      '**Diagnosis:** 1) Check MTU issues: `ping -M do -s 1472 <host>`, 2) Monitor with tcpdump: `tcpdump -i eth0 host <dest>`, 3) Check for packet loss: `mtr <dest>`, 4) Review VPC Flow Logs for REJECT records, 5) Check security group/NACL changes in CloudTrail, 6) Monitor network metrics in CloudWatch, 7) Check DNS resolution timing, 8) Review connection pool exhaustion in application.',
  },
];

export const networkingBasicsMultipleChoice = [
  {
    id: 'net-mc-1',
    question: 'How many usable IP addresses in a /24 CIDR block?',
    options: ['254', '255', '256', '512'],
    correctAnswer: 0,
    explanation:
      '256 total IPs in /24, minus 2 (network address and broadcast) = 254 usable IPs.',
    difficulty: 'easy',
    topic: 'CIDR',
  },
  {
    id: 'net-mc-2',
    question: 'Security Group vs NACL: Which is stateful?',
    options: ['Security Group', 'NACL', 'Both', 'Neither'],
    correctAnswer: 0,
    explanation:
      'Security Groups are stateful (return traffic auto-allowed). NACLs are stateless (must explicitly allow return traffic in both directions).',
    difficulty: 'medium',
    topic: 'AWS Networking',
  },
  {
    id: 'net-mc-3',
    question: 'Which command shows listening TCP ports with process names?',
    options: ['netstat -tlnp', 'ping -p', 'ifconfig -l', 'route -n'],
    correctAnswer: 0,
    explanation:
      'netstat -tlnp shows TCP (-t) listening (-l) ports in numeric format (-n) with process info (-p). Alternative: ss -tlnp (faster).',
    difficulty: 'easy',
    topic: 'Networking Tools',
  },
  {
    id: 'net-mc-4',
    question: 'AWS VPC subnet 10.0.1.0/24 - what is the first usable IP?',
    options: ['10.0.1.0', '10.0.1.1', '10.0.1.2', '10.0.1.4'],
    correctAnswer: 3,
    explanation:
      'AWS reserves first 4 IPs and last IP. Reserved: .0 (network), .1 (VPC router), .2 (DNS), .3 (future), .255 (broadcast). First usable is 10.0.1.4.',
    difficulty: 'advanced',
    topic: 'AWS VPC',
  },
  {
    id: 'net-mc-5',
    question: 'Which DNS record type maps domain to IP address?',
    options: ['CNAME', 'MX', 'A', 'TXT'],
    correctAnswer: 2,
    explanation:
      'A record maps domain to IPv4 address. AAAA for IPv6. CNAME for aliases. MX for mail servers. TXT for text records.',
    difficulty: 'easy',
    topic: 'DNS',
  },
] as any[];
