/**
 * Multiple choice questions for SSH & Remote Administration
 */

import { MultipleChoiceQuestion } from '../../../types';

export const sshRemoteAdministrationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ssh-mc-1',
    question: 'Which SSH configuration directive disables root login?',
    options: [
      'PermitRootLogin no',
      'RootLogin disabled',
      'AllowRoot no',
      'DenyRoot yes',
    ],
    correctAnswer: 0,
    explanation:
      'PermitRootLogin no in /etc/ssh/sshd_config prevents root user from logging in via SSH. This is a critical security hardening step. Always use a regular user with sudo privileges instead of logging in as root.',
    difficulty: 'easy',
    topic: 'SSH Hardening',
  },
  {
    id: 'ssh-mc-2',
    question: 'What does "ssh -L 8080:localhost:80 user@remote" do?',
    options: [
      'Remote port forwarding',
      'Local port forwarding',
      'Dynamic forwarding',
      'Reverse proxy',
    ],
    correctAnswer: 1,
    explanation:
      'Local port forwarding (-L) forwards local port 8080 to remote localhost:80. Accessing localhost:8080 on your machine connects to port 80 on the remote server. Useful for accessing remote services locally (e.g., databases, web apps).',
    difficulty: 'medium',
    topic: 'SSH Tunneling',
  },
  {
    id: 'ssh-mc-3',
    question:
      'Which AWS service allows SSH-like access WITHOUT opening port 22?',
    options: [
      'EC2 Instance Connect',
      'Systems Manager Session Manager',
      'Direct Connect',
      'VPN',
    ],
    correctAnswer: 1,
    explanation:
      'AWS Systems Manager Session Manager provides shell access to EC2 instances without requiring public IPs, bastion hosts, or SSH port 22 in security groups. It uses IAM for authentication and logs all sessions to CloudTrail and CloudWatch.',
    difficulty: 'easy',
    topic: 'AWS Session Manager',
  },
  {
    id: 'ssh-mc-4',
    question: 'What is the purpose of a bastion host?',
    options: [
      'Load balancing',
      'Jump box to private instances',
      'Database replication',
      'CDN caching',
    ],
    correctAnswer: 1,
    explanation:
      'A bastion host (jump box) is a hardened server in a public subnet that acts as a gateway to access private instances. It provides controlled, audited access to private infrastructure while keeping private instances isolated from the internet.',
    difficulty: 'easy',
    topic: 'Bastion Host',
  },
  {
    id: 'ssh-mc-5',
    question:
      'Which SSH key algorithm is most secure and efficient for new keys?',
    options: ['RSA 2048', 'RSA 4096', 'Ed25519', 'DSA'],
    correctAnswer: 2,
    explanation:
      'Ed25519 is the most secure and efficient modern SSH key algorithm. It offers better security than RSA 4096 with much shorter keys (256-bit), faster operations, and resistance to timing attacks. DSA is deprecated. RSA 2048 is minimum, RSA 4096 is acceptable but slower.',
    difficulty: 'advanced',
    topic: 'SSH Keys',
  },
];
