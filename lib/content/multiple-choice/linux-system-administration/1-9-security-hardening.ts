/**
 * Multiple choice questions for Security Hardening
 */

import { MultipleChoiceQuestion } from '../../../types';

export const securityHardeningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'security-mc-1',
    question: 'Which tool protects against SSH brute force attacks?',
    options: ['iptables', 'fail2ban', 'selinux', 'clamav'],
    correctAnswer: 1,
    explanation:
      'fail2ban monitors log files for failed authentication attempts and automatically bans IPs that show malicious behavior (e.g., repeated failed SSH logins). It works by adding iptables rules to block offending IPs.',
    difficulty: 'easy',
    topic: 'Intrusion Prevention',
  },
  {
    id: 'security-mc-2',
    question: 'What does SELinux enforcing mode do?',
    options: [
      'Logs violations only',
      'Blocks violations and logs',
      'Disables SELinux',
      'Allows all access',
    ],
    correctAnswer: 1,
    explanation:
      'SELinux enforcing mode actively blocks policy violations and logs them to audit.log. Permissive mode logs violations but allows them (useful for troubleshooting). Disabled mode turns off SELinux entirely (not recommended in production).',
    difficulty: 'easy',
    topic: 'SELinux',
  },
  {
    id: 'security-mc-3',
    question:
      'Which AWS feature prevents SSRF attacks on EC2 metadata service?',
    options: ['Security Groups', 'IMDSv2', 'WAF', 'KMS'],
    correctAnswer: 1,
    explanation:
      'IMDSv2 (Instance Metadata Service version 2) requires a session token obtained via PUT request, preventing SSRF attacks that use GET requests. It adds a layer of protection by requiring an additional HTTP hop that SSRF payloads cannot typically perform.',
    difficulty: 'advanced',
    topic: 'AWS Security',
  },
  {
    id: 'security-mc-4',
    question: 'What is the principle of least privilege?',
    options: [
      'Give everyone admin access',
      'Grant minimum permissions needed',
      'Use root for everything',
      'Disable all permissions',
    ],
    correctAnswer: 1,
    explanation:
      'Principle of least privilege means granting users, services, and systems only the minimum permissions necessary to perform their tasks. This reduces attack surface and limits damage from compromised accounts or services.',
    difficulty: 'easy',
    topic: 'Security Principles',
  },
  {
    id: 'security-mc-5',
    question: 'Which tool performs file integrity monitoring on Linux?',
    options: ['fail2ban', 'aide', 'selinux', 'clamav'],
    correctAnswer: 1,
    explanation:
      'AIDE (Advanced Intrusion Detection Environment) creates a database of file hashes and attributes, then checks for unauthorized changes. It detects if system files, configs, or binaries have been modified (potential rootkit or malware). ClamAV is antivirus, fail2ban is brute force protection.',
    difficulty: 'medium',
    topic: 'Intrusion Detection',
  },
];
