/**
 * Multiple choice questions for Debugging Production Issues
 */

import { MultipleChoiceQuestion } from '../../../types';

export const debuggingProductionIssuesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'debug-mc-1',
      question: 'Which tool traces system calls made by a process?',
      options: ['ltrace', 'strace', 'ptrace', 'dtrace'],
      correctAnswer: 1,
      explanation:
        'strace traces system calls (kernel interface) like open, read, write, connect. ltrace traces library calls (userspace libraries). ptrace is the underlying API used by debuggers. dtrace is Solaris/BSD tracing framework. Use strace to debug file access, network issues, or process hangs.',
      difficulty: 'easy',
      topic: 'Debugging Tools',
    },
    {
      id: 'debug-mc-2',
      question: 'What command captures network packets for analysis?',
      options: ['netstat', 'ss', 'tcpdump', 'iftop'],
      correctAnswer: 2,
      explanation:
        "tcpdump captures network packets and can save to pcap files for analysis in Wireshark. netstat/ss show current connections but don't capture packets. iftop shows bandwidth usage in real-time. Use tcpdump to debug connection timeouts, protocol issues, or analyze traffic patterns.",
      difficulty: 'easy',
      topic: 'Network Debugging',
    },
    {
      id: 'debug-mc-3',
      question: 'How do you find which process is using port 8000?',
      options: [
        'netstat -tulpn | grep 8000',
        'lsof -i :8000',
        'ss -tulpn | grep 8000',
        'All of the above',
      ],
      correctAnswer: 3,
      explanation:
        'All three commands can identify which process is using a port. lsof -i :8000 shows file descriptor info. netstat -tulpn (deprecated) and ss -tulpn (modern replacement) show socket stats. ss is faster for large connection counts. grep filters for port 8000.',
      difficulty: 'medium',
      topic: 'Network Troubleshooting',
    },
    {
      id: 'debug-mc-4',
      question:
        'What performance impact does strace have on a running process?',
      options: ['None', 'Low (5-10%)', 'Medium (50%)', 'High (2-100x slower)'],
      correctAnswer: 3,
      explanation:
        'strace has HIGH performance impact (2-100x slowdown) because it intercepts every system call. Use cautiously in production, prefer limited tracing (-e flag for specific syscalls), or use on low-traffic instances. For performance profiling, use perf instead (much lower overhead).',
      difficulty: 'advanced',
      topic: 'Performance Impact',
    },
    {
      id: 'debug-mc-5',
      question:
        'A process crashed with segfault. What file contains crash information?',
      options: [
        '/var/log/crash.log',
        'Core dump file',
        '/var/log/syslog',
        'stderr',
      ],
      correctAnswer: 1,
      explanation:
        'Core dump file (location set by kernel.core_pattern) contains full memory snapshot at crash time. Analyze with gdb to see backtrace, variable values, thread states. Enable with "ulimit -c unlimited". syslog may show "segmentation fault" message but not detailed crash info. Core dumps are essential for debugging crashes.',
      difficulty: 'medium',
      topic: 'Core Dumps',
    },
  ];
