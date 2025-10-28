/**
 * Multiple choice questions for Process & Resource Management
 */

import { MultipleChoiceQuestion } from '../../../types';

export const processResourceManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'proc-mc-1',
      question: 'What does "ulimit -n 65536" do?',
      options: [
        'Sets max processes',
        'Sets max open files',
        'Sets max memory',
        'Sets max CPU',
      ],
      correctAnswer: 1,
      explanation:
        'ulimit -n sets the maximum number of open file descriptors for the current shell and its child processes. 65536 is a common production value. Use -u for max processes, no ulimit flag for memory (use cgroups/systemd).',
      difficulty: 'easy',
      topic: 'Resource Limits',
    },
    {
      id: 'proc-mc-2',
      question:
        'Which systemd directive prevents a service from being OOM killed?',
      options: [
        'MemoryMax',
        'OOMScoreAdjust=-1000',
        'ProtectOOM=yes',
        'NoOOMKill=true',
      ],
      correctAnswer: 1,
      explanation:
        'OOMScoreAdjust=-1000 in a systemd service file sets the process OOM score to -1000, which tells the kernel to never kill this process during OOM situations. Use carefully only for critical services like databases. MemoryMax only sets a limit.',
      difficulty: 'advanced',
      topic: 'OOM Management',
    },
    {
      id: 'proc-mc-3',
      question: 'What is a "nice" value of -20?',
      options: [
        'Lowest priority',
        'Highest priority',
        'Default priority',
        'Disabled',
      ],
      correctAnswer: 1,
      explanation:
        'Nice values range from -20 (highest priority) to +19 (lowest priority). Default is 0. Negative nice values require root privileges. Lower nice value = higher CPU priority = "less nice" to other processes.',
      difficulty: 'medium',
      topic: 'Process Priority',
    },
    {
      id: 'proc-mc-4',
      question: 'Which systemd directive limits CPU usage to 2 full cores?',
      options: ['CPULimit=2', 'CPUQuota=200%', 'CPUCores=2', 'CPUMax=2'],
      correctAnswer: 1,
      explanation:
        'CPUQuota=200% limits a service to 200% CPU, which equals 2 full CPU cores (100% per core). CPUQuota=50% would be half a core. This prevents a service from consuming all available CPU and starving other processes.',
      difficulty: 'medium',
      topic: 'CPU Limits',
    },
    {
      id: 'proc-mc-5',
      question:
        'Which AWS instance family is best for memory-intensive workloads like databases?',
      options: [
        't3 (burstable)',
        'c5 (compute)',
        'r5 (memory)',
        'm5 (general)',
      ],
      correctAnswer: 2,
      explanation:
        'R5/R6 instances are memory-optimized with high memory-to-CPU ratios, ideal for databases, in-memory caches (Redis), and big data processing. C5 is compute-optimized, T3 is burstable, M5 is balanced general-purpose.',
      difficulty: 'easy',
      topic: 'EC2 Instance Types',
    },
  ];
