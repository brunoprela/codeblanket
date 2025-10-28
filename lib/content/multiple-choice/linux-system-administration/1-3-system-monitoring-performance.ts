/**
 * Multiple choice questions for System Monitoring & Performance
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systemMonitoringPerformanceMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'sys-mon-mc-1',
      question:
        'A server has 4 CPU cores and shows a load average of 8.5, 7.2, 6.5. The CPU usage is only 30%. What is most likely the problem?',
      options: [
        'The load average is normal for a 4-core system',
        'Processes are waiting for I/O, indicated by high I/O wait time',
        'The CPU metric is incorrect and actual usage is higher',
        'The system needs more RAM to reduce load average',
      ],
      correctAnswer: 1,
      explanation:
        'Load average represents the number of processes in the run queue (runnable + uninterruptible sleep). A load of 8.5 on a 4-core system with only 30% CPU usage indicates processes are waiting for resources other than CPU - typically disk I/O. Check with vmstat for high "wa" (I/O wait) values. This is a classic symptom of disk bottleneck.',
      difficulty: 'advanced',
      topic: 'Performance Analysis',
    },
    {
      id: 'sys-mon-mc-2',
      question:
        'In iostat output, a disk shows %util at 100% and await at 250ms. What does this indicate?',
      options: [
        'The disk is performing optimally at maximum throughput',
        'The disk is saturated and requests are experiencing high latency',
        'The disk cache is full and needs clearing',
        'The disk needs defragmentation',
      ],
      correctAnswer: 1,
      explanation:
        '%util at 100% means the disk was busy 100% of the time (saturated). await of 250ms means average wait time for I/O requests is 250ms, which is extremely high (normal is <10ms for SSD, <20ms for HDD). This indicates the disk cannot keep up with the I/O demand. Solutions: upgrade to higher IOPS volume (gp3 or io2), optimize application I/O patterns, or add read replicas.',
      difficulty: 'medium',
      topic: 'Disk I/O',
    },
    {
      id: 'sys-mon-mc-3',
      question:
        'What is the significance of "steal time" (%st) in CPU metrics on EC2 instances?',
      options: [
        'CPU time stolen by malicious processes',
        'Time the hypervisor is using CPU meant for your instance',
        'CPU time used by kernel processes',
        'Time lost due to context switching',
      ],
      correctAnswer: 1,
      explanation:
        'Steal time (%st) shows the percentage of time a virtual CPU waits for a real CPU while the hypervisor is servicing another virtual CPU. On EC2, consistently high steal time (>5-10%) indicates either your instance is too small, or there\'s "noisy neighbor" contention. Solution: upgrade instance size or switch to a different instance family.',
      difficulty: 'advanced',
      topic: 'Cloud Performance',
    },
    {
      id: 'sys-mon-mc-4',
      question:
        'In free -h output, "free" memory is low but "available" memory is high. What does this mean?',
      options: [
        'The system is out of memory and needs more RAM',
        'Memory metrics are corrupted and should be ignored',
        'Linux is using free memory for cache, which is normal and beneficial',
        'There is a memory leak that needs investigation',
      ],
      correctAnswer: 2,
      explanation:
        'Linux aggressively uses free memory for file system cache to improve performance. This cache is automatically released when applications need memory. "available" memory shows how much is truly available for new applications (including reclaimable cache). Low "free" with high "available" is completely normal. Only worry if "available" is low.',
      difficulty: 'medium',
      topic: 'Memory Management',
    },
    {
      id: 'sys-mon-mc-5',
      question:
        'Which metric best indicates that a process is stuck waiting for I/O operations?',
      options: [
        'High CPU usage in top',
        'Process state "D" (uninterruptible sleep) in ps output',
        'High memory usage in free',
        'Many child processes in pstree',
      ],
      correctAnswer: 1,
      explanation:
        'Process state "D" (uninterruptible sleep) means the process is waiting for I/O and cannot be interrupted by signals. This is normal for short periods, but if many processes are in D state or stay in D state for long periods, it indicates an I/O bottleneck (usually disk). Check with iostat to identify the saturated disk.',
      difficulty: 'advanced',
      topic: 'Process States',
    },
  ];
