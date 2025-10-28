/**
 * Multiple choice questions for Linux Fundamentals for Production
 */

import { MultipleChoiceQuestion } from '../../../types';

export const linuxFundamentalsProductionMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'linux-prod-mc-1',
      question:
        "A production server shows 'No space left on device' error but df -h shows 40% disk usage with 30GB free. What is the most likely cause?",
      options: [
        'The disk is corrupted and df is showing incorrect information',
        'Inodes are exhausted even though disk space is available',
        'A hidden .snapshot directory is consuming the remaining space',
        'The filesystem is in read-only mode due to errors',
      ],
      correctAnswer: 1,
      explanation:
        'Inode exhaustion is a common issue where the filesystem runs out of inodes (file metadata structures) even when disk space is available. Each file requires one inode regardless of size. Check with `df -i` to verify inode usage. This often happens with millions of small files (cache files, sessions, logs).',
      difficulty: 'medium',
      topic: 'File Systems',
    },
    {
      id: 'linux-prod-mc-2',
      question:
        'Which command will correctly set the soft and hard file descriptor limits to 65536 for a systemd service?',
      options: [
        'ulimit -n 65536 in the ExecStart command',
        'LimitNOFILE=65536 in the [Service] section',
        'fs.file-max=65536 in /etc/sysctl.conf',
        'nofile=65536 in /etc/security/limits.conf',
      ],
      correctAnswer: 1,
      explanation:
        "For systemd services, use LimitNOFILE=65536 in the [Service] section of the service file. This is the most reliable method as it applies directly to the service process. While /etc/security/limits.conf works for user logins, it doesn't apply to systemd services. sysctl fs.file-max sets system-wide limits, not per-process limits.",
      difficulty: 'medium',
      topic: 'Process Management',
    },
    {
      id: 'linux-prod-mc-3',
      question:
        'You need to give the "developers" group read-only access to /var/log/app.log without changing the file\'s owner or primary group permissions. What is the best approach?',
      options: [
        'chmod 644 /var/log/app.log',
        'chgrp developers /var/log/app.log && chmod 640 /var/log/app.log',
        'setfacl -m g:developers:r /var/log/app.log',
        "Add developers to the file's primary group with usermod",
      ],
      correctAnswer: 2,
      explanation:
        'ACLs (Access Control Lists) allow fine-grained permissions beyond standard Unix permissions. setfacl -m g:developers:r adds read permission for the developers group without affecting the owner, primary group, or other permissions. This is ideal for giving multiple groups different access levels to the same file.',
      difficulty: 'medium',
      topic: 'File Permissions',
    },
    {
      id: 'linux-prod-mc-4',
      question:
        'A process shows state "D" in ps aux output. What does this mean and why is it concerning?',
      options: [
        'D = Dead (zombie process) - concerning because it indicates parent process issues',
        'D = Delayed (process waiting for timer) - not concerning, normal behavior',
        'D = Uninterruptible sleep (usually I/O wait) - concerning if many processes are stuck',
        'D = Detached (background process) - not concerning, normal for daemons',
      ],
      correctAnswer: 2,
      explanation:
        'State "D" means uninterruptible sleep, typically waiting for I/O operations. Unlike normal sleep ("S"), these processes cannot be interrupted by signals. Many processes in D state often indicates I/O problems (disk saturation, NFS hangs, etc.). These processes contribute to load average. Check with `iostat` to identify I/O issues.',
      difficulty: 'advanced',
      topic: 'Process Management',
    },
    {
      id: 'linux-prod-mc-5',
      question:
        'When should you use XFS over ext4 for an AWS EBS volume in production?',
      options: [
        "Always use XFS because it's newer and faster",
        'When you need large files, high throughput, and online filesystem growth',
        'When you need to shrink the filesystem later',
        "XFS and ext4 are identical; the choice doesn't matter",
      ],
      correctAnswer: 1,
      explanation:
        'XFS excels with large files and high-throughput workloads (data analytics, media storage, large databases). It supports online growth (but not shrinking). XFS is default on Amazon Linux 2023 and RHEL. Use ext4 for general-purpose workloads, smaller files, or when you might need to shrink the filesystem. Both are excellent choices, but workload characteristics should guide your decision.',
      difficulty: 'medium',
      topic: 'File Systems',
    },
  ];
