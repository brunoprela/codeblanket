/**
 * Multiple choice questions for Storage & File Systems
 */

import { MultipleChoiceQuestion } from '../../../types';

export const storageFileSystemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'storage-mc-1',
    question: 'A 500GB gp2 EBS volume provides how many baseline IOPS?',
    options: ['500 IOPS', '1,000 IOPS', '1,500 IOPS', '3,000 IOPS'],
    correctAnswer: 2,
    explanation:
      'gp2 volumes provide 3 IOPS per GB, so 500GB × 3 = 1,500 IOPS baseline. gp2 can burst up to 3,000 IOPS using burst credits, but baseline is 1,500 IOPS. gp3 volumes provide a flat 3,000 IOPS baseline regardless of size (up to 16,000 IOPS with additional cost).',
    difficulty: 'medium',
    topic: 'EBS Volumes',
  },
  {
    id: 'storage-mc-2',
    question:
      'Which EBS volume type should be used for a mission-critical production database requiring 99.999% durability?',
    options: ['gp3', 'gp2', 'io2', 'st1'],
    correctAnswer: 2,
    explanation:
      'io2 volumes provide 99.999% durability (vs 99.8-99.9% for gp3/gp2), making them suitable for mission-critical databases. They also support up to 64,000 IOPS (256,000 with Block Express) and have the highest performance consistency. gp3 is better for cost/performance, but io2 is for maximum durability.',
    difficulty: 'medium',
    topic: 'EBS Volumes',
  },
  {
    id: 'storage-mc-3',
    question:
      'What is the advantage of using UUID instead of device names (e.g., /dev/xvdf) in /etc/fstab?',
    options: [
      'UUIDs are faster to mount',
      'UUIDs are persistent across reboots even if device names change',
      'UUIDs provide better I/O performance',
      'UUIDs enable encryption',
    ],
    correctAnswer: 1,
    explanation:
      'UUIDs (Universally Unique Identifiers) remain constant even if device names change. On EC2, device names like /dev/sdf might appear as /dev/xvdf, or order might change on reboot. UUID ensures the correct filesystem is always mounted. Get UUID with `blkid` command.',
    difficulty: 'easy',
    topic: 'File Systems',
  },
  {
    id: 'storage-mc-4',
    question:
      'In LVM, what is the correct order of components from physical to logical?',
    options: [
      'Physical Volume → Logical Volume → Volume Group',
      'Volume Group → Physical Volume → Logical Volume',
      'Physical Volume → Volume Group → Logical Volume',
      'Logical Volume → Physical Volume → Volume Group',
    ],
    correctAnswer: 2,
    explanation:
      'LVM hierarchy: Physical Volumes (actual disks) → Volume Groups (pools of PVs) → Logical Volumes (virtual partitions). Example: /dev/xvdf (PV) + /dev/xvdg (PV) = data-vg (VG), which contains mysql-lv (LV). This abstraction allows flexible resizing and management.',
    difficulty: 'medium',
    topic: 'LVM',
  },
  {
    id: 'storage-mc-5',
    question:
      'Which mount option reduces I/O by not updating access times on files?',
    options: ['nodiratime', 'noatime', 'relatime', 'defaults'],
    correctAnswer: 1,
    explanation:
      'noatime prevents updating access time (atime) on every file read, reducing I/O operations. This can improve performance by 5-10% on read-heavy workloads. relatime is a compromise that only updates if atime is older than mtime. nodiratime applies only to directories. Common for production: noatime,nodiratime.',
    difficulty: 'easy',
    topic: 'Performance',
  },
];
