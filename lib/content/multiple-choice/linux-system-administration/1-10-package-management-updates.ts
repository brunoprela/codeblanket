/**
 * Multiple choice questions for Package Management & Updates
 */

import { MultipleChoiceQuestion } from '../../../types';

export const packageManagementUpdatesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pkg-mc-1',
      question:
        'Which command installs only security updates on Amazon Linux 2023?',
      options: [
        'dnf update',
        'dnf update --security',
        'dnf upgrade',
        'dnf install security',
      ],
      correctAnswer: 1,
      explanation:
        'dnf update --security (or dnf update-minimal --security) installs only security-related updates. This is ideal for production where you want critical fixes without other package changes that could introduce issues.',
      difficulty: 'medium',
      topic: 'Security Updates',
    },
    {
      id: 'pkg-mc-2',
      question: 'What does "apt-mark hold nginx" do?',
      options: [
        'Uninstall nginx',
        'Prevent nginx from being updated',
        'Update nginx immediately',
        'Download nginx',
      ],
      correctAnswer: 1,
      explanation:
        'apt-mark hold prevents a package from being automatically updated during "apt upgrade". This is useful for keeping specific versions of critical packages. Use "apt-mark unhold" to allow updates again.',
      difficulty: 'easy',
      topic: 'Version Pinning',
    },
    {
      id: 'pkg-mc-3',
      question: 'Which dnf plugin allows locking package versions?',
      options: ['dnf-lock', 'dnf-versionlock', 'dnf-pin', 'dnf-freeze'],
      correctAnswer: 1,
      explanation:
        'dnf-versionlock plugin (python3-dnf-plugin-versionlock) allows locking specific package versions. Use "dnf versionlock add package" to lock and "dnf versionlock list" to show locked packages.',
      difficulty: 'medium',
      topic: 'Version Management',
    },
    {
      id: 'pkg-mc-4',
      question: 'What is the safest way to update production EC2 instances?',
      options: [
        'SSH and run yum update',
        'Create new AMI with updates and use blue-green deployment',
        'Let automatic updates handle it',
        'Update all at once',
      ],
      correctAnswer: 1,
      explanation:
        'Creating a new AMI with updates and using blue-green or rolling deployment via Auto Scaling Groups is safest. This allows testing, gradual rollout, and easy rollback. Direct SSH updates risk inconsistency and lack proper testing.',
      difficulty: 'advanced',
      topic: 'Update Strategy',
    },
    {
      id: 'pkg-mc-5',
      question: 'Which file configures automatic updates on Ubuntu?',
      options: [
        '/etc/apt/apt.conf',
        '/etc/apt/apt.conf.d/50unattended-upgrades',
        '/etc/auto-update.conf',
        '/etc/systemd/system/update.conf',
      ],
      correctAnswer: 1,
      explanation:
        '/etc/apt/apt.conf.d/50unattended-upgrades configures which packages are automatically updated (typically security updates). Enable with "dpkg-reconfigure unattended-upgrades" and configure update frequency in /etc/apt/apt.conf.d/20auto-upgrades.',
      difficulty: 'medium',
      topic: 'Automatic Updates',
    },
  ];
