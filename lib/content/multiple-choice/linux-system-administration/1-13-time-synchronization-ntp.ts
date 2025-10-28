/**
 * Multiple choice questions for Time Synchronization & NTP
 */

import { MultipleChoiceQuestion } from '../../../types';

export const timeSynchronizationNtpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'time-mc-1',
    question: 'What is the IP address of AWS Time Sync Service?',
    options: ['169.254.169.1', '169.254.169.123', '169.254.0.1', '10.0.0.123'],
    correctAnswer: 1,
    explanation:
      '169.254.169.123 is the AWS Time Sync Service available to all EC2 instances via link-local addressing. It provides Stratum 1 time synchronization with microsecond accuracy. This is the recommended NTP server for EC2 instances.',
    difficulty: 'easy',
    topic: 'AWS Time Sync',
  },
  {
    id: 'time-mc-2',
    question: 'Which timezone should production servers use?',
    options: ['Local timezone', 'UTC', 'EST', 'Server admin preference'],
    correctAnswer: 1,
    explanation:
      'Production servers should always use UTC timezone. This avoids daylight saving time confusion, ensures consistency across global infrastructure, simplifies log correlation, and makes database timestamps unambiguous. Application code handles user timezone conversion.',
    difficulty: 'easy',
    topic: 'Timezone Best Practices',
  },
  {
    id: 'time-mc-3',
    question:
      'What command shows current NTP synchronization status in chrony?',
    options: [
      'chronyc status',
      'chronyc tracking',
      'chronyc sync',
      'chronyc check',
    ],
    correctAnswer: 1,
    explanation:
      'chronyc tracking shows detailed synchronization status including reference ID, stratum, system time offset, last offset, RMS offset, frequency, and leap status. Use "chronyc sources" to see available time sources. "chronyc activity" shows source statistics.',
    difficulty: 'medium',
    topic: 'Chrony Commands',
  },
  {
    id: 'time-mc-4',
    question: 'What does "makestep 1.0 3" mean in chrony.conf?',
    options: [
      'Step clock if offset > 1s, up to 3 times',
      'Update every 1s, 3 iterations',
      'Sync every 1 hour, 3 servers',
      'Adjust by 1ms, 3 times',
    ],
    correctAnswer: 0,
    explanation:
      'makestep 1.0 3 allows chrony to step (jump) the system clock if the offset is greater than 1 second, but only for the first 3 clock updates. After 3 updates, chrony will only slew (gradually adjust). This is useful for initial sync where large corrections are expected.',
    difficulty: 'advanced',
    topic: 'Chrony Configuration',
  },
  {
    id: 'time-mc-5',
    question: 'Which port does NTP use?',
    options: ['TCP 123', 'UDP 123', 'TCP 323', 'UDP 323'],
    correctAnswer: 1,
    explanation:
      'NTP uses UDP port 123 for time synchronization. UDP is used because time synchronization requires low latency and can tolerate occasional packet loss. Security groups and NACLs must allow UDP 123 outbound for NTP to work on EC2 instances.',
    difficulty: 'easy',
    topic: 'NTP Protocol',
  },
];
