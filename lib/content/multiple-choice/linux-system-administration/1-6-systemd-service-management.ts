/**
 * Multiple choice questions for Systemd Service Management
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systemdServiceManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'systemd-mc-1',
      question:
        'Which systemd service Type should be used for most modern applications?',
      options: ['forking', 'oneshot', 'simple', 'notify'],
      correctAnswer: 2,
      explanation:
        'Type=simple is the default and appropriate for most modern applications that run in foreground. forking is for traditional daemons that fork to background. oneshot for tasks that exit after completion. notify for applications that signal readiness (sd_notify).',
      difficulty: 'easy',
      topic: 'Service Types',
    },
    {
      id: 'systemd-mc-2',
      question: 'What does "Restart=on-failure" do?',
      options: [
        'Always restart',
        'Never restart',
        'Restart only on non-zero exit',
        'Restart only on signal',
      ],
      correctAnswer: 2,
      explanation:
        'Restart=on-failure restarts the service only on abnormal exit (non-zero exit code, signal, timeout, watchdog). It will NOT restart on clean exit (exit 0). This is ideal for production to auto-recover from crashes while avoiding restart loops on configuration errors.',
      difficulty: 'medium',
      topic: 'Restart Policies',
    },
    {
      id: 'systemd-mc-3',
      question: 'Which command views logs for a specific service?',
      options: [
        'journalctl -u servicename',
        'systemctl logs servicename',
        'tail -f /var/log/service',
        'cat /var/log/syslog',
      ],
      correctAnswer: 0,
      explanation:
        'journalctl -u servicename shows logs for that specific service from the systemd journal. Add -f to follow in real-time, -n 100 for last 100 lines, --since "1 hour ago" for time-based filtering. systemctl logs does not exist.',
      difficulty: 'easy',
      topic: 'Logging',
    },
    {
      id: 'systemd-mc-4',
      question: 'What prevents a service from restarting infinitely in a loop?',
      options: [
        'Restart=always',
        'StartLimitBurst and StartLimitInterval',
        'RestartSec',
        'Type=oneshot',
      ],
      correctAnswer: 1,
      explanation:
        'StartLimitBurst (max restarts) and StartLimitInterval (time window) prevent restart loops. Example: StartLimitBurst=5 and StartLimitInterval=300 allows 5 restarts within 5 minutes. After hitting this limit, systemd stops trying and marks service as failed.',
      difficulty: 'advanced',
      topic: 'Service Protection',
    },
    {
      id: 'systemd-mc-5',
      question:
        'Which security option provides a private /tmp directory for a service?',
      options: [
        'NoNewPrivileges=true',
        'PrivateTmp=true',
        'ProtectSystem=true',
        'ProtectHome=true',
      ],
      correctAnswer: 1,
      explanation:
        'PrivateTmp=true gives the service a private, isolated /tmp directory. This security hardening feature prevents temp file attacks, information leaks, and conflicts with other services. The private /tmp is automatically cleaned up when service stops.',
      difficulty: 'medium',
      topic: 'Security',
    },
  ];
