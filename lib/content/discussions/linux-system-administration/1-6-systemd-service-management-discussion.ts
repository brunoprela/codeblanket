export const systemdServiceManagementDiscussion = [
  {
    id: 1,
    question:
      'Application keeps crashing and restarting in a loop. Design a systemd service configuration that prevents restart storms while ensuring automatic recovery from genuine failures.',
    answer:
      'Use StartLimitInterval and StartLimitBurst: `StartLimitInterval=300` and `StartLimitBurst=5` allows 5 restarts in 5 minutes. Set `Restart=on-failure` to restart only on abnormal exits. Add `RestartSec=10` for 10s delay between restarts. Include ExecStartPre health check. Monitor with `journalctl -u app -f` to identify root cause.',
  },
  {
    id: 2,
    question:
      'Create a production-ready systemd service for a Node.js application with database dependency, resource limits, security hardening, and proper logging.',
    answer:
      '**Complete service file:** `After=postgresql.service`, `Requires=postgresql.service`, `LimitNOFILE=65536`, `MemoryLimit=2G`, `CPUQuota=200%`, `NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectSystem=strict`, `StandardOutput=journal`. Include health check endpoint. Set `Restart=always` with `RestartSec=10`. Use dedicated service user with minimal permissions.',
  },
  {
    id: 3,
    question:
      'How would you migrate from cron jobs to systemd timers? What are the benefits?',
    answer:
      '**Benefits of timers:** Better logging via journalctl, dependencies on services, persistent (run missed executions), randomized delays to prevent thundering herd, easier debugging. **Migration:** Create .service file for task, .timer file with `OnCalendar`, `Persistent=true`. Enable timer instead of cron entry. Example: `OnCalendar=daily` or `OnCalendar=*-*-* 02:00:00` for 2 AM daily.',
  },
];

export const systemdServiceManagementMultipleChoice = [
  {
    id: 'systemd-mc-1',
    question:
      'Which systemd service Type should be used for most modern applications?',
    options: ['forking', 'oneshot', 'simple', 'notify'],
    correctAnswer: 2,
    explanation:
      'Type=simple is default and appropriate for most applications that run in foreground. Forking is for traditional daemons. Oneshot for tasks that exit. Notify for applications that signal readiness.',
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
      'Restart=on-failure restarts service only on abnormal exit (non-zero exit code, signal, timeout). Use for production to auto-recover from crashes while avoiding restart loops on configuration errors.',
    difficulty: 'medium',
    topic: 'Restart Policies',
  },
  {
    id: 'systemd-mc-3',
    question: 'Which command views logs for a specific service?',
    options: [
      'journalctl -u service',
      'systemctl logs service',
      'tail -f /var/log/service',
      'cat /var/log/syslog',
    ],
    correctAnswer: 0,
    explanation:
      'journalctl -u servicename shows logs for that service from systemd journal. Add -f to follow, -n for last N lines, --since for time filtering.',
    difficulty: 'easy',
    topic: 'Logging',
  },
  {
    id: 'systemd-mc-4',
    question: 'What prevents a service restart loop?',
    options: [
      'Restart=always',
      'StartLimitBurst and StartLimitInterval',
      'RestartSec',
      'Type=oneshot',
    ],
    correctAnswer: 1,
    explanation:
      'StartLimitBurst (max restarts) and StartLimitInterval (time window) prevent restart loops. Example: StartLimitBurst=5 and StartLimitInterval=300 allows 5 restarts in 5 minutes, then stops trying.',
    difficulty: 'advanced',
    topic: 'Service Protection',
  },
  {
    id: 'systemd-mc-5',
    question: 'Which option provides a private /tmp for a service?',
    options: [
      'NoNewPrivileges=true',
      'PrivateTmp=true',
      'ProtectSystem=true',
      'ProtectHome=true',
    ],
    correctAnswer: 1,
    explanation:
      'PrivateTmp=true gives service a private /tmp directory, isolated from other processes. Security hardening feature that prevents temp file attacks and information leaks.',
    difficulty: 'medium',
    topic: 'Security',
  },
] as any[];
