/**
 * Multiple choice questions for Health Checks & Readiness Probes section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const healthChecksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the difference between a liveness probe and a readiness probe?',
    options: [
      'They are the same thing',
      'Liveness checks if app is running (restart if fails), Readiness checks if ready for traffic (remove from LB if fails)',
      'Liveness is for staging, Readiness is for production',
      'Liveness checks dependencies, Readiness checks the app',
    ],
    correctAnswer: 1,
    explanation:
      'Liveness probe checks "Is app running?" - Minimal checks, restart container if fails. Readiness probe checks "Ready for traffic?" - Check dependencies (database, cache), remove from load balancer if fails (don\'t restart). Key difference: Liveness failure → restart (fixes deadlock/crash). Readiness failure → stop traffic (handles dependency issues). Never check database in liveness (causes restart loop when DB down). Always check dependencies in readiness (safe to fail without restart).',
  },
  {
    id: 'mc2',
    question: 'What should a liveness probe check?',
    options: [
      'Database connectivity, cache status, and all dependencies',
      'Only if the application can respond (minimal check)',
      'External API availability',
      'Disk space and memory usage',
    ],
    correctAnswer: 1,
    explanation:
      'Liveness should check ONLY if application can respond - minimal check like returning HTTP 200. Do NOT check: Database (restart won\'t fix DB down), cache, external APIs, dependencies. Why: Liveness failure → restart container. Restarting won\'t fix external dependency issues, just creates infinite restart loop. Check dependencies in READINESS instead (removes from traffic without restarting). Liveness answers: "Is app deadlocked or crashed?" not "Are my dependencies healthy?"',
  },
  {
    id: 'mc3',
    question: 'What happens when a Kubernetes readiness probe fails?',
    options: [
      'The pod is immediately deleted',
      'The pod is restarted',
      'The pod is removed from service endpoints (stops receiving traffic) but keeps running',
      'Nothing happens',
    ],
    correctAnswer: 2,
    explanation:
      "When readiness fails, Kubernetes removes pod from service endpoints (load balancer stops sending traffic), but pod keeps running. This allows graceful handling of: Database temporarily down (readiness fails → no traffic → DB recovers → readiness passes → traffic resumes), cache warming (readiness fails during warmup → passes when ready), dependency issues. Pod is NOT restarted (that's liveness). This prevents sending requests to pods that can't handle them while allowing recovery without restart.",
  },
  {
    id: 'mc4',
    question:
      'What is graceful shutdown, and how do readiness probes enable it?',
    options: [
      'Immediately terminating all connections',
      'Marking readiness as failed on SIGTERM so new requests stop, then completing in-flight requests before shutdown',
      'Shutting down slowly over hours',
      'Never shutting down',
    ],
    correctAnswer: 1,
    explanation:
      'Graceful shutdown: (1) Receive SIGTERM signal, (2) Mark readiness as failed (return 503) immediately, (3) Load balancer stops sending new requests (1-2 seconds), (4) Complete in-flight requests (up to 25-30 seconds), (5) Close connections cleanly, exit. Without graceful shutdown: Pod terminates immediately → in-flight requests aborted → users see connection errors. With graceful shutdown: New requests stop → existing requests complete → no user errors. Kubernetes grace period (default 30s) gives time to complete requests.',
  },
  {
    id: 'mc5',
    question: 'When should you use a startup probe?',
    options: [
      'For all applications',
      'For applications that take > 30 seconds to start (load data, warm cache)',
      'Never, liveness is sufficient',
      'Only for databases',
    ],
    correctAnswer: 1,
    explanation:
      'Use startup probe for slow-starting applications (> 30 seconds): Loading large datasets, JVM warm-up, cache pre-population, model loading. Startup probe gives long timeout (e.g., 30 attempts × 10s = 5 minutes) before liveness checks begin. Without startup probe: Liveness probe starts immediately → App still starting → Liveness fails after 30 seconds → Restart → Still starting → Infinite restart loop. With startup probe: Wait up to 5 minutes for startup → Then begin liveness checks → Prevents false restarts during legitimate long startup.',
  },
];
