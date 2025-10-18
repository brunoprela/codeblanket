/**
 * Quiz questions for Microservices Deployment section
 */

export const microservicesdeploymentQuiz = [
  {
    id: 'q1-deploy',
    question:
      'Compare Rolling Update, Blue-Green, and Canary deployment strategies. When would you use each?',
    sampleAnswer:
      'Rolling Update: Gradually replace pods one by one (v1→v1→v1 becomes v2→v1→v1 becomes v2→v2→v1...). Pros: zero downtime, automatic rollback. Use for: normal deployments, low-risk changes. Blue-Green: Two environments (Blue=current, Green=new), switch instantly. Pros: instant rollback, test before switch. Use for: high-stakes deployments, need quick rollback. Canary: Gradually shift traffic (10%→25%→50%→100%). Pros: real user testing, lowest risk. Use for: risky changes, want real user feedback. Example: Rolling for bug fixes, Blue-Green for major releases, Canary for algorithm changes where you monitor metrics before full rollout.',
    keyPoints: [
      'Rolling: gradual pod replacement, zero downtime, automatic rollback',
      'Blue-Green: two environments, instant switch, 2x resources',
      'Canary: gradual traffic shift, lowest risk, requires service mesh',
      'Choose based on risk level and resource availability',
      'All strategies enable zero-downtime deployments',
    ],
  },
  {
    id: 'q2-deploy',
    question:
      'How do you handle database migrations in microservices without causing downtime?',
    sampleAnswer:
      'Use backward-compatible migrations in multiple steps: (1) Add new column as NULLABLE, deploy service that uses it optionally, (2) Backfill data, (3) Make column NOT NULL in next deployment, (4) Remove old column in another deployment. Example: Renaming "address" to "shipping_address": Add shipping_address (nullable) → deploy service writing to both → backfill data → deploy service using only shipping_address → make NOT NULL → remove address column. Each step is backward-compatible so old and new service versions coexist during rolling deployment. NEVER make breaking changes in single step (adding NOT NULL column immediately breaks old service version).',
    keyPoints: [
      'Backward-compatible migrations in multiple steps',
      'Add column as nullable first',
      'Deploy service, then backfill, then enforce constraints',
      'Each step must work with previous service version',
      'Multiple deployments needed for breaking changes',
    ],
  },
  {
    id: 'q3-deploy',
    question:
      'Explain liveness vs readiness probes in Kubernetes. Why do you need both?',
    sampleAnswer:
      "Liveness probe: Is container alive? If fails, Kubernetes restarts the container. Checks if service is deadlocked or crashed. Readiness probe: Is container ready for traffic? If fails, Kubernetes removes pod from service load balancer (doesn't restart). Checks if dependencies are available. Need both because: Liveness prevents permanent failures (restart if stuck), Readiness prevents routing traffic to unhealthy instances (remove from LB temporarily). Example: Service starting up - readiness fails (not ready yet), liveness passes (not stuck). Service running but database down - readiness fails (can't serve requests), liveness passes (service itself is ok).",
    keyPoints: [
      'Liveness: Is service alive? (restart if fails)',
      'Readiness: Is service ready for traffic? (remove from LB if fails)',
      'Liveness prevents permanent failures (deadlocks, crashes)',
      'Readiness prevents routing to unhealthy instances',
      'Both needed for automatic recovery without manual intervention',
    ],
  },
];
