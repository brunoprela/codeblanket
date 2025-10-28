export const productionDeploymentQuiz = [
  {
    id: 1,
    question:
      'Design a complete production deployment architecture for a FastAPI application that needs to handle 10,000 concurrent users with high availability. The architecture must include: load balancer, multiple API servers with Gunicorn+Uvicorn, PostgreSQL database with replication, Redis for caching, monitoring with Prometheus/Grafana, logging with ELK stack, and zero-downtime deployment strategy. Explain how to calculate the number of Gunicorn workers, implement health checks, configure connection pooling, and handle graceful shutdowns. Include infrastructure diagrams and configuration files.',
    answer: `Complete production architecture with load balancer (Nginx) → API servers (Gunicorn with (2*CPU)+1 workers, Uvicorn worker class) → PostgreSQL primary/replica → Redis cluster. Includes connection pooling (pool_size=20 per worker), health checks (liveness/readiness), monitoring setup, and zero-downtime deployment with rolling updates.`,
  },
  {
    id: 2,
    question:
      'Compare three deployment strategies for FastAPI: blue-green, rolling updates, and canary deployments. For each strategy, explain: (1) how it works, (2) rollback process, (3) database migration handling, (4) pros and cons, and (5) when to use each. Implement a Kubernetes deployment with rolling update strategy including proper health checks, resource limits, and graceful shutdown. What happens if a new deployment has a bug discovered after 50% rollout?',
    answer: `Comprehensive comparison of deployment strategies. Blue-green: instant switch between environments (fast rollback, requires 2x resources). Rolling: gradual replacement (resource efficient, slower rollback). Canary: small percentage testing (safest, most complex). Includes Kubernetes manifests with health checks and rollback procedures.`,
  },
  {
    id: 3,
    question:
      'Design a comprehensive monitoring and observability strategy for production FastAPI deployment. The system must track: API response times, error rates, request throughput, database connection pool usage, Redis cache hit/miss ratio, CPU/memory usage, and custom business metrics. Implement Prometheus metrics collection, Grafana dashboards, alerting rules (when to alert on-call engineer), structured logging with context (request IDs), and integration with error tracking (Sentry). Show how to debug a production issue using logs and metrics.',
    answer: `Complete observability stack with Prometheus metrics (request_duration, error_rate, db_pool_usage), Grafana dashboards (golden signals: latency, traffic, errors, saturation), alerting rules (error rate > 5% for 5min), structured logging with request IDs for tracing, and Sentry integration for error tracking. Includes debugging workflow using correlated logs and metrics.`,
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `fastapi-deployment-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));

export const productionDeploymentMultipleChoice = [
  {
    id: 1,
    question:
      'What is the role of Gunicorn when deploying FastAPI to production?',
    options: [
      'Gunicorn manages multiple Uvicorn worker processes, providing process management, graceful restarts, and automatic worker restart on crash',
      'Gunicorn is a replacement for Uvicorn',
      'Gunicorn is only needed for sync frameworks, not FastAPI',
      'Gunicorn handles database connections',
    ],
    correctAnswer: 0,
    explanation:
      "Gunicorn is a process manager, Uvicorn is the ASGI server. Pattern: gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker. Gunicorn spawns 4 Uvicorn processes, each serving requests. Benefits: If one worker crashes, Gunicorn restarts it automatically. Graceful restarts (SIGHUP) for zero-downtime deployments. Load balancing across workers. Monitor worker health. Gunicorn doesn't replace Uvicorn (option 2) - they work together. FastAPI needs Uvicorn (ASGI), Gunicorn provides process management (option 3). Database connections handled by app code (option 4). Production formula: (2 * CPU_cores) + 1 workers.",
  },
  {
    id: 2,
    question:
      'How do you calculate the optimal number of Gunicorn workers for a production deployment?',
    options: [
      'Use the formula (2 * CPU_cores) + 1 workers for I/O-bound applications like FastAPI',
      'Always use 4 workers regardless of CPU',
      'Use as many workers as RAM allows',
      'One worker per expected concurrent user',
    ],
    correctAnswer: 0,
    explanation:
      'Formula: (2 * CPU_cores) + 1. Example: 4 CPU server = (2 * 4) + 1 = 9 workers. Why: Each worker handles async requests (not just 1), I/O-bound work yields during wait (CPU available for others), the +1 ensures CPU always busy, 2x accounts for context switching. Too few workers = underutilized CPU. Too many = excessive context switching, memory usage. For 4-core server: 9 workers, each handling hundreds of concurrent requests via async. This is for I/O-bound (FastAPI). CPU-bound would use workers = CPU_cores. Fixed 4 workers (option 2) wastes resources. RAM-based (option 3) ignores CPU. One per user (option 4) is massive overkill.',
  },
  {
    id: 3,
    question:
      'What is the difference between liveness and readiness probes in Kubernetes health checks?',
    options: [
      'Liveness checks if app is running (restart if fails), readiness checks if app is ready to serve traffic (remove from load balancer if fails)',
      'They are the same thing',
      'Liveness is for startup, readiness is for shutdown',
      'Only one is needed in production',
    ],
    correctAnswer: 0,
    explanation:
      'Liveness probe: "Is the application alive?" Returns 200 = alive, fail = Kubernetes restarts pod. Simple check: GET /health returns {"status": "healthy"}. Use for detecting deadlocks, infinite loops. Readiness probe: "Is the application ready to serve traffic?" Checks dependencies (database, Redis connected). Returns 200 = ready for traffic, fail = remove from load balancer but don\'t restart. Use during startup (connecting to DB) or degraded state (DB connection lost temporarily). Pattern: Liveness = simple ping, Readiness = check all dependencies. Both needed (option 4) for production robustness. They serve different purposes (option 2).',
  },
  {
    id: 4,
    question:
      'What is a zero-downtime deployment and how do rolling updates achieve it?',
    options: [
      'Rolling updates gradually replace old pods with new ones while keeping some old pods serving traffic, ensuring continuous availability',
      'All servers restart simultaneously',
      'Deployment happens outside business hours',
      'Users experience no slowdown',
    ],
    correctAnswer: 0,
    explanation:
      "Rolling update strategy: Start with 3 pods running v1, create 1 new pod with v2, wait for health check to pass, if healthy add to load balancer, remove 1 v1 pod, repeat until all v1 replaced. At every moment, some pods are serving traffic. Configuration: maxSurge: 1 (1 extra pod during update), maxUnavailable: 1 (1 pod can be down). Benefits: No downtime, gradual rollout (catch issues early), automatic rollback if health checks fail. Not simultaneous restart (option 2) which causes downtime. Time-based deployment (option 3) doesn't guarantee zero downtime. Option 4 is vague - zero-downtime means service availability, not necessarily performance.",
  },
  {
    id: 5,
    question:
      'Why should you use Docker multi-stage builds for FastAPI production images?',
    options: [
      'Multi-stage builds reduce final image size by separating build dependencies from runtime, resulting in smaller, more secure production images',
      'Multi-stage builds make the app run faster',
      'Multi-stage builds are required for FastAPI',
      'Multi-stage builds enable multiple environments in one image',
    ],
    correctAnswer: 0,
    explanation:
      "Multi-stage build pattern: Stage 1 (builder): FROM python:3.11, install build tools (gcc, make), pip install packages. Stage 2 (final): FROM python:3.11-slim, copy only installed packages from builder, copy application code. Result: Build dependencies (gcc, pip cache) not in final image. Benefits: Smaller images (500MB → 150MB), faster deployments, fewer attack vectors (no build tools in production), lower storage costs. Doesn't affect runtime performance (option 2). Not required (option 3) but highly recommended. Not about multiple environments (option 4) - use separate images for that. Example: Builder installs psycopg2 (needs gcc), final image only has compiled psycopg2 (no gcc).",
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `production-deployment-mc-${idx + 1}`,
  ...q,
}));
