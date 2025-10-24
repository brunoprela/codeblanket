export const deploymentStrategiesQuiz = [
  {
    id: 'pllm-q-12-1',
    question:
      'Design a CI/CD pipeline for an LLM application that includes automated testing, Docker builds, Kubernetes deployment, and rollback capabilities. How do you ensure zero-downtime deployments?',
    sampleAnswer:
      'CI/CD Pipeline: 1) Code push triggers GitHub Actions, 2) Run unit tests (5min, must pass), 3) Run integration tests on staging (10min), 4) Build Docker image with version tag, 5) Push to registry (ECR/Docker Hub), 6) Deploy to staging Kubernetes, 7) Run smoke tests, 8) Manual approval for production, 9) Rolling update in production, 10) Monitor health, 11) Auto-rollback on errors. Zero-downtime strategy: Rolling update (update pods one at a time), readiness probes (dont send traffic to new pods until ready), liveness probes (restart unhealthy pods), connection draining (wait 30s for in-flight requests), maintain minimum replicas (50% available during update). Rollback: Keep previous 3 images, automatic rollback if error rate >5% for 1min, manual rollback via kubectl rollout undo, test rollback process monthly. Implementation: Dockerfile with multi-stage build, Kubernetes deployment YAML with 3 replicas, horizontal pod autoscaler, service with LoadBalancer. Pre-deployment checks: Database migrations (run before deploy, backwards compatible), configuration validation, integration test suite passes. Post-deployment: Monitor metrics for 10min, compare error rates to baseline, verify key endpoints, check logs for errors.',
    keyPoints: [
      'Complete pipeline: test → build → staging → production with approval gates',
      'Rolling updates with health checks for zero-downtime',
      'Automatic rollback based on error rate with manual override',
    ],
  },
  {
    id: 'pllm-q-12-2',
    question:
      'Compare blue-green deployment, canary deployment, and rolling updates for LLM applications. When would you choose each strategy and what are the trade-offs?',
    sampleAnswer:
      'Blue-green: Two identical environments (blue=current, green=new), deploy to green, test thoroughly, switch traffic 100% to green, keep blue as instant rollback. Pros: instant rollback, thorough testing pre-switch, zero downtime. Cons: 2x resources during deployment, all-or-nothing switch, database migrations tricky. Use for: major releases, critical updates, need high confidence. Rolling update: Update pods one at a time (10% → 30% → 100%), automatic by Kubernetes. Pros: gradual rollout, resource efficient, built-in. Cons: mixed versions running simultaneously, slower rollout, harder rollback. Use for: routine updates, confident in changes, resource-constrained. Canary: Deploy to small subset (5% traffic), monitor metrics, gradually increase (10% → 25% → 50% → 100%), rollback if issues. Pros: early issue detection, gradual validation, minimal user impact if problems. Cons: complex traffic routing, requires monitoring, slower. Use for: risky changes, new features, want real user validation. For LLM apps: Canary best for prompt changes (A/B test prompts), rolling update for code changes, blue-green for model upgrades. Implementation: Istio for canary traffic splitting, Argo Rollouts for progressive delivery, monitor LLM-specific metrics (response quality, latency, cost), automated rollback on quality degradation.',
    keyPoints: [
      'Blue-green: instant rollback but 2x resources, best for major releases',
      'Canary: gradual validation with real traffic, best for risky changes',
      'Rolling: resource-efficient and built-in, best for routine updates',
    ],
  },
  {
    id: 'pllm-q-12-3',
    question:
      'How would you containerize an LLM application with proper security, optimization, and configuration management? Include Dockerfile best practices and Docker Compose for local development.',
    sampleAnswer:
      'Dockerfile best practices: Multi-stage build (builder stage for dependencies, runtime stage for app), use slim base image (python:3.11-slim), pin versions (FROM python:3.11-slim), non-root user (USER appuser), minimal layers (combine RUN commands), .dockerignore (exclude tests, .git), layer caching (COPY requirements before code), health check (HEALTHCHECK CMD curl localhost:8000/health). Security: Scan with trivy, no secrets in image (use env vars), read-only filesystem where possible, drop capabilities, update packages regularly. Optimization: Multi-stage reduces size (500MB → 150MB), layer caching speeds builds (only rebuild changed layers), use BuildKit for parallel builds. Example Dockerfile: FROM python:3.11-slim AS builder; WORKDIR /app; COPY requirements.txt .; RUN pip install --no-cache-dir -r requirements.txt; FROM python:3.11-slim; RUN useradd -m appuser; COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages; COPY . .; USER appuser; CMD ["uvicorn", "main:app",]. Docker Compose for dev: Include app, Redis, PostgreSQL, Celery worker, volumes for hot reload, environment variables from .env, depends_on for startup order, health checks, local network.',
    keyPoints: [
      'Multi-stage build with slim base and non-root user for security',
      'Layer caching and optimization for fast builds',
      'Complete local environment with Docker Compose including all services',
    ],
  },
];
