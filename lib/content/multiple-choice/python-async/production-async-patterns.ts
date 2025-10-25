import { MultipleChoiceQuestion } from '@/lib/types';

export const productionAsyncPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pap-mc-1',
    question: 'Why is graceful shutdown important in production?',
    options: [
      'It makes shutdown faster',
      'To allow in-flight requests to complete and cleanup resources before termination',
      'It is required by Docker',
      'It reduces memory usage',
    ],
    correctAnswer: 1,
    explanation:
      'Graceful shutdown allows proper cleanup before termination: What happens: Receive SIGTERM (from orchestrator/user). Set shutdown event (stop accepting new work). Wait grace period (e.g. 30s) for in-flight work to complete. Cleanup resources (close connections, save state). Exit. Without: In-flight requests aborted (data loss). Connections not closed (resource leak). State not saved (corruption). With: Requests complete (no data loss). Clean connection closure. State properly saved. Implementation: loop.add_signal_handler(SIGTERM, shutdown); shutdown_event.set(); await wait_for(gather(*tasks), timeout=30); await cleanup(). Container orchestration: Kubernetes sends SIGTERM, waits terminationGracePeriodSeconds (default 30s), then SIGKILL. Critical for: Data integrity, resource cleanup, zero-downtime deployments.',
  },
  {
    id: 'pap-mc-2',
    question:
      'What is the purpose of health check endpoints in async services?',
    options: [
      'To make services run faster',
      'To allow orchestrators to monitor service health and restart if unhealthy',
      'To log errors',
      'To test the service',
    ],
    correctAnswer: 1,
    explanation:
      'Health checks enable automated monitoring and recovery: Purpose: Orchestrator (Kubernetes, ECS) polls /health endpoint. If unhealthy: Restart service. If degraded: Stop routing traffic (readiness). Implementation: Check dependencies: database, cache, external APIs. Return: {"status": "healthy/degraded/unhealthy", "checks": {...}}. Status codes: 200 (healthy), 503 (unhealthy). Kubernetes: livenessProbe: Restart if unhealthy. readinessProbe: Remove from load balancer if not ready. Benefits: Automatic recovery (restart unhealthy services). Prevent cascading failures (remove degraded from LB). Visibility (dashboard showing health). Example: @app.get("/health"); async def health(): db_ok = await check_db(); redis_ok = await check_redis(); status = "healthy" if db_ok and redis_ok else "unhealthy"; return {"status": status}. Critical for production reliability!',
  },
  {
    id: 'pap-mc-3',
    question: 'Why use uvloop in production?',
    options: [
      'It is required for asyncio',
      'It provides 2-4× performance improvement over default asyncio event loop',
      'It adds new features',
      'It is easier to use',
    ],
    correctAnswer: 1,
    explanation:
      'uvloop is high-performance asyncio event loop implementation: Performance: Default asyncio: Pure Python event loop. uvloop: Cython-based, uses libuv (same as Node.js). Speedup: 2-4× faster than default. Benchmarks: 10,000 HTTP requests: Default asyncio: 2.0s. uvloop: 0.5s (4× faster). Usage: import uvloop; asyncio.set_event_loop_policy(uvloop.EventLoopPolicy()); asyncio.run(main()). Or: uvloop.run(main()). Benefits: Higher throughput (more requests/sec). Lower latency (faster response). Better scalability (handle more connections). Compatible: Drop-in replacement (no code changes). Use in: Production (always), Development (optional). Trade-off: Slight increase in complexity (C extension dependency). Recommendation: Use uvloop in production for free 2-4× performance boost!',
  },
  {
    id: 'pap-mc-4',
    question:
      'What is the recommended database connection pool size for production?',
    options: [
      'Always 1',
      'min_size=10, max_size=20 (typical), adjust based on load',
      'As large as possible',
      'Equal to number of CPU cores',
    ],
    correctAnswer: 1,
    explanation:
      'Connection pool sizing depends on workload: Typical: min_size=10 (keep 10 connections warm), max_size=20 (allow burst to 20). Why: min_size: Avoids cold start (creating connection ~100ms). Ensures capacity for normal load. max_size: Handles traffic spikes. Prevents overwhelming database. Sizing formula: max_size ≈ (expected concurrent queries) × 1.5. Example: 50 concurrent queries → max_size = 75. Too small: Pool exhaustion (requests queue/timeout). Not enough capacity. Too large: Wastes database resources (each connection = memory). May hit database connection limit. Configuration: await create_pool(min_size=10, max_size=20, command_timeout=60, max_inactive_connection_lifetime=300). Monitor: Pool usage (idle vs active). Adjust if consistently at max. Best practice: Start with 10/20, monitor, adjust based on metrics.',
  },
  {
    id: 'pap-mc-5',
    question: 'Why should all external calls have timeouts in production?',
    options: [
      'To make them faster',
      'To prevent indefinite hangs and resource exhaustion when services are slow/down',
      'It is required by Python',
      'To reduce memory usage',
    ],
    correctAnswer: 1,
    explanation:
      "Timeouts prevent cascading failures and resource exhaustion: Without timeout: Service A calls Service B. Service B hangs (doesn't respond). Service A waits forever. All Service A workers blocked (exhausted). Service A becomes unavailable (cascading failure). With timeout: Service A calls Service B with 5s timeout. Service B hangs. After 5s: TimeoutError. Service A logs error, returns fallback. Service A remains available. Implementation: HTTP: async with session.get(url, timeout=30) as resp. Database: await pool.fetch(query, timeout=60). Any operation: await wait_for(operation(), timeout=10). Benefits: Fast failure detection. Resource protection (connections released). Cascading failure prevention. Best practices: Set reasonable timeouts (30s HTTP, 60s queries). Use circuit breaker with timeouts. Log timeout errors. Critical: Every external call MUST have timeout in production!",
  },
];
