export const debuggingProductionIssuesDiscussion = [
  {
    id: 1,
    question:
      'Production API suddenly showing 5-second response times (normal is 50ms). No code changes. CPU, memory, disk I/O all look normal. How do you debug this systematically?',
    answer:
      "**Systematic approach:** 1) **Timeline:** Check when it started exactly, correlate with deployments, config changes, traffic patterns. 2) **Scope:** All endpoints or specific ones? All users or region-specific? 3) **Quick checks:** `curl -w '@-' -o /dev/null API_URL` to measure. Check load balancer/ALB metrics for backend latency vs total latency. 4) **Network:** `tcpdump -i eth0 port 8000` captures request/response. Time between SYN and first byte. If network delay is 5s, not application. 5) **Application tracing:** Enable detailed logging, check for external API calls (timeouts?), database queries (N+1 problem?). 6) **strace:** `sudo strace -T -p $(pgrep app) | grep -v EAGAIN` shows syscall timings. Look for 5s delays. 7) **Discovery:** Found `connect()` to external auth service taking 5s (DNS resolution issue). **Fix:** Update DNS cache, add Circuit Breaker pattern. **Prevention:** Add distributed tracing, alert on external API latency.",
  },
  {
    id: 2,
    question:
      'Application crashes randomly every few hours with segfault. No pattern in logs. Design a comprehensive debugging strategy to identify the root cause.',
    answer:
      "**Multi-phase debugging:** **Phase 1 - Enable core dumps:** `ulimit -c unlimited`, `kernel.core_pattern=/tmp/core-%e-%p-%t` in sysctl. **Phase 2 - Wait for crash:** Configure monitoring alert when crash occurs. **Phase 3 - Analyze core:** `gdb /path/to/binary /tmp/core.12345`, run `bt full` for backtrace, `info threads` for all threads, examine crashed thread's variables. **Phase 4 - Reproduce:** If pattern found (e.g., NULL pointer at line 234), try to reproduce in staging with specific inputs. **Phase 5 - Root cause:** Common causes: 1) Race condition (multithreading bug), 2) Buffer overflow, 3) Use-after-free, 4) NULL pointer dereference. **Phase 6 - Fix:** Add NULL checks, fix race condition with proper locking, validate input sizes. **Phase 7 - Verify:** Run with valgrind/ASan in staging, load test extensively. **Prevention:** Enable AddressSanitizer in staging builds, implement comprehensive unit tests, add assertions for invariants, monitor crash rates.",
  },
  {
    id: 3,
    question:
      "You're debugging a slow database query. Query takes 10 seconds, but database CPU and disk I/O show minimal activity during execution. What's happening and how do you diagnose?",
    answer:
      "**Hypothesis:** Query is **waiting**, not working. **Diagnosis:** 1) **Check locks:** `SELECT * FROM pg_locks WHERE NOT granted;` (PostgreSQL) or `SHOW PROCESSLIST;` (MySQL). Likely waiting for table lock or row lock. 2) **Find blocker:** `SELECT pid, usename, query, wait_event FROM pg_stat_activity WHERE state='active';` Shows what's holding locks. 3) **Query plan:** `EXPLAIN ANALYZE SELECT...` May show sequential scan instead of index scan (but that would use CPU/IO). 4) **Network:** `tcpdump port 5432` Check if network latency between app and database. 5) **Connection pooling:** Check if waiting for available connection in pool. **Discovery:** Long-running transaction holding table lock. Transaction started 1 hour ago, never committed. **Root cause:** Application code started transaction, encountered error, didn't rollback. **Fix:** 1) Kill blocker: `SELECT pg_terminate_backend(PID);` 2) Fix application: Proper transaction management, `try/finally` with rollback. 3) Set `statement_timeout=30s` to prevent long-running queries. **Prevention:** Monitor lock waits, alert on transactions > 1 minute, implement proper error handling with automatic rollback.",
  },
];
