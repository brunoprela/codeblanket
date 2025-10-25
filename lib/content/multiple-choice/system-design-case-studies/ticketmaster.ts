/**
 * Design Ticketmaster Multiple Choice Questions
 */

export const ticketmasterMultipleChoice = [
  {
    id: 'ticketmaster-redis-cleanup',
    question:
      'When implementing seat holds in Ticketmaster, you use Redis with `SETEX seat:A1:hold user_123 600` to reserve a seat for 10 minutes. The user completes payment after 8 minutes. What is the correct cleanup procedure to release the Redis lock?',
    options: [
      'Leave the lock to expire automatically after 600 seconds; Redis will handle cleanup',
      'Execute `DEL seat:A1:hold` immediately after payment confirmation to release the lock',
      'Use `EXPIRE seat:A1:hold 0` to force immediate expiration of the key',
      'Execute `RENAME seat:A1:hold seat:A1:sold` to mark the seat as sold',
    ],
    correctAnswer: 1,
    explanation:
      "After payment confirmation, you should explicitly delete the Redis lock using `DEL seat:A1:hold` to immediately release the lock and free up resources. While the lock would expire automatically after 600 seconds (Option A), explicitly deleting it is better practice because: (1) it frees Redis memory immediately, (2) it prevents race conditions if the lock expiration happens during another operation, and (3) it makes the state transition atomic—once payment succeeds, the lock is no longer needed. Option C (`EXPIRE seat:A1:hold 0`) would also remove the key but is less idiomatic than DEL. Option D (RENAME) doesn't release the lock; it changes the key name, which doesn't help and could cause confusion in the system. The pattern should be: acquire lock → verify availability → process payment → delete lock → update database to 'sold' status.",
  },
  {
    id: 'ticketmaster-admission-rate',
    question:
      'During a Taylor Swift concert sale, 500,000 users are in the virtual queue and the system needs to admit users gradually. The current database CPU is at 85%, Redis memory is at 70%, and API latency p95 is 800ms. What should the queue admission rate controller do?',
    options: [
      'Increase admission rate from 1000 to 2000 users/minute to clear the queue faster',
      'Maintain current admission rate of 1000 users/minute since metrics are within acceptable range',
      'Decrease admission rate from 1000 to 500 users/minute due to high database CPU and API latency',
      'Stop admissions entirely until database CPU drops below 50%',
    ],
    correctAnswer: 2,
    explanation:
      'The system should decrease the admission rate to 500 users/minute because two critical metrics are showing stress: database CPU at 85% (target < 80%) and API latency p95 at 800ms (target < 500ms). These metrics indicate the system is approaching capacity limits, and admitting more users would risk cascading failures—high latency leads to connection timeouts, retries amplify load, and the system could crash. Option A (increase rate) would worsen the problem by adding more load to an already stressed system. Option B (maintain rate) ignores warning signs; the metrics show degradation, not stability. Option D (stop entirely) is overly aggressive—a gradual decrease to 500 users/minute reduces load while still processing the queue. The dynamic rate controller should continuously monitor these metrics and adjust admission rates accordingly: decrease by 50% if metrics exceed thresholds, increase by 50% if metrics are healthy (CPU < 60%, latency < 300ms). This provides backpressure without completely halting service.',
  },
  {
    id: 'ticketmaster-distributed-lock',
    question:
      "Two users (User A and User B) simultaneously try to purchase seat 'A1' at exactly the same millisecond. The system uses Redis distributed locks with `SET seat:A1:lock NX EX 600`. What is the guaranteed outcome with this approach?",
    options: [
      'Both users acquire the lock simultaneously, resulting in double-booking unless additional database checks are performed',
      'Exactly one user acquires the lock and can proceed; the other receives an immediate failure response',
      "Both requests are queued in Redis, and they're processed sequentially based on arrival order",
      'The user with faster network latency acquires the lock; the outcome is non-deterministic',
    ],
    correctAnswer: 1,
    explanation:
      "With Redis distributed locks using `SET NX` (set if not exists), exactly one user will acquire the lock atomically, and the other will receive an immediate failure response. Redis guarantees atomicity for single commands—even if two SET NX commands arrive in the same millisecond, Redis processes them sequentially (Redis is single-threaded for command execution). The first command to reach Redis will succeed and set the key; the second command will fail because the key already exists (NX = only set if not exists). This makes option B correct. Option A is wrong because Redis NX prevents double-locking; the atomicity guarantee ensures mutual exclusion. Option C is incorrect because Redis doesn't queue SET commands—it processes them immediately and returns success or failure. Option D is misleading; while network latency affects which request arrives first, once at Redis, the outcome is deterministic (first arrival wins). This is why Redis distributed locks are preferred for high-concurrency scenarios like Ticketmaster—they provide strong consistency guarantees without database-level contention.",
  },
  {
    id: 'ticketmaster-cleanup-optimization',
    question:
      "A background cleanup job runs every minute to release expired seat holds: `UPDATE seats SET status='available' WHERE status='held' AND held_until < NOW()`. During peak load, this query starts taking 5 seconds to execute, blocking other database operations. What is the best optimization?",
    options: [
      'Add an index on the held_until column: `CREATE INDEX idx_held_until ON seats (held_until)`',
      "Change the query to process in batches: `UPDATE seats SET status='available' WHERE seat_id IN (SELECT seat_id FROM seats WHERE status='held' AND held_until < NOW() LIMIT 100)`",
      'Add a composite index on (status, held_until): `CREATE INDEX idx_status_held ON seats (status, held_until)` and process in batches',
      'Move cleanup logic to Redis TTL expiration and eliminate the database cleanup job entirely',
    ],
    correctAnswer: 2,
    explanation:
      "The best solution is Option C: add a composite index on (status, held_until) and process updates in batches. Here\'s why this is optimal: The composite index `idx_status_held` allows the database to efficiently find rows where status='held' AND held_until < NOW() without scanning the entire table. The index on (status, held_until) is better than just held_until (Option A) because the query filters on both columns—the database can use the status part to narrow down to only 'held' seats, then use held_until within that subset. Additionally, processing in batches (e.g., LIMIT 100) prevents locking the entire seats table, which is critical during peak load when concurrent seat purchases are happening. Option A (index on held_until only) would still require filtering all rows by status. Option B (batching only) without the composite index would still scan slowly. Option D (Redis TTL only) is risky because if Redis fails or loses data, expired holds would never be cleaned up—the database must remain the source of truth. The best practice is: composite index + batched updates + Redis TTL as a performance optimization layer, with database as the authoritative cleanup mechanism.",
  },
  {
    id: 'ticketmaster-bot-detection',
    question:
      "To implement bot detection, you're analyzing user behavior patterns. A session shows: page load at 10:00:00.000, 'Select Seat' click at 10:00:00.053, payment form filled at 10:00:00.127, 'Purchase' click at 10:00:00.198. What is the most likely assessment?",
    options: [
      'Likely human user; timing shows natural human reaction speeds (50-200ms between actions)',
      'Likely bot; humans cannot interact this quickly—genuine users take 2-5 seconds between actions',
      'Inconclusive; need additional signals like mouse movement patterns and CAPTCHA scores',
      'Likely bot using recorded replay; the consistent sub-200ms timing indicates automation',
    ],
    correctAnswer: 2,
    explanation:
      "The correct answer is Option C: inconclusive—additional signals are needed before labeling this as a bot. While the timing is suspiciously fast (53ms to select seat, 74ms to fill form, 71ms to click purchase), timing alone is insufficient for bot detection because: (1) the user could have pre-filled payment info via browser autofill or password manager, (2) they might be using keyboard shortcuts or tab navigation rather than clicking, (3) network jitter could make timestamps appear closer than actual user actions, and (4) false positives hurt legitimate users. Option A is too lenient—while 50-200ms can be human, filling an entire payment form in 74ms is highly unusual without autofill. Option B is too strict—declaring bot based solely on speed would flag power users and cause false positives. Option D assumes replay attacks, but we'd need additional evidence (identical sequences across sessions). Modern bot detection requires multiple signals: behavioral patterns (mouse movement paths, scroll behavior, keystroke dynamics), device fingerprinting (same device, multiple accounts), CAPTCHA risk scores, IP reputation, and ML models trained on labeled bot/human datasets. A robust system might assign this session a medium-risk score (0.4-0.6) and request CAPTCHA verification before allowing purchase.",
  },
];
