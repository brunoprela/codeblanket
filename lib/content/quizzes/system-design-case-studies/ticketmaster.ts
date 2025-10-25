/**
 * Design Ticketmaster Quiz
 */

export const ticketmasterQuiz = [
  {
    id: 'ticketmaster-concurrency-control',
    question:
      'How would you prevent double-booking when 100,000 users simultaneously try to purchase the same seat in a high-demand concert? Compare pessimistic database locking, optimistic locking, and Redis distributed locks, and explain which approach you would choose and why.',
    sampleAnswer:
      "I would use Redis distributed locks as the primary mechanism with database state as the source of truth. The approach works as follows: When a user selects a seat, attempt to acquire a distributed lock in Redis using SET NX (set if not exists) with a 10-minute expiration: `redis.set('seat:A1:lock', user_id, nx=True, ex=600)`. If the lock is acquired, check the database to verify the seat is still available, then update the seat status to 'held'. If the lock cannot be acquired, immediately return an error indicating the seat is taken. This approach is superior to pessimistic locking because it doesn't create database bottlenecks—pessimistic locks force sequential processing at the database level, limiting throughput to ~100-200 transactions/second per seat. With Redis locks, we can handle 10,000+ lock attempts per second with sub-10ms latency. Optimistic locking using version numbers would result in excessive conflicts during high-demand sales, forcing users to retry repeatedly and creating poor UX. Redis distributed locks provide the speed of optimistic locking with the certainty of pessimistic locking, while keeping the database as the authoritative source of truth for auditing and recovery scenarios.",
    keyPoints: [
      'Redis distributed locks (SET NX) provide fast, atomic seat reservation without database bottlenecks',
      'Pessimistic database locking creates contention and limits throughput to hundreds of transactions/second',
      'Optimistic locking causes high conflict rates during concurrent purchases, requiring excessive retries',
      'Redis locks expire automatically (10 minutes) to handle abandoned sessions without manual cleanup',
      'Database remains source of truth for seat status, with Redis providing the concurrency control layer',
    ],
  },
  {
    id: 'ticketmaster-virtual-queue',
    question:
      'Design a virtual queue system for Ticketmaster that can handle 500,000 users trying to access a sale that opens at 10:00 AM for 10,000 tickets. How would you assign queue positions fairly, gradually admit users to the purchase flow, and dynamically adjust admission rates based on system load?',
    sampleAnswer:
      "I would implement a Redis-based virtual queue using sorted sets. When users arrive (typically 10-30 minutes before sale start), assign each a queue position using a random score: `redis.zadd('queue:event_123', random.random(), user_id)`. The random score ensures fairness—no one gets priority based on arrival time within the pre-queue window, preventing the 'fastest internet wins' problem. Once the sale opens at 10:00 AM, gradually admit users from the queue using `ZRANGE queue:event_123 0 999` to get the first 1000 users, then issue them admission tokens (JWTs valid for 10 minutes). The token allows them to browse seats and complete purchases. Dynamically adjust the admission rate based on system metrics: monitor database CPU (<80%), Redis memory (<85%), and API latency (<500ms p95). If metrics are healthy, increase admission rate (e.g., 1000→2000 users/minute); if system is stressed, decrease (1000→500 users/minute). Maintain WebSocket connections with queued users to provide real-time position updates: 'You are #45,632 in line. Estimated wait: 30 minutes.' When users complete purchases or their tokens expire, admit the next batch from the queue. This approach prevents site crashes while providing predictable wait times and fairness.",
    keyPoints: [
      'Redis sorted sets (ZADD with random scores) provide fair queue ordering without favoring early arrivals',
      'Gradual admission (1000 users/minute initially) prevents overwhelming backend systems',
      'Dynamic rate adjustment based on real-time system metrics (CPU, memory, latency) maintains stability',
      'Admission tokens (JWT with 10-minute expiration) grant time-limited access to purchase flow',
      'WebSocket real-time updates keep users informed of queue position and estimated wait time',
    ],
  },
  {
    id: 'ticketmaster-bot-prevention',
    question:
      'What strategies would you implement to detect and prevent bots from buying all tickets in a popular concert sale, while minimizing friction for legitimate users? Discuss multiple defense layers including rate limiting, device fingerprinting, behavioral analysis, and verified fan programs.',
    sampleAnswer:
      "I would implement a multi-layered defense strategy. Layer 1: Invisible reCAPTCHA v3 runs on every page, providing a risk score (0-1). Scores below 0.3 trigger visual CAPTCHA challenges. Layer 2: Rate limiting at multiple levels—5 seat selection attempts per minute per user, 10 seat holds per hour per credit card, and 10 API requests per second per IP address. Layer 3: Device fingerprinting to detect multiple 'users' from the same device by tracking user agent, screen resolution, timezone, installed fonts, and canvas fingerprinting. If 50 accounts originate from identical device fingerprints, flag all as suspicious. Layer 4: Behavioral analysis using machine learning to distinguish human patterns (gradual mouse movement, natural scroll patterns, 2-5 seconds between clicks) from bot patterns (instant clicks, no mouse movement, superhuman speed). Train a model on labeled data (known bots vs humans) to score each session. Layer 5: Verified Fan programs for high-demand artists—require pre-registration with name, email, phone number, and credit card weeks in advance, then verify via SMS code before sale day. This makes it economically infeasible for scalpers to create thousands of verified accounts. Layer 6: Transaction limits (4 tickets max per user per event) prevent hoarding. This layered approach increases bot prevention cost while keeping friction low for real fans.",
    keyPoints: [
      'Multi-layered defense (CAPTCHA, rate limiting, fingerprinting, behavioral analysis) increases bot operation costs',
      'Invisible reCAPTCHA v3 provides continuous risk scoring without disrupting legitimate user experience',
      'Device fingerprinting detects bot farms running multiple accounts from same hardware',
      'Behavioral analysis (mouse patterns, timing, scrolling) distinguishes human users from automated scripts',
      'Verified Fan pre-registration creates economic barrier for scalpers (SMS verification, credit card requirement)',
    ],
  },
];
