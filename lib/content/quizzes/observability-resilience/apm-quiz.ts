/**
 * Quiz questions for APM section
 */

export const apmQuiz = [
  {
    id: 'q1',
    question:
      'What is the difference between APM (Application Performance Monitoring) and basic observability (logs, metrics, traces)? When is investing in an APM platform worth it?',
    sampleAnswer:
      'Basic observability provides raw telemetry data (logs, metrics, traces), while APM synthesizes this data into actionable insights with intelligence and context. **Differences**: (1) **Intelligence**: Basic: You query logs/metrics manually. APM: Automatic error grouping, root cause suggestions, anomaly detection. (2) **Context**: Basic: Separate tools for logs, metrics, traces. APM: Correlated view—click error → see trace → see logs → see metrics, all in one UI. (3) **Code-Level**: Basic: Service-level visibility. APM: Function-level profiling, slow database queries, N+1 detection. (4) **User Experience**: Basic: Server-side metrics only. APM: Real User Monitoring (RUM) with actual user experience data. (5) **Business Metrics**: Basic: Technical metrics. APM: Business transaction monitoring (revenue per transaction, conversion funnel). **Example**: Basic observability: "Error rate is 5%" (from metrics). APM: "Error rate 5% in checkout service, affecting premium users, caused by NullPointerException in PaymentController.java:42, started 10 minutes ago after v2.3.1 deployment, affecting $50K/hour revenue." **When APM Is Worth It**: (1) **Scale**: 10+ microservices where correlation is complex. (2) **Revenue Impact**: Downtime costs > $10K/hour. (3) **Team Size**: Multiple teams need visibility. (4) **Complexity**: Can\'t manually correlate logs/metrics/traces. (5) **User Experience**: Need RUM data for actual user performance. **When Basic Observability Sufficient**: Small team (< 5 engineers), monolith or few services, low revenue impact, can manually debug with logs. **Cost Consideration**: APM platforms cost $30-100 per host/month. For 100 instances, that\'s $3K-10K/month. Worth it if faster MTTR saves more in revenue or engineering time.',
    keyPoints: [
      'Observability = raw data, APM = intelligent insights',
      'APM adds: error grouping, anomaly detection, code-level profiling',
      'APM correlates logs/metrics/traces in single UI',
      'Worth it for: 10+ services, high revenue impact, complex debugging',
      'Cost $30-100/host/month, justify with faster MTTR and revenue protection',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between Real User Monitoring (RUM) and Synthetic Monitoring. What are the use cases for each, and why should you use both?',
    sampleAnswer:
      "RUM and Synthetic Monitoring measure performance from different perspectives and serve complementary purposes. **Real User Monitoring (RUM)**: (1) **What**: Measures actual user experience by collecting performance data from real users' browsers/apps. (2) **Data Collected**: Page load time, Time to First Byte, Largest Contentful Paint, user interactions, errors in browser. (3) **Pros**: Real user conditions (various networks, devices, locations), actual user behavior patterns, captures real issues users face. (4) **Cons**: Only detects issues after users affected, no data until users access the feature, affected by user's device/network (not just your system). (5) **Use Cases**: Understand true user experience, measure Core Web Vitals, identify slow user journeys, prioritize optimization based on real impact. **Synthetic Monitoring**: (1) **What**: Automated bots that simulate user interactions and measure performance from predefined locations. (2) **Setup**: Scripts that navigate site (login → checkout → purchase), run every N minutes from multiple locations. (3) **Pros**: Proactive (detects issues before users), consistent baseline, works 24/7 even when no real users, can test from specific geographies. (4) **Cons**: Doesn't capture real user diversity, scripts can break with UI changes, simulated load isn't real traffic patterns. (5) **Use Cases**: Uptime monitoring, regression detection after deployments, geographic performance testing, SLA verification. **Why Use Both**: Synthetic: Detects issues before users (canary in coal mine). Example: Synthetic detects checkout broken at 2am before any customers wake up. RUM: Validates real user experience. Example: Synthetic shows 1.2s load time, but RUM shows p99 of 5s for users in Asia on mobile—reveals geographic/device issue synthetic missed. **Concrete Example**: E-commerce site: Synthetic: Monitors checkout flow every 5 minutes from US, EU, Asia. Alerts if checkout fails or exceeds 3s. RUM: Collects data from all real users, shows that 10% of users on iPhone 8 + 3G experience 8s checkout. **Best Practice**: Use synthetic for proactive monitoring and alerting. Use RUM to understand real user pain points and prioritize optimization.",
    keyPoints: [
      'RUM = real users, actual experience, reactive (after users affected)',
      'Synthetic = bots, simulated interactions, proactive (before users affected)',
      'RUM pros: Real conditions, actual behavior; cons: Only see after impact',
      "Synthetic pros: Proactive, consistent baseline; cons: Doesn't capture diversity",
      'Use both: Synthetic for early detection, RUM for real user insights',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you use APM to identify and fix an N+1 query problem? Walk through the detection and resolution process.',
    sampleAnswer:
      'N+1 query problem occurs when code makes 1 query to fetch a list, then N queries to fetch related data for each item in the list. APM tools excel at detecting this. **Detection Process**: (1) **APM Alert**: Datadog APM shows /api/posts endpoint has high p99 latency (2.5s) and unusually high database query count (25 queries per request). (2) **Transaction Trace**: Click on slow transaction sample to see detailed trace. Trace shows: GET /api/posts (2.5s total) → Get all posts query (50ms) "SELECT * FROM posts" → Loop of 20 queries (2.4s total): "SELECT * FROM users WHERE id=1" (120ms), "SELECT * FROM users WHERE id=2" (120ms), ..., "SELECT * FROM users WHERE id=20" (120ms). (3) **Code-Level Insight**: APM shows the problematic code: PostController.java:42 inside loop calling userRepository.findById() 20 times. (4) **Identify Pattern**: Each post fetches its author individually instead of batching. **Root Cause**: ORM lazy loading—posts.author triggers individual query per post. (5) **Solution Options**: (A) Eager Loading: Use JOIN to fetch posts with authors in single query: "SELECT posts.*, users.* FROM posts JOIN users ON posts.author_id = users.id". (B) Batch Loading: Collect all author_ids, fetch all authors in one query: author_ids = [1, 2, ..., 20], then "SELECT * FROM users WHERE id IN (1,2,...,20)". (C) DataLoader Pattern: Use DataLoader to automatically batch and cache. (6) **Implementation**: Change ORM query from posts.include(:author) to posts.joins(:author) for eager loading. Result: 1 query instead of 21. (7) **Verification**: Deploy fix, check APM: p99 latency drops from 2.5s to 150ms (16x improvement!). Database query count drops from 25 to 1 per request. (8) **Automated Detection**: Configure APM alert: If database queries per request > 10, alert (catches N+1 early). **APM Advantages**: (1) Visual trace shows exact query pattern. (2) Code-level attribution identifies exact line. (3) Before/after comparison shows improvement. (4) Can set up alerts to catch regressions.',
    keyPoints: [
      'APM detects N+1: High query count, latency in loop pattern',
      'Transaction trace shows: 1 list query + N individual queries',
      'Code-level profiling identifies exact line causing issue',
      'Fix with eager loading (JOIN) or batch loading (IN clause)',
      'Verify: Query count drops, latency improves dramatically',
    ],
  },
];
