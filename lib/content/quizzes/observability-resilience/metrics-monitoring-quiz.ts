/**
 * Quiz questions for Metrics & Monitoring section
 */

export const metricsMonitoringQuiz = [
  {
    id: 'q1',
    question:
      'Explain what cardinality means in the context of metrics, why high cardinality is problematic, and how to avoid it. Provide examples of high-cardinality and low-cardinality labels.',
    sampleAnswer:
      'Cardinality is the number of unique time series created by all combinations of label values for a metric. **Example**: Metric http_requests_total with labels {method, path, status}. If method has 4 values (GET, POST, PUT, DELETE), path has 10 values, and status has 5 values, cardinality = 4 × 10 × 5 = 200 unique time series. **Why High Cardinality Is Problematic**: (1) Storage Explosion: Each time series requires storage. Millions of time series = terabytes of data. (2) Query Performance: Querying across millions of series is slow, dashboards time out. (3) Memory Usage: Prometheus keeps recent data in memory. High cardinality exhausts RAM. (4) Cost: Cloud metric services (Datadog, New Relic) charge per time series, costs skyrocket. (5) System Crash: Prometheus can OOM-kill with too many series. **High-Cardinality Labels (AVOID)**: user_id (millions of unique users), request_id (infinite unique values), email_address (millions), IP_address (millions), timestamp (infinite), session_id (billions). Each unique value creates a new time series. **Low-Cardinality Labels (GOOD)**: http_method (4-5 values), http_status_code (~20 values), region (5-10 values), service_name (10-100 values), environment (3-4 values: prod, staging, dev). **How to Avoid**: (1) Never use user_id or request_id as labels. (2) Limit label values to bounded sets (< 1000 unique values per label). (3) Use logs for high-cardinality data ("user_id: 123" in log entry, not metric label). (4) Aggregate at collection time (count by status code, not by individual requests). (5) Use label_replace to normalize labels (combine /api/users/1, /api/users/2 into /api/users/:id). (6) Monitor your cardinality: active_series metric in Prometheus. **Real Example**: Bad: http_requests{user_id="user123"} creates millions of series. Good: http_requests{status="200"} creates ~20 series, then query logs for specific users.',
    keyPoints: [
      'Cardinality = number of unique time series (combinations of label values)',
      'High cardinality causes storage explosion, slow queries, high costs',
      'High-cardinality labels: user_id, request_id, IP addresses',
      'Low-cardinality labels: HTTP method, status code, region',
      'Use logs for high-cardinality data, metrics for aggregated data',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the RED method and USE method for monitoring. When should you use each, and what metrics do they focus on? Provide examples.',
    sampleAnswer:
      'RED and USE are complementary monitoring frameworks that focus on different aspects of systems. **RED Method (for Services)**: Focuses on user-facing services and request-based workloads. (1) Rate: Requests per second. Example: http_requests_total rate (100 req/s). (2) Errors: Failed requests rate. Example: 5xx responses / total requests (0.5% error rate). (3) Duration: Latency distribution. Example: p50=100ms, p99=500ms. **When to Use RED**: Microservices, APIs, web servers, anything handling requests. It answers: "Is my service healthy from the user\'s perspective?" Example: E-commerce API should track: Rate (orders/minute), Errors (failed checkouts %), Duration (checkout latency). **USE Method (for Resources)**: Focuses on infrastructure and resource utilization. (1) Utilization: % time resource is busy. Example: CPU at 65%, Disk at 80%. (2) Saturation: Degree of queued work. Example: CPU run queue=2 (tasks waiting), Memory swapping. (3) Errors: Error events. Example: Disk errors, network drops. **When to Use USE**: Infrastructure monitoring (servers, databases, networks). It answers: "Are my resources overloaded?" Example: Database server should track: Utilization (CPU%, disk I/O%), Saturation (connection pool queue, query queue), Errors (connection failures, disk errors). **Complementary Usage**: For a typical web application: (1) Use RED for your API layer (request-based): Rate, Errors, Duration of HTTP requests. (2) Use USE for your infrastructure: CPU, memory, disk utilization/saturation/errors. (3) Together they give complete picture: RED shows user impact, USE shows resource health. **Red Flag**: If USE shows low utilization but RED shows high errors, likely an application bug (not resource constrained). If USE shows high saturation and RED shows high latency, need to scale resources.',
    keyPoints: [
      'RED (Rate, Errors, Duration) for request-based services',
      'USE (Utilization, Saturation, Errors) for infrastructure resources',
      'RED answers: Is service healthy for users?',
      'USE answers: Are resources overloaded?',
      'Use both together: RED for services, USE for infrastructure',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the difference between a Counter and a Gauge in metrics? Provide examples of when to use each, and explain what happens if you use the wrong type.',
    sampleAnswer:
      "Counter and Gauge are fundamentally different metric types with different semantics and use cases. **Counter**: Monotonically increasing value that only goes up (or resets to 0). (1) Starts at 0. (2) Only increases (never decreases). (3) Resets on process restart. (4) Examples: http_requests_total (increments on each request), errors_total (increments on each error), bytes_sent_total (accumulates bytes). (5) Query with rate(): rate(http_requests_total[5m]) gives requests per second. **Gauge**: Point-in-time value that can go up or down. (1) Current value right now. (2) Can increase or decrease. (3) Represents instantaneous state. (4) Examples: cpu_usage_percent (currently 65%), active_connections (currently 47), memory_used_bytes (current memory), temperature_celsius (current temp), queue_length (current items in queue). (5) Query with avg_over_time(): avg_over_time(cpu_usage_percent[5m]) gives average CPU. **Using Wrong Type - Problems**: (1) Counter as Gauge: If you use counter for CPU usage, it only increases even when CPU drops, making it useless. You'd see CPU going from 65 → 80 → 95 even if actual CPU went 65 → 80 → 40. (2) Gauge as Counter: If you use gauge for request count, it might go down on restart or when you reset it, losing historical data. rate() function won't work correctly because it expects monotonic increases. **Example Scenario**: Total HTTP Requests: Use COUNTER (http_requests_total). Query: rate(http_requests_total[5m]) = 100 req/s. Active HTTP Connections: Use GAUGE (http_connections_active). Query: http_connections_active = 47 connections right now. **Rule of Thumb**: If you're counting things that accumulate over time (requests, errors, bytes), use Counter. If you're measuring current state (CPU, memory, queue depth), use Gauge.",
    keyPoints: [
      'Counter: Monotonically increasing, only goes up, resets on restart',
      'Gauge: Point-in-time value, can go up or down',
      'Counter examples: requests_total, errors_total, bytes_sent',
      'Gauge examples: cpu_usage, active_connections, memory_used',
      'Query counters with rate(), gauges with current value or avg_over_time()',
    ],
  },
];
