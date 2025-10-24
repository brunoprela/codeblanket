/**
 * Multiple choice questions for Metrics & Monitoring section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const metricsMonitoringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is cardinality in the context of metrics?',
    options: [
      'The number of data points in a time series',
      'The number of unique time series created by combinations of label values',
      'The accuracy of metric measurements',
      'The frequency of metric collection',
    ],
    correctAnswer: 1,
    explanation:
      'Cardinality is the number of unique time series created by all combinations of label values. For example, a metric with labels {method, status, region} where method has 4 values, status has 5 values, and region has 10 values creates 4 × 5 × 10 = 200 time series (cardinality = 200). High cardinality (millions of unique combinations) causes storage explosion, slow queries, and high costs. This is why you should never use user_id or request_id as labels.',
  },
  {
    id: 'mc2',
    question:
      'What type of metric is http_requests_total, and how should it be queried?',
    options: [
      'Gauge, query with avg_over_time()',
      'Counter, query with rate()',
      'Histogram, query with histogram_quantile()',
      'Summary, query directly',
    ],
    correctAnswer: 1,
    explanation:
      "http_requests_total is a Counter (monotonically increasing, only goes up, resets on restart). Counters should be queried with rate() or increase() functions. For example, rate(http_requests_total[5m]) gives requests per second over the last 5 minutes. You never query a counter's raw value because it just keeps increasing—rate() converts it to a meaningful per-second measurement.",
  },
  {
    id: 'mc3',
    question: 'What does the RED method stand for, and when should it be used?',
    options: [
      'Reliability, Efficiency, Durability - for infrastructure',
      'Rate, Errors, Duration - for request-based services',
      'Read, Execute, Debug - for troubleshooting',
      'Response, Error, Delay - for network monitoring',
    ],
    correctAnswer: 1,
    explanation:
      'RED method stands for Rate (requests per second), Errors (error rate), and Duration (latency). It\'s used for monitoring request-based services like APIs and microservices. It answers: "Is my service healthy from the user\'s perspective?" For example, track: Rate (1000 req/s), Errors (0.5% error rate), Duration (p99 = 200ms). RED is complementary to USE method (Utilization, Saturation, Errors) which is for infrastructure resources.',
  },
  {
    id: 'mc4',
    question:
      'Which of the following is a high-cardinality label that should be AVOIDED in metrics?',
    options: [
      'http_method (GET, POST, PUT, DELETE)',
      'user_id (millions of unique users)',
      'region (us-east-1, eu-west-1, ap-south-1)',
      'http_status_code (200, 404, 500)',
    ],
    correctAnswer: 1,
    explanation:
      'user_id should be avoided as a metric label because it has millions of unique values (high cardinality), creating millions of time series. This causes: storage explosion, slow queries, high costs, and can crash Prometheus. Use logs for high-cardinality data. Low-cardinality labels like http_method (4 values), region (5-10 values), and status_code (~20 values) are safe because they have bounded sets of values.',
  },
  {
    id: 'mc5',
    question:
      'What is the difference between a Counter and a Gauge in Prometheus?',
    options: [
      'Counter measures current value, Gauge measures total over time',
      'Counter only increases (monotonic), Gauge can go up or down',
      'Counter is for errors, Gauge is for success',
      'There is no difference, they are the same',
    ],
    correctAnswer: 1,
    explanation:
      "Counter is monotonically increasing (only goes up, resets on restart), used for accumulating values like http_requests_total, errors_total. Query with rate(). Gauge is a point-in-time value that can increase or decrease, used for current state like cpu_usage_percent, active_connections, memory_used_bytes. Query with current value or avg_over_time(). Using the wrong type breaks queries (e.g., rate() on gauge doesn't work correctly).",
  },
];
