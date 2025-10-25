/**
 * Quiz questions for DoorDash Architecture section
 */

export const doordasharchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain DoorDash\'s dispatch optimization system. How does it assign orders to Dashers while minimizing delivery time and considering multiple constraints?",
    sampleAnswer:
      "DoorDash uses batch matching with optimization algorithm. Flow: (1) Collect orders and available Dashers every 30 seconds. (2) For each order, identify candidate Dashers within reasonable range (H3 geospatial indexing). (3) Calculate cost for each (order, Dasher) pair: estimated pickup time, delivery time, Dasher acceptance probability (decline history), stacking potential (can Dasher take 2 orders?), future orders (don't waste good Dasher on suboptimal order). (4) Formulate as optimization problem: Minimize total delivery time subject to constraints (on-time delivery, Dasher capacity ≤2, Dasher shift hours). (5) Solve using Mixed Integer Linear Programming (MILP) or greedy heuristics (MILP for small problems <1000 orders, greedy for scale). (6) Assign orders to Dashers. Trade-offs: Batch delay (30s) vs global optimization. Greedy (nearest Dasher) is suboptimal - better assignment exists considering future demand. Result: 15% improvement in delivery time vs greedy baseline.",
    keyPoints: [
      'Batch matching every 30 seconds, H3 indexing for candidate selection',
      'Cost function: Pickup time, delivery time, acceptance probability, stacking, future demand',
      'Optimization: MILP for small problems, greedy for scale',
      '15% delivery time improvement vs greedy nearest-Dasher assignment',
    ],
  },
  {
    id: 'q2',
    question:
      'How does DoorDash handle real-time location tracking and ETA prediction for orders? What data and models are involved?',
    sampleAnswer:
      'Real-time tracking: (1) Dasher app sends GPS every 4 seconds via WebSocket. (2) Location Gateway updates Redis (current location, TTL 60s) and Cassandra (history). (3) H3 hex indexing for nearby Dasher queries. ETA prediction: Multi-stage. (1) Restaurant prep time - ML model predicts based on: restaurant (historical prep times), order complexity (item count, customizations), time of day (lunch rush slower), current orders in queue (restaurant has 10 pending orders). (2) Dasher pickup time - Routing service calculates drive time to restaurant using: historical traffic patterns, real-time traffic (Google Maps API), distance. (3) Delivery time - Similar routing calculation to customer. (4) Total ETA = prep time + pickup time + delivery time + buffers. Update ETA every 30 seconds as Dasher moves. ML models: Gradient boosted trees trained on millions of historical orders. Features: Restaurant features, order features, time features, traffic features. Accuracy: Within 5 minutes for 80% of orders.',
    keyPoints: [
      'GPS updates every 4s → Redis (current) + Cassandra (history) + H3 indexing',
      'ETA = prep time + pickup time + delivery time, updated every 30s',
      'ML models (gradient boosted trees): Restaurant, order, traffic features',
      'Accuracy: 80% of predictions within 5 minutes',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe DoorDash\'s experimentation platform. How do they run thousands of A/B tests safely and analyze results?",
    sampleAnswer:
      'DoorDash experimentation platform enables thousands of concurrent A/B tests. Components: (1) Experiment configuration - Define experiment: variants (control, treatment), traffic allocation (5% treatment), target users (new users, SF Bay Area), metrics (order rate, revenue). (2) Assignment service - When user opens app, query assignment service with user_id. Service hashes user_id + experiment_id → deterministic bucket (0-99). If bucket < 5 → treatment, else control. Cache assignment in Redis. (3) Metrics pipeline - User actions (order, add to cart) logged to Kafka. Flink processes stream, joins with experiment assignments, aggregates metrics per experiment. Write to data warehouse (Snowflake). (4) Analysis dashboard - Statistical significance testing (t-test), confidence intervals, multiple comparison correction (Bonferroni). Visualizations (conversion funnel, time series). Safety: (1) Guardrail metrics - Monitor error rates, crash rates. Auto-stop experiment if guardrails violated. (2) Gradual rollout - Start at 1%, increase to 5%, 10%, 50% if metrics positive. (3) Experiment review - Senior engineers review high-risk experiments before launch.',
    keyPoints: [
      'Assignment service: Hash (user_id, experiment_id) → deterministic bucket',
      'Metrics pipeline: Kafka + Flink for stream processing, Snowflake for analysis',
      'Safety: Guardrail metrics, auto-stop, gradual rollout, experiment review',
      'Analysis: Statistical testing (t-test), confidence intervals, visualizations',
    ],
  },
];
