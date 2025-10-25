/**
 * Multiple choice questions for Design Uber section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const uberMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Uber tracks 500,000 active drivers, each sending location updates every 5 seconds. How many location updates per second must the system process?',
    options: [
      '10,000 updates/sec',
      '100,000 updates/sec',
      '300,000 updates/sec',
      '500,000 updates/sec',
    ],
    correctAnswer: 1,
    explanation:
      '500,000 drivers × (1 update / 5 seconds) = 500,000 / 5 = 100,000 updates/second. Plus riders tracking drivers add another 50K/sec (riders update less frequently). Total: ~150K updates/sec. This is why Uber uses Kafka for ingestion (handles millions of msgs/sec) and Cassandra for storage (write-optimized). Each update: 20 bytes × 100K = 2 MB/sec = 172 GB/day. Stored for 30 days = 5.2 TB location history. This scale requires purpose-built time-series storage.',
  },
  {
    id: 'mc2',
    question:
      'Why does Uber use WebSocket connections instead of HTTP polling for real-time location updates?',
    options: [
      'WebSockets are newer technology',
      'HTTP polling: Client requests every 3 sec = overhead; WebSocket: Server pushes updates = efficient',
      'WebSockets are more secure',
      'HTTP cannot send location data',
    ],
    correctAnswer: 1,
    explanation:
      'HTTP POLLING: Rider polls GET /driver/location every 3 seconds. 1M riders × 1 req/3 sec = 333K req/sec. Each req: TCP handshake + headers (~500 bytes overhead). Total: 166 MB/sec overhead. Latency: Up to 3 seconds stale. WEBSOCKET: Server pushes location to rider immediately when driver moves. 1M riders = 1M persistent connections (~7.5 GB memory). Each update: 50 bytes (no handshake). Latency: Real-time (<100ms). RESULT: WebSocket 10x more efficient, lower latency, better UX. Trade-off: Stateful connections (must maintain 1M connections across gateway servers) vs stateless HTTP (easier to scale). For real-time tracking, WebSocket is industry standard.',
  },
  {
    id: 'mc3',
    question:
      'A rider in San Francisco requests a ride. Uber finds 20 drivers within 5 km. What is the optimal matching strategy?',
    options: [
      'Broadcast request to all 20 drivers, first to accept wins',
      'Send to closest driver, wait 15 seconds, if decline/timeout try next driver iteratively',
      'Run auction where drivers bid on the ride',
      'Randomly assign to one driver',
    ],
    correctAnswer: 1,
    explanation:
      "ITERATIVE MATCHING (Production): Send to closest driver (best UX - shortest wait). Wait 15 seconds (long enough for response, short enough for fallback). If decline/timeout, immediately try next. Average 2-3 drivers before match. Total time: 20-45 seconds. BROADCAST (Option A): All 20 drivers receive notification simultaneously. First to accept wins. PROBLEM: (1) Spammy - all drivers interrupted, 19 waste time reviewing. (2) Race condition if multiple accept. (3) Drivers hesitate (waiting to see if anyone accepts). RESULT: Lower acceptance rate, poor driver experience. AUCTION (Option C): Complex, slow (drivers must submit bids). Against Uber\'s UX (instant matching). RANDOM (Option D): Ignores distance, poor UX (far driver assigned when close one available). PRODUCTION CHOICE: Iterative because it optimizes for: (1) Rider UX (closest driver, fastest pickup), (2) Driver UX (only interrupted if likely match), (3) Fast matching (15-30 sec target). Fallback: After 5 declines, expand radius and broadcast.",
  },
  {
    id: 'mc4',
    question:
      'Uber calculates ETA as "John is 4 minutes away". The driver is 2 km away. Which approach provides the most accurate ETA?',
    options: [
      'Straight-line distance / average speed: 2 km / 40 km/h = 3 min',
      'Call Google Maps API with current location, destination, and departure_time=now (includes traffic)',
      'Use historical average for this route',
      'Ask the driver to estimate',
    ],
    correctAnswer: 1,
    explanation:
      'HAVERSINE (Option A): Straight-line distance, assumes constant speed. ERROR: Ignores actual roads (must follow streets), traffic (rush hour vs midnight), stoplights. Typical error: ±5-10 minutes. MAPS API (Correct): Request: GET /directions?origin=driver_lat,lon&destination=rider_lat,lon&departure_time=now. Response: duration_in_traffic: 240 seconds (4 min). Includes: Actual road network, current traffic conditions, historical patterns, stoplights/turns. Accuracy: ±1-2 minutes. Cost: $0.005 per request. HISTORICAL (Option C): "This route usually takes 5 min". Ignores current traffic (accident, road closure). Less accurate than real-time. UBER PRODUCTION: Use Haversine for initial estimate (instant, free), switch to Maps API after driver accepts (accurate, critical for UX). Update ETA every 30-60 seconds as driver moves. Cache common routes (airport → downtown) to reduce API costs.',
  },
  {
    id: 'mc5',
    question:
      'Uber processes payments after trip completion. Why is using an idempotency key critical?',
    options: [
      'Idempotency keys encrypt payment data',
      'Without idempotency: Network failure during charge → Retry → Rider charged twice',
      'Idempotency keys are required by law',
      'Idempotency keys make payments faster',
    ],
    correctAnswer: 1,
    explanation:
      'PROBLEM: Trip completes, app calls stripe.charge($30). Network timeout before response received. App retries → Rider charged $30 twice. SOLUTION: Idempotency key (unique per trip): stripe.charge($30, idempotency_key="trip_abc123"). First request: Charge succeeds, Stripe stores key. Retry with same key: Stripe returns original charge (no duplicate charge). IMPLEMENTATION: idempotency_key = f"trip_{trip_id}_{timestamp}". Ensures uniqueness per trip. Works for any external API call. STRIPE BEHAVIOR: Stores keys for 24 hours. After 24 hours, same key creates new charge (different trip). BEST PRACTICE: Always use idempotency for payment/financial APIs. One duplicate charge = customer trust lost. This is critical in distributed systems where network failures are common.',
  },
];
