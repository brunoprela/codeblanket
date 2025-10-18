/**
 * Quiz questions for Synchronous vs Asynchronous Communication section
 */

export const syncvsasynccommunicationQuiz = [
  {
    id: 'q1',
    question:
      'You are designing a video upload feature for a social media platform. Should the video encoding be synchronous or asynchronous? How would you design the user experience?',
    sampleAnswer:
      'Video encoding should be ASYNCHRONOUS. REASONING: Video encoding is a long-running operation (takes 2-10 minutes for HD video). User cannot wait 10 minutes for HTTP response (timeout, terrible UX). Encoding is CPU-intensive, better to offload to background workers. User doesn\'t need encoding to complete before continuing (can post without encoding done). DESIGN: (1) UPLOAD PHASE (Synchronous): User selects video → POST /upload → Generate presigned S3 URL → Upload directly to S3 (chunked upload for large files) → Return upload_id to user. Latency: 5-30 seconds (just upload, no encoding). User sees: "Video uploaded! Processing..." (2) PROCESSING PHASE (Asynchronous): Upload complete → API publishes "video.uploaded" event to Kafka/SQS. Event payload: { upload_id, s3_key, user_id }. Background workers (pool of 100 workers) consume events. Each worker: Downloads video from S3, encodes multiple resolutions (1080p, 720p, 480p, 360p), generates thumbnails, uploads encoded videos back to S3, publishes "video.encoded" event. Latency: 2-10 minutes (user not waiting). (3) STATUS TRACKING: User polls GET /video/{upload_id}/status → Returns { status: "processing", progress: 45% }. Alternative: WebSocket push updates to user. When encoding complete, status becomes "ready". (4) NOTIFICATION: When encoding complete, publish "video.ready" event. Notification service sends push notification to user: "Your video is ready!". User Experience: Upload video → See "Processing..." → Continue using app → Get notification when ready → Video available for viewing. TRADE-OFF: User can\'t immediately watch video (2-10 min delay), but user isn\'t blocked waiting. System can scale workers independently. Failed encoding can be retried without affecting user. SCALE: Can handle millions of uploads/day by adding more workers.',
    keyPoints: [
      'Video encoding: Async (long-running, 2-10 minutes)',
      'Upload: Sync to S3 (fast, <30 seconds)',
      'Background workers process encoding queue',
      'User polls status or receives push notification when ready',
      'UX: Upload completes fast, encoding happens in background',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the Saga pattern for distributed transactions. Compare choreography vs orchestration approaches with a concrete example.',
    sampleAnswer:
      'SAGA PATTERN: Manages distributed transactions across multiple services using async communication. Ensures eventual consistency without distributed locks/2PC. PROBLEM: User books a trip: Book flight ($500) + book hotel ($200) + rent car ($100). Three services, three databases. Need all-or-nothing behavior (atomicity). Can\'t use database transaction (distributed). CHOREOGRAPHY APPROACH (Event-driven, no coordinator): (1) User books trip → Trip Service publishes "trip.requested" event. (2) Flight Service consumes event → Books flight → Publishes "flight.booked" event. (3) Hotel Service consumes "flight.booked" → Books hotel → Publishes "hotel.booked" event. (4) Car Service consumes "hotel.booked" → Rents car → Publishes "car.rented" event. (5) SUCCESS: All steps complete, trip confirmed. FAILURE SCENARIO: Hotel Service fails to book (no availability). (1) Hotel Service publishes "hotel.booking.failed" event. (2) Flight Service consumes failure → Publishes "flight.cancel" event → Cancels flight reservation. (3) Trip Service consumes → Notifies user "Trip booking failed, refunded." PROS: Loose coupling (services don\'t know about each other). Scalable (no central coordinator). CONS: Hard to understand flow (events cascade). No single place to see saga state. Circular dependency risk (A listens to B, B listens to C, C listens to A). ORCHESTRATION APPROACH (Coordinator controls flow): Trip Saga Orchestrator: (1) User books trip → Saga Orchestrator starts. (2) Orchestrator → Calls Flight Service (async via queue) → Waits for response. (3) Flight success → Orchestrator → Calls Hotel Service → Waits for response. (4) Hotel success → Orchestrator → Calls Car Service → Waits for response. (5) All success → Orchestrator → Updates trip status "confirmed". FAILURE SCENARIO: Hotel booking fails. (1) Orchestrator detects hotel failure. (2) Orchestrator calls compensating transaction: Flight Service cancellation. (3) Orchestrator updates trip status "failed". (4) Orchestrator notifies user. PROS: Clear flow (orchestrator defines order). Easy to see saga state (in orchestrator). Can add new steps without changing existing services. CONS: Orchestrator is single point of coupling. Orchestrator can become complex. RECOMMENDATION: Use orchestration for complex multi-step sagas (easier to maintain). Use choreography for simple event-driven flows. EXAMPLE IMPLEMENTATION (Orchestration with temporal/workflow engine): Use Temporal, Camunda, or AWS Step Functions to model saga. Workflow defines steps + compensations. Automatic retry, timeout handling.',
    keyPoints: [
      'Saga: Distributed transaction pattern using async communication + compensation',
      'Choreography: Event-driven, decentralized, harder to track',
      'Orchestration: Central coordinator, easier to understand and maintain',
      'Compensating transactions undo previous steps on failure',
      'Use orchestration for complex multi-step flows',
    ],
  },
  {
    id: 'q3',
    question:
      'Design an order processing system for an e-commerce site that uses both synchronous and asynchronous communication. Clearly identify which parts are sync vs async and why.',
    sampleAnswer:
      'E-commerce order processing - Hybrid sync/async design: SYNCHRONOUS PARTS (User waiting, critical path): (1) USER PLACES ORDER: User clicks "Place Order" → POST /orders. API validates request (sync): Check user authenticated, validate shipping address, validate payment method. If validation fails → Return 400 error immediately. Latency budget: <100ms. (2) INVENTORY CHECK: Order service → Sync call to Inventory Service: "Check stock for items [A, B, C]". Inventory service locks items temporarily (optimistic lock). Returns: { available: true } or { available: false, out_of_stock: ["A"] }. If out of stock → Return "Item A unavailable" to user. Latency: <50ms. (3) PAYMENT PROCESSING: Order service → Sync call to Payment Service: "Charge $299.99". Payment service calls Stripe API (external, sync). Returns: { status: "success", charge_id: "ch_123" } or { status: "failed", reason: "insufficient_funds" }. If payment fails → Return "Payment declined" to user. Latency: <300ms. (4) CREATE ORDER: Order service writes order to database (sync). Status: "confirmed", includes charge_id. Latency: <20ms. (5) RETURN TO USER: Return order details to user: { order_id: "ORD123", status: "confirmed", estimated_delivery: "Dec 20" }. TOTAL LATENCY: ~470ms (acceptable for checkout). ASYNCHRONOUS PARTS (Background, non-critical): (6) PUBLISH ORDER.PLACED EVENT: Order service publishes to Kafka topic "orders". Event: { order_id, user_id, items, total, timestamp }. No wait for consumers. (7) INVENTORY SERVICE CONSUMES: Listens to "orders" topic. Updates inventory counts (decrease stock). If this fails, retry via queue. Not critical for user (order already confirmed). (8) SHIPPING SERVICE CONSUMES: Creates shipment in shipping system. Calls FedEx API to create shipping label. Publishes "shipment.created" event. Latency: 2-5 seconds (user not waiting). (9) EMAIL SERVICE CONSUMES: Sends order confirmation email. If email fails, retry from dead letter queue. User still has order (email is nice-to-have). Latency: 1-10 seconds. (10) RECOMMENDATION ENGINE CONSUMES: Updates user purchase history for ML model. Batch processes orders for product recommendations. Latency: Minutes to hours (analytics). (11) FRAUD DETECTION CONSUMES: Analyzes order for fraud patterns. If fraud detected (async, 30 seconds later), publishes "order.flagged" event. Order service cancels order, refunds payment. User gets email: "Order cancelled due to security review." WHY HYBRID: User waits only for critical path (payment, inventory check). Fast checkout experience (<500ms). Background services (email, shipping, analytics) happen async. System resilient: If email service down, order still succeeds. Scalable: Can add new consumers (analytics, fraud detection) without changing order service. ARCHITECTURE: API Gateway → Order Service (sync) → Payment/Inventory Services (sync). Order Service → Kafka (async) → Multiple consumers. Dead letter queue for failed async tasks.',
    keyPoints: [
      'Synchronous: Inventory check, payment processing, order creation (user waiting)',
      'Asynchronous: Email, shipping, analytics, fraud detection (background)',
      'Critical path sync: Fast response <500ms, strong consistency',
      'Non-critical async: Better scalability, resilience, loose coupling',
      'Kafka for async events, dead letter queue for retries',
    ],
  },
];
