/**
 * Quiz questions for Data Management in Microservices section
 */

export const datamanagementmicroservicesQuiz = [
  {
    id: 'q1-data',
    question:
      "Why can't you use database JOINs in microservices? How do you query data that spans multiple services?",
    sampleAnswer:
      "JOINs don't work in microservices because each service has its own database - you can't JOIN across network boundaries between separate databases. Solutions: (1) API Composition - application makes multiple API calls and combines results in memory (simple but has latency), (2) Data Duplication - store denormalized data in each service (Order Service stores product name/price, not just ID), update via events, (3) CQRS with Read Models - create specialized read databases that aggregate data from multiple services via events. Choose based on query patterns and consistency requirements. API composition for simple queries, CQRS for complex dashboards, data duplication for frequently accessed data.",
    keyPoints: [
      'Each service has own database (database per service)',
      "Can't JOIN across network/database boundaries",
      'API Composition: multiple calls, combine in app',
      'Data Duplication: store denormalized copies, update via events',
      'CQRS: dedicated read models for complex queries',
    ],
  },
  {
    id: 'q2-data',
    question:
      'What are the trade-offs of the "database per service" pattern? When might you bend this rule?',
    sampleAnswer:
      'Trade-offs: Pros: (1) Loose coupling (services evolve independently), (2) Technology diversity (best DB for each use case), (3) Independent scaling, (4) Fault isolation. Cons: (1) No JOINs across services, (2) Distributed transactions require Saga pattern, (3) Data duplication, (4) Eventual consistency instead of immediate. Bend the rule when: Starting from monolith (Strangler Fig - extract services gradually while sharing DB initially), read-only sharing for analytics/reporting (but better to use events), very early startup (pragmatic to start simple). However, plan migration path to true database per service as you scale.',
    keyPoints: [
      'Pros: autonomy, tech diversity, scaling, isolation',
      'Cons: no JOINs, distributed transactions, duplication, eventual consistency',
      'Bend for: monolith migration (Strangler Fig), analytics',
      'Always plan migration to true database per service',
      "Don't share for write operations (breaks encapsulation)",
    ],
  },
  {
    id: 'q3-data',
    question:
      'Your Order Service stores product price. Product Service updates the price. Should old orders update? How do you handle this?',
    sampleAnswer:
      "Usually old orders should NOT update - they should show the historical price at purchase time (auditing, receipts). For pending orders, it depends on business rules. Implementation: (1) Store snapshot of product data when order created (productName, productPrice at time of order), (2) Product Service publishes ProductPriceChanged event, (3) Order Service can listen and update pending orders if needed, but completed orders stay unchanged. Use status field to determine: ORDER_STATUS = COMPLETED → don't update (historical), ORDER_STATUS = PENDING → optionally update if business requires. This is data duplication with event-driven updates - eventual consistency is fine here.",
    keyPoints: [
      'Historical orders should show price at purchase time',
      'Store snapshot of product data when creating order',
      'Product Service publishes ProductPriceChanged event',
      'Order Service updates based on order status (pending vs completed)',
      'Event-driven updates maintain eventual consistency',
    ],
  },
];
