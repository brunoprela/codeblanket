/**
 * Multiple choice questions for Service Decomposition Strategies section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const servicedecompositionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-decomposition-1',
    question:
      "You're decomposing an e-commerce monolith. Which decomposition is BEST aligned with business capabilities?",
    options: [
      'Frontend Service, Backend Service, Database Service, Cache Service',
      'Product Service, Order Service, Payment Service, Shipping Service, Customer Service',
      'UserRegistrationService, UserLoginService, UserProfileService, UserPasswordService',
      'Read Service, Write Service, Analytics Service',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 decomposes by business capability (what the business does). Each service maps to a clear business function. Option 1 is technical decomposition (how, not what) - every feature would touch all layers. Option 3 is over-granular (nanoservices) - all User-related functions should be one service. Option 4 is technical (CQRS pattern) not business-driven. Business capability decomposition is recommended because it aligns with how business stakeholders think and typically matches organizational structure.',
  },
  {
    id: 'mc-decomposition-2',
    question: 'In Domain-Driven Design, what is a "bounded context"?',
    options: [
      'A security boundary that limits access to sensitive data',
      'A clear boundary within which a domain model is defined and consistent',
      'The maximum size (lines of code) that a microservice should have',
      'A transaction boundary where ACID properties are guaranteed',
    ],
    correctAnswer: 1,
    explanation:
      'A bounded context is a key DDD concept: it\'s a boundary within which a domain model is consistent and has specific meaning. For example, "Customer" in the Sales context (leads, opportunities) is different from "Customer" in the Billing context (payment methods, invoices). Each bounded context typically becomes a microservice. It\'s not about security (option 1), size limits (option 3), or transactions (option 4), but about semantic boundaries in the business domain.',
  },
  {
    id: 'mc-decomposition-3',
    question:
      'You have User, Order, and OrderItems tables that are frequently joined. Product table is referenced by OrderItems but rarely joined. How should you decompose?',
    options: [
      'One service for all tables (User, Order, OrderItems, Product)',
      'User Service (User table) and Order Service (Order, OrderItems, Product tables)',
      'Each table gets its own service (4 services)',
      'Order Service (User, Order, OrderItems) and Product Service (Product), with ProductID as reference in Orders',
    ],
    correctAnswer: 3,
    explanation:
      "Option 4 correctly recognizes that User/Order/OrderItems are tightly coupled (always accessed together) and should stay together, while Product has an independent lifecycle managed by a different team. Order Service stores ProductID as a reference, accepting eventual consistency (product details might change after order is placed, which is usually acceptable). Option 1 (monolithic) doesn't give microservices benefits. Option 2 incorrectly couples Product with Order. Option 3 (nanoservices) creates too much granularity and requires distributed joins.",
  },
  {
    id: 'mc-decomposition-4',
    question:
      'What is the main problem with this architecture: Order Service, Payment Service, and Inventory Service all connecting to the same PostgreSQL database?',
    options: [
      "PostgreSQL can't handle connections from multiple services",
      "It creates a distributed monolith - services can't deploy independently and are coupled through database",
      'It violates security best practices for payment data',
      'PostgreSQL is not suitable for microservices architecture',
    ],
    correctAnswer: 1,
    explanation:
      'Shared database is a critical anti-pattern in microservices. When services share a database: (1) schema changes affect all services, (2) services can\'t deploy independently, (3) services are coupled through database schemas/tables, (4) you lose technology flexibility per service. This creates a "distributed monolith" - microservices complexity without the benefits. The solution is "database per service" pattern where each service owns its data. Option 1 is factually wrong (PostgreSQL handles multiple connections fine). Option 3, while potentially true, isn\'t the main architectural problem. Option 4 is false - PostgreSQL works fine with microservices if each service has its own database.',
  },
  {
    id: 'mc-decomposition-5',
    question:
      'Your API Gateway calls Order Service, which calls User Service, which calls Product Service, which calls Inventory Service (4 sequential network calls). What pattern should you use to reduce latency?',
    options: [
      'Merge all services back into a monolith',
      'Use caching at every layer to avoid repeated calls',
      'Create an Order Aggregation Service that makes parallel calls to User, Product, and Inventory services',
      'Implement database replication to speed up queries',
    ],
    correctAnswer: 2,
    explanation:
      "Sequential network calls create latency (waterfall effect). An Aggregation Service (Backend for Frontend pattern) makes parallel calls and combines results, reducing total time. Example: instead of 4 sequential calls (40ms total), make 3 parallel calls from aggregation service (10ms total). Option 1 (merge to monolith) is too drastic. Option 2 (caching) helps but doesn't solve sequential dependency. Option 4 (database replication) doesn't address service-to-service communication latency. Aggregation services are common for mobile/web BFFs (Backend for Frontend) where UI needs data from multiple services.",
  },
];
