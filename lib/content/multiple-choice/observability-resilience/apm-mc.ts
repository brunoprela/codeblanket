/**
 * Multiple choice questions for APM section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apmMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does APM stand for, and what is its primary purpose?',
    options: [
      'Application Programming Model - for code architecture',
      'Application Performance Monitoring - for comprehensive application visibility',
      'Automated Process Management - for deployment',
      'Advanced Protocol Messaging - for communication',
    ],
    correctAnswer: 1,
    explanation:
      'APM stands for Application Performance Monitoring. It provides comprehensive visibility into application behavior by combining metrics, traces, logs, and code-level insights. Unlike basic observability (raw telemetry), APM synthesizes data into actionable insights with features like: automatic error grouping, root cause suggestions, code-level profiling, Real User Monitoring (RUM), and business transaction tracking.',
  },
  {
    id: 'mc2',
    question:
      'What is the N+1 query problem, and how does APM help identify it?',
    options: [
      'Making N queries when 1 would suffice, APM shows query patterns in traces',
      'Having N+1 database servers',
      'N users causing 1 database failure',
      'A specific type of SQL syntax error',
    ],
    correctAnswer: 0,
    explanation:
      "N+1 query problem occurs when code makes 1 query to fetch a list, then N additional queries to fetch related data for each item (instead of using a JOIN). APM detects this by showing: (1) High query count per request, (2) Loop pattern in transaction trace (20 identical queries), (3) Code-level attribution to the exact loop. Example: Fetch 20 posts → Make 20 separate queries for authors instead of one JOIN or batch query. APM's visual trace reveals this immediately.",
  },
  {
    id: 'mc3',
    question: 'What is Real User Monitoring (RUM) in APM?',
    options: [
      'Testing by real users in staging',
      "Collecting performance data from actual users' browsers/apps",
      'Monitoring user login frequency',
      'Tracking user behavior with analytics',
    ],
    correctAnswer: 1,
    explanation:
      "Real User Monitoring (RUM) collects performance data from actual users' browsers or mobile apps, measuring their real experience. It captures: Page load time, Time to First Byte, Largest Contentful Paint, JavaScript errors, user interactions. Unlike synthetic monitoring (bots simulating users), RUM shows actual conditions: various devices, network speeds, geographies. Essential for understanding true user experience and Core Web Vitals.",
  },
  {
    id: 'mc4',
    question: 'What is a Service Map in APM, and why is it valuable?',
    options: [
      'A geographic map showing server locations',
      'A visual representation of service dependencies with health and performance',
      'A deployment pipeline diagram',
      'A database schema visualization',
    ],
    correctAnswer: 1,
    explanation:
      'A Service Map is a visual representation of microservice dependencies, showing: (1) Which services call which services, (2) Request rate between services, (3) Error rate per connection, (4) Latency per connection, (5) Health status (green/yellow/red). Value: Understand architecture, identify critical paths, spot cascading failures, plan migrations, debug cross-service issues. Example: Discover that payment-service calls auth-service which is failing at 10% rate.',
  },
  {
    id: 'mc5',
    question:
      'When is investing in an APM platform worth the cost ($30-100/host/month)?',
    options: [
      'For all projects, regardless of size',
      'For complex distributed systems (10+ services) with high revenue impact',
      'Only for frontend applications',
      'Never, basic logging is sufficient',
    ],
    correctAnswer: 1,
    explanation:
      'APM is worth the investment when: (1) 10+ microservices where manual correlation is complex, (2) Downtime costs > $10K/hour revenue impact, (3) Multiple teams need visibility, (4) Faster MTTR (Mean Time To Recovery) justifies cost, (5) Real User Monitoring data is valuable. Not worth for: Small teams (<5 engineers), monolith/few services, low revenue impact, can manually debug with basic logs. Cost vs benefit analysis: If APM reduces MTTR by 1 hour and downtime costs $50K/hour, $3K/month APM cost pays for itself.',
  },
];
