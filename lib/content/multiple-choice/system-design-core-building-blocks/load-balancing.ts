/**
 * Multiple choice questions for Load Balancing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const loadbalancingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Your application has 10 servers behind a Round Robin load balancer. 8 servers have 16GB RAM, 2 servers have 64GB RAM. What problem will you likely encounter?',
    options: [
      'All servers will be equally loaded',
      'The 2 powerful servers will be underutilized, 8 weaker servers may be overloaded',
      'Round Robin will automatically detect server capacity and adjust',
      'This configuration will work optimally',
    ],
    correctAnswer: 1,
    explanation:
      'Round Robin distributes requests equally (10% to each server) regardless of server capacity. The 2 servers with 64GB RAM will handle 10% of traffic each (underutilized), while the 8 servers with 16GB RAM also get 10% each (may be overloaded if requests are memory-intensive). Solution: Use Weighted Round Robin with higher weights for powerful servers, or Least Connections/Least Response Time to adapt to actual load.',
  },
  {
    id: 'mc2',
    question:
      'Which load balancing algorithm is BEST for WebSocket connections (long-lived, persistent connections)?',
    options: ['Round Robin', 'Random', 'Least Connections', 'IP Hash'],
    correctAnswer: 2,
    explanation:
      "Least Connections is best for WebSocket connections because: (1) WebSockets are long-lived (can last hours). (2) Round Robin doesn't account for existing connections - Server A might have 100 active WebSocket connections while Server B has 10, but Round Robin still sends requests equally. (3) Least Connections routes new connections to the server with fewest active connections, balancing load based on actual work. IP Hash could work but doesn't balance load well if users are unevenly distributed.",
  },
  {
    id: 'mc3',
    question:
      "Your users complain they're randomly logged out. You discover the load balancer uses Round Robin and sessions are stored in server memory. What is the recommended solution?",
    options: [
      'Switch to IP Hash for sticky sessions',
      'Use cookie-based sticky sessions',
      'Store sessions in Redis (shared storage) and keep any load balancing algorithm',
      'Increase server memory',
    ],
    correctAnswer: 2,
    explanation:
      "Storing sessions in Redis (shared storage) is the recommended solution because: (1) Any server can retrieve any session (truly stateless). (2) No data loss if server crashes. (3) Easy to scale (add/remove servers). (4) Doesn't rely on sticky sessions which have problems (server failure, uneven load). Options 1 and 2 (sticky sessions) are quick fixes but problematic: server crash loses all sessions, prevents true horizontal scaling, uneven load distribution. This is standard practice in distributed systems.",
  },
  {
    id: 'mc4',
    question:
      'What is the main advantage of Layer 7 load balancing over Layer 4?',
    options: [
      'Layer 7 is faster and has higher throughput',
      'Layer 7 can route based on HTTP content (URL path, headers, cookies)',
      'Layer 7 works with any protocol (TCP, UDP, HTTP)',
      'Layer 7 is simpler to configure',
    ],
    correctAnswer: 1,
    explanation:
      "Layer 7 can route based on HTTP content: URL paths (/api/users → User Service, /api/posts → Post Service), headers (X-Mobile: true → mobile backend), cookies (A/B testing). Layer 4 only sees IP addresses and ports, cannot inspect HTTP content. Trade-off: Layer 7 is SLOWER (must parse HTTP) but much more intelligent. Layer 4 is faster but can't do content-based routing. For microservices architecture, Layer 7's content-based routing is essential despite performance cost.",
  },
  {
    id: 'mc5',
    question:
      'Your load balancer health check is set to check every 5 seconds with unhealthy threshold of 3. A server crashes. How long until the load balancer stops sending traffic to it?',
    options: ['5 seconds', '10 seconds', '15 seconds', 'Immediately'],
    correctAnswer: 2,
    explanation:
      '15 seconds. Calculation: (1) Health check every 5 seconds. (2) Unhealthy threshold = 3 (need 3 consecutive failures). (3) Time = 5 seconds × 3 = 15 seconds. So 15 seconds pass before server marked unhealthy and removed from rotation. During these 15 seconds, some user requests will still go to the crashed server (failed requests). To reduce this window: Decrease health check interval (e.g., every 2 seconds → 6 second detection time). But: More frequent health checks = more network overhead. Trade-off between fast failure detection and overhead.',
  },
];
