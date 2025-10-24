/**
 * Multiple choice questions for Netflix Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const netflixarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'Which database does Netflix primarily use for its streaming infrastructure?',
        options: [
            'MySQL with master-slave replication',
            'Cassandra for high availability and write scalability',
            'MongoDB for document-based user profiles',
            'PostgreSQL with read replicas',
        ],
        correctAnswer: 1,
        explanation: 'Netflix uses Cassandra as its primary database for streaming infrastructure. Cassandra provides high availability (multi-datacenter replication), write scalability (handles millions of writes per second), and tunable consistency. Netflix chose Cassandra because it can tolerate datacenter failures while continuing to serve traffic, aligning with their chaos engineering philosophy.',
    },
    {
        id: 'mc2',
        question: 'What is the purpose of Netflix\'s Chaos Monkey tool?',
        options: [
            'To test load balancer failover during peak traffic',
            'To randomly terminate EC2 instances to ensure service resilience',
            'To simulate network latency between microservices',
            'To stress test the recommendation algorithm',
        ],
        correctAnswer: 1,
        explanation: 'Chaos Monkey randomly terminates EC2 instances during business hours to force engineers to build resilient services. If a service crashes when Chaos Monkey kills an instance, it reveals a reliability gap. This proactive approach ensures Netflix\'s systems can handle instance failures gracefully with circuit breakers, retries, fallbacks, and auto-recovery mechanisms.',
    },
    {
        id: 'mc3',
        question: 'How does Netflix\'s Zuul API Gateway contribute to the architecture?',
        options: [
            'It serves video chunks directly from S3 to reduce latency',
            'It routes client requests to appropriate microservices and handles cross-cutting concerns',
            'It transcodes video files into multiple resolutions',
            'It manages user authentication tokens in memory',
        ],
        correctAnswer: 1,
        explanation: 'Zuul is Netflix\'s API Gateway that routes client requests to the appropriate microservices. It handles cross-cutting concerns like authentication, rate limiting, dynamic routing, monitoring, and resiliency patterns. Zuul acts as a single entry point for client requests and can route to 700+ backend microservices based on request paths and configurations.',
    },
    {
        id: 'mc4',
        question: 'What is Netflix\'s EVCache and why is it important?',
        options: [
            'A video encoding cache that stores transcoded segments',
            'A distributed in-memory cache based on Memcached for sub-millisecond latency',
            'A content delivery network for streaming videos globally',
            'An event sourcing system for user activity logs',
        ],
        correctAnswer: 1,
        explanation: 'EVCache is Netflix\'s distributed in-memory caching solution built on top of Memcached. It provides sub-millisecond read latency and is crucial for serving user profiles, recommendations, and frequently accessed data. EVCache is deployed across multiple AWS availability zones for high availability and handles millions of requests per second, significantly reducing load on backend databases.',
    },
    {
        id: 'mc5',
        question: 'How many microservices does Netflix operate as part of its architecture?',
        options: [
            'Approximately 50 microservices',
            'Approximately 150 microservices',
            'Approximately 700+ microservices',
            'Approximately 2000+ microservices',
        ],
        correctAnswer: 2,
        explanation: 'Netflix operates 700+ microservices as part of its architecture. This extensive microservices ecosystem enables independent scaling, deployment, and fault isolation. Each service handles a specific business capability (user management, recommendations, video playback, etc.) and can be deployed thousands of times daily without coordinating with other teams. This requires sophisticated tooling for service discovery (Eureka), circuit breaking (Hystrix), and distributed tracing.',
    },
];

