/**
 * Networking & Communication Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { httphttpsfundamentalsSection } from '../sections/system-design-networking/http-https-fundamentals';
import { tcpvsudpSection } from '../sections/system-design-networking/tcp-vs-udp';
import { websocketsrealtimeSection } from '../sections/system-design-networking/websockets-realtime';
import { dnssystemSection } from '../sections/system-design-networking/dns-system';
import { rpcremoteprocedurecallSection } from '../sections/system-design-networking/rpc-remote-procedure-call';
import { graphqlSection } from '../sections/system-design-networking/graphql';
import { servicediscoverySection } from '../sections/system-design-networking/service-discovery';
import { networkprotocolsSection } from '../sections/system-design-networking/network-protocols';
import { ratelimitingSection } from '../sections/system-design-networking/rate-limiting';
import { apiversioningSection } from '../sections/system-design-networking/api-versioning';

// Import quizzes
import { httphttpsfundamentalsQuiz } from '../quizzes/system-design-networking/http-https-fundamentals';
import { tcpvsudpQuiz } from '../quizzes/system-design-networking/tcp-vs-udp';
import { websocketsrealtimeQuiz } from '../quizzes/system-design-networking/websockets-realtime';
import { dnssystemQuiz } from '../quizzes/system-design-networking/dns-system';
import { rpcremoteprocedurecallQuiz } from '../quizzes/system-design-networking/rpc-remote-procedure-call';
import { graphqlQuiz } from '../quizzes/system-design-networking/graphql';
import { servicediscoveryQuiz } from '../quizzes/system-design-networking/service-discovery';
import { networkprotocolsQuiz } from '../quizzes/system-design-networking/network-protocols';
import { ratelimitingQuiz } from '../quizzes/system-design-networking/rate-limiting';
import { apiversioningQuiz } from '../quizzes/system-design-networking/api-versioning';

// Import multiple choice
import { httphttpsfundamentalsMultipleChoice } from '../multiple-choice/system-design-networking/http-https-fundamentals';
import { tcpvsudpMultipleChoice } from '../multiple-choice/system-design-networking/tcp-vs-udp';
import { websocketsrealtimeMultipleChoice } from '../multiple-choice/system-design-networking/websockets-realtime';
import { dnssystemMultipleChoice } from '../multiple-choice/system-design-networking/dns-system';
import { rpcremoteprocedurecallMultipleChoice } from '../multiple-choice/system-design-networking/rpc-remote-procedure-call';
import { graphqlMultipleChoice } from '../multiple-choice/system-design-networking/graphql';
import { servicediscoveryMultipleChoice } from '../multiple-choice/system-design-networking/service-discovery';
import { networkprotocolsMultipleChoice } from '../multiple-choice/system-design-networking/network-protocols';
import { ratelimitingMultipleChoice } from '../multiple-choice/system-design-networking/rate-limiting';
import { apiversioningMultipleChoice } from '../multiple-choice/system-design-networking/api-versioning';

export const systemDesignNetworkingModule: Module = {
  id: 'system-design-networking',
  title: 'Networking & Communication',
  description:
    'Master networking protocols, communication patterns, and distributed system communication',
  category: 'System Design',
  difficulty: 'Medium',
  estimatedTime: '3-4 hours',
  prerequisites: [],
  icon: '🌐',
  keyTakeaways: [
    'HTTP is stateless, request-response protocol; HTTPS adds encryption and authentication via TLS',
    'HTTP/2 multiplexes requests and compresses headers; HTTP/3 uses QUIC (UDP) for better performance',
    'TCP provides reliability, ordering, flow control; UDP provides speed with no guarantees',
    'WebSocket enables full-duplex real-time communication; use message broker (Redis pub/sub) for scaling',
    'DNS translates domains to IPs via hierarchical system (root → TLD → authoritative); caching with TTL',
    'RPC allows calling remote functions as if they were local; gRPC uses HTTP/2 + Protocol Buffers',
    'gRPC has 4 communication patterns: Unary, Server Streaming, Client Streaming, Bidirectional Streaming',
    'GraphQL allows clients to request exactly the data they need; solve N+1 problem with DataLoader',
    'Service Discovery enables dynamic service location; use client-side (Netflix Eureka) or server-side (Consul) discovery',
    'Service Mesh (Istio, Linkerd) provides observability, traffic management, and security for microservices',
    'MQTT ideal for IoT pub/sub with QoS levels; AMQP provides enterprise message queuing with RabbitMQ',
    'WebRTC enables P2P audio/video; use STUN/TURN for NAT traversal',
    'Token Bucket algorithm recommended for rate limiting (allows bursts, memory efficient)',
    'Rate limit BEFORE authentication to prevent DDoS; use Redis + Lua scripts for distributed rate limiting',
    'URL path versioning (/api/v2) most common for public REST APIs',
    'Deprecation strategy: Announce → Monitor → Whitelist → Shutdown (4-6 months minimum)',
    'GraphQL avoids versioning by deprecating fields instead of creating new versions',
    'Always set timeouts/deadlines on RPC calls; retry transient errors with exponential backoff',
  ],
  learningObjectives: [
    'Understand HTTP/HTTPS fundamentals and how TLS/SSL provides security',
    'Master HTTP methods, status codes, headers, and caching strategies',
    'Compare HTTP/1.1, HTTP/2, and HTTP/3 protocols and their performance characteristics',
    'Understand TCP vs UDP trade-offs and when to use each protocol',
    'Learn TCP reliability mechanisms: three-way handshake, flow control, congestion control',
    'Master WebSocket architecture for real-time bidirectional communication and scaling patterns',
    'Design DNS infrastructure for global distribution, failover, and DDoS protection',
    'Implement RPC systems using gRPC with streaming patterns and error handling',
    'Design GraphQL schemas and solve common performance issues (N+1, caching)',
    'Implement service discovery patterns for dynamic microservices architecture',
    'Choose appropriate network protocols (MQTT, AMQP, WebRTC) for different use cases',
    'Design distributed rate limiting systems using Redis and token bucket algorithm',
    'Implement API versioning strategies and manage deprecation lifecycle',
    'Build production-ready networking systems with monitoring, security, and fault tolerance',
  ],
  sections: [
    {
      ...httphttpsfundamentalsSection,
      quiz: httphttpsfundamentalsQuiz,
      multipleChoice: httphttpsfundamentalsMultipleChoice,
    },
    {
      ...tcpvsudpSection,
      quiz: tcpvsudpQuiz,
      multipleChoice: tcpvsudpMultipleChoice,
    },
    {
      ...websocketsrealtimeSection,
      quiz: websocketsrealtimeQuiz,
      multipleChoice: websocketsrealtimeMultipleChoice,
    },
    {
      ...dnssystemSection,
      quiz: dnssystemQuiz,
      multipleChoice: dnssystemMultipleChoice,
    },
    {
      ...rpcremoteprocedurecallSection,
      quiz: rpcremoteprocedurecallQuiz,
      multipleChoice: rpcremoteprocedurecallMultipleChoice,
    },
    {
      ...graphqlSection,
      quiz: graphqlQuiz,
      multipleChoice: graphqlMultipleChoice,
    },
    {
      ...servicediscoverySection,
      quiz: servicediscoveryQuiz,
      multipleChoice: servicediscoveryMultipleChoice,
    },
    {
      ...networkprotocolsSection,
      quiz: networkprotocolsQuiz,
      multipleChoice: networkprotocolsMultipleChoice,
    },
    {
      ...ratelimitingSection,
      quiz: ratelimitingQuiz,
      multipleChoice: ratelimitingMultipleChoice,
    },
    {
      ...apiversioningSection,
      quiz: apiversioningQuiz,
      multipleChoice: apiversioningMultipleChoice,
    },
  ],
};
