/**
 * Multiple choice questions for RPC (Remote Procedure Call) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const rpcremoteprocedurecallMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'rpc-grpc-transport',
    question:
      'What transport protocol and serialization format does gRPC use by default?',
    options: [
      'HTTP/1.1 with JSON',
      'HTTP/2 with Protocol Buffers',
      'TCP with XML',
      'WebSocket with MessagePack',
    ],
    correctAnswer: 1,
    explanation:
      'gRPC uses HTTP/2 as the transport protocol (enabling multiplexing, streaming, header compression) and Protocol Buffers (protobuf) for serialization. This combination provides high performance through binary encoding and efficient network usage. HTTP/1.1 with JSON is used by REST APIs and is much slower.',
  },
  {
    id: 'rpc-streaming-pattern',
    question:
      'Which gRPC streaming pattern would be most appropriate for implementing a real-time chat application where multiple users send and receive messages simultaneously?',
    options: [
      'Unary RPC',
      'Server Streaming RPC',
      'Client Streaming RPC',
      'Bidirectional Streaming RPC',
    ],
    correctAnswer: 3,
    explanation:
      'Bidirectional Streaming RPC is ideal for chat applications because both client and server need to send streams of messages independently and simultaneously. The client streams outgoing messages while receiving incoming messages from the server. Unary RPC would require polling, server streaming only allows server→client, and client streaming only allows client→server.',
  },
  {
    id: 'rpc-error-handling',
    question:
      "You're implementing a gRPC client that calls a downstream service. The call fails with status code UNAVAILABLE. What is the most appropriate action?",
    options: [
      'Immediately return an error to the user',
      'Retry the request with exponential backoff',
      'Log the error and continue without retrying',
      'Switch to a different RPC method',
    ],
    correctAnswer: 1,
    explanation:
      'UNAVAILABLE is a transient error indicating the service is temporarily down or overloaded. The correct approach is to retry with exponential backoff (e.g., 100ms, 200ms, 400ms, 800ms) up to a maximum number of attempts. This gives the service time to recover. Immediately returning an error provides poor user experience, and not retrying at all misses the opportunity for the call to succeed.',
  },
  {
    id: 'rpc-vs-rest',
    question:
      'In which scenario would REST be a better choice than gRPC for API design?',
    options: [
      'Internal microservices communication requiring low latency',
      'Public API that needs to be easily accessible from browsers without additional tooling',
      'High-throughput streaming of binary data between services',
      'Polyglot environment where multiple languages need strongly-typed interfaces',
    ],
    correctAnswer: 1,
    explanation:
      'REST is better for public APIs consumed by browsers because it works natively with HTTP/1.1, requires no special tooling, supports standard HTTP caching, and can be tested with curl/Postman. gRPC requires gRPC-Web for browsers, which adds complexity. For internal microservices, streaming, and polyglot environments, gRPC is typically superior due to performance and strong typing.',
  },
  {
    id: 'rpc-connection-management',
    question:
      'What is the most important practice for managing gRPC client connections in a high-traffic Node.js application?',
    options: [
      'Create a new client instance for every request',
      'Reuse a single client instance across all requests',
      'Create a client pool and rotate through clients',
      'Close and recreate the client every 100 requests',
    ],
    correctAnswer: 1,
    explanation:
      "Reusing a single gRPC client instance across all requests is critical because: (1) HTTP/2 automatically multiplexes multiple concurrent RPCs over a single TCP connection, (2) Creating new connections has significant overhead (TCP handshake, TLS handshake), (3) gRPC clients maintain connection pools internally. Creating a new client per request would cause severe performance degradation. Unlike HTTP/1.1 where connection pooling is necessary, HTTP/2's multiplexing makes a single client instance optimal.",
  },
];
