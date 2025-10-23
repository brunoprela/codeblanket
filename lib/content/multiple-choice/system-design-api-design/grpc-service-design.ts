/**
 * Multiple choice questions for gRPC Service Design section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const grpcservicedesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'grpc-q1',
    question:
      'What is the main advantage of gRPC over REST for microservices communication?',
    options: [
      'gRPC is easier to implement and debug',
      'gRPC uses binary Protocol Buffers over HTTP/2, providing better performance',
      'gRPC works in browsers without any additional setup',
      'gRPC supports more programming languages',
    ],
    correctAnswer: 1,
    explanation:
      "gRPC uses Protocol Buffers (binary, compact) over HTTP/2 (multiplexing, compression), making it significantly faster than JSON/REST. However, it's harder to debug (not human-readable), requires proxies for browsers, and both support many languages.",
  },
  {
    id: 'grpc-q2',
    question:
      'Which gRPC streaming type would you use for a real-time chat application?',
    options: [
      'Unary RPC (request-response)',
      'Server streaming (one request, multiple responses)',
      'Client streaming (multiple requests, one response)',
      'Bidirectional streaming (both send multiple messages)',
    ],
    correctAnswer: 3,
    explanation:
      'Real-time chat requires bidirectional streaming where both client and server continuously send and receive messages. Server streaming is one-way (server→client), client streaming is opposite (client→server), unary is single request-response.',
  },
  {
    id: 'grpc-q3',
    question:
      'What is the purpose of field numbers in Protocol Buffer definitions?',
    options: [
      'To define the order fields appear in JSON output',
      'To serve as unique, permanent identifiers for backward compatibility',
      'To indicate required vs optional fields',
      'To specify the default value for each field',
    ],
    correctAnswer: 1,
    explanation:
      "Field numbers (1, 2, 3...) are permanent identifiers used in binary encoding. Changing them breaks compatibility. They're not related to JSON order, optionality (proto3 all fields optional), or defaults. Never reuse or change field numbers.",
  },
  {
    id: 'grpc-q4',
    question: 'How does gRPC handle timeouts to prevent hanging requests?',
    options: [
      'Server automatically cancels requests after 30 seconds',
      'Client sets deadline in metadata; server checks if deadline exceeded',
      'HTTP/2 has built-in timeout mechanism',
      'Protocol Buffers include timeout field',
    ],
    correctAnswer: 1,
    explanation:
      'gRPC uses deadlines: client specifies timeout when making call, gRPC propagates it as deadline in metadata, server can check if deadline exceeded and abort. No default timeout (can hang forever), not automatic, not in HTTP/2 or protobuf.',
  },
  {
    id: 'grpc-q5',
    question: 'Why might you choose REST over gRPC for a public API?',
    options: [
      'REST is faster and more efficient',
      'REST has better streaming capabilities',
      'REST works natively in browsers and is human-readable for easier debugging',
      'REST supports more authentication methods',
    ],
    correctAnswer: 2,
    explanation:
      'REST works in browsers without proxies, JSON is human-readable for debugging, and HTTP caching is straightforward. gRPC is actually faster, has better streaming, and both support various auth methods. gRPC requires gRPC-Web proxy for browsers.',
  },
];
