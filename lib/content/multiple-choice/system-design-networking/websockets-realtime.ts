/**
 * Multiple choice questions for WebSockets & Real-Time Communication section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const websocketsrealtimeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'websocket-upgrade',
    question:
      'What HTTP status code does a server return during a successful WebSocket upgrade handshake?',
    options: [
      '200 OK',
      '101 Switching Protocols',
      '201 Created',
      '301 Moved Permanently',
    ],
    correctAnswer: 1,
    explanation:
      '101 Switching Protocols is the status code that indicates the server is switching from HTTP to WebSocket protocol. This happens during the upgrade handshake after the client sends an Upgrade: websocket header.',
  },
  {
    id: 'websocket-vs-http',
    question:
      'Which of the following is the PRIMARY advantage of WebSocket over regular HTTP for a chat application?',
    options: [
      'WebSocket is more secure than HTTPS',
      'WebSocket uses less bandwidth by eliminating HTTP headers on every message',
      'WebSocket connections can survive server restarts',
      'WebSocket supports multiple simultaneous requests',
    ],
    correctAnswer: 1,
    explanation:
      "The primary advantage is reduced overhead. After the initial handshake, WebSocket frames have only 2-14 bytes of overhead compared to hundreds of bytes of HTTP headers. WebSocket is not inherently more secure (both can be encrypted), doesn't survive server restarts better, and HTTP/2 also supports multiplexing.",
  },
  {
    id: 'websocket-scaling',
    question:
      'Why is scaling WebSocket servers more challenging than scaling stateless HTTP servers?',
    options: [
      'WebSocket uses more CPU than HTTP',
      'WebSocket connections are stateful and long-lived, making load balancing difficult',
      'WebSocket can only handle 100 concurrent connections per server',
      'WebSocket requires dedicated hardware',
    ],
    correctAnswer: 1,
    explanation:
      'WebSocket connections are stateful (server maintains connection state) and long-lived (can last hours/days). This makes traditional round-robin load balancing ineffective - you need sticky sessions or a message broker to coordinate between servers. HTTP is stateless, so any server can handle any request.',
  },
  {
    id: 'websocket-heartbeat',
    question:
      'What is the purpose of implementing a heartbeat/ping-pong mechanism in WebSocket connections?',
    options: [
      'To encrypt the messages between client and server',
      "To detect broken connections that TCP doesn't immediately report",
      'To compress data before sending it',
      'To authenticate the user every few seconds',
    ],
    correctAnswer: 1,
    explanation:
      "Heartbeat (ping-pong) detects broken connections that TCP doesn't immediately report, such as when a router disconnects or a mobile device loses signal. Without heartbeat, the server might hold resources for dead connections. It doesn't handle encryption, compression, or authentication.",
  },
  {
    id: 'websocket-vs-sse',
    question:
      'When would Server-Sent Events (SSE) be a better choice than WebSocket?',
    options: [
      'When you need bidirectional communication',
      'When you need to send binary data',
      'When you only need server-to-client updates and want simpler implementation',
      'When you need to scale to millions of connections',
    ],
    correctAnswer: 2,
    explanation:
      'SSE is simpler than WebSocket and has built-in auto-reconnect, making it perfect for one-way server-to-client updates like live scores or notifications. For bidirectional communication, binary data, or complex scaling, WebSocket is better. Both can scale to millions of connections.',
  },
];
