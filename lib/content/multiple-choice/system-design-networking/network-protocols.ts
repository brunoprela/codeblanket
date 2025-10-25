/**
 * Multiple choice questions for Network Protocols section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const networkprotocolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'network-protocol-mqtt-qos',
    question: 'In MQTT, what is the difference between QoS 1 and QoS 2?',
    options: [
      'QoS 1 is faster but less reliable than QoS 2',
      'QoS 1 guarantees at-least-once delivery (may have duplicates), QoS 2 guarantees exactly-once delivery',
      'QoS 1 uses TCP, QoS 2 uses UDP',
      'QoS 1 is for small messages, QoS 2 is for large messages',
    ],
    correctAnswer: 1,
    explanation:
      'QoS 1 (At least once) acknowledges message receipt but may deliver duplicates if acknowledgment is lost. QoS 2 (Exactly once) uses a four-way handshake to guarantee exactly-once delivery with no duplicates, at the cost of higher latency. Both use TCP. QoS 0 is fire-and-forget with no acknowledgment.',
  },
  {
    id: 'network-protocol-webrtc-nat',
    question: 'What is the purpose of STUN and TURN servers in WebRTC?',
    options: [
      'STUN encrypts video streams, TURN compresses audio',
      'STUN discovers the public IP address for NAT traversal, TURN relays traffic when direct P2P connection fails',
      'STUN stores video recordings, TURN transcodes video formats',
      'STUN handles signaling, TURN handles media transport',
    ],
    correctAnswer: 1,
    explanation:
      'STUN (Session Traversal Utilities for NAT) helps clients discover their public IP address and port mappings to establish direct peer-to-peer connections through NATs. TURN (Traversal Using Relays around NAT) acts as a relay server when direct P2P fails due to strict NATs or firewalls, routing traffic through the server. Neither handles encryption (DTLS-SRTP does) or signaling (WebSocket/HTTP do).',
  },
  {
    id: 'network-protocol-amqp-exchange',
    question:
      'In AMQP (RabbitMQ), what is the difference between a fanout exchange and a topic exchange?',
    options: [
      'Fanout is faster because it uses UDP instead of TCP',
      'Fanout broadcasts messages to all bound queues ignoring routing keys; topic uses pattern matching on routing keys',
      'Fanout stores messages persistently, topic does not',
      'Fanout works with MQTT, topic works with HTTP',
    ],
    correctAnswer: 1,
    explanation:
      'A fanout exchange broadcasts every message to all queues bound to it, ignoring routing keys entirely (useful for pub/sub). A topic exchange routes messages based on routing key pattern matching using wildcards (* matches one word, # matches multiple). For example, routing key "user.created" matches patterns "user.*" and "*.created". Both use TCP and can have persistent messages.',
  },
  {
    id: 'network-protocol-ssh-tunnel',
    question:
      'You need to access a MySQL database running on a remote server that only allows connections from localhost. Which SSH tunneling technique should you use?',
    options: [
      'Dynamic port forwarding with -D flag',
      'Remote port forwarding with -R flag',
      'Local port forwarding with -L flag',
      'Reverse port forwarding with -X flag',
    ],
    correctAnswer: 2,
    explanation:
      "Local port forwarding (ssh -L 3307:localhost:3306 user@remote) forwards a local port (3307) to a remote destination (localhost:3306 from the remote server's perspective). This allows you to connect to localhost:3307 on your machine, which tunnels through SSH to the remote server's localhost:3306. Dynamic forwarding creates a SOCKS proxy, remote forwarding exposes local services remotely, and -X is for X11 forwarding.",
  },
  {
    id: 'network-protocol-comparison',
    question:
      'Which protocol would be most appropriate for a battery-powered IoT sensor that sends temperature readings every minute to a cloud service?',
    options: [
      'HTTP/HTTPS with polling',
      'WebSocket for continuous connection',
      'MQTT with QoS 1',
      'WebRTC for real-time updates',
    ],
    correctAnswer: 2,
    explanation:
      "MQTT is ideal for IoT devices because: (1) It\'s extremely lightweight (2-byte header vs HTTP's typical 100+ bytes), (2) Maintains a persistent connection with low overhead, (3) QoS 1 provides reliable delivery with acknowledgments, (4) Designed for unreliable networks. HTTP polling wastes battery with frequent connection overhead. WebSocket is heavier than MQTT. WebRTC is for P2P video/audio, not sensor data.",
  },
];
