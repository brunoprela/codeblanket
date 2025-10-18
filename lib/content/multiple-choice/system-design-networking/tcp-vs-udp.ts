/**
 * Multiple choice questions for TCP vs UDP section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const tcpvsudpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tcp-handshake',
    question:
      'During a TCP three-way handshake, what is the minimum amount of time (in RTTs) before data can start being transmitted?',
    options: ['0 RTT (data can be sent with SYN)', '0.5 RTT', '1 RTT', '2 RTT'],
    correctAnswer: 2,
    explanation:
      'The TCP three-way handshake requires 1 full RTT: Client sends SYN, server responds with SYN-ACK (0.5 RTT), client sends ACK and can then send data (1 RTT total). TCP Fast Open can reduce this to 0 RTT for subsequent connections, but standard TCP requires 1 RTT.',
  },
  {
    id: 'udp-use-case',
    question:
      'Which of the following is the BEST reason to use UDP for live video streaming instead of TCP?',
    options: [
      'UDP provides better error correction for video frames',
      'UDP is more secure than TCP for video transmission',
      'Old video frames become useless after their timestamp, making retransmission pointless',
      'UDP automatically compresses video data',
    ],
    correctAnswer: 2,
    explanation:
      "For live streaming, if a video frame is lost, retransmitting it (as TCP would do) is useless because the timestamp has passed. It's better to skip the lost frame and display the next one. UDP doesn't provide error correction, isn't inherently more secure, and doesn't compress data.",
  },
  {
    id: 'tcp-reliability',
    question:
      'How does TCP detect that a packet has been lost and needs retransmission?',
    options: [
      'The receiver sends an error message',
      'TCP uses checksums to detect corrupted packets',
      'Retransmission timer expires OR three duplicate ACKs received',
      'The router notifies the sender via ICMP',
    ],
    correctAnswer: 2,
    explanation:
      "TCP detects packet loss in two ways: 1) Retransmission timer expires without receiving an ACK, or 2) Receiving three duplicate ACKs (fast retransmit). Checksums detect corruption, not loss. Routers don't directly notify TCP about packet loss.",
  },
  {
    id: 'quic-benefit',
    question: 'What is the primary advantage of QUIC over traditional TCP+TLS?',
    options: [
      'QUIC uses less bandwidth than TCP',
      'QUIC can establish a connection and send encrypted data in 0-1 RTT vs 2-3 RTT for TCP+TLS',
      'QUIC guarantees zero packet loss',
      'QUIC works without encryption',
    ],
    correctAnswer: 1,
    explanation:
      "QUIC's main advantage is connection establishment speed. TCP+TLS requires 2-3 RTT (TCP handshake + TLS handshake), while QUIC combines them into 0-1 RTT. QUIC doesn't use less bandwidth, can't prevent packet loss, and requires encryption by design.",
  },
  {
    id: 'tcp-flow-control',
    question:
      'What mechanism does TCP use to prevent a fast sender from overwhelming a slow receiver?',
    options: [
      'Congestion control',
      'Flow control with sliding window',
      'Automatic packet dropping',
      'Round-robin scheduling',
    ],
    correctAnswer: 1,
    explanation:
      "TCP uses flow control with a sliding window. The receiver advertises its buffer size in each ACK, and the sender won't send more data than the window allows. Congestion control is different - it prevents overwhelming the network (not the receiver).",
  },
];
