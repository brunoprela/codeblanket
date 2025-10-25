/**
 * Multiple choice questions for WhatsApp Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const whatsapparchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Which end-to-end encryption protocol does WhatsApp use?',
    options: [
      'PGP (Pretty Good Privacy)',
      'TLS 1.3 with perfect forward secrecy',
      'Signal Protocol with Double Ratchet',
      'AES-256 with RSA key exchange',
    ],
    correctAnswer: 2,
    explanation:
      "WhatsApp uses the Signal Protocol with Double Ratchet algorithm for end-to-end encryption. Each message has its own encryption key derived through ratcheting, providing forward secrecy (compromising today's key doesn't reveal past messages). The server routes encrypted messages but never has decryption keys. Users can verify encryption by comparing safety numbers (key fingerprints) to prevent man-in-the-middle attacks.",
  },
  {
    id: 'mc2',
    question:
      'What programming language and framework is WhatsApp primarily built on?',
    options: [
      'Java with Spring Boot',
      'Erlang/OTP with Mnesia database',
      'Go with Redis',
      'Node.js with MongoDB',
    ],
    correctAnswer: 1,
    explanation:
      "WhatsApp is built on Erlang/OTP (Open Telecom Platform). Erlang\'s lightweight processes enable millions of concurrent connections per server (2-3M, 10x industry average). The Actor model simplifies distributed systems, and OTP provides fault tolerance patterns. Mnesia, Erlang's in-memory distributed database, stores user sessions and routing tables with microsecond latency. This architecture enables 50 engineers to handle 100 billion messages daily.",
  },
  {
    id: 'mc3',
    question:
      'How many concurrent connections can one WhatsApp server typically handle?',
    options: [
      'Approximately 100,000 connections',
      'Approximately 500,000 connections',
      'Approximately 2-3 million connections',
      'Approximately 10 million connections',
    ],
    correctAnswer: 2,
    explanation:
      "One WhatsApp server handles 2-3 million concurrent connections, which is 10x the industry average. This is achieved through Erlang\'s lightweight processes (each connection = one Erlang process with minimal overhead), custom FreeBSD kernel tuning for network stack optimization, and stateless server design (no message storage on servers, only routing). This efficiency is a key reason WhatsApp operates with minimal infrastructure and engineering team.",
  },
  {
    id: 'mc4',
    question:
      'How long does WhatsApp store encrypted messages for offline users?',
    options: ['7 days', '30 days', '90 days', 'Indefinitely until delivered'],
    correctAnswer: 1,
    explanation:
      "WhatsApp stores encrypted messages for offline users for up to 30 days. Messages are queued in the server (still encrypted), and when the user comes online, messages are pushed to their device. After 30 days, undelivered messages are deleted. This balances message delivery reliability with storage costs. Once delivered and acknowledged, messages are deleted from servers, as they're stored on user devices.",
  },
  {
    id: 'mc5',
    question:
      'How does WhatsApp handle multi-device support while maintaining end-to-end encryption?',
    options: [
      'All devices share the same encryption key',
      'Messages are encrypted separately for each device the recipient owns',
      'Primary device decrypts and forwards to other devices',
      'Server decrypts once and re-encrypts for each device',
    ],
    correctAnswer: 1,
    explanation:
      'For multi-device support (added 2021), WhatsApp encrypts messages separately for each recipient device. If a user has phone, laptop, tablet, and web client, the sender encrypts the message 4 times (once per device). Each device has its own identity key and decrypts independently. Message history is synced by the primary device (phone) re-encrypting history for new devices. This maintains E2E encryption but increases encryption overhead (4x messages).',
  },
];
