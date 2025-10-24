/**
 * Quiz questions for WhatsApp Architecture section
 */

export const whatsapparchitectureQuiz = [
    {
        id: 'q1',
        question: 'Explain WhatsApp\'s end-to-end encryption protocol (Signal Protocol). How does it ensure message privacy while delivering messages reliably?',
        sampleAnswer: 'WhatsApp uses Signal Protocol for E2E encryption. Setup: (1) Each client generates long-term identity key pair (public/private). (2) Upload public key to WhatsApp server. (3) When A messages B first time, fetch B\'s public key from server. (4) Perform Double Ratchet handshake (Diffie-Hellman key exchange) to establish shared secret. (5) Derive message keys from shared secret using ratcheting algorithm. Messaging: (1) A encrypts message with current message key. (2) Send ciphertext to WhatsApp server. (3) Server routes to B (store if B offline). (4) B decrypts with corresponding key. Server never has decryption keys (only ciphertexts). Forward secrecy: Keys rotate with each message, compromising today\'s key doesn\'t reveal past messages. Reliability: Server stores encrypted messages for 30 days if recipient offline, delivers when online. Authentication: Compare safety numbers (key fingerprints) to prevent MITM.',
        keyPoints: [
            'Signal Protocol: Double Ratchet handshake, per-message keys',
            'Server routes encrypted messages, never has decryption keys',
            'Forward secrecy: Key rotation prevents historical decryption',
            'Reliability: Server stores encrypted messages up to 30 days for offline users',
        ],
    },
    {
        id: 'q2',
        question: 'How does WhatsApp achieve its legendary efficiency, handling 100 billion messages/day with only ~50 engineers and minimal servers?',
        sampleAnswer: 'WhatsApp efficiency principles: (1) Erlang/OTP - Built on Erlang VM optimized for concurrency, fault tolerance. Each user connection = lightweight process (millions per server). Actor model simplifies distributed systems. (2) Mnesia database - In-memory distributed database (comes with Erlang). Stores user sessions, routing tables. Fast (microsecond latency), no external dependencies. (3) Stateless servers - Servers don\'t store message history, just route. Messages stored encrypted on sender/recipient devices. Reduces storage cost. (4) Minimal features - No ads, no analytics, no social graph algorithms. Focus on core messaging. (5) FreeBSD + custom tuning - Optimized network stack, kernel tuning for millions of concurrent connections. Result: 1 server handles 2-3 million concurrent connections (10x industry average). 50 engineers because: simple architecture, no complexity of ads/analytics, Erlang reliability reduces operational burden.',
        keyPoints: [
            'Erlang/OTP: Lightweight processes, millions of connections per server',
            'Mnesia: In-memory distributed database for routing (microsecond latency)',
            'Stateless servers: Route only, don\'t store messages (stored on devices)',
            'Minimal features + FreeBSD tuning: 1 server = 2-3M connections',
        ],
    },
    {
        id: 'q3',
        question: 'Describe WhatsApp\'s approach to handling offline message delivery and message synchronization across multiple devices.',
        sampleAnswer: 'Offline delivery: (1) A sends message to B, B is offline. (2) WhatsApp server stores encrypted message in queue (message_id, recipient_id, ciphertext). (3) B comes online, establishes WebSocket connection. (4) Server pushes queued messages to B. (5) B sends ACK for each message. (6) Server deletes delivered messages from queue. Messages stored max 30 days, then deleted. Multi-device sync (added 2021): Challenge: E2E encryption + multiple devices. Solution: (1) Each device has own identity key. (2) Sender encrypts message separately for each recipient device (4 copies if recipient has phone + laptop + tablet + web). (3) Server routes to all online devices, queues for offline devices. (4) Each device decrypts independently. Message history sync: (1) When adding new device, primary device (phone) re-encrypts message history for new device. (2) Transfer via server (still E2E encrypted). (3) New device decrypts and stores locally. Trade-off: More encryption overhead (4x messages) vs multi-device support.',
        keyPoints: [
            'Offline: Server queues encrypted messages up to 30 days, delivers when online',
            'Multi-device: Encrypt message separately for each device (4x messages)',
            'History sync: Primary device re-encrypts history for new device',
            'Trade-off: Encryption overhead vs E2E security + multi-device',
        ],
    },
];

