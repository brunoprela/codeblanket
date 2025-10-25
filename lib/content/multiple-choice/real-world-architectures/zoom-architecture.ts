/**
 * Multiple choice questions for Zoom Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const zoomarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What video routing architecture does Zoom use for small meetings (2-40 participants)?',
    options: [
      'Peer-to-peer (P2P) direct connections',
      'SFU (Selective Forwarding Unit) without transcoding',
      'MCU (Multipoint Control Unit) with server compositing',
      'Hybrid P2P with fallback relay',
    ],
    correctAnswer: 1,
    explanation:
      "Zoom uses SFU (Selective Forwarding Unit) mode for small meetings (2-40 participants). Each participant sends one video stream to Zoom\'s Multimedia Router (SFU), which forwards streams to other participants without transcoding. Participants upload 1 stream and download N-1 streams. This balances latency, quality, and bandwidth better than P2P (which becomes impractical beyond 4-5 participants) or MCU (which requires server CPU for compositing).",
  },
  {
    id: 'mc2',
    question: "What is Zoom\'s typical end-to-end latency for video calls?",
    options: [
      '50-100 milliseconds',
      '150-250 milliseconds',
      '400-500 milliseconds',
      '800-1000 milliseconds',
    ],
    correctAnswer: 1,
    explanation:
      'Zoom achieves 150-250 milliseconds end-to-end latency. This is accomplished through tight jitter buffering (50ms for video, 20ms for audio), Forward Error Correction to avoid retransmissions, and routing users to nearest data centers. This low latency is critical for natural conversation flow. Zoom prioritizes latency over maximum quality, choosing to reduce resolution/bitrate when network conditions degrade rather than increasing buffer size.',
  },
  {
    id: 'mc3',
    question: 'How many co-located data centers does Zoom own globally?',
    options: [
      '5 major data centers',
      '17 co-located data centers',
      '50 cloud regions',
      '100+ edge locations',
    ],
    correctAnswer: 1,
    explanation:
      'Zoom owns infrastructure in 17 co-located data centers globally (US, Europe, Asia, Australia) rather than using public cloud. This gives Zoom cost advantages and control over the video routing infrastructure. Each datacenter has Multimedia Routers (SFU servers), with each router handling 1000-2000 concurrent streams. Additionally, Zoom operates 100s of PoPs (Points of Presence) for TURN relay servers at edge locations for lowest latency.',
  },
  {
    id: 'mc4',
    question:
      "What bandwidth overhead does Zoom\'s Forward Error Correction (FEC) add?",
    options: [
      '1-5% overhead',
      '10-20% overhead',
      '30-40% overhead',
      '50% overhead',
    ],
    correctAnswer: 1,
    explanation:
      "Zoom\'s Forward Error Correction adds 10-20% bandwidth overhead by sending redundant parity packets. This allows reconstructing lost packets without retransmission, which would add latency. For example, sending packets [1,2,3] plus parity [P123], if packet 2 is lost, it can be recovered using [1,3,P123]. This trade-off of bandwidth for latency is crucial for real-time communication, enabling resilience to 10-20% packet loss.",
  },
  {
    id: 'mc5',
    question: 'How does Zoom adapt video quality to network conditions?',
    options: [
      'Fixed quality based on subscription tier',
      'Dynamic adjustment from 1080p@2.5Mbps down to 360p@600kbps',
      'User manually selects quality before meeting',
      'Always maintains highest quality, buffers if needed',
    ],
    correctAnswer: 1,
    explanation:
      'Zoom dynamically adapts video quality based on real-time network monitoring (packet loss, latency, jitter). On good networks, Zoom uses 1080p at 2.5 Mbps. On poor networks, it drops to 360p at 600 kbps or lower. Audio is always prioritized (minimum 100 kbps). This adaptive bitrate ensures meetings continue smoothly even on degraded networks. Video codec choice (H.264 for hardware acceleration) and resolution/bitrate adjustments happen automatically without user intervention.',
  },
];
