/**
 * Quiz questions for Zoom Architecture section
 */

export const zoomarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Zoom's video routing architecture. How does it decide between peer-to-peer, SFU (Selective Forwarding Unit), and MCU (Multipoint Control Unit) modes?",
    sampleAnswer:
      'Zoom uses adaptive video routing based on meeting size and network conditions. (1) 1-on-1 calls: Peer-to-peer (P2P) when possible. Clients exchange ICE candidates, establish direct connection via STUN/TURN. Lowest latency, no server bandwidth cost. Falls back to relay (TURN server) if P2P fails (strict firewalls). (2) Small meetings (2-40 participants): SFU mode. Each participant sends one stream to Zoom SFU (Multimedia Router). SFU forwards streams to other participants without transcoding. Participants receive N-1 streams (3 people = receive 2 streams). Bandwidth: Upload 1 stream, download N-1 streams. (3) Large meetings (40+ participants): SFU with gallery view optimization. SFU sends active speaker + 3-4 gallery thumbnails (not all 100 participants). Webinar mode (1000+ participants): MCU-like, server composites active speaker video, sends single stream to viewers. Decision factors: Meeting size, client bandwidth, CPU capability (mobile vs desktop). Result: Optimized quality and latency for each scenario.',
    keyPoints: [
      '1-on-1: P2P for lowest latency, fallback to TURN relay',
      'Small groups (2-40): SFU forwards streams without transcoding',
      'Large meetings: SFU with gallery optimization (active speaker + thumbnails)',
      'Webinars (1000+): Server compositing (MCU-like) for scalability',
    ],
  },
  {
    id: 'q2',
    question:
      'How does Zoom achieve low-latency video with jitter buffering, FEC (Forward Error Correction), and adaptive bitrate?',
    sampleAnswer:
      'Zoom low-latency optimizations: (1) Jitter buffer - Network packets arrive with variable delay. Buffer holds 50-200ms of video/audio, smooths playout. Trade-off: Latency vs smoothness. Zoom uses 50ms for video (tight), 20ms for audio (tighter). (2) Forward Error Correction - Send redundant data (XOR parity packets). If packet lost, reconstruct from parity. Example: Send packets [1,2,3] + parity [P123]. If packet 2 lost, recover using [1,3,P123]. Adds 10-20% bandwidth overhead but avoids retransmission latency. (3) Adaptive bitrate - Monitor network conditions (packet loss, latency, jitter). Dynamically adjust video resolution and bitrate. Good network: 1080p at 2.5 Mbps. Poor network: Drop to 360p at 600 kbps. Audio always prioritized (minimum 100 kbps). (4) Codec selection - Use H.264 (hardware accelerated) for most clients, VP8 for web. Recent: AV1 for bandwidth savings. Result: 150-250ms end-to-end latency, resilient to 10-20% packet loss.',
    keyPoints: [
      'Jitter buffer: 50ms video, 20ms audio (tight for low latency)',
      'FEC: Send redundant data (10-20% overhead), avoid retransmission',
      'Adaptive bitrate: 1080p@2.5Mbps → 360p@600kbps based on network',
      'Audio prioritized always, hardware-accelerated H.264, 150-250ms latency',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Zoom's data center architecture and global infrastructure. How does it handle 300M+ daily meeting participants?",
    sampleAnswer:
      'Zoom infrastructure: (1) Data centers - 17 co-located data centers globally (US, Europe, Asia, Australia). Own infrastructure (not cloud) for cost and control. Each DC has Multimedia Routers (SFU servers). (2) PoP (Points of Presence) - 100s of edge locations for TURN relay servers. Route users to nearest PoP for lowest latency. (3) Cloud peering - Direct connections to AWS, Azure, GCP for customers using cloud. (4) Meeting routing - User starts meeting → Assigned to nearest DC based on geolocation. Participants join, routed to nearest PoP, connected to meeting DC. (5) Load balancing - If DC reaches capacity, route new meetings to next nearest DC. Web Socket Manager distributes connections across Multimedia Routers. (6) Redundancy - Each DC has redundant power, network, servers. If DC fails, meetings migrated to backup DC (60-second reconnection). Capacity: Each Multimedia Router handles 1000-2000 concurrent streams. 10,000s of routers globally. Scale: 300M daily participants, peaks at 3M concurrent meetings.',
    keyPoints: [
      '17 owned data centers + 100s of PoPs (edge relay servers)',
      'Route users to nearest DC/PoP for low latency, cloud peering',
      'Load balancing across DCs, Web Socket Manager across routers',
      'Redundancy: DC failover with 60s reconnection, 1000-2000 streams/router',
    ],
  },
];
