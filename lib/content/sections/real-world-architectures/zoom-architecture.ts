/**
 * Zoom Architecture Section
 */

export const zoomarchitectureSection = {
    id: 'zoom-architecture',
    title: 'Zoom Architecture',
    content: `Zoom is a video conferencing platform that exploded in popularity, especially during the COVID-19 pandemic. With 300+ million daily meeting participants at peak, Zoom's architecture must handle massive scale video/audio streaming with minimal latency while ensuring quality and reliability.

## Overview

**Scale**: 300M+ daily meeting participants (peak), 3.3 trillion meeting minutes annually, 99.9%+ uptime

**Key Challenges**: Low-latency video/audio, bandwidth optimization, scalability, global distribution, quality under poor networks

## Core Components

### 1. Video Architecture

**Multimedia Router (MMR)**:
- Core of Zoom's architecture
- Routes video/audio between participants
- Performs: Transcoding, mixing, layout composition

**How it works**:
\`\`\`
Meeting with 10 participants:

Option 1: Mesh (P2P) - Each participant sends to all others
- Requires 9 uploads per participant
- Doesn't scale beyond ~4 participants
- High bandwidth consumption

Option 2: SFU (Selective Forwarding Unit) - Used by Zoom
- Each participant sends once to MMR
- MMR forwards to all other participants
- Participants download 9 streams (manageable)
- Scales to 100s of participants

Option 3: MCU (Multipoint Control Unit) - Zoom also uses
- MMR mixes all videos into composite
- Each participant downloads 1 stream
- Lower bandwidth for participants
- Higher CPU for MMR
\`\`\`

**Zoom's Hybrid Approach**:
- Small meetings (<10): SFU (forward individual streams)
- Large meetings (>10): MCU (composite video for some, individual for host)
- Gallery view: Composite of 25-49 participants (MCU)
- Speaker view: Individual streams (SFU)

---

### 2. Audio Processing

**Audio Codecs**:
- **Opus**: Primary codec (excellent quality, low latency)
- Adaptive bitrate: 8-64 kbps based on network
- Packet loss concealment (PLC)
- Echo cancellation, noise suppression

**Audio Mixing**:
- MMR mixes multiple audio streams
- Active speaker detection (VAD - Voice Activity Detection)
- Prioritize active speakers in mix

---

### 3. Video Encoding and Adaptive Bitrate

**Video Codecs**:
- **H.264**: Primary codec (widely supported)
- Hardware acceleration (GPU encoding/decoding)
- Multiple quality layers (simulcast):
  - 1080p @ 2.5 Mbps
  - 720p @ 1.5 Mbps
  - 360p @ 600 kbps
  - 180p @ 150 kbps

**Adaptive Bitrate**:
\`\`\`
Client measures:
- Available bandwidth
- Packet loss
- Latency
- CPU usage

Adjusts:
- Video resolution (1080p → 720p → 360p)
- Frame rate (30fps → 15fps → 5fps)
- Bitrate per layer
\`\`\`

**Simulcast**:
- Sender uploads multiple quality layers simultaneously
- MMR selects appropriate layer per recipient
- Smooth quality transitions

---

### 4. Network Optimization

**Protocol**: UDP (not TCP)
- Lower latency (no retransmissions)
- Custom protocol for reliability (ARQ - Automatic Repeat Request)
- Forward error correction (FEC)

**NAT Traversal**:
- STUN servers (discover public IP)
- TURN servers (relay when P2P fails)
- ICE (Interactive Connectivity Establishment)

**Packet Loss Handling**:
- FEC: Send redundant packets
- Packet loss concealment
- Jitter buffer (smooth playback)
- Graceful degradation (reduce quality vs freeze)

---

### 5. Global Infrastructure

**MMR Deployment**:
- 17 co-located datacenters worldwide
- Route participants to nearest MMR (lowest latency)
- Failover: If datacenter unavailable, route to next nearest

**Latency Optimization**:
- P50 latency: <50ms (video start time)
- P99 latency: <200ms
- Latency budget: Network (30ms) + Processing (20ms)

**Load Balancing**:
- Participants routed based on: Geography, CPU load, network conditions
- Dynamic rebalancing during meeting (if MMR overloaded)

---

### 6. Zoom Meeting Lifecycle

**1. Scheduling**:
\`\`\`
- User schedules meeting via web/app
- Generate meeting ID (11 digits)
- Store in database (MySQL)
- Send calendar invite with meeting URL
\`\`\`

**2. Joining**:
\`\`\`
- User clicks link → Zoom client/web
- Authenticate (if required)
- Query backend: Which MMR to use?
- Backend assigns MMR (nearest, least loaded)
- Establish WebRTC connection to MMR
- Start sending/receiving video/audio
\`\`\`

**3. In-Meeting**:
\`\`\`
- Participants send media to MMR
- MMR forwards/mixes media
- Real-time adjustments (bitrate, quality)
- Recording (if enabled) → Store to S3
- Transcription (if enabled) → ASR (Automatic Speech Recognition)
\`\`\`

**4. Ending**:
\`\`\`
- Host ends meeting
- Disconnect all participants
- Process recording (convert to MP4)
- Upload to cloud storage
- Generate meeting analytics
\`\`\`

---

### 7. Zoom Features Implementation

**Screen Sharing**:
- Capture screen at 30fps
- H.264 encode (optimized for screen content)
- Higher resolution than camera (text readability)
- Prioritize screen share over participant videos

**Virtual Backgrounds**:
- AI segmentation (remove background)
- Replace with image/video
- GPU-accelerated (TensorFlow Lite)
- Runs on client (reduces CPU load)

**Chat**:
- WebSocket for real-time messages
- Messages stored in database
- History available post-meeting

**Reactions**:
- Unicode emojis sent as metadata
- Displayed overlay on participant video
- Cleared after 5 seconds

**Breakout Rooms**:
- Split meeting into sub-meetings
- Separate MMR instance per room
- Host can move participants between rooms
- Rejoin main room when closed

---

### 8. Security

**End-to-End Encryption (E2EE)**:
- AES-256-GCM encryption
- Key generated per meeting
- Keys never sent to Zoom servers (true E2EE)
- Trade-off: No cloud recording, transcription with E2EE

**Meeting Security**:
- Password protection
- Waiting room (host admits participants)
- Lock meeting (no new participants)
- Mute/remove participants (host controls)

**Zoom Bombing Prevention**:
- Screen sharing: Host only (default)
- Remove participant feature
- Report to Trust & Safety team

---

### 9. Scalability

**Horizontal Scaling**:
- Add more MMR servers (datacenters or cloud)
- Each MMR handles 100-500 concurrent meetings
- Stateless (meetings can migrate between MMRs)

**Cloud Scaling**:
- AWS, Oracle Cloud for overflow capacity
- Auto-scaling during peak hours
- Cost optimization (spot instances for non-critical)

**Meeting Capacity**:
- Free: 40 minutes, 100 participants
- Paid: Unlimited time, up to 1000 participants
- Webinar: Up to 50K attendees (view-only for most)

---

### 10. Zoom Phone and Zoom Rooms

**Zoom Phone** (VoIP service):
- SIP protocol (Session Initiation Protocol)
- PSTN gateway (call regular phones)
- Call routing, IVR (Interactive Voice Response)

**Zoom Rooms** (Hardware for conference rooms):
- Dedicated Zoom client on hardware
- Calendar integration (book rooms)
- Touch screen controls
- Multiple cameras/mics (large rooms)

---

## Technology Stack

**Video/Audio**: Custom C++ MMR (high performance), WebRTC
**Codecs**: H.264, Opus, VP8 (fallback)
**Backend**: Java, Node.js
**Data**: MySQL, MongoDB, Redis
**Infrastructure**: Own datacenters + AWS, Oracle Cloud
**AI**: TensorFlow Lite (virtual backgrounds, noise suppression)
**Monitoring**: Custom tools, Datadog

---

## Key Lessons

1. **UDP over TCP** for real-time media (lower latency, custom reliability)
2. **MMR** (router) architecture scales better than mesh P2P
3. **Simulcast** enables adaptive quality per recipient
4. **Global infrastructure** (17 datacenters) reduces latency
5. **Graceful degradation**: Quality reduces under poor network vs freezing
6. **Hardware acceleration**: GPU for encoding/decoding, AI features

---

## Interview Tips

**Q: How does Zoom handle video conferencing at scale?**

A: Use Multimedia Router (MMR) architecture. Participants send video/audio to MMR (SFU - Selective Forwarding Unit), MMR forwards to other participants. For large meetings, MMR mixes videos into composite (MCU - Multipoint Control Unit). Adaptive bitrate: Sender uploads multiple quality layers (simulcast: 1080p, 720p, 360p). MMR selects appropriate layer per recipient based on bandwidth. Codecs: H.264 (video), Opus (audio) with hardware acceleration. Protocol: UDP (low latency) with custom reliability (FEC, ARQ). Global deployment: 17 datacenters, route to nearest MMR. Handle 100-500 meetings per MMR server. Scale horizontally: Add MMR servers, auto-scale cloud resources during peak.

**Q: How does Zoom optimize for poor network conditions?**

A: Adaptive bitrate: Client measures bandwidth, packet loss, latency. Dynamically adjusts: (1) Resolution (1080p → 720p → 360p → 180p). (2) Frame rate (30fps → 15fps → 5fps). (3) Bitrate per layer. Simulcast: Sender uploads multiple layers, MMR selects best for each recipient. Protocol: UDP for low latency, FEC (forward error correction) for packet loss. Packet loss concealment: Audio (interpolate), video (repeat frames). Jitter buffer: Smooth playback despite variable latency. Graceful degradation: Reduce quality vs freeze. Prioritization: Audio > screen share > active speaker video > participant videos.

---

## Summary

Zoom's architecture handles massive-scale video conferencing: MMR (Multimedia Router) for routing/mixing, simulcast for adaptive quality, UDP with custom reliability, global datacenter deployment (17 locations), H.264/Opus codecs with hardware acceleration. Success from low latency, graceful degradation under poor networks, horizontal scalability, and simple user experience.
`,
};

