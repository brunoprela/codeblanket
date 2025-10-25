/**
 * Zoom Architecture Section
 */

export const zoomarchitectureSection = {
  id: 'zoom-architecture',
  title: 'Zoom Architecture',
  content: `Zoom is a video conferencing platform that became essential during the COVID-19 pandemic. With 300+ million daily meeting participants at peak, Zoom's architecture must handle massive-scale video/audio streaming with minimal latency while ensuring quality and reliability even under poor network conditions. This section explores the technical systems behind Zoom.

## Overview

Zoom's scale and challenges:
- **300+ million daily meeting participants** (peak during pandemic)
- **3.3 trillion annual meeting minutes**
- **2.6 trillion connections** annually  
- **99.9%+ uptime**
- **Average latency**: <150ms globally
- **17 co-located datacenters** worldwide + cloud infrastructure

### Key Architectural Challenges

1. **Low-latency video/audio**: Real-time streaming with <150ms end-to-end latency
2. **Quality under poor networks**: Maintain experience with packet loss, jitter, bandwidth constraints
3. **Scalability**: Handle 100-1000+ participants per meeting
4. **Global distribution**: Low latency worldwide (17 datacenters)
5. **Device heterogeneity**: Support web, mobile, desktop, room systems on varied networks
6. **Security**: End-to-end encryption while maintaining quality

---

## Core Architecture: Multimedia Router (MMR)

Zoom's architecture centers on the **Multimedia Router (MMR)** - custom-built servers that route and process video/audio streams.

**Why Not Peer-to-Peer?**

\`\`\`
Mesh (P2P) architecture:
- Each participant sends video to all others
- 10-participant meeting: Each uploads 9 streams, downloads 9 streams
- Doesn't scale beyond 4-5 participants
- High bandwidth consumption (9x upload = infeasible)

MCU (Multipoint Control Unit):
- Server mixes all videos into one composite stream
- Each participant uploads 1 stream, downloads 1 stream
- Low bandwidth for participants
- High CPU for server (encoding/decoding)
- Latency introduced by mixing

SFU (Selective Forwarding Unit) - Zoom's approach:
- Server forwards video streams without transcoding
- Each participant uploads 1 stream, downloads N streams
- Low CPU for server (no encoding/decoding)
- Scalable to 100+ participants
- Used by Zoom for most scenarios
\`\`\`

**Zoom's Hybrid Approach**:

Zoom uses SFU + MCU based on meeting size and view:

\`\`\`
Small meeting (<10 participants):
- SFU mode: Forward individual streams
- Each participant receives all other streams
- Gallery view: Client displays all videos locally

Large meeting (>10 participants):
- Gallery view: MCU mode
  * MMR mixes 25-49 participants into composite
  * Reduces bandwidth for participants
  * Each participant downloads 1 composite stream
- Speaker view: SFU mode
  * Active speaker's stream forwarded individually
  * High quality for speaker, low for others

Webinar (100-10,000 participants):
- Hosts/panelists: SFU (interactive)
- Attendees: View-only stream (like broadcast)
\`\`\`

---

## MMR (Multimedia Router) Deep Dive

**MMR Responsibilities**:

1. **Routing**: Forward video/audio between participants
2. **Transcoding**: Convert between codecs/resolutions (when necessary)
3. **Mixing**: Combine multiple streams (gallery view)
4. **Recording**: Capture meeting video/audio
5. **Layout**: Arrange participants in grid/speaker view
6. **Encryption**: Handle end-to-end encryption

**MMR Architecture**:

\`\`\`
Client A ─────┐
              │
Client B ─────┼────→ MMR Server ────┬───→ Client C
              │                     │
Client D ─────┘                     └───→ Client E

Each MMR handles:
- 100-500 concurrent meetings
- 10,000-50,000 concurrent participants
- Written in C++ (high performance)
- Runs on custom hardware + cloud (AWS, Oracle Cloud, Azure)
\`\`\`

**MMR Capacity Planning**:

\`\`\`
Single MMR capacity:
- Small meetings (3-5 people): 300-500 meetings
- Medium meetings (10-20 people): 100-200 meetings  
- Large meetings (100+ people): 5-10 meetings

Auto-scaling:
- Monitor CPU usage (<80% target)
- Spin up new MMRs when needed
- Migrate meetings to less-loaded MMRs
\`\`\`

---

## Video Encoding and Adaptive Bitrate

**Video Codecs**:

\`\`\`
Primary: H.264/AVC
- Widely supported (all devices)
- Hardware acceleration (GPU encoding/decoding)
- Profiles: Baseline (web), Main (desktop), High (recording)

Fallback: VP8
- For browsers without H.264
- Slightly less efficient

Future: H.265/HEVC, AV1
- Better compression (50% less bandwidth)
- Limited device support currently
\`\`\`

**Simulcast** (Multiple Quality Layers):

Zoom uses simulcast: sender uploads multiple resolutions simultaneously.

\`\`\`
Participant sends:
- Layer 1: 720p @ 1.5 Mbps (high quality)
- Layer 2: 360p @ 600 Kbps (medium quality)
- Layer 3: 180p @ 150 Kbps (low quality)

MMR selects appropriate layer per recipient:
- Strong network recipient → Gets 720p layer
- Weak network recipient → Gets 360p or 180p layer
- No re-encoding needed (just forwarding)
\`\`\`

**Adaptive Bitrate Algorithm**:

\`\`\`python
class AdaptiveBitrateController:
    def __init__(self):
        self.target_bitrate = 1500  # Start at 1.5 Mbps
        self.packet_loss = 0
        self.available_bandwidth = 0
        
    def update(self, network_stats):
        self.packet_loss = network_stats.packet_loss_rate
        self.available_bandwidth = network_stats.bandwidth
        
        # Adjust bitrate based on packet loss
        if self.packet_loss > 5%:
            self.target_bitrate *= 0.8  # Reduce 20%
        elif self.packet_loss < 1% and self.available_bandwidth > self.target_bitrate * 1.5:
            self.target_bitrate *= 1.1  # Increase 10%
        
        # Clamp bitrate
        self.target_bitrate = clamp(self.target_bitrate, 150, 3000)  # 150 Kbps to 3 Mbps
        
        # Select resolution based on bitrate
        if self.target_bitrate >= 1200:
            return Resolution.HD_720p, 30  # fps
        elif self.target_bitrate >= 600:
            return Resolution.SD_360p, 30
        else:
            return Resolution.LOW_180p, 15  # Lower fps too
\`\`\`

**Frame Rate Adaptation**:

\`\`\`
Good network: 30 fps (smooth)
Moderate packet loss: 15 fps (acceptable)
High packet loss: 5-10 fps (choppy but functional)
Very poor network: Freeze video, keep audio only
\`\`\`

---

## Audio Processing

Audio is more critical than video (users tolerate video degradation but not audio).

**Audio Codecs**:

\`\`\`
Primary: Opus
- Excellent quality at low bitrates
- Bitrate: 8-64 Kbps (adaptive)
- Latency: <20ms
- Handles packet loss gracefully (PLC - Packet Loss Concealment)

Audio prioritization:
- Audio packets marked higher priority than video
- Router QoS (Quality of Service) tags
- Zoom sends audio even if video paused
\`\`\`

**Audio Enhancements**:

**1. Echo Cancellation (AEC)**:
- Remove echo from microphone picking up speaker output
- Essential for laptop meetings (speaker and mic close together)
- Algorithm: Adaptive filter, learns speaker-to-mic delay

**2. Noise Suppression**:
- Remove background noise (typing, dog barking, traffic)
- AI-based (trained on millions of noise samples)
- Real-time processing (low latency)

**3. Auto Gain Control (AGC)**:
- Normalize volume levels (loud and quiet speakers)
- Prevents sudden volume spikes

**4. Active Speaker Detection**:
- Identify who is speaking (for speaker view, gallery highlighting)
- Based on voice activity detection (VAD) + volume level

**Audio Mixing**:

For large meetings, MMR mixes audio:

\`\`\`
10 participants speaking simultaneously:
- MMR receives 10 audio streams
- Mix streams (sum waveforms, normalize to prevent clipping)
- Send mixed audio to all participants
- Result: Participants hear all speakers blended

Active speaker enhancement:
- Boost active speaker volume by 3-6 dB
- Reduce background speakers
- Improves intelligibility
\`\`\`

---

## Network Optimization

Zoom is famous for working well on poor networks. How?

**Protocol: UDP (not TCP)**:

\`\`\`
Why UDP?
- Lower latency (no retransmissions, no head-of-line blocking)
- Real-time traffic prefers dropping old packets over delayed delivery
- TCP retransmissions increase latency (unacceptable for video)

Reliability mechanisms:
- Forward Error Correction (FEC): Send redundant data
- Negative Acknowledgment (NACK): Request missing packets (selective)
- Jitter buffer: Smooth out network jitter
\`\`\`

**Forward Error Correction (FEC)**:

\`\`\`
Zoom sends redundant packets:
- For every 10 data packets, send 2-3 FEC packets
- FEC packets allow reconstructing lost data packets
- Trade-off: 20-30% extra bandwidth for resilience

Example:
- Packets 1-10 sent + FEC_A, FEC_B
- If packet 3 lost, reconstruct from packets 1,2,4-10 + FEC
- Avoid retransmission (no latency increase)
\`\`\`

**Jitter Buffer**:

\`\`\`
Problem: Network jitter causes packets to arrive at variable times
Solution: Jitter buffer (30-100ms)

Algorithm:
- Buffer incoming packets for 50ms (configurable)
- Play packets in order (smooth playback)
- If packet doesn't arrive in time, use FEC or PLC
- Adaptive: Increase buffer if high jitter, decrease if low

Trade-off: Increased latency vs smooth playback
\`\`\`

**Packet Loss Concealment (PLC)**:

\`\`\`
Video PLC:
- Repeat last frame (freeze video momentarily)
- Interpolate between frames (if brief loss)

Audio PLC (Opus codec built-in):
- Predict missing samples based on previous audio
- Fade out if extended loss
- Users barely notice <5% packet loss
\`\`\`

**Bandwidth Adaptation**:

\`\`\`
Zoom continuously measures available bandwidth:
- Send RTCP reports (statistics about connection quality)
- Estimate bandwidth using packet arrival patterns
- Adjust bitrate within 1-2 seconds of network change

Graceful degradation:
1. High bandwidth: 720p 30fps + screen share 1080p 30fps
2. Medium bandwidth: 360p 30fps + screen share 720p 15fps
3. Low bandwidth: 180p 15fps + screen share 480p 5fps
4. Very low bandwidth: Audio only, no video
\`\`\`

---

## NAT Traversal and Firewall Penetration

Participants are behind corporate firewalls, NATs, restrictive networks. Zoom must establish connections.

**ICE (Interactive Connectivity Establishment)**:

\`\`\`
1. STUN (Session Traversal Utilities for NAT):
   - Client contacts STUN server
   - Discovers public IP address
   - Maps internal IP:port to external IP:port

2. Direct connection attempt (if possible):
   - Client A → STUN → public IP A
   - Client B → STUN → public IP B
   - Try direct UDP hole punching
   - If successful: Peer-to-peer path (lower latency)

3. TURN (Traversal Using Relays around NAT):
   - If direct connection fails (symmetric NAT, firewall)
   - Route traffic through TURN relay server
   - Higher latency but works reliably
   - Zoom's MMR acts as TURN server

Connection preference:
1. Direct (P2P) - lowest latency
2. Via MMR (relay) - higher latency but works everywhere
\`\`\`

**Firewall Traversal**:

\`\`\`
Zoom tries multiple ports:
- UDP: 3478, 8801, 8802, 8803
- TCP: 8801, 8802 (fallback)
- HTTPS: 443 (last resort, tunnels over HTTP)

Most restrictive: HTTPS tunnel over port 443
- Works in almost all corporate networks
- Higher latency (HTTP overhead)
- Used by <5% of connections
\`\`\`

---

## Global Infrastructure

Zoom operates 17 co-located datacenters worldwide + cloud resources.

**Datacenter Locations**:
- North America: US East, US West, US Central, Canada
- Europe: UK, Germany, Netherlands
- Asia: Japan, Hong Kong, Singapore, India, Australia
- South America: Brazil
- Middle East/Africa: UAE

**MMR Routing**:

\`\`\`
Participant joins meeting:
1. Client queries Zoom API: Which MMR should I use?
2. Zoom backend considers:
   - Geographic location (closest datacenter)
   - MMR load (CPU usage)
   - Network path quality (latency, packet loss)
3. Returns MMR IP address and port
4. Client establishes connection to MMR
\`\`\`

**Multi-Region Meetings**:

\`\`\`
Participants from different regions:
- US participant → US MMR
- Europe participant → Europe MMR
- Asia participant → Asia MMR

Zoom uses "cascading MMRs":
- MMRs communicate with each other
- Video streams forwarded between MMRs
- Optimized paths (not all-to-all)

Example:
10 US participants + 5 Europe participants + 3 Asia participants

Architecture:
US MMR ←→ Europe MMR ←→ Asia MMR
  (10)       (5)          (3)

US MMR sends 1 composite stream to Europe MMR
Europe MMR sends 1 composite stream to US MMR and Asia MMR
Result: Reduced inter-region bandwidth
\`\`\`

---

## Features Implementation

### Screen Sharing

\`\`\`
Screen capture:
- OS-level screen capture API (Windows, macOS, Linux)
- Capture frequency: 10-30 fps (adaptive)
- Encoding: H.264 (optimized for screen content)
  * Screen content: Text, graphics, slides
  * Higher resolution than camera (1080p-4K)
  * Lower frame rate than camera (10-15 fps often sufficient)

Bandwidth optimization:
- Only send changed regions (dirty rectangles)
- High compression for static content
- Prioritize over participant videos (screen more important)

Advanced: Multiple screen sharing
- Educators: Share multiple screens
- Each screen is separate stream
- Participants can choose which to view
\`\`\`

### Virtual Backgrounds

\`\`\`
AI-based background removal:
- Deep learning model (TensorFlow Lite)
- Segment person from background (semantic segmentation)
- Replace background with image/video/blur
- Real-time processing (30 fps)

Model:
- Input: Video frame (640x360)
- Output: Binary mask (person vs background)
- Inference time: <10ms (GPU accelerated)
- Model size: ~2MB (runs on mobile)

GPU acceleration:
- Desktop: NVIDIA CUDA, AMD ROCm
- Mobile: Metal (iOS), OpenGL ES (Android)
- Fallback: CPU (slower, ~20 fps)
\`\`\`

### Breakout Rooms

\`\`\`
Architecture:
- Host creates breakout rooms (Room A, Room B, Room C)
- Assign participants to rooms
- Technically: Create sub-MMR sessions
  * Each room is separate meeting
  * Separate video/audio streams
  * Host can broadcast to all rooms
  * Host can join any room

Data Model:
Main meeting ID: 123456
Breakout Room A ID: 123456-A
Breakout Room B ID: 123456-B

When host closes rooms:
- Migrate participants back to main meeting
- Merge MMR sessions
- Participants see main meeting again
\`\`\`

### Recording

\`\`\`
Local recording:
- Client captures audio/video locally
- Encodes to MP4 (H.264 + AAC)
- Stores on local disk
- No server involvement (free feature)

Cloud recording:
- MMR captures meeting audio/video
- Encodes to MP4 (multiple quality levels)
- Uploads to AWS S3 (or Zoom's storage)
- Generates shareable link
- Post-processing: Speaker view, gallery view, transcription

Transcription (AI):
- Automatic Speech Recognition (ASR)
- Model: Whisper (OpenAI) or custom
- Generate subtitles/closed captions
- Searchable transcript (find keyword in recording)
\`\`\`

---

## Security

**End-to-End Encryption (E2EE)**:

\`\`\`
How it works:
1. Meeting host enables E2EE (opt-in)
2. Participants generate public/private key pairs
3. Exchange public keys via Zoom server
4. Encrypt media with shared secret key (AES-256-GCM)
5. Keys never leave client devices
6. MMR cannot decrypt (truly end-to-end)

Trade-offs:
- Pro: Maximum security (Zoom can't access content)
- Con: No cloud recording, no transcription, no phone dial-in
- Con: Slightly higher CPU usage (encryption overhead)

Zoom's default (not E2EE):
- TLS encrypted in transit (client ↔ MMR)
- Encrypted at rest (stored recordings)
- MMR can decrypt (needed for recording, transcription)
- Still secure, but not "zero-trust"
\`\`\`

**Waiting Room**:
- Host manually admits participants (prevent Zoom bombing)

**Passcode**:
- Require passcode to join (prevent unauthorized access)

**Screen Sharing Lock**:
- Only host can share screen (prevent disruptions)

---

## Technology Stack

### Core

- **C++**: MMR (multimedia router), video/audio processing
- **WebRTC**: Browser-based clients (modified)
- **Custom protocols**: Proprietary RTP/RTCP extensions

### Backend

- **Go**: Backend services (API, meeting management)
- **Java**: Some services
- **Python**: Data processing, ML models

### Data Storage

- **MySQL**: Meeting metadata, user accounts
- **Redis**: Session storage, rate limiting
- **S3**: Cloud recordings, file storage

### AI/ML

- **TensorFlow Lite**: Virtual backgrounds, noise suppression
- **Whisper**: Transcription (ASR)

### Infrastructure

- **Own datacenters**: 17 co-located facilities
- **AWS, Oracle Cloud, Azure**: Additional capacity (auto-scaling)
- **Custom hardware**: Optimized servers for MMR

### Monitoring

- **Custom tools**: Internal monitoring
- **Datadog**: Metrics, logs
- **PagerDuty**: Alerting, on-call

---

## Key Lessons

### 1. UDP Essential for Real-Time

TCP retransmissions unacceptable for video. UDP with FEC, NACK, PLC provides reliability without latency penalty.

### 2. Adaptive Bitrate Critical

Networks vary wildly (broadband to 3G). Continuous bandwidth adaptation ensures quality for all users.

### 3. Audio > Video

Users tolerate video degradation but not audio issues. Prioritize audio packets, use Opus codec, apply noise suppression.

### 4. Hybrid SFU+MCU

Small meetings: SFU (low latency). Large meetings: MCU for gallery view (low bandwidth). Best of both worlds.

### 5. Global Infrastructure Reduces Latency

17 datacenters worldwide ensure <150ms latency globally. Proximity matters for real-time.

---

## Interview Tips

**Q: How does Zoom handle video conferencing at scale?**

A: Use Multimedia Router (MMR) architecture. Participants send video/audio to MMR (centralized server), MMR forwards to other participants (SFU - Selective Forwarding Unit). For large meetings, MMR mixes videos into composite (MCU - Multipoint Control Unit) to reduce bandwidth. Simulcast: Sender uploads multiple quality layers (720p, 360p, 180p), MMR selects appropriate layer per recipient based on bandwidth. Adaptive bitrate: Continuously measure network quality (packet loss, bandwidth), adjust resolution/frame rate within 1-2 seconds. Protocol: UDP for low latency, Forward Error Correction (FEC) for reliability, jitter buffer (30-100ms) for smooth playback. Global deployment: 17 datacenters, route participants to nearest MMR (<150ms latency). Scale: Each MMR handles 100-500 concurrent meetings, auto-scale by adding MMRs.

**Q: How does Zoom optimize for poor network conditions?**

A: Multiple techniques: (1) Adaptive bitrate: Measure bandwidth and packet loss every second, adjust video resolution (720p→360p→180p) and frame rate (30fps→15fps→5fps) dynamically. (2) Forward Error Correction (FEC): Send redundant packets (20-30% overhead), reconstruct lost packets without retransmission. (3) Jitter buffer: Buffer packets for 30-100ms (adaptive based on jitter), smooth out variable network delays. (4) Packet Loss Concealment: Video - repeat last frame, Audio - predict missing samples (Opus codec). (5) Audio prioritization: Mark audio packets higher priority (QoS), maintain audio even if video paused. (6) Graceful degradation: Very poor network → audio-only mode. Result: Zoom usable even at 5-10% packet loss where competitors fail.

**Q: Why does Zoom use UDP instead of TCP for video?**

A: Real-time video requires low latency; TCP retransmissions increase latency unacceptably. UDP characteristics: (1) No retransmissions: Drop old packets instead of delaying delivery. (2) No head-of-line blocking: Single lost packet doesn't block subsequent packets. (3) Lower latency: ~50ms faster than TCP for video. Reliability via UDP: (1) Forward Error Correction (FEC): Redundant packets allow reconstructing losses. (2) Selective NACK: Request specific lost packets if critical. (3) Packet Loss Concealment: Predict/interpolate missing data. (4) Jitter buffer: Smooth out timing variations. Trade-off: UDP requires custom reliability mechanisms (FEC, NACK) but enables real-time performance impossible with TCP.

---

## Summary

Zoom's architecture handles massive-scale video conferencing with quality and reliability:

**Key Takeaways**:

1. **MMR architecture**: SFU for small meetings (low latency), MCU for large meetings (low bandwidth)
2. **Simulcast**: Multiple quality layers (720p, 360p, 180p), MMR selects per recipient's bandwidth
3. **UDP with FEC**: Low latency protocol, Forward Error Correction for reliability without retransmissions
4. **Adaptive bitrate**: Continuous bandwidth measurement, adjust resolution/frame rate dynamically
5. **Audio prioritization**: Opus codec, echo cancellation, noise suppression, maintain audio over video
6. **Global infrastructure**: 17 datacenters worldwide, route to nearest MMR (<150ms latency)
7. **Network resilience**: Jitter buffer, packet loss concealment, graceful degradation
8. **E2EE option**: End-to-end encryption available (trade-off: no cloud recording/transcription)

Zoom's success from focus on reliability under poor network conditions, global low-latency infrastructure, and optimized video/audio processing.
`,
};
