/**
 * TCP vs UDP Section
 */

export const tcpvsudpSection = {
  id: 'tcp-vs-udp',
  title: 'TCP vs UDP',
  content: `TCP and UDP are the two primary transport-layer protocols in the Internet Protocol Suite. Understanding when to use each is crucial for system design decisions.

## The Transport Layer

The **transport layer** (Layer 4 in OSI model) is responsible for:
- End-to-end communication between applications
- Reliability (if needed)
- Flow control
- Multiplexing (multiple applications on one host)

Two main protocols: **TCP** and **UDP**

---

## TCP (Transmission Control Protocol)

**TCP** is a **connection-oriented**, **reliable**, **ordered** delivery protocol.

### Key Characteristics

**1. Connection-Oriented**:
- Must establish connection before sending data
- Three-way handshake

**2. Reliable**:
- Guarantees delivery
- Retransmits lost packets
- Acknowledges received packets

**3. Ordered**:
- Packets arrive in order sent
- Reorders out-of-order packets

**4. Flow Control**:
- Prevents sender from overwhelming receiver
- Sliding window protocol

**5. Congestion Control**:
- Detects network congestion
- Reduces send rate to avoid packet loss

---

## TCP Three-Way Handshake

Before data can be sent, TCP establishes a connection:

\`\`\`
Client                                Server
  |                                     |
  |------ SYN (seq=1000) -------------->|
  |       "Let\'s establish connection"  |
  |                                     |
  |<----- SYN-ACK (seq=5000, ack=1001) -|
  |       "Acknowledged, here's my seq" |
  |                                     |
  |------ ACK (ack=5001) -------------->|
  |       "Got it, let's start"         |
  |                                     |
  |====== Data transmission ============|
\`\`\`

**Why three-way?**
- Prevents old duplicate connections from causing confusion
- Both sides agree on initial sequence numbers
- Ensures both sides are ready

**Overhead**: 
- 1 round-trip time (RTT) before data can be sent
- For high-latency connections (e.g., 100ms RTT), that's 100ms delay
- This is why TCP can be slow for short requests

---

## TCP Data Transfer

Once connected, TCP ensures reliable delivery:

\`\`\`
Client                                Server
  |                                     |
  |------ Packet 1 (seq=1000) --------->|
  |       Data: "Hello"                 |
  |                                     |
  |<----- ACK (ack=1005) ---------------|
  |       "Got it"                      |
  |                                     |
  |------ Packet 2 (seq=1005) --------->|
  |       Data: "World"                 |
  |                                     |
  |  X--- Packet lost ---------X        |
  |                                     |
  |       [Timeout]                     |
  |                                     |
  |------ Retransmit Packet 2 --------->|
  |       Data: "World"                 |
  |                                     |
  |<----- ACK (ack=1010) ---------------|
\`\`\`

**Key mechanisms**:
- **Sequence numbers**: Track which bytes have been sent
- **Acknowledgments (ACKs)**: Confirm receipt
- **Retransmission timer**: Resend if ACK not received
- **Duplicate ACKs**: Signal lost packet (fast retransmit)

---

## TCP Flow Control

**Problem**: Sender might be faster than receiver

**Solution**: **Sliding window**

\`\`\`
Sender\'s view:
[Sent & ACKed][Sent, awaiting ACK][Can send now][Can't send yet]
              Window size = what receiver can handle
\`\`\`

Receiver tells sender: "I have 64KB buffer available"
Sender won't send more than 64KB before getting ACKs

**TCP window size**:
- Advertised in every ACK
- Dynamically adjusted based on receiver's buffer

---

## TCP Congestion Control

**Problem**: Too much traffic can conquer the network

**Solution**: TCP congestion control algorithms

### Slow Start

Start sending slowly, increase exponentially:
\`\`\`
Round 1: Send 1 packet
Round 2: Send 2 packets (if ACKed)
Round 3: Send 4 packets
Round 4: Send 8 packets
...
\`\`\`

Increase until:
- Packet loss occurs (network congested)
- Slow start threshold reached

### Congestion Avoidance

After slow start, increase linearly:
- Add 1 MSS (Maximum Segment Size) per RTT

### Fast Retransmit / Fast Recovery

If 3 duplicate ACKs received:
- Assume packet lost (don't wait for timeout)
- Retransmit immediately
- Cut congestion window in half

**Algorithms**:
- **TCP Reno**: Basic congestion control
- **TCP Cubic**: Modern (Linux default), better for high-bandwidth networks
- **BBR** (Bottleneck Bandwidth and RTT): Google\'s algorithm, optimizes for throughput

---

## UDP (User Datagram Protocol)

**UDP** is **connectionless**, **unreliable**, **unordered** protocol.

### Key Characteristics

**1. Connectionless**:
- No handshake
- Just send packets (datagrams)
- No connection state

**2. Unreliable**:
- No delivery guarantee
- Packets can be lost
- No retransmission

**3. Unordered**:
- Packets may arrive out of order
- No reordering

**4. No Flow Control**:
- Sender sends as fast as it wants

**5. No Congestion Control**:
- Doesn't adapt to network conditions

**6. Minimal Overhead**:
- 8-byte header (vs TCP's 20+ bytes)
- No connection setup latency

---

## UDP Datagram Structure

\`\`\`
Client                                Server
  |                                     |
  |------ Datagram 1 ------------------>|
  |       Data: "Hello"                 |
  |                                     |
  |------ Datagram 2 ------------------>|
  |       Data: "World"                 |
  |                                     |
  |------ Datagram 3 ------------------>|
  |  X--- Lost -----------------X       |
  |                                     |
  |       No retransmission!            |
  |       No notification!              |
\`\`\`

**No guarantees**:
- Datagram 2 might arrive before Datagram 1
- Datagram 3 might never arrive
- Application must handle these cases

---

## TCP vs UDP Comparison

| Feature | TCP | UDP |
|---------|-----|-----|
| **Connection** | Connection-oriented | Connectionless |
| **Reliability** | Guaranteed delivery | No guarantee |
| **Ordering** | Ordered | Unordered |
| **Speed** | Slower (overhead) | Faster (minimal overhead) |
| **Header size** | 20-60 bytes | 8 bytes |
| **Flow control** | Yes | No |
| **Congestion control** | Yes | No |
| **Use case** | Reliable delivery needed | Speed more important than reliability |
| **Examples** | HTTP, FTP, SSH, Email | DNS, VoIP, Video streaming, Gaming |

---

## When to Use TCP

Use TCP when:
- **Reliability is critical**
- **Order matters**
- **You can tolerate latency**

### Use Cases

**1. HTTP/HTTPS** (Web traffic):
- Must receive entire HTML page
- Order matters (can't render page with missing parts)
- User can wait 100ms

**2. File Transfer (FTP, SFTP)**:
- Can't have corrupted files
- Every byte must arrive
- Speed less critical than reliability

**3. Email (SMTP, IMAP)**:
- Can't lose emails
- Order not critical, but reliability is

**4. SSH/Remote Access**:
- Every keystroke must arrive
- Out-of-order commands would be confusing

**5. Database Connections**:
- Queries and results must be reliable
- Transactions require reliability

---

## When to Use UDP

Use UDP when:
- **Speed is more important than reliability**
- **Real-time is critical**
- **You can handle packet loss**

### Use Cases

**1. DNS (Domain Name System)**:
- Queries are small (single packet)
- If lost, just retry (application-level retry)
- TCP handshake would double latency
\`\`\`
TCP DNS: 
  1. SYN, SYN-ACK, ACK (1 RTT)
  2. Query, Response (1 RTT)
  Total: 2 RTT

UDP DNS:
  1. Query, Response (1 RTT)
  Total: 1 RTT
\`\`\`

**2. Video Streaming (Live)**:
- Old frames are useless (timestamp passed)
- Better to skip lost frame than wait for retransmit
- Example: Zoom, YouTube Live, Twitch

**3. Online Gaming**:
- Player position updates 60 times/second
- If packet lost, next update is coming in 16ms anyway
- Retransmitting old position is useless
- Example: Counter-Strike, Call of Duty

**4. VoIP (Voice over IP)**:
- Human voice can tolerate small losses
- Delay causes awkward pauses
- Better to have slight audio glitch than 200ms delay

**5. IoT Sensor Data**:
- Sending temperature every second
- Lost reading not critical (next one coming soon)
- Millions of devices, TCP overhead prohibitive

**6. Metrics/Logging**:
- StatsD sends metrics via UDP
- Losing occasional metric acceptable
- High volume, low individual importance

---

## Hybrid Approaches

### **QUIC (Quick UDP Internet Connections)**

- **Used by**: HTTP/3, Google services
- **Idea**: UDP + reliability built in application layer

**Why QUIC?**

TCP has limitations:
- Head-of-line blocking (one lost packet blocks everything)
- Slow connection setup (1-2 RTT)
- Hard to update (baked into OS kernel)

QUIC advantages:
- **0-RTT connection**: Establish connection + send data in one round trip
- **No head-of-line blocking**: Independent streams
- **Survives IP changes**: Mobile devices switching networks
- **Encrypted by default**: Built-in TLS 1.3

\`\`\`
TCP + TLS:
  1. TCP handshake (SYN, SYN-ACK, ACK)
  2. TLS handshake
  Total: 2-3 RTT before data

QUIC:
  1. Connection + encryption + data
  Total: 0-1 RTT
\`\`\`

**Adoption**:
- HTTP/3 uses QUIC
- Google search, YouTube, Gmail use QUIC
- ~25% of internet traffic now uses QUIC

### **RTP (Real-time Transport Protocol)**

- Built on UDP
- Adds sequence numbers and timestamps
- Used for VoIP, video conferencing
- Application decides what to do with lost packets

### **SCTP (Stream Control Transmission Protocol)**

- Combines benefits of TCP and UDP
- Multiple streams in one connection
- No head-of-line blocking
- Less widely supported

---

## TCP Optimizations

### **TCP Fast Open (TFO)**

Skip the full 3-way handshake on subsequent connections:
\`\`\`
First connection: Normal 3-way handshake
Server gives client a cookie

Subsequent connections:
Client sends SYN + Cookie + Data (all in one packet!)
Server validates cookie, sends SYN-ACK + Response
\`\`\`

**Benefit**: Saves 1 RTT

**Adoption**: Supported by Linux, macOS, iOS

### **TCP BBR (Bottleneck Bandwidth and RTT)**

- Google\'s congestion control algorithm
- Measures actual bottleneck bandwidth
- Better than Cubic for high-bandwidth networks
- 2-5x faster for YouTube, Google services

### **TCP Keepalive**

- Sends periodic probes to check if connection alive
- Prevents middleboxes from timing out connection
- Configurable interval (default: 2 hours)

---

## Real-World Examples

### **Netflix**

- **Video streaming**: TCP (not UDP!)
- Why? Video is pre-recorded, not live
- Can buffer ahead
- Every byte must arrive (or video corrupted)
- Uses adaptive bitrate over TCP

### **Zoom**

- **Video/audio**: UDP primary, TCP fallback
- UDP for real-time with custom reliability layer
- Falls back to TCP if firewall blocks UDP
- Implements own jitter buffer, packet loss concealment

### **Discord**

- **Voice chat**: UDP with custom reliability
- **Text chat**: TCP (must be reliable)
- **File transfer**: TCP

### **Google Meet**

- Uses WebRTC over UDP (SRTP protocol)
- Falls back to TCP if UDP blocked
- Implements own congestion control

### **Cloudflare**

- Supports QUIC (HTTP/3) for websites
- Falls back to TCP (HTTP/2) if client doesn't support

---

## Common Mistakes

### ❌ **Using UDP without considering packet loss**
\`\`\`python
# Bad: Just send and forget
udp_socket.sendto (critical_data, address)
# If packet lost, data gone forever!
\`\`\`

**Fix**: Implement application-level acknowledgments for critical data

### ❌ **Using TCP for real-time when UDP better**
- Example: Live video streaming over TCP
- Problem: Retransmits delay entire stream

### ❌ **Assuming UDP is always faster**
- For large data transfers, TCP's congestion control helps
- UDP can congest network if sending too fast

### ❌ **Not handling TCP connection failures**
- TCP connections can break
- Need timeouts, reconnection logic

### ❌ **Forgetting about firewalls**
- Many corporate firewalls block UDP
- Need TCP fallback mechanism

---

## Interview Tips

### **Question: "When would you use UDP over TCP?"**

**Good answer structure**:
1. "UDP is appropriate when speed/latency is more important than reliability"
2. Give concrete examples:
   - Live video streaming (old frames useless)
   - Online gaming (position updates every 16ms)
   - DNS (small request, can retry if lost)
3. Mention you'd handle packet loss at application layer if needed
4. Note that UDP is connectionless, so lower overhead

### **Question: "Explain TCP three-way handshake"**

**Hit these points**:
1. Client sends SYN with initial sequence number
2. Server responds SYN-ACK (acknowledges + sends its sequence)
3. Client sends ACK
4. Purpose: Agree on sequence numbers, ensure both sides ready
5. Overhead: 1 RTT before data can be sent

### **Question: "How does TCP ensure reliability?"**

**Key mechanisms**:
1. Sequence numbers (track bytes)
2. Acknowledgments (confirm receipt)
3. Retransmission timers (resend if ACK not received)
4. Checksums (detect corruption)
5. Flow control (prevent overwhelming receiver)

### **Question: "What is QUIC and why was it created?"**

**Answer**:
- QUIC is UDP-based protocol with reliability built in
- Solves TCP problems:
  - Head-of-line blocking (one lost packet blocks all streams)
  - Slow connection setup (1-2 RTT)
  - Can't update TCP (in kernel)
- Used by HTTP/3
- Benefits: 0-RTT connection, independent streams, survives IP changes

---

## Best Practices

### **1. Use TCP for reliability, UDP for speed**
- Default to TCP unless you have specific reason for UDP
- Don't reinvent TCP on top of UDP (unless you're QUIC)

### **2. Implement timeouts**
- TCP connections can hang
- Set appropriate socket timeouts
\`\`\`python
socket.settimeout(30)  # 30 seconds
\`\`\`

### **3. TCP: Enable keepalive for long-lived connections**
- Detects broken connections
- Prevents middlebox timeouts

### **4. UDP: Implement application-level reliability if needed**
- Sequence numbers
- Acknowledgments
- Retransmission logic

### **5. Consider QUIC for modern applications**
- Better performance than TCP
- Built-in encryption
- Check if your stack supports it

### **6. Handle UDP firewall issues**
- Many enterprises block UDP
- Always have TCP fallback

### **7. Monitor packet loss**
- High packet loss indicates network issues
- TCP automatically adapts, but monitor performance

---

## Key Takeaways

1. **TCP**: Connection-oriented, reliable, ordered - use for reliability
2. **UDP**: Connectionless, unreliable, fast - use for speed/real-time
3. **TCP handshake**: 3-way (SYN, SYN-ACK, ACK) before data transfer
4. **TCP reliability**: Sequence numbers, ACKs, retransmission, flow control
5. **UDP use cases**: DNS, live video, gaming, VoIP - speed matters
6. **QUIC**: Modern UDP-based protocol combining TCP benefits with UDP speed
7. **Trade-off**: TCP reliability vs UDP speed - choose based on requirements`,
};
