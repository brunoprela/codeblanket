/**
 * Network Protocols Section
 */

export const networkprotocolsSection = {
  id: 'network-protocols',
  title: 'Network Protocols',
  content: `Understanding network protocols is essential for system design. This section covers key protocols beyond HTTP, including SMTP, FTP, SSH, MQTT, AMQP, and WebRTC.
    
    ## Application Layer Protocols
    
    ### **1. SMTP (Simple Mail Transfer Protocol)**
    
    **Purpose**: Email transmission between mail servers
    
    **Port**: 25 (plain), 587 (submission with TLS)
    
    **How it Works**:
    \`\`\`
    User → Email Client → SMTP Server (sender) → SMTP Server (recipient) → Email Client → User
    \`\`\`
    
    **Example Session**:
    \`\`\`
    telnet smtp.example.com 25
    
    S: 220 smtp.example.com ESMTP Postfix
    C: HELO client.example.com
    S: 250 smtp.example.com
    C: MAIL FROM:<sender@example.com>
    S: 250 OK
    C: RCPT TO:<recipient@example.com>
    S: 250 OK
    C: DATA
    S: 354 End data with <CR><LF>.<CR><LF>
    C: Subject: Test Email
    C: 
    C: This is the email body.
    C: .
    S: 250 OK: queued as 12345
    C: QUIT
    S: 221 Bye
    \`\`\`
    
    **Node.js Example**:
    \`\`\`javascript
    const nodemailer = require('nodemailer');
    
    const transporter = nodemailer.createTransport({
      host: 'smtp.example.com',
      port: 587,
      secure: false, // STARTTLS
      auth: {
        user: 'sender@example.com',
        pass: 'password'
      }
    });
    
    await transporter.sendMail({
      from: 'sender@example.com',
      to: 'recipient@example.com',
      subject: 'Hello',
      text: 'Email body',
      html: '<p>Email body</p>'
    });
    \`\`\`
    
    **Use Cases**:
    - Sending transactional emails (order confirmations, password resets)
    - Email notifications
    - Newsletters
    
    ---
    
    ### **2. FTP/SFTP (File Transfer Protocol)**
    
    **FTP**: Plain text file transfer (Port 21)
    **SFTP**: Secure FTP over SSH (Port 22)
    
    **Problems with FTP**:
    - No encryption (credentials sent in plain text)
    - Requires separate data connection (complex firewalls)
    - Active vs Passive modes confusion
    
    **Better Alternative: SFTP**:
    \`\`\`bash
    # Upload file
    sftp user@server.com
    put local-file.txt /remote/path/
    
    # Download file
    get /remote/path/file.txt local-file.txt
    
    # Sync directory (rsync over SSH)
    rsync -avz --progress /local/dir/ user@server.com:/remote/dir/
    \`\`\`
    
    **Node.js Example (SFTP)**:
    \`\`\`javascript
    const Client = require('ssh2-sftp-client');
    const sftp = new Client();
    
    await sftp.connect({
      host: 'server.com',
      port: 22,
      username: 'user',
      password: 'password'
    });
    
    // Upload
    await sftp.put('/local/file.txt', '/remote/file.txt');
    
    // Download
    await sftp.get('/remote/file.txt', '/local/file.txt');
    
    // List directory
    const list = await sftp.list('/remote/path');
    console.log(list);
    
    await sftp.end();
    \`\`\`
    
    **Use Cases**:
    - Log file collection
    - Backup transfers
    - Data exchange with partners
    - Deployment scripts
    
    ---
    
    ### **3. SSH (Secure Shell)**
    
    **Purpose**: Secure remote shell access and command execution
    
    **Port**: 22
    
    **Key Features**:
    - Encrypted communication
    - Public key authentication
    - Port forwarding (tunneling)
    - File transfer (SCP, SFTP)
    
    **SSH Tunneling**:
    
    **Local Port Forwarding** (access remote service locally):
    \`\`\`bash
    # Access remote MySQL (port 3306) on localhost:3307
    ssh -L 3307:localhost:3306 user@remote-server
    
    # Now connect to localhost:3307 to reach remote MySQL
    mysql -h 127.0.0.1 -P 3307 -u root -p
    \`\`\`
    
    **Remote Port Forwarding** (expose local service remotely):
    \`\`\`bash
    # Expose local service (localhost:8080) on remote server port 8080
    ssh -R 8080:localhost:8080 user@remote-server
    
    # Remote server can now access your localhost:8080 via its port 8080
    \`\`\`
    
    **Dynamic Port Forwarding** (SOCKS proxy):
    \`\`\`bash
    # Create SOCKS proxy on localhost:1080
    ssh -D 1080 user@remote-server
    
    # Configure browser to use SOCKS proxy localhost:1080
    # All traffic routes through remote server
    \`\`\`
    
    **Node.js SSH Client**:
    \`\`\`javascript
    const { Client } = require('ssh2');
    const conn = new Client();
    
    conn.on('ready', () => {
      console.log('Client :: ready');
      
      // Execute command
      conn.exec('uptime', (err, stream) => {
        stream.on('data', (data) => {
          console.log('STDOUT: ' + data);
        });
        
        stream.on('close', () => {
          conn.end();
        });
      });
    });
    
    conn.connect({
      host: 'server.com',
      port: 22,
      username: 'user',
      privateKey: require('fs').readFileSync('/path/to/private/key')
    });
    \`\`\`
    
    ---
    
    ### **4. MQTT (Message Queuing Telemetry Transport)**
    
    **Purpose**: Lightweight pub/sub messaging for IoT devices
    
    **Port**: 1883 (plain), 8883 (TLS)
    
    **Key Features**:
    - Extremely lightweight (header as small as 2 bytes)
    - Pub/sub model with topics
    - QoS levels (0, 1, 2)
    - Retained messages
    - Last Will and Testament (LWT)
    
    **Architecture**:
    \`\`\`
    Publisher → MQTT Broker (Mosquitto/EMQX) → Subscriber
    \`\`\`
    
    **Topic Hierarchy**:
    \`\`\`
    home/living-room/temperature
    home/living-room/humidity
    home/bedroom/temperature
    home/+/temperature          # Wildcard: all rooms' temperature
    home/#                      # Wildcard: everything under home
    \`\`\`
    
    **QoS Levels**:
    - **QoS 0** (At most once): Fire and forget, no acknowledgment
    - **QoS 1** (At least once): Acknowledged, may receive duplicates
    - **QoS 2** (Exactly once): Four-way handshake, guaranteed delivery
    
    **Node.js Example**:
    \`\`\`javascript
    const mqtt = require('mqtt');
    const client = mqtt.connect('mqtt://broker.example.com');
    
    // Subscribe
    client.on('connect', () => {
      client.subscribe('home/+/temperature', (err) => {
        if (!err) {
          console.log('Subscribed');
        }
      });
    });
    
    // Receive messages
    client.on('message', (topic, message) => {
      console.log(\`\${topic}: \${message.toString()}\`);
      // home/living-room/temperature: 72.5
    });
    
    // Publish
    setInterval(() => {
      const temp = (Math.random() * 30 + 60).toFixed(1);
      client.publish('home/living-room/temperature', temp, { qos: 1 });
    }, 5000);
    \`\`\`
    
    **Last Will and Testament** (LWT):
    \`\`\`javascript
    // Set LWT when connecting
    const client = mqtt.connect('mqtt://broker.example.com', {
      will: {
        topic: 'devices/sensor-1/status',
        payload: 'offline',
        qos: 1,
        retain: true
      }
    });
    
    // If client disconnects unexpectedly, broker publishes LWT
    \`\`\`
    
    **Use Cases**:
    - IoT sensor data collection
    - Smart home automation
    - Real-time dashboards
    - Vehicle telemetry
    
    ---
    
    ### **5. AMQP (Advanced Message Queuing Protocol)**
    
    **Purpose**: Enterprise message queuing and routing
    
    **Port**: 5672 (plain), 5671 (TLS)
    
    **Popular Implementation**: RabbitMQ
    
    **Key Features**:
    - Reliable message delivery
    - Complex routing (direct, fanout, topic, headers)
    - Message persistence
    - Transactions
    - Flow control
    
    **Architecture**:
    \`\`\`
    Producer → Exchange → Queue → Consumer
    \`\`\`
    
    **Exchange Types**:
    
    **Direct Exchange** (routing key exact match):
    \`\`\`
    Producer --routing_key: "error"--> Exchange --"error"--> Queue (bound to "error")
    \`\`\`
    
    **Topic Exchange** (routing key pattern match):
    \`\`\`
    Producer --"user.created"--> Exchange --"user.*"--> Queue A
                                          --"*.created"--> Queue B
    \`\`\`
    
    **Fanout Exchange** (broadcast to all queues):
    \`\`\`
    Producer --> Exchange --> Queue A
                          --> Queue B
                          --> Queue C
    \`\`\`
    
    **Node.js Example (RabbitMQ)**:
    \`\`\`javascript
    const amqp = require('amqplib');
    
    // Producer
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    await channel.assertExchange('logs', 'fanout', { durable: false });
    
    setInterval(() => {
      const msg = \`Log message \${Date.now()}\`;
      channel.publish('logs', '', Buffer.from(msg));
      console.log(\`Sent: \${msg}\`);
    }, 1000);
    
    // Consumer
    const connection2 = await amqp.connect('amqp://localhost');
    const channel2 = await connection2.createChannel();
    
    await channel2.assertExchange('logs', 'fanout', { durable: false });
    
    const q = await channel2.assertQueue('', { exclusive: true });
    await channel2.bindQueue(q.queue, 'logs', '');
    
    channel2.consume(q.queue, (msg) => {
      console.log(\`Received: \${msg.content.toString()}\`);
    }, { noAck: true });
    \`\`\`
    
    **Work Queue Pattern**:
    \`\`\`javascript
    // Multiple workers consume from same queue
    // Each message delivered to only ONE worker (load balancing)
    
    // Worker 1
    channel.prefetch(1); // Only take one message at a time
    channel.consume('task_queue', (msg) => {
      const task = msg.content.toString();
      console.log(\`Worker 1 processing: \${task}\`);
      
      // Simulate work
      setTimeout(() => {
        channel.ack(msg); // Acknowledge completion
      }, 1000);
    });
    
    // Worker 2 (same code)
    // Messages distributed: W1, W2, W1, W2, W1, W2...
    \`\`\`
    
    **Use Cases**:
    - Task queues (image processing, email sending)
    - Event-driven architectures
    - Microservices communication
    - Job scheduling
    
    ---
    
    ### **6. WebRTC (Web Real-Time Communication)**
    
    **Purpose**: Peer-to-peer audio, video, and data transfer in browsers
    
    **Protocols Used**:
    - **STUN**: Discover public IP address (NAT traversal)
    - **TURN**: Relay traffic when peer-to-peer fails
    - **ICE**: Combines STUN/TURN to establish connection
    - **SDP**: Session description (offer/answer)
    - **DTLS-SRTP**: Encrypted media transport
    
    **Architecture**:
    \`\`\`
    Peer A ←→ Signaling Server ←→ Peer B
      ↓                              ↓
      +--------- Direct P2P ---------+
      (via STUN/TURN if needed)
    \`\`\`
    
    **Connection Flow**:
    \`\`\`
    1. Peer A creates offer (SDP)
    2. Peer A sends offer to signaling server
    3. Signaling server forwards to Peer B
    4. Peer B creates answer (SDP)
    5. Peer B sends answer back
    6. ICE candidates exchanged
    7. Direct peer-to-peer connection established
    \`\`\`
    
    **Simple WebRTC Example**:
    \`\`\`javascript
    // Peer A (caller)
    const peerA = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        {
          urls: 'turn:turn.example.com',
          username: 'user',
          credential: 'password'
        }
      ]
    });
    
    // Add local stream
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true
    });
    stream.getTracks().forEach(track => peerA.addTrack(track, stream));
    
    // Create offer
    const offer = await peerA.createOffer();
    await peerA.setLocalDescription(offer);
    
    // Send offer to Peer B via signaling server
    signalingServer.send({ type: 'offer', sdp: offer });
    
    // Handle ICE candidates
    peerA.onicecandidate = (event) => {
      if (event.candidate) {
        signalingServer.send({ type: 'ice-candidate', candidate: event.candidate });
      }
    };
    
    // Receive answer from Peer B
    signalingServer.on('answer', async (answer) => {
      await peerA.setRemoteDescription(answer);
    });
    
    // Peer B (answerer)
    const peerB = new RTCPeerConnection({ iceServers: [...] });
    
    // Receive offer from Peer A
    signalingServer.on('offer', async (offer) => {
      await peerB.setRemoteDescription(offer);
      
      // Add local stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      stream.getTracks().forEach(track => peerB.addTrack(track, stream));
      
      // Create answer
      const answer = await peerB.createAnswer();
      await peerB.setLocalDescription(answer);
      
      // Send answer back
      signalingServer.send({ type: 'answer', sdp: answer });
    });
    
    // Display remote stream
    peerB.ontrack = (event) => {
      remoteVideo.srcObject = event.streams[0];
    };
    \`\`\`
    
    **Use Cases**:
    - Video conferencing (Zoom, Google Meet)
    - Voice calls (WhatsApp Web, Discord)
    - Screen sharing
    - P2P file transfer
    - Live streaming
    
    ---
    
    ## Protocol Comparison
    
    | **Protocol** | **Transport** | **Use Case** | **Pros** | **Cons** |
    |--------------|---------------|--------------|----------|----------|
    | **HTTP/HTTPS** | TCP | Web requests | Universal, cacheable | Stateless, overhead |
    | **WebSocket** | TCP | Real-time bidirectional | Full-duplex, efficient | No HTTP caching |
    | **MQTT** | TCP | IoT pub/sub | Lightweight, QoS | Limited routing |
    | **AMQP** | TCP | Enterprise messaging | Complex routing, reliable | Heavier, complex |
    | **gRPC** | TCP (HTTP/2) | Microservices RPC | Fast, streaming | Browser support limited |
    | **WebRTC** | UDP (SRTP) | P2P audio/video | Low latency, P2P | Complex setup |
    
    ---
    
    ## When to Use Each Protocol
    
    ### **Use MQTT When:**
    - IoT devices with limited bandwidth
    - Millions of publishers/subscribers
    - Need QoS guarantees
    - Battery-powered devices (efficient)
    
    ### **Use AMQP When:**
    - Enterprise message queuing
    - Complex routing requirements
    - Need guaranteed delivery and ordering
    - Transactions required
    
    ### **Use WebRTC When:**
    - Real-time audio/video required
    - Peer-to-peer preferred (low latency)
    - Browser-based communication
    - Screen sharing needed
    
    ### **Use WebSocket When:**
    - Real-time updates (chat, notifications)
    - Bidirectional communication
    - Lower latency than HTTP polling
    - Browser support essential
    
    ---
    
    ## Key Takeaways
    
    1. **SMTP** for email transmission (use with TLS on port 587)
    2. **SFTP/SCP** for secure file transfer (never plain FTP)
    3. **SSH** enables secure remote access and tunneling
    4. **MQTT** ideal for IoT pub/sub with QoS levels
    5. **AMQP** provides enterprise-grade message queuing with RabbitMQ
    6. **WebRTC** enables P2P audio/video with NAT traversal via STUN/TURN
    7. **Choose protocol based on**: latency requirements, reliability needs, client capabilities
    8. **Security**: Always use TLS (SMTPS, SFTP, MQTTS, AMQPS)
    9. **IoT**: MQTT preferred over HTTP (lighter, persistent connections)
    10. **Video calls**: WebRTC for P2P, or RTMP/HLS for streaming`,
};
