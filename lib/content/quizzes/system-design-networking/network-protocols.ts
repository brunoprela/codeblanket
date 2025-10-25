/**
 * Quiz questions for Network Protocols section
 */

export const networkprotocolsQuiz = [
  {
    id: 'network-protocol-iot-architecture',
    question:
      "Design a scalable IoT platform for 1 million smart home devices (sensors, cameras, thermostats) sending data to the cloud. Choose appropriate protocols for device-to-cloud communication, cloud-to-device commands, video streaming, and mobile app notifications. Justify your protocol choices, explain the architecture, and describe how you'd handle device authentication, message persistence, and fault tolerance.",
    sampleAnswer: `**IoT Platform Architecture for 1M Smart Home Devices**
    
    **1. Device Categories and Protocol Selection**
    
    | **Device Type** | **Data Pattern** | **Protocol** | **Justification** |
    |-----------------|------------------|--------------|-------------------|
    | **Sensors** (temperature, humidity, motion) | Periodic telemetry (1/min) | **MQTT** | Lightweight, persistent connection, QoS support |
    | **Thermostats** (bidirectional control) | Telemetry + commands | **MQTT** | Pub/sub for commands, efficient bidirectional |
    | **Cameras** (video streaming) | Continuous video | **RTSP → HLS/DASH** | RTSP from device, HLS/DASH to clients |
    | **Mobile App** (real-time notifications) | Push notifications | **WebSocket + FCM/APNS** | WebSocket for live data, FCM/APNS for offline push |
    
    **2. High-Level Architecture**
    
    \`\`\`
    Devices (1M)
        ↓
    MQTT Broker Cluster (EMQX)
        ↓
    Message Router (Kafka)
        ↓
        +--> Stream Processing (Flink) --> Time-Series DB (TimescaleDB)
        +--> Rule Engine (Drools) --> Alerts (SNS/SQS)
        +--> Video Ingestion --> S3 + CloudFront
        ↓
    API Gateway (GraphQL/REST)
        ↓
    Mobile/Web Apps
    \`\`\`
    
    **3. Device-to-Cloud Communication (MQTT)**
    
    **Why MQTT**:
    - **Lightweight**: 2-byte header vs HTTP's 100+ bytes
    - **Persistent connection**: No connection overhead per message
    - **QoS levels**: Guarantee delivery for critical data
    - **Last Will Testament**: Detect offline devices
    - **Topic hierarchy**: Organize devices efficiently
    
    **Topic Structure**:
    \`\`\`
    devices/{device_id}/telemetry      # Device publishes data
    devices/{device_id}/commands       # Cloud publishes commands
    devices/{device_id}/status         # Online/offline status (LWT)
    devices/{device_id}/errors         # Error reporting
    \`\`\`
    
    **Device Implementation** (Temperature Sensor):
    \`\`\`javascript
    const mqtt = require('mqtt');
    
    // Connect with TLS and auth
    const client = mqtt.connect('mqtts://mqtt.example.com:8883', {
      clientId: \`device-\${deviceId}\`,
      username: deviceId,
      password: deviceSecret,
      clean: false, // Persist session
      will: {
        topic: \`devices/\${deviceId}/status\`,
        payload: 'offline',
        qos: 1,
        retain: true
      },
      reconnectPeriod: 5000 // Auto-reconnect
    });
    
    client.on('connect', () => {
      // Publish online status
      client.publish(\`devices/\${deviceId}/status\`, 'online', {
        qos: 1,
        retain: true
      });
      
      // Subscribe to commands
      client.subscribe(\`devices/\${deviceId}/commands\`, { qos: 1 });
    });
    
    // Publish telemetry every minute
    setInterval(() => {
      const data = {
        temperature: readTemperature(),
        humidity: readHumidity(),
        battery: readBattery(),
        timestamp: Date.now()
      };
      
      client.publish(
        \`devices/\${deviceId}/telemetry\`,
        JSON.stringify (data),
        { qos: 1 } // At-least-once delivery
      );
    }, 60000);
    
    // Handle commands
    client.on('message', (topic, message) => {
      if (topic === \`devices/\${deviceId}/commands\`) {
        const command = JSON.parse (message.toString());
        handleCommand (command);
      }
    });
    \`\`\`
    
    **4. MQTT Broker Cluster (EMQX)**
    
    **Configuration for 1M Devices**:
    \`\`\`yaml
    # emqx.conf
    node.max_ports = 2097152  # Support 2M connections
    
    # Cluster configuration
    cluster.discovery = k8s
    cluster.k8s.apiserver = https://kubernetes.default.svc:443
    cluster.k8s.namespace = iot-platform
    
    # Connection limits per node
    mqtt.max_packet_size = 1MB
    mqtt.max_clientid_len = 256
    mqtt.max_topic_alias = 65535
    
    # Session persistence
    mqtt.session_expiry_interval = 7200s  # 2 hours
    
    # Resource limits
    listener.tcp.external.max_connections = 500000  # 500K per node (2 nodes = 1M)
    listener.tcp.external.acceptors = 64
    listener.tcp.external.max_conn_rate = 1000  # 1K connections/sec
    
    # TLS
    listener.ssl.external.handshake_timeout = 15s
    listener.ssl.external.keyfile = /etc/certs/key.pem
    listener.ssl.external.certfile = /etc/certs/cert.pem
    
    # Authentication (PostgreSQL backend)
    auth.pgsql.server = postgres:5432
    auth.pgsql.username_query = SELECT password_hash FROM devices WHERE device_id = \${username}
    
    # ACL (topic-level permissions)
    auth.pgsql.acl_query = SELECT allow, topic, action FROM device_acls WHERE device_id = \${username}
    \`\`\`
    
    **Scaling**:
    - **2 EMQX nodes**: 500K connections each = 1M total
    - **Auto-scaling**: Add nodes when connections >400K (80% capacity)
    - **Load balancing**: AWS NLB with TCP passthrough
    
    **5. Message Persistence and Fault Tolerance**
    
    **Challenge**: Device publishes while broker restarts → message lost?
    
    **Solution 1: MQTT Persistent Sessions**:
    \`\`\`javascript
    // Device connects with clean=false
    const client = mqtt.connect('mqtts://mqtt.example.com', {
      clean: false, // Persist session
      clientId: deviceId // Same client ID on reconnect
    });
    
    // Broker stores:
    // - Subscriptions
    // - Unacknowledged messages
    // - QoS 1/2 messages
    \`\`\`
    
    **Solution 2: Bridge to Kafka for Guaranteed Persistence**:
    \`\`\`yaml
    # EMQX rule engine bridges to Kafka
    rules:
      - sql: SELECT * FROM "devices/+/telemetry"
        actions:
          - kafka_produce:
              topic: device-telemetry
              partition: \${clientid}  # Same device always same partition
              key: \${device_id}
              value: \${payload}
    \`\`\`
    
    **Kafka Configuration**:
    \`\`\`properties
    # Replication for fault tolerance
    replication.factor=3
    min.insync.replicas=2
    
    # Retention
    log.retention.hours=168  # 7 days
    log.segment.bytes=1073741824  # 1GB segments
    
    # Partitions (for parallelism)
    partitions=100  # 10K devices per partition
    \`\`\`
    
    **6. Device Authentication and Security**
    
    **Authentication Flow**:
    \`\`\`
    1. Device provisioning:
       - Generate device_id and device_secret
       - Store in PostgreSQL with hash
       
    2. Device connection:
       - MQTT username = device_id
       - MQTT password = device_secret
       - EMQX queries PostgreSQL for validation
       
    3. Topic-level ACL:
       - Device can only publish to devices/{device_id}/*
       - Device can only subscribe to devices/{device_id}/commands
    \`\`\`
    
    **Device Provisioning**:
    \`\`\`javascript
    // Provisioning API
    app.post('/api/devices', async (req, res) => {
      const deviceId = uuidv4();
      const deviceSecret = crypto.randomBytes(32).toString('hex');
      const secretHash = await bcrypt.hash (deviceSecret, 10);
      
      await db.devices.create({
        device_id: deviceId,
        password_hash: secretHash,
        owner_id: req.user.id,
        created_at: new Date()
      });
      
      // Create ACL rules
      await db.device_acls.createMany([
        {
          device_id: deviceId,
          topic: \`devices/\${deviceId}/telemetry\`,
          action: 'publish',
          allow: true
        },
        {
          device_id: deviceId,
          topic: \`devices/\${deviceId}/commands\`,
          action: 'subscribe',
          allow: true
        }
      ]);
      
      res.json({
        device_id: deviceId,
        device_secret: deviceSecret, // Return ONCE, never again!
        mqtt_host: 'mqtts://mqtt.example.com:8883'
      });
    });
    \`\`\`
    
    **7. Cloud-to-Device Commands**
    
    **Challenge**: Send command to specific device
    
    **Solution**:
    \`\`\`javascript
    // API endpoint
    app.post('/api/devices/:deviceId/commands', async (req, res) => {
      const { deviceId } = req.params;
      const { command, params } = req.body;
      
      // Check device ownership
      const device = await db.devices.findOne({
        where: { device_id: deviceId, owner_id: req.user.id }
      });
      
      if (!device) {
        return res.status(404).json({ error: 'Device not found' });
      }
      
      // Check device is online
      const isOnline = await redis.get(\`device:\${deviceId}:online\`);
      if (!isOnline) {
        return res.status(503).json({ error: 'Device offline' });
      }
      
      // Publish command to device's command topic
      const message = JSON.stringify({
        command,
        params,
        timestamp: Date.now(),
        request_id: uuidv4()
      });
      
      await mqttClient.publish(
        \`devices/\${deviceId}/commands\`,
        message,
        { qos: 1 }
      );
      
      res.json({ status: 'sent' });
    });
    \`\`\`
    
    **8. Video Streaming Architecture**
    
    **Challenge**: 100K cameras streaming video
    
    **Architecture**:
    \`\`\`
    Camera (RTSP) → Media Server (Wowza/Kurento) → S3 (recordings)
                            ↓
                      HLS/DASH (adaptive)
                            ↓
                      CloudFront CDN → Mobile/Web App
    \`\`\`
    
    **Why not MQTT for video**:
    - MQTT designed for small messages (KB)
    - Video requires GB/hour bandwidth
    - RTSP optimized for continuous streams
    
    **Recording Flow**:
    \`\`\`javascript
    // Camera pushes RTSP stream
    // rtsp://camera-ip:554/stream
    
    // Media server ingests and creates HLS segments
    ffmpeg -i rtsp://camera-ip:554/stream \\
      -codec: copy \\
      -hls_time 6 \\
      -hls_list_size 10 \\
      -hls_flags delete_segments \\
      /tmp/camera-123/stream.m3u8
    
    // Upload segments to S3
    aws s3 sync /tmp/camera-123/ s3://video-streams/camera-123/
    \`\`\`
    
    **9. Mobile App Communication**
    
    **Real-Time Updates (WebSocket)**:
    \`\`\`javascript
    // User subscribes to their devices
    const ws = new WebSocket('wss://api.example.com/ws');
    
    ws.send(JSON.stringify({
      type: 'subscribe',
      devices: ['device-1', 'device-2', 'device-3',]
    }));
    
    ws.onmessage = (event) => {
      const data = JSON.parse (event.data);
      // data: { device_id: 'device-1', temperature: 72.5, ... }
      updateUI(data);
    };
    \`\`\`
    
    **Push Notifications (FCM/APNS)**:
    \`\`\`javascript
    // Rule engine triggers alert
    if (temperature > 85) {
      await sendPushNotification({
        token: userDeviceToken,
        title: 'High Temperature Alert',
        body: \`Living room: \${temperature}°F\`,
        data: { device_id: deviceId, type: 'alert' }
      });
    }
    \`\`\`
    
    **10. Monitoring and Observability**
    
    **Metrics to Track**:
    \`\`\`yaml
    # MQTT Broker
    - emqx_connections_count: Current connections
    - emqx_messages_received_rate: Messages/sec
    - emqx_messages_sent_rate: Messages/sec
    - emqx_session_count: Active sessions
    
    # Devices
    - devices_online_count: Online devices by type
    - messages_per_device_p50/p95: Message frequency
    - device_battery_level: Battery health
    
    # Infrastructure
    - kafka_consumer_lag: Processing delay
    - timescaledb_write_rate: Ingestion rate
    - api_latency_p99: API performance
    \`\`\`
    
    **Key Takeaways**:
    
    1. **MQTT for IoT**: Lightweight, QoS, persistent connections
    2. **EMQX cluster**: 500K connections per node, bridge to Kafka
    3. **Kafka for persistence**: Guaranteed message delivery, replay capability
    4. **Device auth**: Unique device_id + secret, topic-level ACL
    5. **Commands**: Publish to devices/{device_id}/commands with QoS 1
    6. **Video**: RTSP → HLS/DASH → CDN (not MQTT)
    7. **Mobile**: WebSocket for real-time, FCM/APNS for offline push
    8. **Scaling**: Horizontal scaling of MQTT brokers, partitioned Kafka
    9. **Fault tolerance**: MQTT sessions + Kafka replication
    10. **Monitor**: Connection count, message rate, consumer lag, battery health`,
    keyPoints: [
      'MQTT for IoT devices: Lightweight, persistent connections, QoS support, Last Will Testament',
      'EMQX cluster: 500K connections per node, TLS authentication, topic-level ACL',
      'Bridge MQTT to Kafka for guaranteed persistence and replay capability',
      'Video streaming: RTSP from device → HLS/DASH to clients (not MQTT)',
      'Mobile notifications: WebSocket for real-time + FCM/APNS for offline push',
      'Device authentication: Unique device_id + secret, stored in PostgreSQL',
      'Scaling: Horizontal MQTT brokers, partitioned Kafka (100 partitions for 1M devices)',
      'Fault tolerance: MQTT persistent sessions + Kafka replication (3x)',
    ],
  },
  {
    id: 'network-protocol-cdn-design',
    question:
      'Design a Content Delivery Network (CDN) for a global media streaming service serving 100M users. Choose protocols for content delivery, origin fetch, cache invalidation, and real-time analytics. Explain how you would handle cache coherence, origin shield, HTTP/3 benefits, and measure CDN performance. Include specific protocol choices and architectural decisions.',
    sampleAnswer: `**CDN Architecture for Global Media Streaming**

**1. High-Level Architecture**

\`\`\`
Users (100M)
    ↓
Edge Servers (1000+ PoPs globally)
    ↓ (cache miss)
Regional Origin Shields (10 regions)
    ↓ (cache miss)
Origin Servers (Content Source)
    ↑
CDN Control Plane (Cache invalidation, analytics)
\`\`\`

**2. Protocol Selection**

| **Component** | **Protocol** | **Justification** |
|---------------|--------------|-------------------|
| **User → Edge** | HTTP/3 (QUIC) | 0-RTT, connection migration, better mobile performance |
| **Edge → Origin Shield** | HTTP/2 with TLS 1.3 | Multiplexing, header compression, fast handshake |
| **Origin Shield → Origin** | HTTP/2 + gRPC | Efficient for metadata + content fetch |
| **Cache Invalidation** | gRPC streaming | Real-time bidirectional updates |
| **Analytics** | Protocol Buffers over HTTPS | Compact, efficient serialization |

---

**3. User → Edge: HTTP/3 (QUIC over UDP)**

**Why HTTP/3**:
- **0-RTT resumption**: Returning users connect instantly
- **Connection migration**: Mobile users changing networks (WiFi ↔ 4G) don't drop connections
- **Head-of-line blocking elimination**: Lost packets only block affected stream
- **Better congestion control**: BBR (Bottleneck Bandwidth and RTT) vs cubic

**Edge Server Configuration** (NGINX/Caddy):

\`\`\`nginx
http {
    # HTTP/3 support
    listen 443 quic reuseport;
    listen 443 ssl http2;  # Fallback to HTTP/2
    
    # Alt-Svc header to advertise HTTP/3
    add_header Alt-Svc 'h3=":443"; ma=86400';
    
    # QUIC settings
    quic_gso on;
    quic_retry on;
    
    # SSL/TLS
    ssl_protocols TLSv1.3;
    ssl_early_data on;  # 0-RTT
    
    # Caching
    proxy_cache cdn_cache;
    proxy_cache_valid 200 24h;
    proxy_cache_valid 404 1m;
    proxy_cache_key $scheme$host$request_uri;
    
    # Cache lock (prevent thundering herd)
    proxy_cache_lock on;
    proxy_cache_lock_timeout 5s;
    
    location /video/ {
        # Serve from cache if available
        proxy_cache_use_stale error timeout updating;
        
        # On cache miss, fetch from origin shield
        proxy_pass https://origin-shield-\${geo};
        
        # Add cache status header
        add_header X-Cache-Status $upstream_cache_status;
        add_header X-CDN-Pop $server_name;
    }
}
\`\`\`

**Benefits of HTTP/3 for Streaming**:
- **15-30% faster video start**: 0-RTT eliminates 1 round trip
- **50% fewer rebuffers on mobile**: Connection migration prevents stalls
- **Better throughput**: No HOL blocking means one lost packet doesn't block entire stream

---

**4. Edge → Origin Shield: HTTP/2 with TLS 1.3**

**Why Origin Shield**:
- Reduces origin load by aggregating requests from multiple edges
- Collapses duplicate requests (100 edge servers requesting same video → 1 request to origin)
- Provides second-tier caching

**Origin Shield Architecture**:

\`\`\`
Edge Servers (US West) ────┐
Edge Servers (US East) ────┼──→ Origin Shield (US)  ─┐
Edge Servers (US Central) ─┘                          │
                                                       ├──→ Origin Servers
Edge Servers (Europe) ─────────→ Origin Shield (EU) ─┤
Edge Servers (Asia) ───────────→ Origin Shield (APAC)─┘
\`\`\`

**Why HTTP/2 for Edge → Origin Shield**:
- **Multiplexing**: 1 connection handles many concurrent video requests
- **Header compression**: HPACK reduces redundant headers
- **Server push**: Origin Shield can push manifest + first segment together
- **TLS 1.3**: Faster handshake (1-RTT vs 2-RTT in TLS 1.2)

**Edge Server → Origin Shield Request**:

\`\`\`javascript
// Edge server making request to origin shield
const http2 = require('http2');

const client = http2.connect('https://origin-shield-us.example.com');

const req = client.request({
  ':path': '/video/movie123/segment_00042.ts',
  'x-cdn-edge-id': process.env.EDGE_ID,
  'x-cdn-pop': process.env.POP_LOCATION,
  'x-forwarded-for': originalClientIP
});

req.on('response', (headers) => {
  const cacheStatus = headers['x-cache-status',];
  // 'hit' or 'miss' from origin shield
  
  if (cacheStatus === 'miss') {
    metrics.increment('origin_shield_miss');
  }
});

req.on('data', (chunk) => {
  // Stream video data to client
  clientResponse.write (chunk);
});

req.on('end', () => {
  clientResponse.end();
  client.close();
});
\`\`\`

---

**5. Cache Invalidation: gRPC Streaming**

**Challenge**: When content is updated (live sports score, breaking news), invalidate cached copies across 1000+ edge servers.

**gRPC Bidirectional Streaming Solution**:

\`\`\`protobuf
// cdn_control.proto
service CDNControl {
  // Each edge server opens persistent connection to control plane
  rpc StreamInvalidations (stream InvalidationAck) returns (stream InvalidationRequest);
}

message InvalidationRequest {
  string request_id = 1;
  repeated string cache_keys = 2;  // URLs or patterns to invalidate
  int64 timestamp = 3;
  InvalidationType type = 4;  // PURGE, SOFT_PURGE, TAG_BASED
}

message InvalidationAck {
  string request_id = 1;
  string edge_server_id = 2;
  InvalidationStatus status = 3;
  int32 keys_invalidated = 4;
}

enum InvalidationType {
  PURGE = 0;         // Delete from cache immediately
  SOFT_PURGE = 1;    // Mark stale, revalidate on next request
  TAG_BASED = 2;     // Invalidate all content with specific tag
}
\`\`\`

**Edge Server Implementation**:

\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('cdn_control.proto');
const cdnControl = grpc.loadPackageDefinition (packageDefinition).CDNControl;

const client = new cdnControl.CDNControl(
  'control-plane.example.com:50051',
  grpc.credentials.createSsl()
);

// Open bidirectional stream
const stream = client.StreamInvalidations();

// Listen for invalidation requests from control plane
stream.on('data', async (request) => {
  console.log(\`Received invalidation: \${request.request_id}\`);
  
  let keysInvalidated = 0;
  
  for (const key of request.cache_keys) {
    if (request.type === 'PURGE') {
      // Delete from cache
      await cache.delete (key);
      keysInvalidated++;
    } else if (request.type === 'SOFT_PURGE') {
      // Mark as stale
      await cache.markStale (key);
      keysInvalidated++;
    } else if (request.type === 'TAG_BASED') {
      // Find all content with tag and invalidate
      const keys = await cache.findByTag (key);
      await cache.deleteMany (keys);
      keysInvalidated += keys.length;
    }
  }
  
  // Send acknowledgment back to control plane
  stream.write({
    request_id: request.request_id,
    edge_server_id: process.env.EDGE_ID,
    status: 'COMPLETED',
    keys_invalidated: keysInvalidated
  });
});

stream.on('error', (error) => {
  console.error('Stream error:', error);
  // Reconnect with exponential backoff
  setTimeout(() => reconnect(), 5000);
});

// Heartbeat to keep connection alive
setInterval(() => {
  stream.write({
    edge_server_id: process.env.EDGE_ID,
    status: 'HEARTBEAT'
  });
}, 30000);
\`\`\`

**Control Plane (Initiating Invalidation)**:

\`\`\`javascript
// When content is updated, broadcast invalidation to all edge servers
async function invalidateContent (cacheKeys) {
  const requestId = generateUUID();
  const request = {
    request_id: requestId,
    cache_keys: cacheKeys,
    timestamp: Date.now(),
    type: 'PURGE'
  };
  
  // Track acknowledgments
  const acks = new Map();
  const targetEdgeServers = await getActiveEdgeServers();
  
  // Send to all connected edge servers
  for (const [edgeId, stream] of connectedEdges) {
    stream.write (request);
    acks.set (edgeId, { sent: Date.now(), received: false });
  }
  
  // Wait for acks (or timeout after 10 seconds)
  await Promise.race([
    waitForAllAcks (acks),
    timeout(10000)
  ]);
  
  // Log completion
  const successCount = Array.from (acks.values()).filter (a => a.received).length;
  console.log(\`Invalidation completed: \${successCount}/\${targetEdgeServers.length} edge servers\`);
  
  // Alert if <95% success
  if (successCount / targetEdgeServers.length < 0.95) {
    sendAlert('Cache invalidation partial failure');
  }
}
\`\`\`

**Why gRPC Streaming**:
- **Real-time**: Sub-second invalidation propagation
- **Bidirectional**: Edge servers can ack completion
- **Efficient**: Single connection per edge server (not HTTP polling)
- **Reliable**: Built-in retries, flow control

---

**6. Real-Time Analytics: Protocol Buffers over HTTPS**

**Edge Server → Analytics Pipeline**:

\`\`\`protobuf
// analytics.proto
message AccessLog {
  string edge_server_id = 1;
  string client_ip = 2;
  string url = 3;
  int32 http_status = 4;
  int64 bytes_sent = 5;
  int32 response_time_ms = 6;
  string cache_status = 7;  // hit, miss, stale
  string user_agent = 8;
  string geo_country = 9;
  int64 timestamp = 10;
}

message AnalyticsBatch {
  repeated AccessLog logs = 1;
}
\`\`\`

**Edge Server Batching & Sending**:

\`\`\`javascript
const logs = [];

// Log each request
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    logs.push({
      edge_server_id: process.env.EDGE_ID,
      client_ip: req.ip,
      url: req.url,
      http_status: res.statusCode,
      bytes_sent: res.getHeader('content-length'),
      response_time_ms: Date.now() - start,
      cache_status: res.getHeader('x-cache-status'),
      user_agent: req.headers['user-agent',],
      geo_country: geoip.lookup (req.ip).country,
      timestamp: Date.now()
    });
  });
  
  next();
});

// Batch and send every 10 seconds
setInterval (async () => {
  if (logs.length === 0) return;
  
  const batch = { logs: logs.splice(0, 10000) };  // Max 10K per batch
  const serialized = AnalyticsBatch.encode (batch).finish();
  
  // Send compressed protobuf to analytics pipeline
  await fetch('https://analytics.example.com/ingest', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-protobuf',
      'Content-Encoding': 'gzip'
    },
    body: gzip (serialized)
  });
}, 10000);
\`\`\`

**Why Protocol Buffers**:
- **Compact**: 5-10x smaller than JSON
- **Fast**: ~10x faster serialization/deserialization
- **Schema evolution**: Add fields without breaking old clients
- **Cross-language**: Same .proto works in Go, Python, Node, etc.

---

**7. Performance Measurement**

**Key CDN Metrics**:

\`\`\`javascript
// Real User Monitoring (RUM)
const metrics = {
  // Latency
  ttfb: 'Time to First Byte',  // Target: <50ms
  ttlb: 'Time to Last Byte',    // Video download time
  
  // Cache efficiency
  cache_hit_ratio: 'Cache hits / Total requests',  // Target: >95%
  origin_offload: '1 - (Origin requests / Total requests)',  // Target: >95%
  
  // Quality of Experience
  video_startup_time: 'Time to start playback',  // Target: <2s
  rebuffer_ratio: 'Rebuffer events / Video views',  // Target: <1%
  
  // Throughput
  throughput: 'Mbps delivered to user',
  concurrent_streams: 'Active video streams',
  
  // Errors
  error_rate: '5xx / Total requests',  // Target: <0.01%
  origin_errors: '5xx from origin / Origin requests'
};
\`\`\`

**Performance Comparison**:

| **Metric** | **HTTP/1.1** | **HTTP/2** | **HTTP/3 (QUIC)** |
|------------|--------------|------------|-------------------|
| **TTFB (returning user)** | 150ms | 100ms | **50ms** (0-RTT) |
| **Video startup** | 3s | 2.5s | **1.8s** |
| **Rebuffer rate (mobile)** | 5% | 3% | **1.5%** (connection migration) |
| **Throughput (lossy network)** | 5 Mbps | 7 Mbps | **10 Mbps** (better congestion control) |

---

**8. Cache Coherence Strategy**

**Multi-Tier Caching**:

\`\`\`
Edge Cache (1000 servers)
    ↓
Origin Shield Cache (10 servers)
    ↓
Origin Cache (CDN-friendly caching headers)
\`\`\`

**Cache-Control Headers from Origin**:

\`\`\`http
HTTP/1.1 200 OK
Cache-Control: public, max-age=86400, s-maxage=604800, stale-while-revalidate=3600
CDN-Cache-Control: max-age=604800
Surrogate-Control: max-age=2592000
Vary: Accept-Encoding
ETag: "abc123"
X-Cache-Tag: movie:123, category:action
\`\`\`

**Tag-Based Invalidation**:

\`\`\`javascript
// When movie 123 is updated
await invalidateContent([
  '/video/movie123/*',           // Specific path
  'tag:movie:123',                // All content tagged with this movie
  'tag:category:action'           // All action movies
]);
\`\`\`

---

**Key Takeaways**:

1. **HTTP/3 for users**: 0-RTT, connection migration, better mobile performance
2. **HTTP/2 for backend**: Multiplexing, efficient for edge-to-origin communication
3. **gRPC for control plane**: Real-time cache invalidation with bidirectional streams
4. **Protocol Buffers for analytics**: Compact, fast, schema evolution
5. **Origin shield**: Reduces origin load by 95%+, provides second-tier caching
6. **Tag-based invalidation**: Efficient way to purge related content
7. **Performance**: HTTP/3 reduces TTFB by 50%, rebuffers by 66% on mobile
8. **Monitoring**: Cache hit ratio >95%, TTFB <50ms, rebuffer rate <1%`,
    keyPoints: [
      'HTTP/3 (QUIC) for user-facing: 0-RTT, connection migration, eliminates head-of-line blocking',
      'HTTP/2 for edge-to-origin: Multiplexing, header compression, efficient backend communication',
      'gRPC streaming for cache invalidation: Real-time bidirectional updates across 1000+ edge servers',
      'Protocol Buffers for analytics: 5-10x smaller than JSON, faster serialization',
      'Origin Shield: Collapses requests, reduces origin load by 95%, second-tier caching',
      'Tag-based cache invalidation: Purge related content efficiently (movie:123, category:action)',
      'Performance gains: 50% faster TTFB, 66% fewer mobile rebuffers with HTTP/3',
      'Key metrics: Cache hit ratio >95%, TTFB <50ms, rebuffer rate <1%',
    ],
  },
  {
    id: 'network-protocol-p2p-streaming',
    question:
      'Design a peer-to-peer (P2P) video streaming protocol for a live streaming platform to reduce CDN bandwidth costs by 70%. Explain how you would handle peer discovery, chunk distribution, incentive mechanisms, and fallback to CDN. Compare WebRTC Data Channels vs BitTorrent-style protocols, and discuss security, NAT traversal, and quality of experience trade-offs.',
    sampleAnswer: `**P2P Live Streaming Protocol Design**

**Goal**: Reduce CDN costs by 70% while maintaining <2s latency for live streams.

---

**1. High-Level Architecture**

\`\`\`
Live Stream Source
    ↓
CDN Edge Servers (30% traffic)
    ↓
Tracker/Signaling Server (WebSocket)
    ↓
P2P Mesh Network (70% traffic)
    ↓
Viewers (100M)
\`\`\`

**Hybrid Approach**: CDN + P2P
- **CDN**: Seed the first chunk, serve users without P2P capability
- **P2P**: Distribute most chunks peer-to-peer
- **Fallback**: Switch to CDN if P2P fails

---

**2. Protocol Selection: WebRTC Data Channels vs BitTorrent**

| **Aspect** | **WebRTC Data Channels** | **BitTorrent** |
|------------|--------------------------|----------------|
| **Latency** | **Low (1-2s)** | High (10-30s) |
| **Browser Support** | **Native** | Needs plugin |
| **NAT Traversal** | **Built-in (STUN/TURN)** | Requires setup |
| **Use Case** | **Live streaming** | File distribution |
| **Complexity** | Moderate | Low |
| **Connection Setup** | Fast (100-200ms) | Slow (handshake) |

**Choice**: **WebRTC Data Channels** for live streaming

**Why**:
- Native browser support (no plugins)
- Low latency (suitable for live)
- Built-in NAT traversal (STUN/TURN)
- Encrypted by default (DTLS)

---

**3. Peer Discovery & Signaling**

**Signaling Server** (WebSocket):

\`\`\`javascript
// Client connects to tracker
const ws = new WebSocket('wss://tracker.example.com');

ws.onopen = () => {
  // Register as peer for this stream
  ws.send(JSON.stringify({
    type: 'join',
    stream_id: 'stream-12345',
    peer_id: generatePeerId(),
    capabilities: {
      upload_bandwidth: measureUploadBandwidth(),
      webrtc_support: true,
      nat_type: detectNATType()
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse (event.data);
  
  if (message.type === 'peers') {
    // Tracker sends list of peers
    const peers = message.peers;
    
    // Connect to 5-10 peers
    peers.slice(0, 10).forEach (peer => {
      connectToPeer (peer);
    });
  }
};
\`\`\`

**Tracker Server** (Assigns peers):

\`\`\`javascript
const streams = new Map(); // stream_id -> Set of peers

wss.on('connection', (ws) => {
  ws.on('message', (data) => {
    const message = JSON.parse (data);
    
    if (message.type === 'join') {
      const { stream_id, peer_id, capabilities } = message;
      
      if (!streams.has (stream_id)) {
        streams.set (stream_id, new Set());
      }
      
      const peers = streams.get (stream_id);
      peers.add({ peer_id, ws, capabilities, joined_at: Date.now() });
      
      // Send list of existing peers to new joiner
      const peerList = Array.from (peers)
        .filter (p => p.peer_id !== peer_id)
        .slice(0, 20); // Top 20 peers
      
      ws.send(JSON.stringify({
        type: 'peers',
        peers: peerList.map (p => ({
          peer_id: p.peer_id,
          upload_bandwidth: p.capabilities.upload_bandwidth
        }))
      }));
      
      // Notify existing peers about new joiner
      for (const peer of peers) {
        if (peer.peer_id !== peer_id) {
          peer.ws.send(JSON.stringify({
            type: 'peer_joined',
            peer: { peer_id, capabilities }
          }));
        }
      }
    }
  });
});
\`\`\`

**Peer Selection Strategy**:
1. **Geographic proximity**: Prefer peers in same region (lower latency)
2. **Upload bandwidth**: Prefer peers with high upload capacity
3. **Chunk availability**: Connect to peers with chunks you need
4. **Connection limit**: Maintain 5-10 active connections

---

**4. Chunk Distribution Protocol**

**Chunk Format**:

\`\`\`javascript
interface VideoChunk {
  stream_id: string;
  chunk_id: number;     // Incremental sequence number
  timestamp: number;    // Playback timestamp
  data: ArrayBuffer;    // Video data (HLS segment)
  duration: number;     // Chunk duration (2-4 seconds)
  size: number;         // Bytes
  hash: string;         // SHA-256 for integrity
}
\`\`\`

**P2P Connection Setup** (WebRTC):

\`\`\`javascript
async function connectToPeer (peerInfo) {
  const pc = new RTCPeerConnection({
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      {
        urls: 'turn:turn.example.com:3478',
        username: 'user',
        credential: 'pass'
      }
    ]
  });
  
  // Create data channel
  const dataChannel = pc.createDataChannel('video-chunks', {
    ordered: false,  // Don't wait for lost packets
    maxRetransmits: 0  // No retransmissions (live stream)
  });
  
  dataChannel.onopen = () => {
    console.log('Connected to peer:', peerInfo.peer_id);
    
    // Request needed chunks
    requestChunks (dataChannel, getNeededChunks());
  };
  
  dataChannel.onmessage = (event) => {
    const chunk = decodeChunk (event.data);
    
    // Verify chunk integrity
    if (verifyChunk (chunk)) {
      // Add to buffer
      chunkBuffer.set (chunk.chunk_id, chunk);
      
      // Forward to other peers (become a relay)
      relayChunkToOtherPeers (chunk);
    }
  };
  
  // ICE/SDP signaling via tracker
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      ws.send(JSON.stringify({
        type: 'ice_candidate',
        target_peer: peerInfo.peer_id,
        candidate: event.candidate
      }));
    }
  };
  
  const offer = await pc.createOffer();
  await pc.setLocalDescription (offer);
  
  ws.send(JSON.stringify({
    type: 'offer',
    target_peer: peerInfo.peer_id,
    sdp: offer
  }));
}
\`\`\`

**Chunk Request/Response**:

\`\`\`javascript
// Request chunks
function requestChunks (dataChannel, chunkIds) {
  dataChannel.send(JSON.stringify({
    type: 'request',
    chunk_ids: chunkIds
  }));
}

// Handle chunk requests
dataChannel.onmessage = (event) => {
  const message = JSON.parse (event.data);
  
  if (message.type === 'request') {
    // Send requested chunks
    message.chunk_ids.forEach (id => {
      const chunk = chunkBuffer.get (id);
      if (chunk) {
        dataChannel.send (encodeChunk (chunk));
        metrics.increment('chunks_uploaded');
      }
    });
  }
};
\`\`\`

---

**5. Chunk Selection Strategy (Rarest First)**

\`\`\`javascript
function getNeededChunks() {
  const currentPlayhead = video.currentTime;
  const chunkDuration = 2; // 2 seconds per chunk
  const currentChunkId = Math.floor (currentPlayhead / chunkDuration);
  
  // Priority:
  // 1. Next 3 chunks (critical for playback)
  // 2. Future 10 chunks (buffering)
  // 3. Rarest chunks in network
  
  const needed = [];
  
  // Critical chunks
  for (let i = 0; i < 3; i++) {
    const id = currentChunkId + i;
    if (!chunkBuffer.has (id)) {
      needed.push({ id, priority: 'critical', rarity: 0 });
    }
  }
  
  // Buffer chunks
  for (let i = 3; i < 10; i++) {
    const id = currentChunkId + i;
    if (!chunkBuffer.has (id)) {
      const rarity = getChunkRarity (id);
      needed.push({ id, priority: 'buffer', rarity });
    }
  }
  
  // Sort by priority then rarity (rarest first)
  return needed
    .sort((a, b) => {
      if (a.priority === 'critical' && b.priority !== 'critical') return -1;
      if (a.priority !== 'critical' && b.priority === 'critical') return 1;
      return b.rarity - a.rarity;
    })
    .map (c => c.id);
}

function getChunkRarity (chunkId) {
  // Query peers for chunk availability
  let peersWithChunk = 0;
  
  for (const peer of connectedPeers) {
    if (peer.availableChunks.has (chunkId)) {
      peersWithChunk++;
    }
  }
  
  return 1 / (peersWithChunk + 1);  // Higher = rarer
}
\`\`\`

---

**6. Incentive Mechanism (Tit-for-Tat)**

**Problem**: Leechers (download but don't upload) hurt the network.

**Solution**: BitTorrent-style reciprocity

\`\`\`javascript
class PeerManager {
  constructor() {
    this.peers = new Map();
    this.uploadedToPeer = new Map();
    this.downloadedFromPeer = new Map();
  }
  
  recordUpload (peerId, bytes) {
    this.uploadedToPeer.set(
      peerId,
      (this.uploadedToPeer.get (peerId) || 0) + bytes
    );
  }
  
  recordDownload (peerId, bytes) {
    this.downloadedFromPeer.set(
      peerId,
      (this.downloadedFromPeer.get (peerId) || 0) + bytes
    );
  }
  
  // Periodically evaluate peers (every 30 seconds)
  evaluatePeers() {
    const ratios = new Map();
    
    for (const peer of this.peers.values()) {
      const uploaded = this.uploadedToPeer.get (peer.id) || 0;
      const downloaded = this.downloadedFromPeer.get (peer.id) || 0;
      
      // Sharing ratio
      const ratio = downloaded > 0 ? uploaded / downloaded : 0;
      ratios.set (peer.id, ratio);
    }
    
    // Unchoke top 4 contributors
    const topPeers = Array.from (ratios.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([peerId]) => peerId);
    
    // Unchoke 1 random peer (optimistic unchoking)
    const randomPeer = getRandomPeer (this.peers);
    topPeers.push (randomPeer.id);
    
    // Update peer states
    for (const peer of this.peers.values()) {
      if (topPeers.includes (peer.id)) {
        peer.unchoke();  // Allow downloading from this peer
      } else {
        peer.choke();    // Stop sending data to this peer
      }
    }
  }
}
\`\`\`

**Visualization**:

\`\`\`
Peer A (uploads 10 MB, downloads 5 MB) → Ratio: 2.0 → ✅ Unchoked
Peer B (uploads 8 MB, downloads 4 MB)  → Ratio: 2.0 → ✅ Unchoked
Peer C (uploads 2 MB, downloads 10 MB) → Ratio: 0.2 → ❌ Choked
Peer D (uploads 0 MB, downloads 5 MB)  → Ratio: 0.0 → ❌ Choked
Peer E (random)                         → Ratio: N/A → ✅ Unchoked (optimistic)
\`\`\`

---

**7. CDN Fallback**

**When to fall back to CDN**:
1. **P2P unavailable**: No peers or poor connectivity
2. **Buffer starvation**: Not receiving chunks fast enough
3. **High latency**: P2P latency >2s
4. **Quality degradation**: Excessive buffering or rebuffering

\`\`\`javascript
class VideoPlayer {
  constructor() {
    this.p2pEnabled = true;
    this.cdnFallbackActive = false;
    this.bufferHealth = 10; // seconds of buffer
  }
  
  async loadChunk (chunkId) {
    // Try P2P first
    if (this.p2pEnabled && !this.cdnFallbackActive) {
      const chunk = await this.loadFromP2P(chunkId, { timeout: 2000 });
      
      if (chunk) {
        this.bufferHealth += chunk.duration;
        return chunk;
      }
    }
    
    // Fallback to CDN
    console.log('Falling back to CDN for chunk', chunkId);
    const chunk = await this.loadFromCDN(chunkId);
    
    // Check if we should disable P2P
    if (this.bufferHealth < 3) {
      this.cdnFallbackActive = true;
      setTimeout(() => {
        this.cdnFallbackActive = false;  // Retry P2P in 30s
      }, 30000);
    }
    
    return chunk;
  }
  
  async loadFromP2P(chunkId, options) {
    return Promise.race([
      this.requestChunkFromPeers (chunkId),
      timeout (options.timeout)
    ]).catch(() => null);
  }
  
  async loadFromCDN(chunkId) {
    const response = await fetch(\`https://cdn.example.com/chunks/\${chunkId}.ts\`);
    return response.arrayBuffer();
  }
}
\`\`\`

---

**8. NAT Traversal (STUN/TURN)**

**NAT Types**:
1. **Full Cone**: Easy, direct P2P works
2. **Restricted Cone**: Moderate, needs STUN
3. **Port Restricted Cone**: Moderate, needs STUN
4. **Symmetric**: Hard, needs TURN relay

**Solution**: Use STUN first, fallback to TURN

\`\`\`javascript
const iceServers = [
  // Public STUN servers (free)
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  
  // TURN relay servers (costs money)
  {
    urls: 'turn:turn.example.com:3478',
    username: getTurnCredentials().username,
    credential: getTurnCredentials().password
  }
];

const pc = new RTCPeerConnection({ iceServers });

pc.onicecandidate = (event) => {
  if (event.candidate) {
    console.log('ICE candidate type:', event.candidate.type);
    // host, srflx (STUN), relay (TURN)
    
    if (event.candidate.type === 'relay') {
      metrics.increment('turn_relay_used');
      // TURN relay costs money - track usage
    }
  }
};
\`\`\`

**TURN Relay Costs**:
- TURN used for ~15-20% of peers (symmetric NAT)
- Relay bandwidth: ~30% of CDN bandwidth
- **Net savings**: 70% P2P + 15% TURN + 15% CDN = **55% CDN cost reduction**

---

**9. Security**

**Threats**:
1. **Malicious chunks**: Peer sends corrupted data
2. **Sybil attack**: Single entity creates many fake peers
3. **Eclipse attack**: Isolate peer from honest peers
4. **Privacy**: IP addresses visible to peers

**Mitigations**:

\`\`\`javascript
// 1. Chunk verification (SHA-256 hash)
function verifyChunk (chunk) {
  const hash = sha256(chunk.data);
  return hash === chunk.hash;
}

// 2. Peer reputation system
class ReputationSystem {
  constructor() {
    this.scores = new Map();
  }
  
  recordBadChunk (peerId) {
    const score = this.scores.get (peerId) || 100;
    this.scores.set (peerId, score - 10);
    
    if (score - 10 < 50) {
      blockPeer (peerId);
    }
  }
  
  recordGoodChunk (peerId) {
    const score = this.scores.get (peerId) || 100;
    this.scores.set (peerId, Math.min (score + 1, 100));
  }
}

// 3. Sybil resistance (proof of work)
function generatePeerId() {
  let nonce = 0;
  let hash;
  
  do {
    hash = sha256(\`\${Date.now()}-\${nonce}\`);
    nonce++;
  } while (!hash.startsWith('0000')); // Require 4 leading zeros
  
  return hash;
}

// 4. Privacy (use TURN relay to hide IP)
const pc = new RTCPeerConnection({
  iceTransportPolicy: 'relay'  // Force TURN (hide real IP)
});
\`\`\`

---

**10. Quality of Experience**

**Metrics**:

\`\`\`javascript
const qoe = {
  startup_time: 'Time to first frame',  // Target: <2s
  rebuffer_rate: 'Rebuffer events / minute',  // Target: <0.1
  average_bitrate: 'Average video quality',  // Target: 1080p
  p2p_ratio: 'P2P bytes / Total bytes',  // Target: >70%
  cdn_fallback_rate: 'CDN requests / Total',  // Target: <30%
  peer_count: 'Active peer connections',  // Target: 5-10
  upload_contribution: 'Bytes uploaded',  // Encourage sharing
};
\`\`\`

**User Experience**:
- **Transparent**: User doesn't notice P2P vs CDN
- **Adaptive**: Falls back to CDN seamlessly
- **Privacy-aware**: Option to disable P2P
- **Bandwidth control**: Limit upload/download rates

---

**Key Takeaways**:

1. **WebRTC Data Channels**: Best for live P2P streaming (low latency, native support)
2. **Hybrid CDN + P2P**: CDN seeds, P2P distributes (70% cost savings)
3. **Rarest-first strategy**: Prioritize chunks fewer peers have
4. **Tit-for-tat incentives**: Reward uploaders, throttle leechers
5. **CDN fallback**: Seamless switch when P2P fails
6. **STUN/TURN**: NAT traversal (~15% peers need TURN relay)
7. **Security**: Chunk hashing, reputation system, proof-of-work peer IDs
8. **Net savings**: 55-70% CDN cost reduction after TURN relay costs`,
    keyPoints: [
      'WebRTC Data Channels over BitTorrent: Low latency (1-2s), native browser support, built-in NAT traversal',
      'Hybrid CDN + P2P: CDN seeds first chunk and serves fallback, P2P handles 70% of traffic',
      'Peer discovery: WebSocket tracker assigns 5-10 peers based on geography, bandwidth, chunk availability',
      'Rarest-first chunk selection: Prioritize chunks fewer peers have to maximize distribution',
      'Tit-for-tat incentives: Unchoke top 4 contributors + 1 random peer to prevent leeching',
      'NAT traversal: STUN for most peers (~85%), TURN relay for symmetric NAT (~15%)',
      'Security: SHA-256 chunk verification, reputation system, proof-of-work peer IDs',
      'Cost savings: 70% P2P - 15% TURN = 55% net CDN cost reduction',
    ],
  },
];
