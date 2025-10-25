/**
 * Quiz questions for gRPC Service Design section
 */

export const grpcservicedesignQuiz = [
  {
    id: 'grpc-d1',
    question:
      'Design a gRPC service for a video streaming platform with user management, video upload, and live streaming. Define the proto file and explain your RPC type choices.',
    sampleAnswer: `Complete gRPC service design for video streaming platform:

\`\`\`protobuf
syntax = "proto3";

package streaming;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// ========== User Service ==========

service UserService {
  // Unary: Simple CRUD
  rpc CreateUser(CreateUserRequest) returns (User);
  rpc GetUser(GetUserRequest) returns (User);
  rpc UpdateUser(UpdateUserRequest) returns (User);
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);
  
  // Server streaming: List with potentially large results
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Get user's subscriptions
  rpc GetSubscriptions(GetUserRequest) returns (stream Channel);
}

message User {
  string id = 1;
  string username = 2;
  string email = 3;
  string avatar_url = 4;
  google.protobuf.Timestamp created_at = 5;
  int64 subscriber_count = 6;
}

// ========== Video Service ==========

service VideoService {
  // Client streaming: Video upload in chunks
  rpc UploadVideo (stream VideoChunk) returns (Video);
  
  // Server streaming: Download video in chunks
  rpc DownloadVideo(VideoRequest) returns (stream VideoChunk);
  
  // Unary: Metadata operations
  rpc GetVideo(VideoRequest) returns (Video);
  rpc UpdateVideo(UpdateVideoRequest) returns (Video);
  rpc DeleteVideo(VideoRequest) returns (google.protobuf.Empty);
  
  // Server streaming: Search results
  rpc SearchVideos(SearchRequest) returns (stream Video);
  
  // Get video comments
  rpc GetComments(VideoRequest) returns (stream Comment);
}

message Video {
  string id = 1;
  string title = 2;
  string description = 3;
  string thumbnail_url = 4;
  string video_url = 5;
  int64 duration_seconds = 6;
  int64 view_count = 7;
  string author_id = 8;
  google.protobuf.Timestamp created_at = 9;
  VideoStatus status = 10;
}

enum VideoStatus {
  PROCESSING = 0;
  READY = 1;
  FAILED = 2;
}

message VideoChunk {
  bytes data = 1;
  int64 offset = 2;
  string video_id = 3;  // Empty for upload, set for download
}

// ========== Live Streaming Service ==========

service LiveStreamService {
  // Bidirectional: Real-time stream
  rpc Stream (stream StreamPacket) returns (stream StreamPacket);
  
  // Server streaming: Watch live stream
  rpc WatchStream(WatchRequest) returns (stream StreamPacket);
  
  // Bidirectional: Live chat
  rpc Chat (stream ChatMessage) returns (stream ChatMessage);
  
  // Unary: Stream management
  rpc StartStream(StartStreamRequest) returns (Stream);
  rpc EndStream(StreamRequest) returns (google.protobuf.Empty);
  rpc GetStream(StreamRequest) returns (Stream);
}

message Stream {
  string id = 1;
  string title = 2;
  string streamer_id = 3;
  int64 viewer_count = 4;
  google.protobuf.Timestamp started_at = 5;
  StreamStatus status = 6;
}

enum StreamStatus {
  LIVE = 0;
  ENDED = 1;
  OFFLINE = 2;
}

message StreamPacket {
  bytes video_data = 1;
  bytes audio_data = 2;
  int64 timestamp_ms = 3;
  string stream_id = 4;
}

message ChatMessage {
  string id = 1;
  string stream_id = 2;
  string user_id = 3;
  string username = 4;
  string message = 5;
  google.protobuf.Timestamp sent_at = 6;
}

// ========== Recommendation Service ==========

service RecommendationService {
  // Server streaming: Personalized recommendations
  rpc GetRecommendations(RecommendationRequest) returns (stream Video);
  
  // Unary: Record view for algorithm
  rpc RecordView(ViewEvent) returns (google.protobuf.Empty);
}

message RecommendationRequest {
  string user_id = 1;
  int32 count = 2;
}

message ViewEvent {
  string user_id = 1;
  string video_id = 2;
  int64 watch_duration_seconds = 3;
  google.protobuf.Timestamp viewed_at = 4;
}

// ========== Common Messages ==========

message GetUserRequest {
  string id = 1;
}

message VideoRequest {
  string id = 1;
}

message SearchRequest {
  string query = 1;
  int32 limit = 2;
  int32 offset = 3;
}
\`\`\`

**RPC Type Justifications**:

1. **Unary (CreateUser, GetVideo)**: Simple CRUD operations
2. **Client Streaming (UploadVideo)**: Client sends video in chunks, server returns metadata when complete
3. **Server Streaming (DownloadVideo, SearchVideos)**: Server sends data in chunks or streams results
4. **Bidirectional (Stream, Chat)**: Real-time two-way communication

**Implementation Considerations**:

- Video chunking for large files
- Stream IDs for reconnection
- Timestamps for ordering
- Enums for status
- Separate services for separation of concerns
- Metadata in separate messages

This design leverages gRPC's strengths: performance, streaming, and type safety.`,
    keyPoints: [
      'Use unary RPC for simple CRUD operations',
      'Client streaming for chunked uploads',
      'Server streaming for large result sets',
      'Bidirectional streaming for real-time communication',
      'Separate services by domain for maintainability',
    ],
  },
  {
    id: 'grpc-d2',
    question:
      'Your gRPC microservices are experiencing intermittent failures and timeouts. Design a comprehensive error handling and retry strategy.',
    sampleAnswer: `Comprehensive gRPC error handling and reliability strategy:

**1. Client-Side Retry Logic**

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// Exponential backoff retry
async function retryCall (callFn, maxRetries = 3) {
  let lastError;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await callFn();
    } catch (error) {
      lastError = error;
      
      // Only retry on transient errors
      const retryableErrors = [
        grpc.status.UNAVAILABLE,
        grpc.status.DEADLINE_EXCEEDED,
        grpc.status.RESOURCE_EXHAUSTED
      ];
      
      if (!retryableErrors.includes (error.code)) {
        throw error;  // Don't retry permanent errors
      }
      
      // Exponential backoff with jitter
      const baseDelay = 100;
      const maxDelay = 5000;
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 100,
        maxDelay
      );
      
      console.log(\`Retry \${attempt + 1}/\${maxRetries} after \${delay}ms\`);
      await sleep (delay);
    }
  }
  
  throw lastError;
}

// Usage
const response = await retryCall(() => 
  new Promise((resolve, reject) => {
    client.getUser(
      { id: '123' },
      { deadline: Date.now() + 5000 },
      (err, response) => {
        if (err) reject (err);
        else resolve (response);
      }
    );
  })
);
\`\`\`

**2. Deadline Propagation**

\`\`\`javascript
// Client sets deadline
const deadline = new Date();
deadline.setSeconds (deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  callback
);

// Server propagates to downstream calls
async function getUser (call, callback) {
  // Get deadline from incoming call
  const deadline = call.deadline;
  
  // Propagate to downstream service
  const response = await downstreamClient.getData(
    { id: call.request.id },
    { deadline: deadline },
    callback
  );
  
  callback (null, response);
}
\`\`\`

**3. Circuit Breaker**

\`\`\`javascript
class CircuitBreaker {
  constructor (threshold = 5, timeout = 60000) {
    this.failureThreshold = threshold;
    this.timeout = timeout;
    this.failureCount = 0;
    this.state = 'CLOSED';  // CLOSED, OPEN, HALF_OPEN
    this.nextAttempt = Date.now();
  }
  
  async call (fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      // Try half-open
      this.state = 'HALF_OPEN';
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }
  
  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.timeout;
    }
  }
}

// Usage
const breaker = new CircuitBreaker();

const response = await breaker.call(() =>
  client.getUser({ id: '123' })
);
\`\`\`

**4. Timeout Strategy**

\`\`\`javascript
// Service-specific timeouts
const TIMEOUTS = {
  userService: 2000,      // Fast service
  analyticsService: 10000, // Slow aggregation
  searchService: 5000
};

function callWithTimeout (client, method, request, serviceName) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + TIMEOUTS[serviceName];
    
    client[method](
      request,
      { deadline },
      (err, response) => {
        if (err) {
          if (err.code === grpc.status.DEADLINE_EXCEEDED) {
            console.error(\`\${serviceName} timeout after \${TIMEOUTS[serviceName]}ms\`);
          }
          reject (err);
        } else {
          resolve (response);
        }
      }
    );
  });
}
\`\`\`

**5. Health Checking**

\`\`\`protobuf
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
  }
  ServingStatus status = 1;
}
\`\`\`

\`\`\`javascript
// Server implementation
const healthServer = {
  check: (call, callback) => {
    callback (null, {
      status: ServingStatus.SERVING
    });
  }
};

// Client health check before calls
async function ensureHealth() {
  const response = await healthClient.check({});
  if (response.status !== ServingStatus.SERVING) {
    throw new Error('Service not healthy');
  }
}
\`\`\`

**6. Error Response Enrichment**

\`\`\`javascript
function enrichError (error, context) {
  return {
    code: error.code,
    message: error.message,
    details: {
      service: context.service,
      method: context.method,
      requestId: context.requestId,
      timestamp: new Date().toISOString(),
      stack: error.stack
    }
  };
}

// Server
async function getUser (call, callback) {
  try {
    const user = await db.findUser (call.request.id);
    if (!user) {
      return callback({
        code: grpc.status.NOT_FOUND,
        message: 'User not found',
        metadata: new grpc.Metadata({
          'request-id': call.metadata.get('request-id')[0]
        })
      });
    }
    callback (null, user);
  } catch (error) {
    callback (enrichError (error, {
      service: 'UserService',
      method: 'getUser',
      requestId: call.metadata.get('request-id')[0]
    }));
  }
}
\`\`\`

**7. Monitoring & Alerting**

\`\`\`javascript
const prometheus = require('prom-client');

const grpcDuration = new prometheus.Histogram({
  name: 'grpc_request_duration_seconds',
  help: 'gRPC request duration',
  labelNames: ['service', 'method', 'status',]
});

const grpcErrors = new prometheus.Counter({
  name: 'grpc_errors_total',
  help: 'gRPC errors',
  labelNames: ['service', 'method', 'code',]
});

// Interceptor
function monitoringInterceptor (call, callback, next) {
  const start = Date.now();
  
  next (call, (err, response) => {
    const duration = (Date.now() - start) / 1000;
    
    grpcDuration.labels(
      call.getPath(),
      err ? 'error' : 'success'
    ).observe (duration);
    
    if (err) {
      grpcErrors.labels(
        call.getPath(),
        err.code
      ).inc();
    }
    
    callback (err, response);
  });
}
\`\`\`

**Best Practice Stack**:

1. **Timeouts**: Always set deadlines
2. **Retries**: Exponential backoff, only transient errors
3. **Circuit breaker**: Fail fast when service down
4. **Health checks**: Verify before calling
5. **Monitoring**: Track errors, latencies, retries
6. **Graceful degradation**: Fallbacks when possible
7. **Idempotency**: Safe to retry operations

Trade-off: More reliability mechanisms = more complexity. Start with timeouts and retries, add circuit breakers as needed.`,
    keyPoints: [
      'Implement exponential backoff retry for transient failures',
      'Always set deadlines to prevent hanging requests',
      'Use circuit breakers to fail fast when services are down',
      'Propagate deadlines through service call chains',
      'Monitor errors and latencies for proactive alerting',
    ],
  },
  {
    id: 'grpc-d3',
    question:
      'Compare gRPC and REST for different scenarios: public API, internal microservices, mobile app, and IoT devices. Which would you choose for each and why?',
    sampleAnswer: `Detailed comparison for different use cases:

**Scenario 1: Public API (Third-Party Developers)**

**Winner: REST**

Reasons:
- Browser support without proxies
- Human-readable JSON for debugging
- Easy to test (curl, Postman, browser)
- Documentation tools (Swagger/OpenAPI)
- HTTP caching (CDN, browser cache)
- Familiar to most developers

gRPC challenges:
- Requires gRPC-Web proxy for browsers
- Binary format hard to debug
- Less familiar to external developers

Example: Stripe, GitHub, Twilio all use REST for public APIs.

**Scenario 2: Internal Microservices**

**Winner: gRPC**

Reasons:
- 7-10x faster (binary protobuf vs JSON)
- Strong typing prevents bugs
- Code generation for multiple languages
- Native bidirectional streaming
- Built-in load balancing
- Efficient for high-traffic internal communication

REST advantages:
- Simpler debugging
- HTTP caching

Trade-off: Performance and type safety > debugging convenience for internal use.

Example: Netflix, Uber use gRPC for internal microservices.

**Implementation**:
\`\`\`protobuf
// Product Service
service ProductService {
  rpc GetProduct(ProductRequest) returns (Product);
  rpc ListProducts(ListRequest) returns (stream Product);
}

// Inventory Service (calls Product Service)
service InventoryService {
  rpc CheckStock(StockRequest) returns (StockResponse);
}
\`\`\`

**Scenario 3: Mobile App (iOS/Android)**

**Winner: gRPC (with considerations)**

Reasons:
- Smaller payload sizes (battery/bandwidth)
- Binary format faster to parse
- Bidirectional streaming for real-time features
- Official support for mobile platforms
- Connection reuse (HTTP/2)

Considerations:
- Initial payload larger (code generation)
- More complex setup than REST
- Debugging harder

REST advantages:
- Simpler, less code
- Easier debugging

Recommendation: gRPC for apps with high data transfer or real-time features, REST for simple CRUD apps.

Example: Google apps use gRPC internally.

**Scenario 4: IoT Devices (Constrained Resources)**

**Winner: gRPC (but consider MQTT)**

Reasons:
- Smaller payloads (critical for limited bandwidth)
- Binary protocol efficient
- HTTP/2 multiplexing reduces connections
- Stream data efficiently

Challenges:
- Memory overhead for protobuf
- TLS required (compute intensive)

Alternative: MQTT for pub/sub patterns

REST challenges:
- JSON parsing expensive
- Larger payloads
- More network overhead

Recommendation: gRPC for direct device-to-cloud, MQTT for device-to-device pub/sub.

**Scenario 5: Real-Time Features (Chat, Collaboration)**

**Winner: gRPC**

Reasons:
- Native bidirectional streaming
- Lower latency than REST + SSE/WebSockets
- Same infrastructure as other services

REST approach:
- WebSockets (different protocol)
- Server-Sent Events (one-way)
- Both require separate infrastructure

Example:
\`\`\`protobuf
service ChatService {
  rpc Chat (stream Message) returns (stream Message);
}
\`\`\`

**Scenario 6: File Upload/Download**

**Winner: REST**

Reasons:
- Native HTTP multipart/form-data
- Progress tracking simpler
- Resume broken uploads
- Direct CDN integration

gRPC approach:
- Stream in chunks (more complex)
- No standard multipart support
- Custom progress tracking

Example: AWS S3, Dropbox use REST/HTTP for uploads.

**Scenario 7: Third-Party Webhook Integration**

**Winner: REST**

Reasons:
- Webhooks are HTTP POST requests
- Easy for third parties to send
- No special client libraries needed
- Ubiquitous support

gRPC: Not suitable (third parties unlikely to have gRPC clients).

**Summary Table**:

| Use Case | Choice | Primary Reason |
|----------|--------|----------------|
| Public API | REST | Browser support, debugging |
| Microservices | gRPC | Performance, type safety |
| Mobile app | gRPC | Efficiency, streaming |
| IoT devices | gRPC | Small payloads, binary |
| Real-time | gRPC | Bidirectional streaming |
| File uploads | REST | Native HTTP support |
| Webhooks | REST | Universal HTTP support |
| Admin dashboard | REST | Debugging ease |

**Hybrid Approach** (Best of Both):

Many companies offer both:
\`\`\`
External: REST API (public developers)
Internal: gRPC (microservices)
Mobile: gRPC (performance)
Web: REST (browser native)
\`\`\`

Example: Google offers both REST and gRPC for most services.

**Decision Framework**:

1. **Performance critical?** → gRPC
2. **Browser required?** → REST
3. **External developers?** → REST
4. **Real-time/streaming?** → gRPC
5. **Simple CRUD?** → REST
6. **Type safety critical?** → gRPC
7. **Debugging ease?** → REST

Choose based on specific constraints, not dogma. Many successful systems use both.`,
    keyPoints: [
      'REST for public APIs (browser support, ease of use)',
      'gRPC for internal microservices (performance, type safety)',
      'gRPC for mobile apps (efficiency, real-time features)',
      'REST for file uploads and webhooks (native HTTP support)',
      'Hybrid approach common: both REST and gRPC in same system',
    ],
  },
];
