/**
 * RPC (Remote Procedure Call) Section
 */

export const rpcremoteprocedurecallSection = {
  id: 'rpc-remote-procedure-call',
  title: 'RPC (Remote Procedure Call)',
  content: `RPC (Remote Procedure Call) is a protocol that allows a program to execute procedures on a remote system as if they were local function calls. Understanding RPC is crucial for designing distributed systems, microservices, and high-performance APIs.

## What is RPC?

**RPC (Remote Procedure Call)** enables a client to call functions on a remote server using the same syntax as local function calls.

**Conceptual Flow**:
\`\`\`
Client Code:
  result = calculateTax(amount, state)
  
Behind the Scenes:
  1. Client stub marshals parameters (serializes)
  2. Message sent over network
  3. Server stub unmarshals parameters
  4. Server executes calculateTax(amount, state)
  5. Result marshaled and sent back
  6. Client receives and unmarshals result
\`\`\`

**Key Characteristics**:
- **Transparency**: Calling remote functions looks like calling local functions
- **Synchronous**: Typically blocking (client waits for response)
- **Strongly Typed**: Interface defined via IDL (Interface Definition Language)
- **Binary Protocol**: Usually more efficient than REST/JSON

---

## Popular RPC Frameworks

### **1. gRPC (Google RPC)**

**Overview**: Modern, high-performance RPC framework using HTTP/2 and Protocol Buffers.

**Key Features**:
- Uses Protocol Buffers (protobuf) for serialization
- HTTP/2 for transport (multiplexing, streaming)
- Support for 4 communication patterns
- Strong typing with code generation
- Built-in authentication, load balancing, health checking

**Example Proto File**:
\`\`\`protobuf
// user.proto
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUser(CreateUserRequest) returns (User);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
  int32 age = 3;
}
\`\`\`

**Node.js gRPC Server**:
\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;

// Implement service methods
function getUser(call, callback) {
  const userId = call.request.id;
  
  // Fetch from database
  const user = {
    id: userId,
    name: 'John Doe',
    email: 'john@example.com',
    age: 30
  };
  
  callback(null, user);
}

function listUsers(call) {
  const users = [
    { id: '1', name: 'Alice', email: 'alice@example.com', age: 25 },
    { id: '2', name: 'Bob', email: 'bob@example.com', age: 30 },
    { id: '3', name: 'Charlie', email: 'charlie@example.com', age: 35 }
  ];
  
  // Stream users one by one
  users.forEach(user => call.write(user));
  call.end();
}

function createUser(call, callback) {
  const newUser = {
    id: generateId(),
    name: call.request.name,
    email: call.request.email,
    age: call.request.age
  };
  
  // Save to database
  callback(null, newUser);
}

// Create and start server
const server = new grpc.Server();
server.addService(userProto.UserService.service, {
  getUser,
  listUsers,
  createUser
});

server.bindAsync(
  '0.0.0.0:50051',
  grpc.ServerCredentials.createInsecure(),
  () => {
    console.log('gRPC server running on port 50051');
    server.start();
  }
);
\`\`\`

**Node.js gRPC Client**:
\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;

const client = new userProto.UserService(
  'localhost:50051',
  grpc.credentials.createInsecure()
);

// Unary RPC (single request, single response)
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    console.error('Error:', error);
    return;
  }
  console.log('User:', user);
});

// Server streaming RPC
const call = client.listUsers({ page_size: 10 });
call.on('data', (user) => {
  console.log('Received user:', user);
});
call.on('end', () => {
  console.log('All users received');
});
call.on('error', (error) => {
  console.error('Stream error:', error);
});

// Create user
client.createUser(
  { name: 'Jane Doe', email: 'jane@example.com', age: 28 },
  (error, user) => {
    if (error) {
      console.error('Error:', error);
      return;
    }
    console.log('Created user:', user);
  }
);
\`\`\`

---

### **2. Apache Thrift**

**Overview**: Cross-language RPC framework developed by Facebook, supports multiple protocols and transports.

**Key Features**:
- Multiple protocols (binary, compact, JSON)
- Multiple transports (TCP, HTTP, framed)
- Code generation for many languages
- Flexible and extensible

**Thrift IDL Example**:
\`\`\`thrift
// user.thrift
namespace js UserService

struct User {
  1: string id,
  2: string name,
  3: string email,
  4: i32 age
}

exception UserNotFound {
  1: string message
}

service UserService {
  User getUser(1: string id) throws (1: UserNotFound notFound),
  list<User> listUsers(1: i32 pageSize),
  User createUser(1: string name, 2: string email, 3: i32 age)
}
\`\`\`

---

### **3. JSON-RPC**

**Overview**: Lightweight RPC protocol using JSON for encoding.

**Key Features**:
- Simple and human-readable
- Transport-agnostic (HTTP, WebSocket, TCP)
- Language-agnostic
- No code generation required

**JSON-RPC Request**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "method": "user.getUser",
  "params": {
    "id": "123"
  },
  "id": 1
}
\`\`\`

**JSON-RPC Response**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "result": {
    "id": "123",
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
  },
  "id": 1
}
\`\`\`

**JSON-RPC Error Response**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "User not found"
  },
  "id": 1
}
\`\`\`

---

## gRPC Communication Patterns

### **1. Unary RPC** (Request-Response)

Single request → Single response (like REST)

\`\`\`protobuf
rpc GetUser(GetUserRequest) returns (User);
\`\`\`

**Use Cases**:
- CRUD operations
- Simple queries
- Most common pattern

---

### **2. Server Streaming RPC**

Single request → Stream of responses

\`\`\`protobuf
rpc ListUsers(ListUsersRequest) returns (stream User);
\`\`\`

**Use Cases**:
- Large result sets
- Real-time updates
- File downloads

**Example**:
\`\`\`javascript
// Server
function listUsers(call) {
  const users = fetchAllUsers(); // Could be millions
  
  users.forEach(user => {
    call.write(user); // Stream one by one
  });
  
  call.end();
}

// Client
const call = client.listUsers({});
call.on('data', (user) => {
  processUser(user); // Process as they arrive
});
\`\`\`

---

### **3. Client Streaming RPC**

Stream of requests → Single response

\`\`\`protobuf
rpc UploadFile(stream FileChunk) returns (UploadResponse);
\`\`\`

**Use Cases**:
- File uploads
- Batch operations
- Collecting metrics

**Example**:
\`\`\`javascript
// Client
const call = client.uploadFile((error, response) => {
  console.log('Upload complete:', response);
});

// Stream file chunks
fileChunks.forEach(chunk => {
  call.write(chunk);
});

call.end();

// Server
function uploadFile(call, callback) {
  const chunks = [];
  
  call.on('data', (chunk) => {
    chunks.push(chunk);
  });
  
  call.on('end', () => {
    const file = Buffer.concat(chunks);
    saveFile(file);
    callback(null, { success: true, size: file.length });
  });
}
\`\`\`

---

### **4. Bidirectional Streaming RPC**

Stream of requests ↔ Stream of responses (independent)

\`\`\`protobuf
rpc Chat(stream ChatMessage) returns (stream ChatMessage);
\`\`\`

**Use Cases**:
- Chat applications
- Real-time collaboration
- Live gaming

**Example**:
\`\`\`javascript
// Client
const call = client.chat();

// Send messages
call.write({ user: 'Alice', message: 'Hello!' });
call.write({ user: 'Alice', message: 'How are you?' });

// Receive messages
call.on('data', (message) => {
  console.log(\`\${message.user}: \${message.message}\`);
});

// Server
function chat(call) {
  call.on('data', (message) => {
    // Broadcast to all connected clients
    broadcastToAll(message);
  });
  
  call.on('end', () => {
    call.end();
  });
}
\`\`\`

---

## RPC vs REST vs GraphQL

| **Aspect** | **RPC (gRPC)** | **REST** | **GraphQL** |
|------------|----------------|----------|-------------|
| **Protocol** | HTTP/2, Binary (Protobuf) | HTTP/1.1, JSON | HTTP, JSON |
| **Performance** | ⚡ Very Fast (binary, multiplexing) | 🐢 Slower (text, sequential) | 🏃 Medium (single endpoint) |
| **Type Safety** | ✅ Strong (code generation) | ❌ Weak (manual validation) | ✅ Strong (schema) |
| **Discoverability** | ⚠️ Requires documentation | ✅ Self-documenting (HATEOAS) | ✅ Self-documenting (introspection) |
| **Caching** | ❌ Difficult (HTTP/2, binary) | ✅ Easy (HTTP caching) | ⚠️ Moderate (requires work) |
| **Streaming** | ✅ Native support | ❌ No native support | ⚠️ Via subscriptions |
| **Browser Support** | ⚠️ Requires gRPC-Web | ✅ Native | ✅ Native |
| **Learning Curve** | ⚠️ Steeper | ✅ Easy | ⚠️ Moderate |
| **Best For** | Microservices, internal APIs | Public APIs, CRUD | Complex queries, mobile |

---

## When to Use RPC

### **✅ Use RPC When:**

1. **Internal Microservices Communication**
   - Services within same organization
   - Strong typing and performance critical
   - Example: Order Service → Inventory Service

2. **High-Performance Requirements**
   - Low latency critical
   - High throughput needed
   - Example: Trading systems, real-time analytics

3. **Streaming Data**
   - Server streaming (logs, metrics)
   - Client streaming (file uploads)
   - Bidirectional (chat, collaboration)

4. **Polyglot Environments**
   - Multiple programming languages
   - Need consistent interfaces
   - Code generation valuable

5. **Complex Operations**
   - Actions that don't map to CRUD
   - Procedure-oriented APIs
   - Example: processOrder(), runAnalysis()

### **❌ Avoid RPC When:**

1. **Public APIs**
   - External developers need access
   - REST more familiar/accessible
   - Caching important

2. **Browser-Only Clients**
   - gRPC-Web adds complexity
   - REST/GraphQL more natural
   - Unless using gRPC-Web proxy

3. **Simple CRUD Operations**
   - REST sufficient
   - No performance requirements
   - Standard HTTP caching desired

4. **Debugging/Testing Priority**
   - Binary protocols harder to inspect
   - curl/Postman not usable
   - Tools less mature

---

## RPC Error Handling

### **gRPC Status Codes**:

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// Common status codes
grpc.status.OK                  // 0 - Success
grpc.status.CANCELLED           // 1 - Operation cancelled
grpc.status.UNKNOWN             // 2 - Unknown error
grpc.status.INVALID_ARGUMENT    // 3 - Invalid argument
grpc.status.DEADLINE_EXCEEDED   // 4 - Timeout
grpc.status.NOT_FOUND           // 5 - Not found
grpc.status.ALREADY_EXISTS      // 6 - Already exists
grpc.status.PERMISSION_DENIED   // 7 - Permission denied
grpc.status.UNAUTHENTICATED     // 16 - Not authenticated
grpc.status.UNAVAILABLE         // 14 - Service unavailable
grpc.status.INTERNAL            // 13 - Internal error
\`\`\`

**Server-Side Error Handling**:
\`\`\`javascript
function getUser(call, callback) {
  const userId = call.request.id;
  
  if (!userId) {
    return callback({
      code: grpc.status.INVALID_ARGUMENT,
      message: 'User ID is required'
    });
  }
  
  const user = database.findUser(userId);
  
  if (!user) {
    return callback({
      code: grpc.status.NOT_FOUND,
      message: \`User \${userId} not found\`
    });
  }
  
  callback(null, user);
}
\`\`\`

**Client-Side Error Handling with Retry**:
\`\`\`javascript
function getUserWithRetry(userId, maxRetries = 3) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    
    function attempt() {
      attempts++;
      
      client.getUser({ id: userId }, (error, user) => {
        if (!error) {
          return resolve(user);
        }
        
        // Retry on transient errors
        if (
          error.code === grpc.status.UNAVAILABLE ||
          error.code === grpc.status.DEADLINE_EXCEEDED
        ) {
          if (attempts < maxRetries) {
            const delay = Math.min(1000 * Math.pow(2, attempts), 10000);
            console.log(\`Retry \${attempts} after \${delay}ms\`);
            setTimeout(attempt, delay);
            return;
          }
        }
        
        // Non-retryable error or max retries exceeded
        reject(error);
      });
    }
    
    attempt();
  });
}
\`\`\`

---

## RPC Performance Optimization

### **1. Connection Pooling**

\`\`\`javascript
// Bad: New connection per request
function makeRequest() {
  const client = new UserService('localhost:50051');
  client.getUser({ id: '123' }, callback);
}

// Good: Reuse connection
const client = new UserService('localhost:50051');

function makeRequest() {
  client.getUser({ id: '123' }, callback);
}
\`\`\`

### **2. Timeouts and Deadlines**

\`\`\`javascript
// Set deadline (absolute time)
const deadline = new Date();
deadline.setSeconds(deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  (error, user) => {
    if (error && error.code === grpc.status.DEADLINE_EXCEEDED) {
      console.error('Request timed out');
    }
  }
);
\`\`\`

### **3. Compression**

\`\`\`javascript
// Enable compression
client.getUser(
  { id: '123' },
  {
    'grpc.default_compression_algorithm': grpc.compressionAlgorithms.gzip,
    'grpc.default_compression_level': grpc.compressionLevels.high
  },
  callback
);
\`\`\`

### **4. Multiplexing**

HTTP/2 automatically multiplexes multiple RPC calls over single TCP connection:

\`\`\`javascript
// All these calls use same connection automatically
client.getUser({ id: '1' }, callback1);
client.getUser({ id: '2' }, callback2);
client.getUser({ id: '3' }, callback3);
// No connection overhead! HTTP/2 multiplexes.
\`\`\`

---

## RPC Security

### **1. TLS/SSL Encryption**

\`\`\`javascript
// Server with TLS
const credentials = grpc.ServerCredentials.createSsl(
  fs.readFileSync('ca.crt'),
  [{
    cert_chain: fs.readFileSync('server.crt'),
    private_key: fs.readFileSync('server.key')
  }]
);

server.bindAsync('0.0.0.0:50051', credentials, () => {
  server.start();
});

// Client with TLS
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.crt')
);

const client = new UserService('localhost:50051', credentials);
\`\`\`

### **2. Authentication with Metadata**

\`\`\`javascript
// Client: Send auth token
const metadata = new grpc.Metadata();
metadata.add('authorization', 'Bearer eyJhbGc...');

client.getUser({ id: '123' }, metadata, callback);

// Server: Verify token
function getUser(call, callback) {
  const metadata = call.metadata;
  const authHeader = metadata.get('authorization')[0];
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Missing or invalid token'
    });
  }
  
  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    // Proceed with request
    callback(null, user);
  } catch (error) {
    callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Invalid token'
    });
  }
}
\`\`\`

### **3. Interceptors for Cross-Cutting Concerns**

\`\`\`javascript
// Client interceptor (add auth to all requests)
function authInterceptor(options, nextCall) {
  return new grpc.InterceptingCall(nextCall(options), {
    start: (metadata, listener, next) => {
      metadata.add('authorization', \`Bearer \${getAuthToken()}\`);
      next(metadata, listener);
    }
  });
}

const client = new UserService(
  'localhost:50051',
  credentials,
  { interceptors: [authInterceptor] }
);

// Server interceptor (logging, auth)
function loggingInterceptor(call, callback) {
  const start = Date.now();
  const method = call.handler.path;
  
  console.log(\`[RPC] \${method} started\`);
  
  // Call original method
  return (originalCall, originalCallback) => {
    originalCallback((error, response) => {
      const duration = Date.now() - start;
      console.log(\`[RPC] \${method} completed in \${duration}ms\`);
      callback(error, response);
    });
  };
}
\`\`\`

---

## Load Balancing RPC Services

### **Client-Side Load Balancing**:

\`\`\`javascript
// gRPC supports DNS-based load balancing
const client = new UserService(
  'dns:///userservice.example.com:50051',
  credentials
);

// Round-robin across all resolved IPs
\`\`\`

### **Server-Side Load Balancing with Envoy**:

\`\`\`yaml
# envoy.yaml
static_resources:
  listeners:
    - address:
        socket_address:
          address: 0.0.0.0
          port_value: 50051
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                http2_protocol_options: {}
                route_config:
                  virtual_hosts:
                    - name: userservice
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route:
                            cluster: userservice_cluster
  clusters:
    - name: userservice_cluster
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      http2_protocol_options: {}
      load_assignment:
        cluster_name: userservice_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: userservice1
                      port_value: 50051
              - endpoint:
                  address:
                    socket_address:
                      address: userservice2
                      port_value: 50051
              - endpoint:
                  address:
                    socket_address:
                      address: userservice3
                      port_value: 50051
\`\`\`

---

## Common RPC Mistakes

### **❌ Mistake 1: No Timeout/Deadline**

\`\`\`javascript
// Bad: No timeout
client.getUser({ id: '123' }, callback);
// If server hangs, client waits forever

// Good: Always set deadline
const deadline = Date.now() + 5000; // 5 seconds
client.getUser({ id: '123' }, { deadline }, callback);
\`\`\`

### **❌ Mistake 2: Not Handling Errors Properly**

\`\`\`javascript
// Bad: Treating all errors the same
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    throw error; // Don't retry transient errors!
  }
});

// Good: Retry transient errors
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    if (error.code === grpc.status.UNAVAILABLE) {
      // Retry with backoff
      retryWithBackoff();
    } else if (error.code === grpc.status.NOT_FOUND) {
      // Don't retry, return 404
      return res.status(404).send('User not found');
    } else {
      // Log and return 500
      logger.error(error);
      return res.status(500).send('Internal error');
    }
  }
});
\`\`\`

### **❌ Mistake 3: Creating New Client Per Request**

\`\`\`javascript
// Bad: Connection overhead
app.get('/user/:id', (req, res) => {
  const client = new UserService('localhost:50051');
  client.getUser({ id: req.params.id }, callback);
});

// Good: Reuse client
const client = new UserService('localhost:50051');

app.get('/user/:id', (req, res) => {
  client.getUser({ id: req.params.id }, callback);
});
\`\`\`

### **❌ Mistake 4: Not Using Streaming for Large Data**

\`\`\`javascript
// Bad: Load all users in memory
rpc GetUsers(Empty) returns (UserList); // Contains array of ALL users

// Good: Stream users
rpc GetUsers(Empty) returns (stream User); // Stream one at a time
\`\`\`

### **❌ Mistake 5: No Monitoring/Observability**

Always add:
- Request duration metrics
- Error rate by status code
- Request rate (QPS)
- Connection pool size

---

## Real-World Example: Microservices with gRPC

**Scenario**: E-commerce platform with Order Service, Inventory Service, Payment Service.

**Architecture**:
\`\`\`
API Gateway (REST)
      |
      v
Order Service (gRPC)
      |
      +---> Inventory Service (gRPC)
      |
      +---> Payment Service (gRPC)
\`\`\`

**order.proto**:
\`\`\`protobuf
syntax = "proto3";

import "inventory.proto";
import "payment.proto";

service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  rpc GetOrder(GetOrderRequest) returns (Order);
}

message CreateOrderRequest {
  string user_id = 1;
  repeated OrderItem items = 2;
  PaymentInfo payment_info = 3;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}

message Order {
  string order_id = 1;
  string user_id = 2;
  repeated OrderItem items = 3;
  string status = 4;
  double total = 5;
}
\`\`\`

**Order Service Implementation**:
\`\`\`javascript
async function createOrder(call, callback) {
  const { user_id, items, payment_info } = call.request;
  
  try {
    // 1. Check inventory
    const inventoryClient = getInventoryClient();
    const inventoryCheck = await new Promise((resolve, reject) => {
      inventoryClient.checkAvailability(
        { items },
        { deadline: Date.now() + 2000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    if (!inventoryCheck.available) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Items not available'
      });
    }
    
    // 2. Process payment
    const paymentClient = getPaymentClient();
    const payment = await new Promise((resolve, reject) => {
      paymentClient.processPayment(
        {
          amount: inventoryCheck.total,
          payment_info
        },
        { deadline: Date.now() + 5000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    if (!payment.success) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Payment failed'
      });
    }
    
    // 3. Reserve inventory
    await new Promise((resolve, reject) => {
      inventoryClient.reserveItems(
        { items, order_id: generateOrderId() },
        { deadline: Date.now() + 2000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    // 4. Create order
    const order = {
      order_id: generateOrderId(),
      user_id,
      items,
      status: 'confirmed',
      total: inventoryCheck.total
    };
    
    await saveOrder(order);
    
    callback(null, order);
    
  } catch (error) {
    logger.error('Order creation failed:', error);
    
    // Rollback if needed
    if (error.code === grpc.status.UNAVAILABLE) {
      return callback({
        code: grpc.status.UNAVAILABLE,
        message: 'Service temporarily unavailable'
      });
    }
    
    callback({
      code: grpc.status.INTERNAL,
      message: 'Failed to create order'
    });
  }
}
\`\`\`

**Benefits of Using gRPC Here**:
1. **Performance**: Binary protocol, HTTP/2 multiplexing
2. **Type Safety**: Protobuf definitions prevent errors
3. **Streaming**: Can stream order updates
4. **Polyglot**: Services can be in different languages
5. **Timeouts**: Built-in deadline handling
6. **Load Balancing**: Native support

---

## Key Takeaways

1. **RPC allows calling remote functions** as if they were local
2. **gRPC uses HTTP/2 + Protocol Buffers** for high performance
3. **4 communication patterns**: Unary, Server Streaming, Client Streaming, Bidirectional
4. **Best for internal microservices**, not public APIs
5. **Always set timeouts/deadlines** to prevent hanging
6. **Retry transient errors** (UNAVAILABLE, DEADLINE_EXCEEDED)
7. **Use TLS for security**, metadata for auth
8. **Streaming better than loading** all data in memory
9. **Connection pooling** critical for performance
10. **Trade-off: Performance vs ease of use** (gRPC vs REST)`,
};
