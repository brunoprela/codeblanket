/**
 * gRPC Service Design Section
 */

export const grpcservicedesignSection = {
  id: 'grpc-service-design',
  title: 'gRPC Service Design',
  content: `gRPC is a high-performance RPC framework using Protocol Buffers, ideal for microservices communication. Understanding gRPC design is essential for building efficient distributed systems.

## What is gRPC?

**gRPC** (gRPC Remote Procedure Call) is an open-source RPC framework developed by Google.

### **Key Characteristics**
- Uses **HTTP/2** for transport
- **Protocol Buffers** (protobuf) for serialization
- Supports **multiple languages**
- **Bidirectional streaming**
- Built-in **authentication**, **load balancing**, and **deadlines**

### **gRPC vs REST**

| Feature | REST | gRPC |
|---------|------|------|
| Protocol | HTTP/1.1 | HTTP/2 |
| Payload | JSON/XML (text) | Protobuf (binary) |
| Performance | Slower | Faster |
| Streaming | No (SSE hack) | Yes (native) |
| Browser support | Native | Requires proxy |
| Human-readable | Yes | No |
| Use case | Public APIs | Microservices |

## Protocol Buffers

**Protobuf** is a language-neutral data serialization format.

### **Define Schema**

\`\`\`protobuf
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUser(CreateUserRequest) returns (User);
  rpc UpdateUser(UpdateUserRequest) returns (User);
  rpc DeleteUser(DeleteUserRequest) returns (Empty);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
  repeated string tags = 5;
  google.protobuf.Timestamp created_at = 6;
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string role = 3;
}

message CreateUserInput {
  string name = 1;
  string email = 2;
  string password = 3;
}

message Empty {}
\`\`\`

**Field Numbers**: Permanent identifiers (1, 2, 3...) for backward compatibility.

## RPC Types

### **1. Unary RPC (Request-Response)**

Simple request-response pattern:

\`\`\`protobuf
rpc GetUser(GetUserRequest) returns (User);
\`\`\`

\`\`\`javascript
// Client
const response = await client.getUser({ id: '123' });
console.log (response.name);
\`\`\`

### **2. Server Streaming**

Client sends one request, server streams multiple responses:

\`\`\`protobuf
rpc ListUsers(ListUsersRequest) returns (stream User);
\`\`\`

\`\`\`javascript
// Client
const call = client.listUsers({ pageSize: 100 });

call.on('data', (user) => {
  console.log('Received user:', user.name);
});

call.on('end', () => {
  console.log('Stream ended');
});
\`\`\`

**Use cases**: Large result sets, real-time updates, log streaming

### **3. Client Streaming**

Client streams multiple requests, server sends one response:

\`\`\`protobuf
rpc UploadUsers (stream CreateUserRequest) returns (UploadSummary);
\`\`\`

\`\`\`javascript
// Client
const call = client.uploadUsers((err, response) => {
  console.log('Uploaded:', response.count);
});

users.forEach (user => call.write (user));
call.end();
\`\`\`

**Use cases**: Batch uploads, log aggregation

### **4. Bidirectional Streaming**

Both client and server stream:

\`\`\`protobuf
rpc Chat (stream ChatMessage) returns (stream ChatMessage);
\`\`\`

\`\`\`javascript
// Client
const call = client.chat();

call.on('data', (message) => {
  console.log('Received:', message.text);
});

call.write({ text: 'Hello!' });
call.write({ text: 'How are you?' });
\`\`\`

**Use cases**: Chat, real-time collaboration, gaming

## Error Handling

**gRPC Status Codes**:

\`\`\`
OK                 = 0   // Success
CANCELLED          = 1   // Client cancelled
INVALID_ARGUMENT   = 3   // Invalid request
NOT_FOUND          = 5   // Resource not found
ALREADY_EXISTS     = 6   // Resource exists
PERMISSION_DENIED  = 7   // No permission
RESOURCE_EXHAUSTED = 8   // Rate limit, quota
FAILED_PRECONDITION = 9  // System state issue
UNIMPLEMENTED      = 12  // Not implemented
INTERNAL           = 13  // Server error
UNAVAILABLE        = 14  // Service unavailable
UNAUTHENTICATED    = 16  // Not authenticated
\`\`\`

**Return Errors**:

\`\`\`javascript
// Server
async getUser (call, callback) {
  const { id } = call.request;
  
  const user = await db.users.findById (id);
  
  if (!user) {
    return callback({
      code: grpc.status.NOT_FOUND,
      message: 'User not found',
      details: \`User \${id} does not exist\`
    });
  }
  
  callback (null, user);
}
\`\`\`

**Error Details** (Rich Errors):

\`\`\`protobuf
import "google/rpc/error_details.proto";

message ErrorResponse {
  google.rpc.BadRequest bad_request = 1;
  google.rpc.RetryInfo retry_info = 2;
}
\`\`\`

## Metadata (Headers)

**Send Metadata**:

\`\`\`javascript
// Client
const metadata = new grpc.Metadata();
metadata.add('authorization', 'Bearer token123');
metadata.add('request-id', 'uuid-123');

client.getUser({ id: '123' }, metadata, (err, response) => {
  // ...
});
\`\`\`

**Receive Metadata**:

\`\`\`javascript
// Server
async getUser (call, callback) {
  const metadata = call.metadata;
  const authToken = metadata.get('authorization')[0];
  
  // Verify token
  const user = await authenticateToken (authToken);
  
  if (!user) {
    return callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Invalid token'
    });
  }
  
  // Process request
}
\`\`\`

## Deadlines and Timeouts

**Client-Side Deadline**:

\`\`\`javascript
// Timeout after 5 seconds
const deadline = new Date();
deadline.setSeconds (deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  (err, response) => {
    if (err && err.code === grpc.status.DEADLINE_EXCEEDED) {
      console.error('Request timed out');
    }
  }
);
\`\`\`

**Server Check**:

\`\`\`javascript
async getUser (call, callback) {
  // Check if client cancelled or deadline exceeded
  if (call.cancelled) {
    return callback({
      code: grpc.status.CANCELLED,
      message: 'Request cancelled'
    });
  }
  
  // Long operation
  const user = await expensiveQuery();
  
  callback (null, user);
}
\`\`\`

## Authentication

### **1. SSL/TLS**

\`\`\`javascript
// Server
const server = new grpc.Server();
const credentials = grpc.ServerCredentials.createSsl(
  fs.readFileSync('ca.pem'),
  [{
    cert_chain: fs.readFileSync('server-cert.pem'),
    private_key: fs.readFileSync('server-key.pem')
  }]
);

server.bindAsync('0.0.0.0:50051', credentials, () => {
  server.start();
});

// Client
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.pem')
);

const client = new UserServiceClient('localhost:50051', credentials);
\`\`\`

### **2. Token-Based (Metadata)**

\`\`\`javascript
// Client interceptor
const authInterceptor = (options, nextCall) => {
  return new grpc.InterceptingCall (nextCall (options), {
    start: (metadata, listener, next) => {
      metadata.add('authorization', \`Bearer \${getToken()}\`);
      next (metadata, listener);
    }
  });
};

const client = new UserServiceClient(
  'localhost:50051',
  credentials,
  { interceptors: [authInterceptor] }
);
\`\`\`

### **3. Mutual TLS (mTLS)**

Both client and server authenticate:

\`\`\`javascript
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.pem'),
  fs.readFileSync('client-key.pem'),  // Client cert
  fs.readFileSync('client-cert.pem')
);
\`\`\`

## Load Balancing

**Client-Side Load Balancing**:

\`\`\`javascript
// Round-robin across multiple servers
const client = new UserServiceClient(
  'dns:///service.example.com',  // Resolves to multiple IPs
  credentials,
  {
    'grpc.lb_policy_name': 'round_robin'
  }
);
\`\`\`

**Server-Side** (with proxy like Envoy)

## Interceptors (Middleware)

**Server Interceptor**:

\`\`\`javascript
const loggingInterceptor = (call, callback, next) => {
  console.log('Request:', call.request);
  const start = Date.now();
  
  next (call, (err, response) => {
    console.log('Duration:', Date.now() - start);
    callback (err, response);
  });
};

server.use (loggingInterceptor);
\`\`\`

**Client Interceptor**:

\`\`\`javascript
const retryInterceptor = (options, nextCall) => {
  return new grpc.InterceptingCall (nextCall (options), {
    start: (metadata, listener, next) => {
      const retryListener = {
        onReceiveStatus: (status, nextStatus) => {
          if (status.code === grpc.status.UNAVAILABLE) {
            // Retry logic
            return retry();
          }
          nextStatus (status);
        }
      };
      next (metadata, retryListener);
    }
  });
};
\`\`\`

## Real-World Example

**Service Definition**:

\`\`\`protobuf
syntax = "proto3";

package ecommerce;

service ProductService {
  rpc GetProduct(GetProductRequest) returns (Product);
  rpc SearchProducts(SearchRequest) returns (stream Product);
  rpc CreateOrder (stream OrderItem) returns (Order);
}

message Product {
  string id = 1;
  string name = 2;
  double price = 3;
  int32 stock = 4;
}

message GetProductRequest {
  string id = 1;
}

message SearchRequest {
  string query = 1;
  int32 limit = 2;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}

message Order {
  string id = 1;
  double total = 2;
  repeated OrderItem items = 3;
}
\`\`\`

## Best Practices

1. **Use protobuf field numbers wisely**: Never reuse, reserve deprecated ones
2. **Enable deadlines**: Prevent hanging requests
3. **Implement retries with backoff**: Handle transient failures
4. **Use streaming for large data**: Don't load everything in memory
5. **Secure with TLS**: Always in production
6. **Monitor with interceptors**: Logging, metrics, tracing
7. **Version your proto files**: Backward compatibility
8. **Document services**: Comments in proto files
9. **Use service mesh**: For advanced traffic management (Istio, Linkerd)
10. **Test with tools**: grpcurl, BloomRPC

## When to Use gRPC

**Use gRPC when**:
- Microservices communication (internal)
- High performance required
- Bidirectional streaming needed
- Type safety important
- Polyglot services (multiple languages)

**Use REST when**:
- Public APIs (browser access)
- Third-party integrations
- Human-readable format needed
- Simple request-response
- HTTP caching important`,
};
