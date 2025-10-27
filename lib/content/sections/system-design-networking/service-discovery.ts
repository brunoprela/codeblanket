/**
 * Service Discovery Section
 */

export const servicediscoverySection = {
  id: 'service-discovery',
  title: 'Service Discovery',
  content: `Service Discovery is the process by which services in a distributed system locate and communicate with each other. Understanding service discovery is crucial for building scalable microservices architectures.
    
    ## What is Service Discovery?
    
    In a dynamic microservices environment, services are constantly starting, stopping, and moving. **Service Discovery** enables services to find each other without hardcoding network locations.
    
    **Without Service Discovery**:
    \`\`\`javascript
    // Hardcoded - breaks when service moves
    const orderService = 'http://order-service-1.internal:8080';
    const response = await fetch(\`\${orderService}/orders/123\`);
    \`\`\`
    
    **With Service Discovery**:
    \`\`\`javascript
    // Dynamic - always finds current location
    const orderService = serviceRegistry.get('order-service');
    const response = await fetch(\`\${orderService}/orders/123\`);
    \`\`\`
    
    ---
    
    ## Service Discovery Patterns
    
    ### **1. Client-Side Discovery**
    
    **Clients query service registry and load balance**:
    
    \`\`\`
    Client
      ↓
      1. Query service registry for "order-service"
      ↓
    Service Registry (returns: [ip1, ip2, ip3])
      ↓
      2. Client chooses ip2 (round-robin)
      ↓
    Order Service Instance 2 (ip2)
    \`\`\`
    
    **Pros**:
    - No extra network hop
    - Client controls load balancing
    - Simple architecture
    
    **Cons**:
    - Clients must implement discovery logic
    - Couples clients to registry
    - Language-specific client libraries
    
    **Example: Netflix Eureka**:
    \`\`\`javascript
    const Eureka = require('eureka-js-client').Eureka;
    
    const client = new Eureka({
      instance: {
        app: 'user-service',
        hostName: 'localhost',
        ipAddr: '127.0.0.1',
        port: { '$': 3000, '@enabled': true },
        vipAddress: 'user-service',
        dataCenterInfo: {
          '@class': 'com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo',
          name: 'MyOwn'
        }
      },
      eureka: {
        host: 'eureka-server',
        port: 8761,
        servicePath: '/eureka/apps/'
      }
    });
    
    client.start();
    
    // Get service instances
    function getService (serviceName) {
      const instances = client.getInstancesByAppId (serviceName);
      // Client-side load balancing (round-robin)
      const instance = instances[Math.floor(Math.random() * instances.length)];
      return \`http://\${instance.ipAddr}:\${instance.port['$']}\`;
    }
    
    const orderServiceUrl = getService('ORDER-SERVICE');
    \`\`\`
    
    ### **2. Server-Side Discovery**
    
    **Load balancer queries registry**:
    
    \`\`\`
    Client
      ↓
      Request to load-balancer.internal
      ↓
    Load Balancer
      ↓
      1. Query service registry for "order-service"
      ↓
    Service Registry (returns: [ip1, ip2, ip3])
      ↓
      2. Load balancer chooses ip2
      ↓
    Order Service Instance 2 (ip2)
    \`\`\`
    
    **Pros**:
    - Clients don't need discovery logic
    - Centralized load balancing
    - Language-agnostic
    
    **Cons**:
    - Extra network hop
    - Load balancer is single point of failure
    - More complex infrastructure
    
    **Example: AWS ELB + ECS**:
    - ECS registers tasks with ELB
    - Clients call ELB DNS name
    - ELB routes to healthy instances
    
    ---
    
    ## Service Registry Patterns
    
    ### **1. Self-Registration**
    
    **Services register themselves**:
    
    \`\`\`javascript
    // Service startup
    const serviceInfo = {
      name: 'user-service',
      id: uuidv4(),
      address: process.env.HOST,
      port: process.env.PORT,
      health: '/health'
    };
    
    await consul.agent.service.register (serviceInfo);
    
    // Heartbeat to maintain registration
    setInterval (async () => {
      await consul.agent.check.pass(\`service:\${serviceInfo.id}\`);
    }, 10000);
    
    // Deregister on shutdown
    process.on('SIGTERM', async () => {
      await consul.agent.service.deregister (serviceInfo.id);
      process.exit(0);
    });
    \`\`\`
    
    **Pros**:
    - Simple - service manages its own lifecycle
    - No external registration logic
    
    **Cons**:
    - Services must implement registration logic
    - Language-specific clients needed
    
    ### **2. Third-Party Registration**
    
    **External registrar registers services**:
    
    \`\`\`
    Kubernetes
      ↓
      Watches service deployments
      ↓
    Registrar (sidecar)
      ↓
      Registers/deregisters with service registry
      ↓
    Consul/Eureka/etcd
    \`\`\`
    
    **Example: Kubernetes + Service**:
    \`\`\`yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: user-service
    spec:
      selector:
        app: user-service
      ports:
        - protocol: TCP
          port: 80
          targetPort: 3000
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: user-service
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: user-service
      template:
        metadata:
          labels:
            app: user-service
        spec:
          containers:
            - name: user-service
              image: user-service:1.0.0
              ports:
                - containerPort: 3000
    \`\`\`
    
    **Kubernetes automatically**:
    - Registers pods with Service
    - Updates endpoints as pods start/stop
    - Provides DNS resolution (user-service.default.svc.cluster.local)
    - Load balances across pods
    
    ---
    
    ## Popular Service Discovery Tools
    
    ### **1. Consul (HashiCorp)**
    
    **Features**:
    - Service registry and health checking
    - Key-value store
    - Multi-datacenter support
    - Service mesh capabilities
    
    **Registration**:
    \`\`\`javascript
    const Consul = require('consul');
    const consul = new Consul({ host: 'consul.service.consul', port: 8500 });
    
    // Register service
    await consul.agent.service.register({
      id: 'user-service-1',
      name: 'user-service',
      address: '10.0.1.5',
      port: 3000,
      check: {
        http: 'http://10.0.1.5:3000/health',
        interval: '10s',
        timeout: '5s'
      }
    });
    
    // Query service
    const services = await consul.health.service({
      service: 'user-service',
      passing: true // Only healthy instances
    });
    
    services.forEach (service => {
      console.log(\`\${service.Service.Address}:\${service.Service.Port}\`);
    });
    \`\`\`
    
    **DNS Interface**:
    \`\`\`bash
    # Query via DNS
    dig @127.0.0.1 -p 8600 user-service.service.consul
    
    # Returns:
    user-service.service.consul. 0 IN A 10.0.1.5
    user-service.service.consul. 0 IN A 10.0.1.6
    user-service.service.consul. 0 IN A 10.0.1.7
    \`\`\`
    
    ### **2. etcd**
    
    **Features**:
    - Distributed key-value store
    - Used by Kubernetes
    - Strong consistency (Raft consensus)
    - Watch mechanism for real-time updates
    
    **Example**:
    \`\`\`javascript
    const { Etcd3 } = require('etcd3');
    const client = new Etcd3();
    
    // Register service with TTL lease
    const lease = client.lease(10); // 10 second TTL
    await lease.put('services/user-service/instance-1').value(JSON.stringify({
      address: '10.0.1.5',
      port: 3000
    }));
    
    // Keep-alive to maintain registration
    await lease.keepalive();
    
    // Query services
    const services = await client.getAll()
      .prefix('services/user-service/')
      .strings();
    
    console.log (services);
    \`\`\`
    
    ### **3. ZooKeeper**
    
    **Features**:
    - Distributed coordination service
    - Used by Kafka, Hadoop
    - Hierarchical namespace
    - Watches for changes
    
    **Example**:
    \`\`\`javascript
    const zookeeper = require('node-zookeeper-client');
    const client = zookeeper.createClient('localhost:2181');
    
    client.once('connected', async () => {
      // Create ephemeral node (disappears when client disconnects)
      await client.create(
        '/services/user-service/instance-1',
        Buffer.from(JSON.stringify({ address: '10.0.1.5', port: 3000 })),
        zookeeper.CreateMode.EPHEMERAL
      );
      
      // Watch for changes
      const children = await client.getChildren('/services/user-service', (event) => {
        console.log('Services changed:', event);
      });
    });
    
    client.connect();
    \`\`\`
    
    ### **4. Kubernetes Service Discovery**
    
    **Built-in DNS**:
    \`\`\`javascript
    // Services automatically get DNS names
    const response = await fetch('http://user-service.default.svc.cluster.local/users/123');
    
    // Short form within same namespace
    const response = await fetch('http://user-service/users/123');
    \`\`\`
    
    **Environment Variables**:
    \`\`\`javascript
    // Kubernetes injects service info
    const userServiceHost = process.env.USER_SERVICE_SERVICE_HOST;
    const userServicePort = process.env.USER_SERVICE_SERVICE_PORT;
    \`\`\`
    
    **API Server**:
    \`\`\`javascript
    const k8s = require('@kubernetes/client-node');
    const kc = new k8s.KubeConfig();
    kc.loadFromDefault();
    
    const k8sApi = kc.makeApiClient (k8s.CoreV1Api);
    
    // Get service endpoints
    const endpoints = await k8sApi.readNamespacedEndpoints('user-service', 'default');
    endpoints.body.subsets.forEach (subset => {
      subset.addresses.forEach (address => {
        console.log(\`\${address.ip}:\${subset.ports[0].port}\`);
      });
    });
    \`\`\`
    
    ---
    
    ## Health Checking
    
    **Critical for service discovery** - only route traffic to healthy instances.
    
    ### **Active Health Checks**:
    
    \`\`\`javascript
    // Health check endpoint
    app.get('/health', async (req, res) => {
      try {
        // Check dependencies
        await database.ping();
        await redis.ping();
        
        res.status(200).json({
          status: 'healthy',
          uptime: process.uptime(),
          timestamp: Date.now()
        });
      } catch (error) {
        res.status(503).json({
          status: 'unhealthy',
          error: error.message
        });
      }
    });
    
    // Consul health check
    await consul.agent.service.register({
      name: 'user-service',
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s',    // Check every 10 seconds
        timeout: '5s',      // Timeout after 5 seconds
        deregistercriticalserviceafter: '1m' // Deregister if unhealthy for 1 min
      }
    });
    \`\`\`
    
    ### **Passive Health Checks**:
    
    \`\`\`javascript
    // Track failures, remove unhealthy instances
    class CircuitBreaker {
      constructor (threshold = 5) {
        this.failureCount = new Map();
        this.threshold = threshold;
      }
      
      recordFailure (instanceId) {
        const count = this.failureCount.get (instanceId) || 0;
        this.failureCount.set (instanceId, count + 1);
        
        if (count + 1 >= this.threshold) {
          console.log(\`Instance \${instanceId} marked unhealthy\`);
          this.removeFromPool (instanceId);
        }
      }
      
      recordSuccess (instanceId) {
        this.failureCount.set (instanceId, 0);
      }
    }
    \`\`\`
    
    ---
    
    ## Load Balancing with Service Discovery
    
    ### **Client-Side Load Balancing**:
    
    \`\`\`javascript
    class ServiceClient {
      constructor (serviceName, consul) {
        this.serviceName = serviceName;
        this.consul = consul;
        this.instances = [];
        this.currentIndex = 0;
        
        // Refresh instances periodically
        this.refreshInstances();
        setInterval(() => this.refreshInstances(), 30000);
      }
      
      async refreshInstances() {
        const services = await this.consul.health.service({
          service: this.serviceName,
          passing: true
        });
        
        this.instances = services.map (s => ({
          address: s.Service.Address,
          port: s.Service.Port
        }));
      }
      
      // Round-robin load balancing
      getNext() {
        if (this.instances.length === 0) {
          throw new Error('No healthy instances');
        }
        
        const instance = this.instances[this.currentIndex];
        this.currentIndex = (this.currentIndex + 1) % this.instances.length;
        
        return \`http://\${instance.address}:\${instance.port}\`;
      }
      
      // Random load balancing
      getRandom() {
        const instance = this.instances[Math.floor(Math.random() * this.instances.length)];
        return \`http://\${instance.address}:\${instance.port}\`;
      }
      
      // Weighted load balancing
      getWeighted() {
        // Implement weighted random selection
        // (instances can have different weights based on capacity)
      }
    }
    
    // Usage
    const userService = new ServiceClient('user-service', consul);
    
    async function makeRequest (userId) {
      const url = userService.getNext();
      return await fetch(\`\${url}/users/\${userId}\`);
    }
    \`\`\`
    
    ---
    
    ## Service Mesh (Advanced Service Discovery)
    
    **Service mesh** adds a dedicated infrastructure layer for service-to-service communication.
    
    **Popular Service Meshes**:
    - **Istio** (Google/IBM/Lyft)
    - **Linkerd** (CNCF)
    - **Consul Connect** (HashiCorp)
    
    **Architecture**:
    \`\`\`
    Service A → Envoy Proxy (sidecar) → Envoy Proxy (sidecar) → Service B
                    ↓                              ↓
              Control Plane (Istio/Linkerd)
                    ↓
             Service Registry
    \`\`\`
    
    **Features**:
    - Automatic service discovery
    - Load balancing
    - Circuit breaking
    - Mutual TLS
    - Request tracing
    - Metrics collection
    - Traffic splitting (canary deployments)
    
    **Example: Istio Virtual Service**:
    \`\`\`yaml
    apiVersion: networking.istio.io/v1beta1
    kind: VirtualService
    metadata:
      name: user-service
    spec:
      hosts:
        - user-service
      http:
        - match:
            - headers:
                x-version:
                  exact: "v2"
          route:
            - destination:
                host: user-service
                subset: v2
        - route:
            - destination:
                host: user-service
                subset: v1
              weight: 90
            - destination:
                host: user-service
                subset: v2
              weight: 10
    \`\`\`
    
    ---
    
    ## When to Use Service Discovery
    
    ### **✅ Use Service Discovery When:**1. **Microservices Architecture**
       - Many services communicating
       - Services scale independently
       - Dynamic environments (containers, cloud)
    
    2. **Auto-Scaling**
       - Instances constantly added/removed
       - Need automatic registration/deregistration
    
    3. **Multi-Region Deployments**
       - Services in different regions
       - Geo-routing based on location
    
    4. **Zero-Downtime Deployments**
       - Rolling updates
       - Blue-green deployments
       - Canary releases
    
    ### **❌ Avoid Service Discovery When:**1. **Monolithic Application**
       - Single application
       - Static deployment
    
    2. **Small Number of Services**
       - 2-3 services
       - Rarely change
       - Can use environment variables
    
    3. **Simple Environment**
       - Single server
       - No auto-scaling
       - Fixed IP addresses
    
    ---
    
    ## Common Mistakes
    
    ### **❌ Mistake 1: No Health Checks**
    
    \`\`\`javascript
    // Bad: Register without health check
    consul.agent.service.register({
      name: 'user-service',
      port: 3000
    });
    // Service stays registered even if crashed!
    
    // Good: Always include health check
    consul.agent.service.register({
      name: 'user-service',
      port: 3000,
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s'
      }
    });
    \`\`\`
    
    ### **❌ Mistake 2: Caching Service Locations Too Long**
    
    \`\`\`javascript
    // Bad: Cache forever
    const userServiceUrl = await getService('user-service');
    // Service might have moved!
    
    // Good: Refresh periodically
    setInterval (async () => {
      this.cachedServices = await refreshServices();
    }, 30000); // 30 seconds
    \`\`\`
    
    ### **❌ Mistake 3: No Graceful Shutdown**
    
    \`\`\`javascript
    // Bad: Abrupt shutdown
    process.on('SIGTERM', () => {
      process.exit(0); // Requests in flight will fail!
    });
    
    // Good: Deregister, then drain
    process.on('SIGTERM', async () => {
      // 1. Deregister from service discovery
      await consul.agent.service.deregister (serviceId);
      
      // 2. Stop accepting new requests
      server.close();
      
      // 3. Wait for in-flight requests to complete
      await waitForRequestsToDrain();
      
      // 4. Exit
      process.exit(0);
    });
    \`\`\`
    
    ### **❌ Mistake 4: Single Point of Failure**
    
    \`\`\`javascript
    // Bad: Single Consul server
    const consul = new Consul({ host: 'consul-server' });
    
    // Good: Consul cluster with multiple nodes
    const consul = new Consul({
      host: 'consul.service.consul', // DNS round-robin to multiple servers
      promisify: true
    });
    \`\`\`
    
    ---
    
    ## Real-World Example: E-Commerce with Service Discovery
    
    **Architecture**:
    \`\`\`
                  Consul Cluster
                        ↓
        +--------------+---------------+--------------+
        |              |               |              |
    API Gateway   Order Service   User Service   Inventory Service
        |              |               |              |
      (3 instances) (5 instances)  (3 instances)  (2 instances)
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    // Order Service
    const consul = new Consul();
    
    // Register on startup
    await consul.agent.service.register({
      id: \`order-service-\${process.env.INSTANCE_ID}\`,
      name: 'order-service',
      address: process.env.HOST,
      port: parseInt (process.env.PORT),
      tags: ['http', 'v1'],
      check: {
        http: \`http://\${process.env.HOST}:\${process.env.PORT}/health\`,
        interval: '10s',
        timeout: '5s'
      }
    });
    
    // Service client for calling other services
    class MicroserviceClient {
      constructor (serviceName) {
        this.serviceName = serviceName;
        this.consul = new Consul();
      }
      
      async call (path, options = {}) {
        // Get healthy instances
        const services = await this.consul.health.service({
          service: this.serviceName,
          passing: true
        });
        
        if (services.length === 0) {
          throw new Error(\`No healthy instances of \${this.serviceName}\`);
        }
        
        // Round-robin
        const service = services[Math.floor(Math.random() * services.length)];
        const url = \`http://\${service.Service.Address}:\${service.Service.Port}\${path}\`;
        
        // Make request with retry
        let attempts = 0;
        while (attempts < 3) {
          try {
            const response = await fetch (url, options);
            if (!response.ok) throw new Error(\`HTTP \${response.status}\`);
            return await response.json();
          } catch (error) {
            attempts++;
            if (attempts >= 3) throw error;
            await new Promise (resolve => setTimeout (resolve, 100 * attempts));
          }
        }
      }
    }
    
    // Create order (calls inventory and user services)
    app.post('/orders', async (req, res) => {
      const { userId, items } = req.body;
      
      // Call user service to validate user
      const userClient = new MicroserviceClient('user-service');
      const user = await userClient.call(\`/users/\${userId}\`);
      
      // Call inventory service to check availability
      const inventoryClient = new MicroserviceClient('inventory-service');
      const availability = await inventoryClient.call('/check', {
        method: 'POST',
        body: JSON.stringify({ items })
      });
      
      if (!availability.available) {
        return res.status(400).json({ error: 'Items not available' });
      }
      
      // Create order
      const order = await db.orders.create({ userId, items });
      res.json (order);
    });
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **Service discovery enables dynamic service location** without hardcoded addresses
    2. **Two patterns**: Client-side (client queries registry) vs Server-side (load balancer queries)
    3. **Two registration patterns**: Self-registration vs Third-party registration
    4. **Popular tools**: Consul, etcd, ZooKeeper, Kubernetes DNS
    5. **Health checks are critical** - only route to healthy instances
    6. **Cache service locations** but refresh periodically (30-60 seconds)
    7. **Graceful shutdown**: Deregister → Stop accepting requests → Drain → Exit
    8. **Service mesh** adds advanced features (mTLS, observability, traffic management)
    9. **Load balancing strategies**: Round-robin, random, weighted, least connections
    10. **Essential for microservices** in dynamic, auto-scaling environments`,
};
