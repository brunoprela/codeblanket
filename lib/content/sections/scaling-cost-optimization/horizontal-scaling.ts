export const horizontalScaling = {
  title: 'Horizontal Scaling',
  content: `

# Horizontal Scaling for LLM Applications

## Introduction

Horizontal scaling (also known as scaling out) is the process of adding more machines or instances to your infrastructure to handle increased load, as opposed to vertical scaling (scaling up) which involves adding more resources to a single machine. For LLM applications, horizontal scaling is crucial because:

- LLM API calls have variable latency and can be long-running
- User traffic patterns can be unpredictable and spiky
- Single servers have natural limits on concurrent connections
- Production systems need high availability and fault tolerance
- Cost efficiency requires elastic scaling based on demand

Unlike traditional web applications, LLM applications have unique challenges: API rate limits, token-based costs, streaming responses, and stateful conversations. This section covers how to design and implement horizontally scalable LLM architectures that handle these challenges while maintaining performance and reliability.

---

## Understanding Horizontal vs Vertical Scaling

### Vertical Scaling (Scaling Up)
- Adding more CPU, RAM, or GPU to a single machine
- Simpler to implement initially
- Has physical limits (maximum machine size)
- Single point of failure
- More expensive at scale
- Downtime required for upgrades

### Horizontal Scaling (Scaling Out)
- Adding more machines to distribute load
- Theoretically unlimited scaling
- Better fault tolerance (no single point of failure)
- More complex architecture required
- Cost-effective at scale
- Zero-downtime deployments possible

For LLM applications, horizontal scaling is almost always the better choice because:
1. **API Call Distribution**: Multiple servers can make parallel API calls
2. **Rate Limit Management**: Distribute requests across multiple API keys
3. **Geographic Distribution**: Serve users from nearby regions
4. **Fault Tolerance**: Continue operating if some servers fail
5. **Cost Optimization**: Scale up during peak hours, down during quiet periods

---

## Stateless Architecture Design

The foundation of horizontal scaling is **stateless design**. Each request should be self-contained and not depend on server-specific state.

### Stateless vs Stateful Servers

**Stateful (❌ Hard to Scale)**:
\`\`\`python
# BAD: Server stores conversation history in memory
class ChatServer:
    def __init__(self):
        # Conversation history stored on this server
        self.conversations = {}
    
    def chat(self, user_id: str, message: str):
        # If this user hits a different server, history is lost!
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({"role": "user", "content": message})
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.conversations[user_id]
        )
        
        self.conversations[user_id].append({
            "role": "assistant", 
            "content": response.choices[0].message.content
        })
        
        return response
\`\`\`

**Stateless (✅ Scales Horizontally)**:
\`\`\`python
# GOOD: State stored in external database
import redis
import json
from typing import List, Dict

class StatelessChatServer:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Fetch conversation from Redis (shared across all servers)"""
        history = self.redis.get(f"conversation:{user_id}")
        return json.loads(history) if history else []
    
    def save_conversation_history(self, user_id: str, messages: List[Dict]):
        """Save conversation to Redis with expiration"""
        self.redis.setex(
            f"conversation:{user_id}",
            3600,  # 1 hour TTL
            json.dumps(messages)
        )
    
    async def chat(self, user_id: str, message: str):
        # Fetch history from shared storage
        messages = self.get_conversation_history(user_id)
        messages.append({"role": "user", "content": message})
        
        # Make LLM call (stateless)
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages
        )
        
        # Update shared storage
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        self.save_conversation_history(user_id, messages)
        
        return response
\`\`\`

### Key Principles for Stateless Design

1. **External State Storage**: Use Redis, databases, or object storage for all state
2. **No In-Memory Caching**: Or use distributed caching (Redis, Memcached)
3. **Idempotent Operations**: Same request produces same result
4. **No Session Affinity Dependency**: Any server can handle any request
5. **Shared Configuration**: Config stored in environment variables or config service

---

## Load Balancing Strategies

Load balancers distribute incoming requests across multiple servers. For LLM applications, choosing the right strategy is crucial.

### Load Balancing Algorithms

#### 1. Round Robin
Distributes requests evenly across all servers in rotation.

\`\`\`python
# Simple round-robin implementation
class RoundRobinBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current = 0
    
    def get_next_server(self) -> str:
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Usage
balancer = RoundRobinBalancer([
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
])

# Requests distributed: server1 → server2 → server3 → server1 → ...
for _ in range(5):
    print(balancer.get_next_server())
\`\`\`

**Pros**: Simple, fair distribution  
**Cons**: Doesn't account for server load or response times

#### 2. Least Connections
Sends requests to the server with fewest active connections.

\`\`\`python
# Least connections balancer
from collections import defaultdict
import asyncio

class LeastConnectionsBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.active_connections = defaultdict(int)
    
    def get_next_server(self) -> str:
        # Choose server with least connections
        return min(self.servers, key=lambda s: self.active_connections[s])
    
    async def make_request(self, endpoint: str, **kwargs):
        server = self.get_next_server()
        self.active_connections[server] += 1
        
        try:
            # Make actual request
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{server}{endpoint}", **kwargs)
            return response
        finally:
            self.active_connections[server] -= 1
\`\`\`

**Pros**: Accounts for actual server load  
**Cons**: Requires tracking connection state

#### 3. Weighted Round Robin
Distributes based on server capacity (more powerful servers get more requests).

\`\`\`python
class WeightedRoundRobinBalancer:
    def __init__(self, servers: Dict[str, int]):
        # servers = {"server1": 3, "server2": 2, "server3": 1}
        # server1 gets 3/6 of traffic, server2 gets 2/6, server3 gets 1/6
        self.weighted_servers = []
        for server, weight in servers.items():
            self.weighted_servers.extend([server] * weight)
        self.current = 0
    
    def get_next_server(self) -> str:
        server = self.weighted_servers[self.current]
        self.current = (self.current + 1) % len(self.weighted_servers)
        return server

# Usage
balancer = WeightedRoundRobinBalancer({
    "http://large-server:8000": 5,   # 5x capacity
    "http://medium-server:8000": 3,  # 3x capacity
    "http://small-server:8000": 2    # 2x capacity
})
\`\`\`

**Pros**: Respects server capabilities  
**Cons**: Static weights don't adapt to real-time load

#### 4. Consistent Hashing (for caching)
Maps users to specific servers consistently (useful for cache locality).

\`\`\`python
import hashlib

class ConsistentHashBalancer:
    def __init__(self, servers: List[str], replicas: int = 3):
        self.servers = servers
        self.replicas = replicas
        self.ring = {}
        self._build_ring()
    
    def _build_ring(self):
        """Build hash ring with virtual nodes"""
        for server in self.servers:
            for i in range(self.replicas):
                key = f"{server}:{i}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                self.ring[hash_value] = server
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_server_for_key(self, key: str) -> str:
        """Get consistent server for a given key (e.g., user_id)"""
        if not self.ring:
            return None
        
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find first server with hash >= key hash
        for ring_key in self.sorted_keys:
            if ring_key >= hash_value:
                return self.ring[ring_key]
        
        # Wrap around to first server
        return self.ring[self.sorted_keys[0]]

# Usage - same user always goes to same server (cache locality)
balancer = ConsistentHashBalancer([
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
])

# User alice will always hit the same server
print(balancer.get_server_for_key("user_alice"))  # Always same result
print(balancer.get_server_for_key("user_bob"))    # Might be different server
\`\`\`

**Pros**: Cache locality, minimal redistribution when servers change  
**Cons**: Can lead to uneven load distribution

### Production Load Balancer Setup

\`\`\`python
# FastAPI application with health checks
from fastapi import FastAPI, Request
import httpx
import asyncio
import time

app = FastAPI()

# Server health tracking
class HealthChecker:
    def __init__(self, servers: List[str], check_interval: int = 30):
        self.servers = servers
        self.healthy_servers = set(servers)
        self.check_interval = check_interval
        self.last_check = {}
    
    async def check_health(self, server: str) -> bool:
        """Check if server is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{server}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def run_health_checks(self):
        """Continuously monitor server health"""
        while True:
            for server in self.servers:
                is_healthy = await self.check_health(server)
                
                if is_healthy:
                    if server not in self.healthy_servers:
                        print(f"✅ Server {server} is now healthy")
                    self.healthy_servers.add(server)
                else:
                    if server in self.healthy_servers:
                        print(f"❌ Server {server} is unhealthy")
                    self.healthy_servers.discard(server)
                
                self.last_check[server] = time.time()
            
            await asyncio.sleep(self.check_interval)
    
    def get_healthy_servers(self) -> List[str]:
        return list(self.healthy_servers)

# Production-ready load balancer
class ProductionLoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.health_checker = HealthChecker(servers)
        self.balancer = LeastConnectionsBalancer(servers)
    
    async def start(self):
        """Start health checking in background"""
        asyncio.create_task(self.health_checker.run_health_checks())
    
    async def forward_request(self, endpoint: str, **kwargs):
        """Forward request to healthy server"""
        healthy_servers = self.health_checker.get_healthy_servers()
        
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        # Update balancer with current healthy servers
        self.balancer.servers = healthy_servers
        
        # Forward request
        return await self.balancer.make_request(endpoint, **kwargs)

# Initialize load balancer
load_balancer = ProductionLoadBalancer([
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
])

@app.on_event("startup")
async def startup():
    await load_balancer.start()

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """Proxy chat requests to backend servers"""
    body = await request.json()
    response = await load_balancer.forward_request("/api/chat", json=body)
    return response.json()
\`\`\`

---

## Auto-Scaling Strategies

Auto-scaling automatically adjusts the number of servers based on demand.

### Metrics-Based Auto-Scaling

\`\`\`python
# Auto-scaler based on metrics
import boto3
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ScalingMetrics:
    cpu_percent: float
    memory_percent: float
    active_requests: int
    avg_response_time: float
    queue_depth: int

class AutoScaler:
    def __init__(
        self,
        min_instances: int = 2,
        max_instances: int = 10,
        target_cpu: float = 70.0,
        target_response_time: float = 2.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_period: int = 300  # 5 minutes
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu
        self.target_response_time = target_response_time
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.last_scaling_action = None
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should add more instances"""
        return (
            metrics.cpu_percent > self.scale_up_threshold or
            metrics.avg_response_time > self.target_response_time * 2 or
            metrics.queue_depth > 100
        )
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we can remove instances"""
        return (
            metrics.cpu_percent < self.scale_down_threshold and
            metrics.avg_response_time < self.target_response_time * 0.5 and
            metrics.queue_depth < 10
        )
    
    def in_cooldown(self) -> bool:
        """Check if we're in cooldown period"""
        if not self.last_scaling_action:
            return False
        
        time_since_last = datetime.now() - self.last_scaling_action
        return time_since_last.total_seconds() < self.cooldown_period
    
    def calculate_desired_instances(
        self, 
        current_instances: int, 
        metrics: ScalingMetrics
    ) -> int:
        """Calculate how many instances we should have"""
        if self.in_cooldown():
            return current_instances
        
        if self.should_scale_up(metrics):
            # Scale up by 50% or add at least 1
            desired = max(
                current_instances + 1,
                int(current_instances * 1.5)
            )
            desired = min(desired, self.max_instances)
            
            if desired > current_instances:
                self.last_scaling_action = datetime.now()
                print(f"📈 Scaling UP: {current_instances} → {desired}")
            
            return desired
        
        elif self.should_scale_down(metrics):
            # Scale down by 25%
            desired = max(
                current_instances - 1,
                int(current_instances * 0.75)
            )
            desired = max(desired, self.min_instances)
            
            if desired < current_instances:
                self.last_scaling_action = datetime.now()
                print(f"📉 Scaling DOWN: {current_instances} → {desired}")
            
            return desired
        
        return current_instances

# Usage with AWS ECS
class ECSAutoScaler:
    def __init__(self, cluster_name: str, service_name: str):
        self.ecs = boto3.client('ecs')
        self.cloudwatch = boto3.client('cloudwatch')
        self.cluster_name = cluster_name
        self.service_name = service_name
        self.autoscaler = AutoScaler()
    
    def get_current_metrics(self) -> ScalingMetrics:
        """Fetch current metrics from CloudWatch"""
        # Get CPU utilization
        cpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName='CPUUtilization',
            Dimensions=[
                {'Name': 'ServiceName', 'Value': self.service_name},
                {'Name': 'ClusterName', 'Value': self.cluster_name}
            ],
            StartTime=datetime.now() - timedelta(minutes=5),
            EndTime=datetime.now(),
            Period=300,
            Statistics=['Average']
        )
        
        avg_cpu = cpu_response['Datapoints'][0]['Average'] if cpu_response['Datapoints'] else 0
        
        # Get memory utilization
        memory_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName='MemoryUtilization',
            Dimensions=[
                {'Name': 'ServiceName', 'Value': self.service_name},
                {'Name': 'ClusterName', 'Value': self.cluster_name}
            ],
            StartTime=datetime.now() - timedelta(minutes=5),
            EndTime=datetime.now(),
            Period=300,
            Statistics=['Average']
        )
        
        avg_memory = memory_response['Datapoints'][0]['Average'] if memory_response['Datapoints'] else 0
        
        return ScalingMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            active_requests=0,  # Would need custom metric
            avg_response_time=0,  # Would need custom metric
            queue_depth=0  # Would need custom metric
        )
    
    def get_current_instance_count(self) -> int:
        """Get current number of running tasks"""
        response = self.ecs.describe_services(
            cluster=self.cluster_name,
            services=[self.service_name]
        )
        return response['services'][0]['desiredCount']
    
    def scale_to(self, desired_count: int):
        """Scale ECS service to desired count"""
        self.ecs.update_service(
            cluster=self.cluster_name,
            service=self.service_name,
            desiredCount=desired_count
        )
        print(f"✅ Scaled to {desired_count} instances")
    
    async def run_autoscaling_loop(self):
        """Continuously monitor and scale"""
        while True:
            try:
                metrics = self.get_current_metrics()
                current_count = self.get_current_instance_count()
                desired_count = self.autoscaler.calculate_desired_instances(
                    current_count, 
                    metrics
                )
                
                if desired_count != current_count:
                    self.scale_to(desired_count)
                
            except Exception as e:
                print(f"❌ Autoscaling error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
\`\`\`

### Predictive Auto-Scaling

\`\`\`python
# Predict traffic patterns and scale proactively
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class PredictiveAutoScaler:
    def __init__(self):
        self.model = LinearRegression()
        self.history = []
    
    def record_traffic(self, timestamp: datetime, request_count: int):
        """Record traffic for learning"""
        self.history.append({
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'request_count': request_count
        })
    
    def train_model(self):
        """Train model on historical data"""
        if len(self.history) < 168:  # Need at least 1 week of data
            return False
        
        df = pd.DataFrame(self.history)
        
        X = df[['hour', 'day_of_week']]
        y = df['request_count']
        
        self.model.fit(X, y)
        return True
    
    def predict_traffic(self, future_timestamp: datetime) -> float:
        """Predict traffic at a future time"""
        X = np.array([[
            future_timestamp.hour,
            future_timestamp.weekday()
        ]])
        
        return self.model.predict(X)[0]
    
    def get_proactive_scaling_plan(self) -> List[Dict]:
        """Generate scaling plan for next 4 hours"""
        plan = []
        current_time = datetime.now()
        
        for i in range(4):
            future_time = current_time + timedelta(hours=i+1)
            predicted_traffic = self.predict_traffic(future_time)
            
            # Estimate required instances (1 instance per 100 req/min)
            required_instances = max(2, int(predicted_traffic / 100) + 1)
            
            plan.append({
                'time': future_time,
                'predicted_traffic': predicted_traffic,
                'required_instances': required_instances
            })
        
        return plan
\`\`\`

---

## Zero-Downtime Deployments

Deploy new code without service interruption.

### Rolling Update Strategy

\`\`\`python
# Rolling update implementation
class RollingDeployment:
    def __init__(
        self, 
        servers: List[str],
        health_check_url: str = "/health",
        batch_size: int = 1,
        health_check_retries: int = 10
    ):
        self.servers = servers
        self.health_check_url = health_check_url
        self.batch_size = batch_size
        self.health_check_retries = health_check_retries
    
    async def check_server_health(self, server: str) -> bool:
        """Check if server is healthy"""
        for attempt in range(self.health_check_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{server}{self.health_check_url}",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(5)  # Wait before retry
        
        return False
    
    async def deploy_to_server(self, server: str, new_version: str):
        """Deploy new version to a single server"""
        print(f"🔄 Deploying version {new_version} to {server}")
        
        # 1. Remove server from load balancer
        await self.remove_from_load_balancer(server)
        print(f"  ⏸️  Removed from load balancer")
        
        # 2. Wait for existing connections to drain
        await asyncio.sleep(30)
        print(f"  ⏳ Drained connections")
        
        # 3. Deploy new code (example: Docker container)
        await self.deploy_new_container(server, new_version)
        print(f"  📦 Deployed new container")
        
        # 4. Health check the new deployment
        is_healthy = await self.check_server_health(server)
        if not is_healthy:
            print(f"  ❌ Health check failed! Rolling back...")
            await self.rollback_server(server)
            raise Exception(f"Deployment to {server} failed")
        
        print(f"  ✅ Health check passed")
        
        # 5. Add server back to load balancer
        await self.add_to_load_balancer(server)
        print(f"  ▶️  Added back to load balancer")
        
        return True
    
    async def rolling_deploy(self, new_version: str):
        """Deploy to all servers in rolling fashion"""
        print(f"🚀 Starting rolling deployment of version {new_version}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total servers: {len(self.servers)}")
        
        # Deploy in batches
        for i in range(0, len(self.servers), self.batch_size):
            batch = self.servers[i:i + self.batch_size]
            print(f"\n📦 Deploying batch {i // self.batch_size + 1}")
            
            # Deploy to all servers in batch concurrently
            tasks = [self.deploy_to_server(server, new_version) for server in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if any deployments failed
            for server, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"❌ Deployment failed on {server}: {result}")
                    print("🛑 Stopping deployment")
                    return False
            
            print(f"✅ Batch deployed successfully")
            
            # Wait before next batch
            if i + self.batch_size < len(self.servers):
                print("⏳ Waiting before next batch...")
                await asyncio.sleep(60)
        
        print(f"\n🎉 Rolling deployment of {new_version} completed successfully!")
        return True

# Usage
deployer = RollingDeployment(
    servers=[
        "http://server1:8000",
        "http://server2:8000", 
        "http://server3:8000",
        "http://server4:8000"
    ],
    batch_size=2  # Deploy 2 servers at a time
)

# Deploy new version
await deployer.rolling_deploy("v2.1.0")
\`\`\`

### Blue-Green Deployment

\`\`\`python
# Blue-Green deployment strategy
class BlueGreenDeployment:
    def __init__(self, load_balancer_url: str):
        self.load_balancer_url = load_balancer_url
        self.blue_servers = []
        self.green_servers = []
        self.active_environment = "blue"
    
    async def deploy_to_green(self, new_version: str):
        """Deploy new version to green environment (inactive)"""
        print(f"🟢 Deploying version {new_version} to GREEN environment")
        
        # Deploy to all green servers
        for server in self.green_servers:
            await self.deploy_server(server, new_version)
            print(f"  ✅ Deployed to {server}")
        
        # Health check all green servers
        all_healthy = True
        for server in self.green_servers:
            if not await self.check_health(server):
                print(f"  ❌ Health check failed: {server}")
                all_healthy = False
        
        if not all_healthy:
            raise Exception("Green environment health check failed")
        
        print("✅ GREEN environment is healthy and ready")
        return True
    
    async def switch_to_green(self):
        """Switch traffic from blue to green"""
        print("🔄 Switching traffic: BLUE → GREEN")
        
        # Update load balancer to point to green
        await self.update_load_balancer_target(self.green_servers)
        
        self.active_environment = "green"
        print("✅ Traffic switched to GREEN")
    
    async def rollback_to_blue(self):
        """Rollback to blue if green has issues"""
        print("⚠️  Rolling back: GREEN → BLUE")
        
        await self.update_load_balancer_target(self.blue_servers)
        
        self.active_environment = "blue"
        print("✅ Rolled back to BLUE")
    
    async def full_deployment(self, new_version: str):
        """Complete blue-green deployment with rollback capability"""
        try:
            # 1. Deploy to inactive environment (green)
            await self.deploy_to_green(new_version)
            
            # 2. Run smoke tests on green
            print("🧪 Running smoke tests on GREEN...")
            smoke_test_passed = await self.run_smoke_tests(self.green_servers)
            
            if not smoke_test_passed:
                raise Exception("Smoke tests failed on green")
            
            # 3. Switch traffic to green
            await self.switch_to_green()
            
            # 4. Monitor for issues
            print("👀 Monitoring GREEN for 5 minutes...")
            await asyncio.sleep(300)
            
            # 5. Check error rates
            error_rate = await self.get_error_rate()
            if error_rate > 0.01:  # More than 1% errors
                print(f"⚠️  High error rate: {error_rate:.2%}")
                await self.rollback_to_blue()
                return False
            
            print("🎉 Deployment successful!")
            
            # 6. Blue is now old version, make it the new "green" for next deployment
            self.blue_servers, self.green_servers = self.green_servers, self.blue_servers
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            await self.rollback_to_blue()
            return False
\`\`\`

---

## Session Affinity (Sticky Sessions)

When stateless design isn't possible, use session affinity carefully.

\`\`\`python
# Session affinity implementation
import hashlib

class SessionAffinityBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.server_weights = {server: 1.0 for server in servers}
    
    def get_server_for_session(self, session_id: str) -> str:
        """Route session to consistent server"""
        # Hash session ID to choose server
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        index = hash_value % len(self.servers)
        return self.servers[index]
    
    async def handle_request_with_affinity(
        self, 
        session_id: str,
        endpoint: str,
        **kwargs
    ):
        """Route request based on session"""
        server = self.get_server_for_session(session_id)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server}{endpoint}",
                cookies={"session_id": session_id},
                **kwargs
            )
        
        return response

# FastAPI with session affinity
from fastapi import FastAPI, Cookie
from typing import Optional

app = FastAPI()
balancer = SessionAffinityBalancer([
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
])

@app.post("/api/chat")
async def chat(
    message: str,
    session_id: Optional[str] = Cookie(None)
):
    if not session_id:
        # Create new session
        session_id = str(uuid.uuid4())
    
    # Route to appropriate server based on session
    response = await balancer.handle_request_with_affinity(
        session_id=session_id,
        endpoint="/api/chat",
        json={"message": message}
    )
    
    return response.json()
\`\`\`

**⚠️ Warning**: Session affinity makes scaling harder. If a server goes down, all sessions on that server are lost. Prefer stateless design when possible.

---

## Production Example: Complete Horizontally Scaled System

\`\`\`python
# Complete production-ready horizontally scaled LLM application
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
import asyncio
import httpx
from typing import List, Dict, Optional
import time
import json

# Request/Response models
class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    response_time: float
    server_id: str

# Production-grade horizontally scaled chat service
class HorizontallyScaledChatService:
    def __init__(
        self,
        redis_url: str,
        server_id: str,
        openai_api_key: str
    ):
        self.redis = None
        self.redis_url = redis_url
        self.server_id = server_id
        self.openai_api_key = openai_api_key
        
        # Metrics for auto-scaling
        self.active_requests = 0
        self.total_requests = 0
        self.response_times = []
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await redis.from_url(self.redis_url)
    
    async def get_conversation(self, conversation_id: str) -> List[Dict]:
        """Fetch conversation from shared Redis"""
        data = await self.redis.get(f"conversation:{conversation_id}")
        return json.loads(data) if data else []
    
    async def save_conversation(self, conversation_id: str, messages: List[Dict]):
        """Save conversation to shared Redis with 1 hour TTL"""
        await self.redis.setex(
            f"conversation:{conversation_id}",
            3600,
            json.dumps(messages)
        )
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat request (stateless, can run on any server)"""
        start_time = time.time()
        self.active_requests += 1
        self.total_requests += 1
        
        try:
            # Generate conversation ID if new
            conversation_id = request.conversation_id or f"conv_{int(time.time())}_{request.user_id}"
            
            # Fetch conversation history from shared storage
            messages = await self.get_conversation(conversation_id)
            messages.append({"role": "user", "content": request.message})
            
            # Call OpenAI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": messages,
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
            
            assistant_message = response.json()["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_message})
            
            # Save updated conversation
            await self.save_conversation(conversation_id, messages)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=assistant_message,
                response_time=response_time,
                server_id=self.server_id
            )
            
        finally:
            self.active_requests -= 1
    
    def get_metrics(self) -> Dict:
        """Get metrics for auto-scaling decisions"""
        return {
            "server_id": self.server_id,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "timestamp": time.time()
        }

# FastAPI application
app = FastAPI()
chat_service = None

@app.on_event("startup")
async def startup():
    global chat_service
    import os
    chat_service = HorizontallyScaledChatService(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        server_id=os.getenv("SERVER_ID", "server-1"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await chat_service.connect()

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - stateless, works on any server"""
    return await chat_service.chat(request)

@app.get("/health")
async def health():
    """Health check for load balancer"""
    return {
        "status": "healthy",
        "server_id": chat_service.server_id,
        "active_requests": chat_service.active_requests
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring/auto-scaling"""
    return chat_service.get_metrics()

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
\`\`\`

---

## Best Practices for Horizontal Scaling

### 1. Design for Failure
- Assume servers will fail
- Use health checks and automatic removal
- Implement retry logic
- Use circuit breakers

### 2. Monitor Everything
- Request rates
- Response times
- Error rates
- Resource utilization (CPU, memory)
- Queue depths

### 3. Test at Scale
- Load test before production
- Simulate server failures
- Test auto-scaling behavior
- Measure actual costs

### 4. Optimize Costs
- Use spot instances for batch processing
- Scale down during low traffic
- Use reserved instances for baseline capacity
- Monitor and optimize API costs

### 5. Security
- Use VPCs and security groups
- Encrypt data in transit and at rest
- Rotate API keys regularly
- Implement rate limiting

---

## Summary

Horizontal scaling is essential for production LLM applications. Key takeaways:

- **Stateless Design**: Store state externally (Redis, databases)
- **Load Balancing**: Distribute requests effectively across servers
- **Auto-Scaling**: Automatically adjust capacity based on demand
- **Zero Downtime**: Deploy without interrupting service
- **Health Checks**: Continuously monitor server health
- **Metrics**: Track everything for informed scaling decisions

With proper horizontal scaling, your LLM application can handle massive traffic while maintaining performance and controlling costs.

`,
  exercises: [
    {
      prompt:
        'Build a stateless chat service that stores conversation history in Redis. Test that conversations work correctly when requests go to different servers.',
      solution: `
# Provided in the complete example above
# Key points:
# 1. Store all state in Redis (external to servers)
# 2. Use conversation_id to fetch history
# 3. Test by running multiple server instances
# 4. Verify same conversation works across servers
`,
    },
    {
      prompt:
        'Implement a least-connections load balancer that tracks active requests and routes new requests to the server with the fewest connections.',
      solution: `
# Implementation shown in LeastConnectionsBalancer class above
# Test by sending concurrent requests and verifying distribution
`,
    },
    {
      prompt:
        'Create an auto-scaler that scales up when average response time exceeds 3 seconds and scales down when CPU drops below 20% for 5 minutes.',
      solution: `
# Extend AutoScaler class with custom thresholds
# Add response time tracking to metrics
# Implement cooldown to prevent flapping
`,
    },
  ],
};
