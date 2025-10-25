export const multiRegionDeployment = {
  title: 'Multi-Region Deployment',
  content: `

# Multi-Region Deployment for LLM Applications

## Introduction

Multi-region deployment means running your application in multiple geographic regions (e.g., US-East, EU-West, Asia-Pacific) simultaneously. For LLM applications serving global users, this provides:

- **30-70% Latency Reduction**: Users connect to nearest region (200ms → 60ms)
- **99.99% Availability**: If one region fails, others continue serving
- **Regulatory Compliance**: Keep EU data in EU, China data in China
- **Disaster Recovery**: Geographic redundancy
- **Better User Experience**: Fast responses worldwide

**Trade-offs**: Increased complexity, higher costs, data synchronization challenges.

---

## Multi-Region Architecture Patterns

### Active-Passive (Failover)

Primary region handles all traffic, secondary regions on standby for disasters.

\`\`\`python
# Simple health check based failover
import requests
from typing import List, Optional

class FailoverManager:
    """Manage failover between regions"""
    
    def __init__(self, regions: List[dict]):
        self.regions = regions  # [{"name": "us-east-1", "url": "https://...", "priority": 1}, ...]
        self.current_region = None
        self._select_primary()
    
    def _select_primary (self):
        """Select highest priority healthy region"""
        for region in sorted (self.regions, key=lambda r: r["priority"]):
            if self._is_healthy (region):
                self.current_region = region
                print(f"✅ Using region: {region['name']}")
                return
        
        raise Exception("No healthy regions available!")
    
    def _is_healthy (self, region: dict) -> bool:
        """Check if region is healthy"""
        try:
            response = requests.get(
                f"{region['url']}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    async def make_request (self, endpoint: str, **kwargs):
        """Make request with automatic failover"""
        max_retries = len (self.regions)
        
        for attempt in range (max_retries):
            try:
                response = await self._call_region(
                    self.current_region,
                    endpoint,
                    **kwargs
                )
                return response
            except Exception as e:
                print(f"❌ {self.current_region['name']} failed: {e}")
                
                # Try next region
                self._select_primary()
                if attempt < max_retries - 1:
                    print(f"🔄 Failing over to {self.current_region['name']}")
                    continue
                raise

# Usage
failover = FailoverManager([
    {"name": "us-east-1", "url": "https://us-api.example.com", "priority": 1},
    {"name": "eu-west-1", "url": "https://eu-api.example.com", "priority": 2},
])

response = await failover.make_request("/api/chat", json={"prompt": "Hello"})
\`\`\`

**Pros**: Simple, cost-effective (only primary running)  
**Cons**: Slow failover (minutes), wasted capacity in standby regions

### Active-Active (Global Load Balancing)

All regions actively serve traffic simultaneously.

\`\`\`python
# Geographic load balancing
class GeographicLoadBalancer:
    """Route users to nearest region"""
    
    def __init__(self):
        self.regions = {
            "us-east-1": {
                "url": "https://us-api.example.com",
                "countries": ["US", "CA", "MX"]
            },
            "eu-west-1": {
                "url": "https://eu-api.example.com",
                "countries": ["GB", "FR", "DE", "IT", "ES"]
            },
            "ap-southeast-1": {
                "url": "https://asia-api.example.com",
                "countries": ["SG", "TH", "MY", "ID"]
            }
        }
    
    def get_region_for_country (self, country_code: str) -> str:
        """Determine best region for user's country"""
        for region, config in self.regions.items():
            if country_code in config["countries"]:
                return region
        
        # Default to US
        return "us-east-1"
    
    async def route_request (self, user_country: str, endpoint: str, **kwargs):
        """Route request to appropriate region"""
        region = self.get_region_for_country (user_country)
        region_url = self.regions[region]["url"]
        
        print(f"Routing {user_country} → {region}")
        
        response = await httpx.post(
            f"{region_url}{endpoint}",
            **kwargs
        )
        
        return response

# FastAPI integration
from fastapi import FastAPI, Request

app = FastAPI()
balancer = GeographicLoadBalancer()

@app.post("/api/chat")
async def chat (request: Request, body: dict):
    # Get user's country from CloudFlare header or GeoIP
    country = request.headers.get("cf-ipcountry", "US")
    
    # Route to nearest region
    response = await balancer.route_request(
        country,
        "/api/chat",
        json=body
    )
    
    return response.json()
\`\`\`

**Pros**: Fast global responses, no failover delay, better resource utilization  
**Cons**: Complex, data synchronization needed, higher costs

---

## AWS Multi-Region Deployment

### Route 53 for Global Routing

\`\`\`python
import boto3

# Configure Route 53 with health checks
route53 = boto3.client('route53')

# Create health checks for each region
health_checks = {}
for region in ['us-east-1', 'eu-west-1', 'ap-southeast-1']:
    response = route53.create_health_check(
        HealthCheckConfig={
            'Type': 'HTTPS',
            'ResourcePath': '/health',
            'FullyQualifiedDomainName': f'{region}.api.example.com',
            'Port': 443,
            'RequestInterval': 30,
            'FailureThreshold': 3,
        }
    )
    health_checks[region] = response['HealthCheck']['Id']

# Create latency-based routing
route53.change_resource_record_sets(
    HostedZoneId='Z1234567890ABC',
    ChangeBatch={
        'Changes': [
            {
                'Action': 'CREATE',
                'ResourceRecordSet': {
                    'Name': 'api.example.com',
                    'Type': 'A',
                    'SetIdentifier': 'us-east-1',
                    'Region': 'us-east-1',
                    'AliasTarget': {
                        'HostedZoneId': 'Z35SXDOTRQ7X7K',
                        'DNSName': 'us-alb.amazonaws.com',
                        'EvaluateTargetHealth': True
                    },
                    'HealthCheckId': health_checks['us-east-1']
                }
            },
            # Repeat for other regions...
        ]
    }
)
\`\`\`

### Cross-Region Database Replication

\`\`\`python
# Using Amazon RDS with cross-region read replicas
import boto3

rds = boto3.client('rds', region_name='us-east-1')

# Create read replica in another region
rds.create_db_instance_read_replica(
    DBInstanceIdentifier='mydb-eu-west-1-replica',
    SourceDBInstanceIdentifier='arn:aws:rds:us-east-1:123456789012:db:mydb',
    DBInstanceClass='db.r5.xlarge',
    AvailabilityZone='eu-west-1a',
    PubliclyAccessible=False,
)

# Application reads from local replica
class MultiRegionDatabase:
    """Access database with regional read replicas"""
    
    def __init__(self, region: str):
        self.region = region
        self.primary_endpoint = "mydb.us-east-1.rds.amazonaws.com"
        self.replica_endpoints = {
            "us-east-1": self.primary_endpoint,
            "eu-west-1": "mydb-eu-west-1-replica.rds.amazonaws.com",
            "ap-southeast-1": "mydb-ap-southeast-1-replica.rds.amazonaws.com"
        }
    
    def get_read_endpoint (self) -> str:
        """Get read endpoint for current region"""
        return self.replica_endpoints.get (self.region, self.primary_endpoint)
    
    def get_write_endpoint (self) -> str:
        """Always write to primary"""
        return self.primary_endpoint

# Usage
db = MultiRegionDatabase (region="eu-west-1")

# Reads from local replica (fast)
connection = psycopg2.connect(
    host=db.get_read_endpoint(),
    database="mydb",
    user="user",
    password="pass"
)

# Writes to primary (may be slower from EU)
write_connection = psycopg2.connect(
    host=db.get_write_endpoint(),
    database="mydb",
    user="user",
    password="pass"
)
\`\`\`

---

## Data Consistency Challenges

### Eventually Consistent Data

\`\`\`python
# Handle eventual consistency
class EventuallyConsistentCache:
    """Cache that handles multi-region replication lag"""
    
    def __init__(self, redis_clusters: dict):
        # Redis cluster per region
        self.clusters = {
            region: redis.from_url (url)
            for region, url in redis_clusters.items()
        }
        self.local_region = os.getenv("AWS_REGION", "us-east-1")
    
    async def set (self, key: str, value: str, ttl: int = 3600):
        """Write to local region"""
        local_redis = self.clusters[self.local_region]
        await local_redis.setex (key, ttl, value)
        
        # Async replication to other regions happens automatically
        # via Redis Enterprise or custom replication
    
    async def get (self, key: str, consistency: str = "eventual"):
        """Read with specified consistency level"""
        
        if consistency == "eventual":
            # Read from local region (may be stale)
            local_redis = self.clusters[self.local_region]
            return await local_redis.get (key)
        
        elif consistency == "strong":
            # Read from all regions, use most recent
            results = await asyncio.gather(*[
                cluster.get (key) 
                for cluster in self.clusters.values()
            ])
            
            # In practice, would compare timestamps/versions
            # For now, return first non-None result
            for result in results:
                if result:
                    return result
        
        return None

# Usage
cache = EventuallyConsistentCache({
    "us-east-1": "redis://us-redis:6379",
    "eu-west-1": "redis://eu-redis:6379",
})

# Write
await cache.set("user:123", json.dumps (user_data))

# Read with eventual consistency (fast, may be stale)
data = await cache.get("user:123", consistency="eventual")

# Read with strong consistency (slower, guaranteed fresh)
data = await cache.get("user:123", consistency="strong")
\`\`\`

---

## Regulatory Compliance (GDPR)

### Data Residency

\`\`\`python
# Ensure EU data stays in EU
class DataResidencyManager:
    """Enforce data residency requirements"""
    
    def __init__(self):
        self.region_requirements = {
            # EU users' data must stay in EU
            "EU": ["eu-west-1", "eu-central-1"],
            # China data must stay in China
            "CN": ["cn-north-1"],
            # Others can use any region
            "OTHER": ["us-east-1", "eu-west-1", "ap-southeast-1"]
        }
    
    def get_allowed_regions (self, user_country: str) -> List[str]:
        """Get regions where user's data can be stored"""
        
        if user_country in ["GB", "FR", "DE", "IT", "ES"]:
            return self.region_requirements["EU"]
        elif user_country == "CN":
            return self.region_requirements["CN"]
        else:
            return self.region_requirements["OTHER"]
    
    def validate_storage_region (self, user_country: str, storage_region: str):
        """Ensure storage region is compliant"""
        allowed = self.get_allowed_regions (user_country)
        
        if storage_region not in allowed:
            raise ValueError(
                f"Cannot store {user_country} data in {storage_region}. "
                f"Allowed regions: {allowed}"
            )

# FastAPI middleware
@app.middleware("http")
async def enforce_data_residency (request: Request, call_next):
    user_country = request.headers.get("cf-ipcountry", "US")
    current_region = os.getenv("AWS_REGION")
    
    # Check if this region can serve this user
    manager = DataResidencyManager()
    allowed_regions = manager.get_allowed_regions (user_country)
    
    if current_region not in allowed_regions:
        # Redirect to compliant region
        compliant_region = allowed_regions[0]
        redirect_url = f"https://{compliant_region}.api.example.com{request.url.path}"
        
        return JSONResponse(
            status_code=307,
            content={"redirect_to": redirect_url},
            headers={"Location": redirect_url}
        )
    
    return await call_next (request)
\`\`\`

---

## Cost Optimization for Multi-Region

### Intelligent Region Selection

\`\`\`python
# Balance cost and latency
class CostAwareRegionSelector:
    """Select region balancing cost and performance"""
    
    def __init__(self):
        # Region costs and latencies
        self.regions = {
            "us-east-1": {"cost_multiplier": 1.0, "base_latency": 50},
            "eu-west-1": {"cost_multiplier": 1.2, "base_latency": 60},
            "ap-southeast-1": {"cost_multiplier": 1.3, "base_latency": 70},
        }
    
    def select_region(
        self,
        user_country: str,
        user_tier: str,
        optimize_for: str = "latency"
    ) -> str:
        """Select best region for user"""
        
        if optimize_for == "cost" && user_tier == "free":
            # Free users get cheapest region
            return "us-east-1"
        
        elif optimize_for == "latency":
            # Premium users get lowest latency
            # In practice, use geographic distance
            if user_country in ["GB", "FR", "DE"]:
                return "eu-west-1"
            elif user_country in ["SG", "JP", "AU"]:
                return "ap-southeast-1"
            else:
                return "us-east-1"
        
        return "us-east-1"
\`\`\`

---

## Monitoring Multi-Region Systems

\`\`\`python
# Comprehensive multi-region monitoring
class MultiRegionMonitor:
    """Monitor health across all regions"""
    
    def __init__(self, regions: List[str]):
        self.regions = regions
    
    async def check_all_regions (self) -> dict:
        """Health check all regions"""
        results = {}
        
        tasks = [
            self._check_region (region)
            for region in self.regions
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for region, response in zip (self.regions, responses):
            if isinstance (response, Exception):
                results[region] = {
                    "healthy": False,
                    "error": str (response)
                }
            else:
                results[region] = {
                    "healthy": True,
                    "latency_ms": response["latency"],
                    "requests_per_sec": response["rps"]
                }
        
        return results
    
    async def _check_region (self, region: str) -> dict:
        """Check single region"""
        url = f"https://{region}.api.example.com/health"
        
        start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get (url, timeout=5.0)
        latency = (time.time() - start) * 1000
        
        data = response.json()
        
        return {
            "latency": latency,
            "rps": data.get("requests_per_sec", 0)
        }

# Dashboard
monitor = MultiRegionMonitor(["us-east-1", "eu-west-1", "ap-southeast-1"])

health = await monitor.check_all_regions()

print("Regional Health:")
for region, status in health.items():
    if status["healthy"]:
        print(f"  ✅ {region}: {status['latency_ms']:.0f}ms, {status['requests_per_sec']} req/s")
    else:
        print(f"  ❌ {region}: {status['error']}")
\`\`\`

---

## Best Practices

### 1. Start with Two Regions
- Primary + one failover region
- Add more as traffic grows
- Each region adds complexity

### 2. Use Managed Services
- AWS Route 53 for routing
- RDS cross-region replicas
- CloudFront for static content

### 3. Test Failover Regularly
- Run chaos engineering tests
- Simulate region failures
- Measure failover time

### 4. Monitor Replication Lag
- Track how far behind replicas are
- Alert on excessive lag (>1s)
- May need to block writes during lag

### 5. Consider Regulatory Requirements
- GDPR, data residency laws
- Some data must stay in region
- May need region-specific instances

---

## Summary

Multi-region deployment provides:

- **30-70% latency reduction** for global users
- **99.99% availability** through geographic redundancy
- **Regulatory compliance** by keeping data in specific regions
- **Disaster recovery** across regions

**Trade-offs**: Higher complexity, costs, data synchronization challenges

**Recommendations**:
- Start with active-passive (simpler)
- Move to active-active as you scale
- Use managed services (Route 53, RDS replicas)
- Test failover regularly

For global applications, multi-region deployment is essential for performance and reliability.

`,
  exercises: [
    {
      prompt:
        'Deploy your LLM application to two AWS regions with Route 53 latency-based routing. Measure latency improvement for users in different continents.',
      solution: `Deploy to us-east-1 and eu-west-1, configure Route 53 with health checks, test from US, Europe, Asia using services like Pingdom. Expected: 50-70% latency reduction for European users.`,
    },
    {
      prompt:
        'Implement automatic failover between regions. Simulate primary region failure and measure failover time.',
      solution: `Use FailoverManager class, deploy health checks, test by shutting down primary region. Expected: <5 minute failover with health check based system.`,
    },
    {
      prompt:
        "Build data residency enforcement that keeps EU users' data in EU regions only, with automatic redirection.",
      solution: `Implement DataResidencyManager middleware, test with VPN from different countries. Verify EU requests never write to US databases.`,
    },
  ],
};
