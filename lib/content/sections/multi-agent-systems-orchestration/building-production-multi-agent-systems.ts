/**
 * Building Production Multi-Agent Systems Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const buildingproductionmultiagentsystemsSection = {
  id: 'building-production-multi-agent-systems',
  title: 'Building Production Multi-Agent Systems',
  content: `# Building Production Multi-Agent Systems

Master deploying, scaling, and maintaining multi-agent systems in production.

## Overview: Production Requirements

Production multi-agent systems need:

- **Reliability**: Handle failures gracefully
- **Scalability**: Grow with demand
- **Observability**: Monitor what's happening
- **Cost Efficiency**: Optimize resource usage
- **Security**: Protect against threats
- **Maintainability**: Easy to update and debug

### Production Challenges

**Coordination at Scale**: Many agents, many tasks  
**State Management**: Consistent state across agents  
**Error Handling**: Cascading failures  
**Cost Control**: LLM API costs add up  
**Monitoring**: What\'s each agent doing?  
**Deployment**: Update without downtime  

## Production Architecture

\`\`\`python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import asyncio
import logging

@dataclass
class ProductionConfig:
    """Configuration for production deployment."""
    # Scaling
    max_concurrent_agents: int = 10
    agent_pool_size: int = 5
    
    # Reliability
    max_retries: int = 3
    timeout_seconds: float = 30.0
    enable_circuit_breaker: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"
    
    # Cost Control
    max_cost_per_hour: float = 100.0
    enable_cost_tracking: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

class ProductionMultiAgentSystem:
    """Production-ready multi-agent system."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.agents: Dict[str, Any] = {}
        self.agent_pool: asyncio.Queue = asyncio.Queue (maxsize=config.agent_pool_size)
        self.metrics = MetricsCollector()
        self.cost_tracker = CostTracker (max_per_hour=config.max_cost_per_hour)
        self.circuit_breaker = CircuitBreaker() if config.enable_circuit_breaker else None
        self.cache = Cache() if config.enable_caching else None
        
        # Setup logging
        logging.basicConfig (level=getattr (logging, config.log_level))
        self.logger = logging.getLogger("MultiAgentSystem")
    
    async def initialize (self):
        """Initialize system."""
        self.logger.info("Initializing production system...")
        
        # Initialize agent pool
        for i in range (self.config.agent_pool_size):
            await self.agent_pool.put (f"agent_{i}")
        
        self.logger.info (f"Initialized with {self.config.agent_pool_size} agents")
    
    async def execute_task(
        self,
        task: Dict[str, Any],
        priority: int = 0
    ) -> Dict[str, Any]:
        """Execute task with production features."""
        task_id = task.get('id', 'unknown')
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.allow_request():
            self.logger.warning (f"Circuit breaker open, rejecting task {task_id}")
            return {"error": "Circuit breaker open", "task_id": task_id}
        
        # Check cache
        if self.cache:
            cached = self.cache.get (task_id)
            if cached:
                self.logger.info (f"Cache hit for task {task_id}")
                self.metrics.record("cache_hit")
                return cached
        
        # Check cost limits
        if self.cost_tracker.would_exceed_limit (estimated_cost=0.01):
            self.logger.error (f"Cost limit would be exceeded")
            return {"error": "Cost limit exceeded", "task_id": task_id}
        
        # Acquire agent from pool
        agent_id = await self.agent_pool.get()
        
        try:
            # Execute with retries
            result = await self._execute_with_retry (agent_id, task)
            
            # Cache result
            if self.cache and not result.get('error'):
                self.cache.set (task_id, result, ttl=self.config.cache_ttl_seconds)
            
            # Track success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            return result
        
        except Exception as e:
            self.logger.error (f"Task {task_id} failed: {e}")
            
            # Track failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            return {"error": str (e), "task_id": task_id}
        
        finally:
            # Return agent to pool
            await self.agent_pool.put (agent_id)
    
    async def _execute_with_retry(
        self,
        agent_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute with retry logic."""
        for attempt in range (self.config.max_retries):
            try:
                result = await asyncio.wait_for(
                    self._execute_single (agent_id, task),
                    timeout=self.config.timeout_seconds
                )
                
                # Track metrics
                self.metrics.record("task_success")
                
                return result
            
            except asyncio.TimeoutError:
                self.logger.warning (f"Attempt {attempt + 1} timed out")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                self.logger.warning (f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def _execute_single(
        self,
        agent_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single task."""
        self.logger.info (f"Agent {agent_id} executing task {task.get('id')}")
        
        # Simulate work
        await asyncio.sleep(0.5)
        
        # Track cost
        cost = 0.01  # Estimated cost
        self.cost_tracker.add_cost (cost)
        
        return {
            "task_id": task.get('id'),
            "result": f"Completed by {agent_id}",
            "cost": cost
        }
    
    async def shutdown (self):
        """Graceful shutdown."""
        self.logger.info("Shutting down system...")
        
        # Wait for all tasks to complete
        # In real system, would drain queues
        
        # Export metrics
        metrics = self.metrics.get_summary()
        self.logger.info (f"Final metrics: {metrics}")
        
        # Close resources
        if self.cache:
            self.cache.close()

# Usage
config = ProductionConfig(
    max_concurrent_agents=10,
    enable_metrics=True,
    enable_cost_tracking=True
)

system = ProductionMultiAgentSystem (config)
await system.initialize()

# Execute tasks
tasks = [
    {"id": "task_1", "type": "analysis"},
    {"id": "task_2", "type": "generation"},
]

results = await asyncio.gather(*[
    system.execute_task (task) for task in tasks
])

await system.shutdown()
\`\`\`

## Circuit Breaker Pattern

\`\`\`python
class CircuitBreaker:
    """Circuit breaker for agent failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.failures = 0
        self.successes = 0
        self.state = "closed"  # closed, open, half_open
        self.last_failure_time = None
    
    def allow_request (self) -> bool:
        """Check if request should be allowed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.timeout_seconds:
                    self.state = "half_open"
                    self.successes = 0
                    return True
            return False
        
        if self.state == "half_open":
            return True
        
        return False
    
    def record_success (self):
        """Record successful request."""
        if self.state == "half_open":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "closed"
                self.failures = 0
        elif self.state == "closed":
            self.failures = max(0, self.failures - 1)
    
    def record_failure (self):
        """Record failed request."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
\`\`\`

## Metrics Collection

\`\`\`python
class MetricsCollector:
    """Collect system metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, int] = {}
        self.timings: Dict[str, List[float]] = {}
    
    def record (self, metric: str, value: float = 1):
        """Record metric."""
        self.metrics[metric] = self.metrics.get (metric, 0) + value
    
    def record_timing (self, operation: str, duration: float):
        """Record operation timing."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append (duration)
    
    def get_summary (self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = dict (self.metrics)
        
        # Add timing statistics
        for operation, durations in self.timings.items():
            if durations:
                summary[f"{operation}_avg"] = sum (durations) / len (durations)
                summary[f"{operation}_min"] = min (durations)
                summary[f"{operation}_max"] = max (durations)
        
        return summary
\`\`\`

## Cost Tracking

\`\`\`python
class CostTracker:
    """Track and limit API costs."""
    
    def __init__(self, max_per_hour: float):
        self.max_per_hour = max_per_hour
        self.costs: List[tuple[float, float]] = []  # (timestamp, cost)
    
    def add_cost (self, cost: float):
        """Add cost."""
        self.costs.append((time.time(), cost))
        
        # Clean old costs (>1 hour)
        cutoff = time.time() - 3600
        self.costs = [(t, c) for t, c in self.costs if t > cutoff]
    
    def get_hourly_cost (self) -> float:
        """Get cost in last hour."""
        return sum (cost for _, cost in self.costs)
    
    def would_exceed_limit (self, estimated_cost: float) -> bool:
        """Check if adding cost would exceed limit."""
        return (self.get_hourly_cost() + estimated_cost) > self.max_per_hour
    
    def get_remaining_budget (self) -> float:
        """Get remaining budget this hour."""
        return max(0, self.max_per_hour - self.get_hourly_cost())
\`\`\`

## Caching Layer

\`\`\`python
import hashlib

class Cache:
    """Simple in-memory cache."""
    
    def __init__(self):
        self.data: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
    
    def _hash_key (self, key: Any) -> str:
        """Hash key for storage."""
        return hashlib.md5(str (key).encode()).hexdigest()
    
    def get (self, key: Any) -> Optional[Any]:
        """Get from cache."""
        hashed = self._hash_key (key)
        
        if hashed in self.data:
            value, expiry = self.data[hashed]
            
            if time.time() < expiry:
                return value
            else:
                # Expired
                del self.data[hashed]
        
        return None
    
    def set (self, key: Any, value: Any, ttl: int = 3600):
        """Set in cache."""
        hashed = self._hash_key (key)
        expiry = time.time() + ttl
        self.data[hashed] = (value, expiry)
    
    def clear_expired (self):
        """Clear expired entries."""
        now = time.time()
        expired = [k for k, (_, exp) in self.data.items() if exp < now]
        for k in expired:
            del self.data[k]
    
    def close (self):
        """Cleanup."""
        self.data.clear()
\`\`\`

## Health Checks

\`\`\`python
class HealthChecker:
    """Monitor system health."""
    
    def __init__(self, system: ProductionMultiAgentSystem):
        self.system = system
    
    async def check_health (self) -> Dict[str, Any]:
        """Comprehensive health check."""
        checks = {
            "agents": await self._check_agents(),
            "costs": self._check_costs(),
            "circuit_breaker": self._check_circuit_breaker(),
            "cache": self._check_cache()
        }
        
        overall = all (c["healthy"] for c in checks.values())
        
        return {
            "healthy": overall,
            "checks": checks,
            "timestamp": time.time()
        }
    
    async def _check_agents (self) -> Dict[str, Any]:
        """Check agent pool."""
        available = self.system.agent_pool.qsize()
        total = self.system.config.agent_pool_size
        
        healthy = available > 0
        
        return {
            "healthy": healthy,
            "available_agents": available,
            "total_agents": total,
            "utilization": 1 - (available / total) if total > 0 else 0
        }
    
    def _check_costs (self) -> Dict[str, Any]:
        """Check cost status."""
        current = self.system.cost_tracker.get_hourly_cost()
        limit = self.system.cost_tracker.max_per_hour
        
        healthy = current < limit * 0.9  # 90% threshold
        
        return {
            "healthy": healthy,
            "current_cost": current,
            "cost_limit": limit,
            "utilization": current / limit if limit > 0 else 0
        }
    
    def _check_circuit_breaker (self) -> Dict[str, Any]:
        """Check circuit breaker status."""
        if not self.system.circuit_breaker:
            return {"healthy": True, "state": "disabled"}
        
        state = self.system.circuit_breaker.state
        healthy = state != "open"
        
        return {
            "healthy": healthy,
            "state": state,
            "failures": self.system.circuit_breaker.failures
        }
    
    def _check_cache (self) -> Dict[str, Any]:
        """Check cache status."""
        if not self.system.cache:
            return {"healthy": True, "state": "disabled"}
        
        size = len (self.system.cache.data)
        
        return {
            "healthy": True,
            "entries": size
        }

# Usage
checker = HealthChecker (system)
health = await checker.check_health()

if health["healthy"]:
    print("✅ System healthy")
else:
    print("❌ System unhealthy:")
    for check_name, check_result in health["checks"].items():
        if not check_result["healthy"]:
            print(f"  - {check_name}: {check_result}")
\`\`\`

## Deployment Strategies

\`\`\`python
class DeploymentManager:
    """Manage rolling deployments."""
    
    def __init__(self):
        self.old_system: Optional[ProductionMultiAgentSystem] = None
        self.new_system: Optional[ProductionMultiAgentSystem] = None
        self.cutover_percentage = 0
    
    async def blue_green_deploy(
        self,
        new_config: ProductionConfig
    ):
        """Blue-green deployment."""
        print("Starting blue-green deployment...")
        
        # Start new system (green)
        print("  1. Starting new system...")
        self.new_system = ProductionMultiAgentSystem (new_config)
        await self.new_system.initialize()
        
        # Health check new system
        print("  2. Health checking new system...")
        checker = HealthChecker (self.new_system)
        health = await checker.check_health()
        
        if not health["healthy"]:
            print("  ❌ New system unhealthy, aborting")
            await self.new_system.shutdown()
            return False
        
        # Cutover
        print("  3. Cutting over traffic...")
        if self.old_system:
            await self.old_system.shutdown()
        
        self.old_system = self.new_system
        self.new_system = None
        
        print("  ✅ Deployment complete")
        return True
    
    async def canary_deploy(
        self,
        new_config: ProductionConfig,
        steps: List[int] = [10, 25, 50, 100]
    ):
        """Canary deployment with gradual traffic shift."""
        print("Starting canary deployment...")
        
        # Start new system
        self.new_system = ProductionMultiAgentSystem (new_config)
        await self.new_system.initialize()
        
        # Gradually increase traffic
        for percentage in steps:
            print(f"  Shifting {percentage}% traffic to new system...")
            self.cutover_percentage = percentage
            
            # Monitor for issues
            await asyncio.sleep(60)  # Wait 1 minute
            
            # Check health
            checker = HealthChecker (self.new_system)
            health = await checker.check_health()
            
            if not health["healthy"]:
                print(f"  ❌ Issues detected, rolling back")
                await self.rollback()
                return False
        
        # Complete cutover
        print("  ✅ Canary successful, completing cutover")
        if self.old_system:
            await self.old_system.shutdown()
        
        self.old_system = self.new_system
        self.new_system = None
        self.cutover_percentage = 0
        
        return True
    
    async def rollback (self):
        """Rollback to old system."""
        print("Rolling back deployment...")
        
        if self.new_system:
            await self.new_system.shutdown()
            self.new_system = None
        
        self.cutover_percentage = 0
        print("  ✅ Rollback complete")
    
    def route_request (self, task: Dict) -> ProductionMultiAgentSystem:
        """Route request to appropriate system."""
        if not self.new_system:
            return self.old_system
        
        # Route based on cutover percentage
        import random
        if random.random() * 100 < self.cutover_percentage:
            return self.new_system
        else:
            return self.old_system

# Usage
deployment = DeploymentManager()

# Blue-green deployment
await deployment.blue_green_deploy (new_config)

# Or canary deployment
await deployment.canary_deploy (new_config, steps=[10, 25, 50, 100])
\`\`\`

## Monitoring Dashboard

\`\`\`python
class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, system: ProductionMultiAgentSystem):
        self.system = system
    
    def get_dashboard_data (self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return {
            "timestamp": time.time(),
            "metrics": self.system.metrics.get_summary(),
            "costs": {
                "hourly": self.system.cost_tracker.get_hourly_cost(),
                "remaining": self.system.cost_tracker.get_remaining_budget()
            },
            "agents": {
                "available": self.system.agent_pool.qsize(),
                "total": self.system.config.agent_pool_size
            },
            "circuit_breaker": {
                "state": self.system.circuit_breaker.state if self.system.circuit_breaker else "disabled",
                "failures": self.system.circuit_breaker.failures if self.system.circuit_breaker else 0
            }
        }
    
    def print_dashboard (self):
        """Print dashboard to console."""
        data = self.get_dashboard_data()
        
        print("\\n" + "="*60)
        print("MULTI-AGENT SYSTEM DASHBOARD")
        print("="*60)
        
        print(f"\\n📊 METRICS:")
        for key, value in data["metrics"].items():
            print(f"  {key}: {value}")
        
        print(f"\\n💰 COSTS:")
        print(f"  Hourly: \\$\{data['costs']['hourly']:.2f}")
        print(f"  Remaining: \\$\{data['costs']['remaining']:.2f}")

print(f"\\n🤖 AGENTS:")
print(f"  Available: {data['agents']['available']}/{data['agents']['total']}")

print(f"\\n🔌 CIRCUIT BREAKER:")
print(f"  State: {data['circuit_breaker']['state']}")
print(f"  Failures: {data['circuit_breaker']['failures']}")

print("=" * 60)

# Usage
dashboard = MonitoringDashboard (system)
dashboard.print_dashboard()
\`\`\`

## Production Checklist

\`\`\`python
class ProductionChecklist:
    """Verify system is production-ready."""
    
    @staticmethod
    def verify_readiness (system: ProductionMultiAgentSystem) -> Dict[str, bool]:
        """Check production readiness."""
        checks = {
            "logging_enabled": system.logger is not None,
            "metrics_enabled": system.config.enable_metrics,
            "cost_tracking": system.config.enable_cost_tracking,
            "error_handling": system.config.max_retries > 0,
            "timeout_configured": system.config.timeout_seconds > 0,
            "circuit_breaker": system.circuit_breaker is not None,
            "caching_enabled": system.config.enable_caching,
            "health_checks": True  # Would check if health endpoint exists
        }
        
        all_passed = all (checks.values())
        
        return {
            "ready": all_passed,
            "checks": checks
        }

# Usage
readiness = ProductionChecklist.verify_readiness (system)

if readiness["ready"]:
    print("✅ System is production-ready")
else:
    print("❌ System not ready:")
    for check, passed in readiness["checks"].items():
        if not passed:
            print(f"  - {check}: FAILED")
\`\`\`

## Best Practices

1. **Start Small**: Deploy to staging first
2. **Monitor Everything**: Logs, metrics, traces
3. **Handle Failures**: Retries, circuit breakers, fallbacks
4. **Control Costs**: Track and limit spending
5. **Cache Aggressively**: Reduce API calls
6. **Health Checks**: Continuous monitoring
7. **Gradual Rollouts**: Canary deployments
8. **Document**: Document architecture and operations

## Production Deployment Steps

1. **Build**: Create production configuration
2. **Test**: Load test in staging
3. **Monitor**: Set up monitoring and alerts
4. **Deploy**: Blue-green or canary deploy
5. **Verify**: Check health and metrics
6. **Monitor**: Watch for issues
7. **Optimize**: Adjust based on metrics

You now understand production multi-agent systems!
`,
};
