export const productionDeployment = {
  title: 'Production Deployment',
  id: 'production-deployment',
  content: `
# Production Deployment

## Introduction

**Production deployment** is releasing trading system changes to live environments without causing downtime or financial loss. In trading, bad deployments can cost millions per minute.

**Why Deployment Strategy Matters:**
- **Zero downtime requirement**: Cannot stop trading during market hours (9:30 AM - 4:00 PM ET)
- **Rollback capability**: Must revert bad changes in <2 minutes
- **Financial risk**: Each deployment could impact active positions worth millions
- **Regulatory compliance**: Must maintain audit trail of all deployments
- **Client impact**: Institutional clients have strict uptime SLAs (99.99%+)

**Deployment Horror Stories:**
- **Knight Capital (2012)**: Bad deployment → $440M loss in 45 minutes
- **Trading firm (2013)**: Database migration during market hours → 2 hours downtime → $50M+ lost opportunity
- **HFT firm (2015)**: Canary rollout caught bug → prevented $100M+ disaster

**Real-World Deployment Practices:**
- **Goldman Sachs**: Blue-green deployments, extensive pre-prod testing, gradual rollout
- **Citadel Securities**: Canary deployments with automated rollback, <1% initial rollout
- **Interactive Brokers**: Weekend deployments only, full regression testing, multiple staging environments
- **Jane Street**: Feature flags for all changes, instant rollback capability

This section covers production deployment strategies for trading systems.

---

## Deployment Strategies Comparison

| Strategy | Downtime | Rollback Time | Complexity | Risk | Use Case |
|----------|----------|---------------|------------|------|----------|
| Big Bang | Hours | Hours | Low | Very High | Legacy systems only |
| Rolling | None | Minutes | Medium | Medium | Stateless services |
| Blue-Green | Seconds | Seconds | Medium | Low | Most trading systems |
| Canary | None | Seconds | High | Very Low | Critical services |
| Feature Flags | None | Instant | High | Very Low | New features |

---

## Blue-Green Deployment

\`\`\`python
"""
Blue-Green Deployment for Trading Systems

Architecture:
- Blue environment: Currently live (v1.0)
- Green environment: New version (v2.0)
- Load balancer: Routes traffic to active environment
- Instant switchover: DNS/load balancer update

Benefits:
- Zero downtime
- Instant rollback (switch back to blue)
- Full testing in production-like environment
- Simple conceptually

Drawbacks:
- 2x infrastructure cost
- Database migrations tricky
- State synchronization needed
"""

from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

class EnvironmentColor(Enum):
    BLUE = "BLUE"
    GREEN = "GREEN"

class EnvironmentStatus(Enum):
    OFFLINE = "OFFLINE"
    DEPLOYING = "DEPLOYING"
    TESTING = "TESTING"
    LIVE = "LIVE"
    DRAINING = "DRAINING"

class Environment:
    """
    Represents one environment (blue or green)
    """
    
    def __init__(
        self,
        color: EnvironmentColor,
        version: str,
        capacity: int = 100
    ):
        self.color = color
        self.version = version
        self.capacity = capacity  # Number of servers
        self.status = EnvironmentStatus.OFFLINE
        self.active_connections = 0
        self.requests_per_second = 0
        self.error_rate = 0.0
        self.deployed_at: Optional[datetime] = None
    
    async def deploy(self, new_version: str):
        """Deploy new version to this environment"""
        print(f"[{self.color.value}] Deploying version {new_version}...")
        
        self.status = EnvironmentStatus.DEPLOYING
        self.version = new_version
        
        # Simulate deployment steps
        await self._pull_docker_images()
        await self._update_configuration()
        await self._start_services()
        await self._run_health_checks()
        
        self.status = EnvironmentStatus.TESTING
        self.deployed_at = datetime.utcnow()
        
        print(f"[{self.color.value}] ✓ Deployment complete")
    
    async def _pull_docker_images(self):
        """Pull Docker images"""
        print(f"  [1/4] Pulling Docker images...")
        await asyncio.sleep(2)
    
    async def _update_configuration(self):
        """Update configuration files"""
        print(f"  [2/4] Updating configuration...")
        await asyncio.sleep(1)
    
    async def _start_services(self):
        """Start all services"""
        print(f"  [3/4] Starting services...")
        await asyncio.sleep(3)
    
    async def _run_health_checks(self):
        """Run health checks"""
        print(f"  [4/4] Running health checks...")
        await asyncio.sleep(2)
    
    async def run_smoke_tests(self) -> bool:
        """
        Run smoke tests before going live
        
        Tests:
        - Service health checks
        - Database connectivity
        - API endpoints responding
        - Order placement (test mode)
        - Position queries working
        - No critical errors in logs
        """
        print(f"[{self.color.value}] Running smoke tests...")
        
        tests = [
            self._test_health_endpoints(),
            self._test_database_connectivity(),
            self._test_order_placement(),
            self._test_position_queries(),
            self._check_error_logs()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        all_passed = all(r is True for r in results if not isinstance(r, Exception))
        
        if all_passed:
            print(f"[{self.color.value}] ✓ All smoke tests passed")
        else:
            print(f"[{self.color.value}] ✗ Smoke tests failed")
            for i, result in enumerate(results):
                if isinstance(result, Exception) or result is False:
                    print(f"    Test {i+1} failed: {result}")
        
        return all_passed
    
    async def _test_health_endpoints(self) -> bool:
        """Test /health endpoints"""
        await asyncio.sleep(0.5)
        return True
    
    async def _test_database_connectivity(self) -> bool:
        """Test database connections"""
        await asyncio.sleep(0.5)
        return True
    
    async def _test_order_placement(self) -> bool:
        """Test order placement (test mode)"""
        await asyncio.sleep(1)
        return True
    
    async def _test_position_queries(self) -> bool:
        """Test position queries"""
        await asyncio.sleep(0.5)
        return True
    
    async def _check_error_logs(self) -> bool:
        """Check for critical errors in logs"""
        await asyncio.sleep(0.3)
        return True
    
    def go_live(self):
        """Make this environment live"""
        self.status = EnvironmentStatus.LIVE
        print(f"[{self.color.value}] Now LIVE (version {self.version})")
    
    def start_draining(self):
        """Start draining connections"""
        self.status = EnvironmentStatus.DRAINING
        print(f"[{self.color.value}] Draining connections...")
    
    async def wait_for_drain(self, timeout_seconds: int = 30):
        """Wait for all connections to drain"""
        elapsed = 0
        while self.active_connections > 0 and elapsed < timeout_seconds:
            print(f"[{self.color.value}] Waiting for {self.active_connections} connections to drain...")
            await asyncio.sleep(5)
            elapsed += 5
            # Simulate connection decrease
            self.active_connections = max(0, self.active_connections - 10)
        
        if self.active_connections == 0:
            print(f"[{self.color.value}] ✓ All connections drained")
        else:
            print(f"[{self.color.value}] ⚠️  Timeout: {self.active_connections} connections remaining")
    
    def take_offline(self):
        """Take environment offline"""
        self.status = EnvironmentStatus.OFFLINE
        print(f"[{self.color.value}] Taken OFFLINE")


class LoadBalancer:
    """
    Load balancer that routes traffic to active environment
    
    Implementations:
    - AWS Application Load Balancer (ALB)
    - NGINX
    - HAProxy
    - DNS weighted routing
    """
    
    def __init__(self):
        self.active_env: Optional[Environment] = None
        self.traffic_split: Dict[EnvironmentColor, int] = {}
    
    def switch_traffic(self, new_env: Environment, instant: bool = True):
        """
        Switch traffic to new environment
        
        Args:
            new_env: Environment to switch to
            instant: If True, instant switch. If False, gradual migration
        """
        if instant:
            print(f"[LoadBalancer] Switching 100% traffic to {new_env.color.value}")
            self.active_env = new_env
            self.traffic_split = {new_env.color: 100}
        else:
            # Gradual migration for canary
            pass
    
    def get_active_environment(self) -> Optional[Environment]:
        """Get currently active environment"""
        return self.active_env


class BlueGreenDeploymentManager:
    """
    Manages blue-green deployments
    
    Workflow:
    1. Blue is live (v1.0)
    2. Deploy new version to green (v2.0)
    3. Run smoke tests on green
    4. Switch traffic from blue to green
    5. Monitor green for issues
    6. If issues: instant rollback to blue
    7. If stable: take blue offline
    """
    
    def __init__(self):
        # Create environments
        self.blue = Environment(EnvironmentColor.BLUE, version="1.0.0")
        self.green = Environment(EnvironmentColor.GREEN, version="1.0.0")
        
        # Load balancer
        self.load_balancer = LoadBalancer()
        
        # Initially blue is live
        self.blue.go_live()
        self.blue.active_connections = 100
        self.load_balancer.switch_traffic(self.blue)
        
        # Deployment history
        self.deployment_history: List[Dict] = []
    
    async def deploy_new_version(self, new_version: str) -> bool:
        """
        Deploy new version using blue-green strategy
        
        Returns: True if successful, False if rolled back
        """
        print("\\n" + "=" * 80)
        print(f"BLUE-GREEN DEPLOYMENT: {new_version}")
        print("=" * 80)
        
        deployment_start = datetime.utcnow()
        
        # Determine which environment to deploy to
        if self.blue.status == EnvironmentStatus.LIVE:
            target_env = self.green
            source_env = self.blue
        else:
            target_env = self.blue
            source_env = self.green
        
        print(f"\\nCurrent: {source_env.color.value} (v{source_env.version})")
        print(f"Target: {target_env.color.value} (deploying v{new_version})\\n")
        
        try:
            # Step 1: Deploy to inactive environment
            print("Step 1/6: Deploying to target environment...")
            await target_env.deploy(new_version)
            
            # Step 2: Run smoke tests
            print("\\nStep 2/6: Running smoke tests...")
            smoke_tests_passed = await target_env.run_smoke_tests()
            
            if not smoke_tests_passed:
                print("\\n❌ Smoke tests failed - ABORTING DEPLOYMENT")
                target_env.take_offline()
                return False
            
            # Step 3: Gradual traffic switch (optional)
            print("\\nStep 3/6: Switching traffic...")
            self.load_balancer.switch_traffic(target_env, instant=True)
            target_env.go_live()
            
            # Step 4: Monitor new environment
            print("\\nStep 4/6: Monitoring new environment (30 seconds)...")
            monitoring_passed = await self._monitor_environment(target_env, duration=10)
            
            if not monitoring_passed:
                print("\\n❌ Monitoring detected issues - ROLLING BACK")
                await self.rollback(source_env)
                return False
            
            # Step 5: Drain old environment
            print("\\nStep 5/6: Draining old environment...")
            source_env.start_draining()
            await source_env.wait_for_drain()
            
            # Step 6: Take old environment offline
            print("\\nStep 6/6: Taking old environment offline...")
            source_env.take_offline()
            
            # Record deployment
            deployment_duration = (datetime.utcnow() - deployment_start).total_seconds()
            self.deployment_history.append({
                'version': new_version,
                'from_env': source_env.color.value,
                'to_env': target_env.color.value,
                'duration_seconds': deployment_duration,
                'timestamp': deployment_start.isoformat(),
                'status': 'SUCCESS'
            })
            
            print("\\n" + "=" * 80)
            print(f"✓ DEPLOYMENT SUCCESSFUL (took {deployment_duration:.1f}s)")
            print(f"  Live: {target_env.color.value} (v{new_version})")
            print(f"  Standby: {source_env.color.value} (v{source_env.version})")
            print("=" * 80 + "\\n")
            
            return True
            
        except Exception as e:
            print(f"\\n❌ DEPLOYMENT FAILED: {e}")
            print("Rolling back...")
            await self.rollback(source_env)
            
            self.deployment_history.append({
                'version': new_version,
                'status': 'FAILED',
                'error': str(e),
                'timestamp': deployment_start.isoformat()
            })
            
            return False
    
    async def _monitor_environment(
        self,
        env: Environment,
        duration: int = 30
    ) -> bool:
        """
        Monitor environment after deployment
        
        Checks:
        - Error rate < 0.1%
        - Request rate > 0 (traffic flowing)
        - No critical errors
        - Latency p99 < 100ms
        """
        print(f"  Monitoring {env.color.value} for {duration} seconds...")
        
        for i in range(duration // 5):
            await asyncio.sleep(5)
            
            # Simulate metrics
            env.error_rate = 0.01  # 0.01% errors (good)
            env.requests_per_second = 1000
            
            print(f"    t+{(i+1)*5}s: RPS={env.requests_per_second}, Errors={env.error_rate*100:.3f}%")
            
            # Check thresholds
            if env.error_rate > 0.001:  # >0.1% error rate
                print(f"    ⚠️  High error rate: {env.error_rate*100:.3f}%")
                return False
            
            if env.requests_per_second < 100:
                print(f"    ⚠️  Low request rate: {env.requests_per_second}")
                return False
        
        print(f"  ✓ Monitoring passed")
        return True
    
    async def rollback(self, previous_env: Environment):
        """
        Instant rollback to previous environment
        
        This is why blue-green is powerful: instant rollback
        """
        print("\\n" + "=" * 80)
        print("🔄 ROLLING BACK")
        print("=" * 80)
        
        # Switch traffic back
        self.load_balancer.switch_traffic(previous_env, instant=True)
        previous_env.go_live()
        
        print(f"✓ Rolled back to {previous_env.color.value} (v{previous_env.version})")
        print("=" * 80 + "\\n")
    
    def get_deployment_history(self) -> List[Dict]:
        """Get deployment history"""
        return self.deployment_history


# Example: Blue-green deployment
async def blue_green_example():
    """Demonstrate blue-green deployment"""
    
    manager = BlueGreenDeploymentManager()
    
    # Deploy v2.0.0
    success = await manager.deploy_new_version("2.0.0")
    
    if success:
        print("\\n✅ Deployment successful!")
    else:
        print("\\n❌ Deployment failed (rolled back)")
    
    # Show deployment history
    print("\\nDeployment History:")
    for deployment in manager.get_deployment_history():
        print(f"  {deployment}")

# asyncio.run(blue_green_example())
\`\`\`

---

## Canary Deployment

\`\`\`python
"""
Canary Deployment: Gradual Rollout

Strategy:
- Start with 1-5% of traffic on new version
- Monitor closely for errors/latency
- Gradually increase: 1% → 5% → 25% → 50% → 100%
- Instant rollback if issues detected

Timeline:
- t+0:    1% on canary
- t+15min: 5% on canary (if metrics good)
- t+30min: 25% on canary
- t+1hr:  50% on canary
- t+2hr:  100% on canary (complete)
"""

from typing import Dict, Tuple

class CanaryDeployment:
    """
    Canary deployment with automated rollout
    """
    
    def __init__(self):
        self.old_version = "1.0.0"
        self.new_version = "2.0.0"
        
        # Traffic split (percentage)
        self.old_traffic_pct = 100
        self.new_traffic_pct = 0
        
        # Metrics
        self.old_error_rate = 0.0
        self.new_error_rate = 0.0
        self.old_latency_p99 = 50.0  # ms
        self.new_latency_p99 = 50.0
        
        # Rollout schedule
        self.rollout_stages = [1, 5, 25, 50, 100]
        self.current_stage = 0
        
        # Thresholds for promotion
        self.max_error_rate_increase = 0.001  # 0.1%
        self.max_latency_increase_pct = 0.20  # 20%
    
    async def start_canary(self):
        """Start canary rollout"""
        print("=" * 80)
        print(f"CANARY DEPLOYMENT: {self.new_version}")
        print("=" * 80)
        
        for stage_pct in self.rollout_stages:
            print(f"\\n--- Stage: {stage_pct}% on canary ---")
            
            # Update traffic split
            await self.update_traffic_split(stage_pct)
            
            # Monitor for duration
            duration = 15 if stage_pct < 50 else 30  # minutes
            print(f"Monitoring for {duration} minutes...")
            
            # Simulate monitoring
            await asyncio.sleep(duration * 0.1)  # Simulate (scaled down)
            
            # Check metrics
            if not await self.check_canary_health():
                print(f"\\n❌ Canary failed at {stage_pct}% - ROLLING BACK")
                await self.rollback()
                return False
            
            print(f"✓ Stage {stage_pct}% passed")
        
        print("\\n" + "=" * 80)
        print("✓ CANARY COMPLETE - 100% on new version")
        print("=" * 80)
        
        return True
    
    async def update_traffic_split(self, canary_pct: int):
        """Update traffic split"""
        self.new_traffic_pct = canary_pct
        self.old_traffic_pct = 100 - canary_pct
        
        print(f"Traffic split: {self.old_traffic_pct}% v{self.old_version}, "
              f"{self.new_traffic_pct}% v{self.new_version}")
    
    async def check_canary_health(self) -> bool:
        """
        Check if canary version is healthy
        
        Compares:
        - Error rate (canary vs baseline)
        - Latency p99 (canary vs baseline)
        - Request success rate
        """
        # Simulate metrics collection
        self.old_error_rate = 0.001  # 0.1%
        self.new_error_rate = 0.0012  # 0.12% (slightly higher but acceptable)
        
        self.old_latency_p99 = 50.0
        self.new_latency_p99 = 55.0  # 10% higher (acceptable)
        
        print(f"\\nMetrics:")
        print(f"  Error rate: baseline={self.old_error_rate*100:.3f}%, canary={self.new_error_rate*100:.3f}%")
        print(f"  Latency p99: baseline={self.old_latency_p99:.1f}ms, canary={self.new_latency_p99:.1f}ms")
        
        # Check error rate
        error_rate_increase = self.new_error_rate - self.old_error_rate
        if error_rate_increase > self.max_error_rate_increase:
            print(f"  ❌ Error rate increased by {error_rate_increase*100:.3f}% (threshold: {self.max_error_rate_increase*100}%)")
            return False
        
        # Check latency
        latency_increase_pct = (self.new_latency_p99 - self.old_latency_p99) / self.old_latency_p99
        if latency_increase_pct > self.max_latency_increase_pct:
            print(f"  ❌ Latency increased by {latency_increase_pct*100:.1f}% (threshold: {self.max_latency_increase_pct*100}%)")
            return False
        
        print(f"  ✓ Metrics within acceptable range")
        return True
    
    async def rollback(self):
        """Instant rollback to old version"""
        print("\\n🔄 Rolling back to 100% old version...")
        await self.update_traffic_split(0)
        print("✓ Rollback complete")


# Example
# asyncio.run(CanaryDeployment().start_canary())
\`\`\`

---

## Production Deployment Checklist

\`\`\`yaml
# =============================================================================
# PRE-DEPLOYMENT CHECKLIST
# =============================================================================

code_quality:
  - [ ] Code reviewed by 2+ senior engineers
  - [ ] All unit tests passing (>95% coverage)
  - [ ] Integration tests passing
  - [ ] End-to-end tests passing
  - [ ] Performance tests passing (latency benchmarks met)
  - [ ] Security scan completed (no critical vulnerabilities)
  - [ ] Linting passed (no errors, warnings acceptable)

database:
  - [ ] Database migrations tested in staging
  - [ ] Rollback migration script ready
  - [ ] Data backup completed (<1 hour old)
  - [ ] Migration estimated time < 30 seconds
  - [ ] No breaking schema changes
  - [ ] Indexes created (no long-running ALTER TABLE during deploy)

dependencies:
  - [ ] All dependencies pinned to specific versions
  - [ ] No major version upgrades (unless explicitly tested)
  - [ ] Security updates reviewed
  - [ ] Docker images built and pushed to registry
  - [ ] Configuration updated in vault/secrets manager

testing:
  - [ ] Smoke tests defined
  - [ ] Load tests completed (2x expected traffic)
  - [ ] Chaos tests passed (simulated failures)
  - [ ] Canary test plan defined
  - [ ] Rollback tested in staging

monitoring:
  - [ ] Metrics dashboards updated
  - [ ] Alerts configured for new version
  - [ ] Log aggregation configured
  - [ ] Error tracking setup (Sentry/DataDog)
  - [ ] Synthetic monitoring setup

operations:
  - [ ] Deployment window scheduled (off-peak hours preferred)
  - [ ] On-call engineer assigned
  - [ ] Backup engineer assigned
  - [ ] Stakeholders notified (trading desk, clients)
  - [ ] Rollback plan documented
  - [ ] Incident response plan updated

compliance:
  - [ ] Change management ticket approved
  - [ ] Audit log entry created
  - [ ] Regulatory notifications sent (if required)
  - [ ] Documentation updated

# =============================================================================
# DEPLOYMENT EXECUTION
# =============================================================================

deployment:
  1. [ ] Pre-deployment announcement (Slack #trading-ops)
  2. [ ] Verify all pre-deployment checks passed
  3. [ ] Take database backup
  4. [ ] Deploy to green environment
  5. [ ] Run database migrations (if any)
  6. [ ] Run smoke tests
  7. [ ] Check error logs (no critical errors)
  8. [ ] Check metrics (latency, throughput, error rate)
  9. [ ] Start canary rollout (1% → 5% → 25% → 50% → 100%)
  10. [ ] Monitor each stage for 15-30 minutes
  11. [ ] If issues: Execute rollback immediately
  12. [ ] If successful: Complete rollout
  13. [ ] Drain old environment
  14. [ ] Take old environment offline
  15. [ ] Post-deployment announcement

monitoring_thresholds:
  error_rate:
    warning: >0.1%
    critical: >1.0%
    action: Rollback immediately
  
  latency_p99:
    warning: >100ms
    critical: >500ms
    action: Investigate, consider rollback
  
  order_success_rate:
    warning: <99%
    critical: <95%
    action: Rollback immediately
  
  fill_rate:
    warning: <95%
    critical: <90%
    action: Rollback immediately

# =============================================================================
# POST-DEPLOYMENT
# =============================================================================

post_deployment:
  - [ ] Monitor for 2 hours post-deployment
  - [ ] Verify all key metrics within normal range
  - [ ] Check error logs for anomalies
  - [ ] Verify order flow working correctly
  - [ ] Verify position tracking accurate
  - [ ] Verify P&L calculations correct
  - [ ] Run reconciliation report (EOD)
  - [ ] Update documentation (if needed)
  - [ ] Post-deployment retrospective (within 24 hours)
  - [ ] Update runbooks (lessons learned)

rollback_triggers:
  immediate_rollback:
    - Error rate >1%
    - Order success rate <95%
    - Critical service down
    - Data corruption detected
    - Regulatory violation
  
  consider_rollback:
    - Error rate >0.1% for >15 minutes
    - Latency p99 >100ms for >30 minutes
    - Fill rate <95% for >15 minutes
    - Customer complaints >10
\`\`\`

---

## Summary

**Deployment Strategy Selection:**

| Scenario | Strategy | Rationale |
|----------|----------|-----------|
| New trading system | Blue-Green | Zero downtime, instant rollback |
| Critical fix during market hours | Blue-Green | Fastest, safest |
| New feature (non-critical) | Canary | Gradual rollout, detect issues early |
| Database schema change | Blue-Green | Test fully before switching |
| Configuration change only | Rolling | Simplest, minimal risk |

**Key Principles:**1. **Never deploy during market hours** (9:30 AM - 4:00 PM ET) unless critical fix
2. **Always have rollback plan** ready before starting
3. **Monitor intensively** for first 2 hours post-deployment
4. **Test everything in staging** first (no surprises in prod)
5. **Automate everything** (humans make mistakes under pressure)
6. **Document everything** (compliance requirement)

**Production Best Practices:**
- Deploy on weekends or after market close (4:00 PM ET)
- Use feature flags for new features (deploy code disabled, enable later)
- Maintain 2 environments minimum (blue/green)
- Keep old version running for ≥24 hours after deployment
- Automate rollback (< 2 minute manual intervention)
- Test rollback procedure monthly

**Next Section**: Module 14.15 - Project: Complete Trading System
`,
};
