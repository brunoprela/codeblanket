export const disasterRecoveryFailover = {
    title: 'Disaster Recovery and Failover',
    id: 'disaster-recovery-failover',
    content: `
# Disaster Recovery and Failover

## Introduction

**Disaster recovery (DR)** ensures trading operations continue after catastrophic failures. In trading, downtime directly equals lost revenue and client trust.

**Why DR is Critical:**
- **Financial impact**: 1 hour of downtime during market hours can cost $1M-$10M+ in lost trading opportunity
- **Regulatory requirements**: SEC Rule 38a-1 requires business continuity plans
- **Client SLAs**: Institutional clients demand 99.99%+ uptime (52 minutes/year downtime maximum)
- **Market events**: Flash crashes, network outages, data center failures happen regularly
- **Competitive pressure**: Competitors will capture your market share during outages

**Real-World DR Examples:**
- **Knight Capital (2012)**: No proper failover → $440M loss in 45 minutes
- **NASDAQ (2013)**: 3-hour outage → Implemented multi-datacenter DR
- **NYSE (2015)**: Trading halt → Enhanced DR procedures
- **Interactive Brokers**: Multi-datacenter active-active, <5 second failover
- **Goldman Sachs**: Geographic redundancy across 3 continents

**DR Metrics:**
- **RTO** (Recovery Time Objective): Time to restore service (target: <5 minutes)
- **RPO** (Recovery Point Objective): Acceptable data loss (target: <1 second)
- **MTTR** (Mean Time To Recovery): Average recovery time
- **MTBF** (Mean Time Between Failures): Average uptime between failures

This section covers production disaster recovery and failover strategies.

---

## DR Strategy: Active-Passive Failover

\`\`\`python
"""
Active-Passive Failover Implementation

Setup:
- Primary datacenter: Handles all production traffic
- Secondary datacenter: Standby, ready to take over
- Health monitoring: Detects primary failure in <10 seconds
- Automated failover: Promotes secondary to primary

RTO: <5 minutes
RPO: <1 second (streaming replication)
"""

import time
import asyncio
from enum import Enum
from typing import Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import redis
import psycopg2

class ServerRole(Enum):
    """Server roles in DR setup"""
    ACTIVE = "ACTIVE"    # Handling production traffic
    PASSIVE = "PASSIVE"  # Standby, replicating data
    UNKNOWN = "UNKNOWN"  # Health check failed

class FailoverReason(Enum):
    """Reasons for triggering failover"""
    HEALTH_CHECK_FAILED = "HEALTH_CHECK_FAILED"
    HIGH_LATENCY = "HIGH_LATENCY"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"

@dataclass
class ServerHealth:
    """Server health status"""
    server_id: str
    role: ServerRole
    is_healthy: bool
    latency_ms: float
    last_heartbeat: datetime
    active_connections: int
    orders_per_second: float

class TradingServer:
    """
    Trading server with health monitoring
    
    Each server:
    - Monitors its own health
    - Sends heartbeats to coordinator
    - Can promote/demote based on role
    """
    
    def __init__(
        self,
        server_id: str,
        datacenter: str,
        initial_role: ServerRole = ServerRole.PASSIVE
    ):
        self.server_id = server_id
        self.datacenter = datacenter
        self.role = initial_role
        self.is_healthy = True
        self.is_processing_orders = False
        
        # Database connections
        self.db_primary: Optional[psycopg2.connection] = None
        self.db_replica: Optional[psycopg2.connection] = None
        
        # Redis (state cache)
        self.redis: Optional[redis.Redis] = None
        
        # Metrics
        self.latency_ms = 0.0
        self.active_connections = 0
        self.orders_processed = 0
        self.last_heartbeat = datetime.utcnow()
    
    def get_health_status(self) -> ServerHealth:
        """Get current health status"""
        return ServerHealth(
            server_id=self.server_id,
            role=self.role,
            is_healthy=self.is_healthy,
            latency_ms=self.latency_ms,
            last_heartbeat=self.last_heartbeat,
            active_connections=self.active_connections,
            orders_per_second=self.orders_processed / 60.0  # Last minute
        )
    
    async def send_heartbeat(self) -> dict:
        """Send heartbeat to monitoring service"""
        self.last_heartbeat = datetime.utcnow()
        
        health = self.get_health_status()
        
        return {
            'server_id': health.server_id,
            'datacenter': self.datacenter,
            'role': health.role.value,
            'is_healthy': health.is_healthy,
            'latency_ms': health.latency_ms,
            'timestamp': health.last_heartbeat.isoformat(),
            'active_connections': health.active_connections,
            'orders_per_second': health.orders_per_second
        }
    
    async def promote_to_active(self):
        """
        Promote server to ACTIVE
        
        Steps:
        1. Verify data is synchronized
        2. Promote database replica to primary
        3. Start accepting orders
        4. Update load balancer
        5. Notify operations team
        """
        print(f"[{self.server_id}] *** PROMOTING TO ACTIVE ***")
        
        # Step 1: Check replication lag
        lag = await self._check_replication_lag()
        if lag > 1000:  # >1 second lag
            print(f"[{self.server_id}] WARNING: Replication lag {lag}ms")
        
        # Step 2: Promote database
        print(f"[{self.server_id}] Promoting database replica to primary...")
        await self._promote_database()
        
        # Step 3: Update role
        self.role = ServerRole.ACTIVE
        
        # Step 4: Start order processing
        print(f"[{self.server_id}] Starting order processing...")
        await self.start_order_processing()
        
        # Step 5: Update load balancer (DNS or HAProxy)
        print(f"[{self.server_id}] Updating load balancer...")
        await self._update_load_balancer()
        
        print(f"[{self.server_id}] ✓ PROMOTION COMPLETE - Now ACTIVE")
    
    async def demote_to_passive(self):
        """
        Demote server to PASSIVE
        
        Steps:
        1. Stop accepting new orders
        2. Wait for pending orders to complete
        3. Demote database to replica
        4. Start replication from new primary
        """
        print(f"[{self.server_id}] *** DEMOTING TO PASSIVE ***")
        
        # Step 1: Stop accepting orders
        print(f"[{self.server_id}] Stopping order processing...")
        await self.stop_order_processing()
        
        # Step 2: Wait for pending orders
        print(f"[{self.server_id}] Waiting for pending orders...")
        await self._wait_for_pending_orders()
        
        # Step 3: Update role
        self.role = ServerRole.PASSIVE
        
        # Step 4: Configure replication
        print(f"[{self.server_id}] Configuring as replica...")
        await self._configure_as_replica()
        
        print(f"[{self.server_id}] ✓ DEMOTION COMPLETE - Now PASSIVE")
    
    async def start_order_processing(self):
        """Start processing orders"""
        self.is_processing_orders = True
        print(f"[{self.server_id}] ✓ Accepting orders")
    
    async def stop_order_processing(self):
        """Stop processing orders"""
        self.is_processing_orders = False
        print(f"[{self.server_id}] ✗ Stopped accepting orders")
    
    async def _check_replication_lag(self) -> float:
        """Check database replication lag (milliseconds)"""
        # In production: Query PostgreSQL replication lag
        # SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000
        return 50.0  # Simulated
    
    async def _promote_database(self):
        """Promote database replica to primary"""
        # In production: Execute pg_promote() on PostgreSQL
        await asyncio.sleep(2)  # Simulate promotion time
    
    async def _configure_as_replica(self):
        """Configure database as replica"""
        # In production: Set up streaming replication
        await asyncio.sleep(1)
    
    async def _update_load_balancer(self):
        """Update load balancer to route traffic"""
        # In production: Update HAProxy/DNS/AWS ELB
        await asyncio.sleep(0.5)
    
    async def _wait_for_pending_orders(self):
        """Wait for pending orders to complete"""
        # In production: Check order queue is empty
        await asyncio.sleep(3)  # Simulate drain time


class FailoverCoordinator:
    """
    Coordinates failover between active and passive servers
    
    Responsibilities:
    - Monitor health of both servers
    - Detect failures (missed heartbeats, high latency)
    - Trigger failover when necessary
    - Prevent split-brain scenarios
    - Log all failover events
    """
    
    def __init__(
        self,
        active_server: TradingServer,
        passive_server: TradingServer,
        failover_threshold: int = 3
    ):
        self.active = active_server
        self.passive = passive_server
        self.failover_threshold = failover_threshold
        
        # State tracking
        self.missed_heartbeats = 0
        self.failover_count = 0
        self.last_failover_time: Optional[datetime] = None
        self.failover_history = []
        
        # Thresholds
        self.max_latency_ms = 1000  # 1 second
        self.heartbeat_interval_sec = 5
        
        print("[Coordinator] Initialized")
        print(f"  Active: {active_server.server_id} ({active_server.datacenter})")
        print(f"  Passive: {passive_server.server_id} ({passive_server.datacenter})")
    
    async def monitor_health(self):
        """
        Continuous health monitoring
        
        Runs every 5 seconds:
        1. Request heartbeat from active server
        2. Check if response is healthy
        3. Track consecutive failures
        4. Trigger failover if threshold exceeded
        """
        print(f"[Coordinator] Starting health monitoring (interval: {self.heartbeat_interval_sec}s)")
        
        while True:
            try:
                # Get heartbeat from active server
                heartbeat = await self.active.send_heartbeat()
                
                # Check health
                if not heartbeat['is_healthy']:
                    self.missed_heartbeats += 1
                    print(f"[Coordinator] ⚠️  Unhealthy heartbeat {self.missed_heartbeats}/{self.failover_threshold}")
                    
                    if self.missed_heartbeats >= self.failover_threshold:
                        await self.trigger_failover(FailoverReason.HEALTH_CHECK_FAILED)
                        self.missed_heartbeats = 0
                
                # Check latency
                elif heartbeat['latency_ms'] > self.max_latency_ms:
                    print(f"[Coordinator] ⚠️  High latency: {heartbeat['latency_ms']}ms")
                    await self.trigger_failover(FailoverReason.HIGH_LATENCY)
                
                else:
                    # Healthy - reset counter
                    if self.missed_heartbeats > 0:
                        print(f"[Coordinator] ✓ Active server recovered")
                    self.missed_heartbeats = 0
                
                await asyncio.sleep(self.heartbeat_interval_sec)
                
            except Exception as e:
                self.missed_heartbeats += 1
                print(f"[Coordinator] ⚠️  Heartbeat failed: {e} ({self.missed_heartbeats}/{self.failover_threshold})")
                
                if self.missed_heartbeats >= self.failover_threshold:
                    await self.trigger_failover(FailoverReason.HEALTH_CHECK_FAILED)
                    self.missed_heartbeats = 0
                
                await asyncio.sleep(self.heartbeat_interval_sec)
    
    async def trigger_failover(self, reason: FailoverReason):
        """
        Execute failover from active to passive
        
        Critical: Must prevent split-brain (both servers active)
        """
        print("\\n" + "=" * 80)
        print(f"🚨 FAILOVER TRIGGERED: {reason.value}")
        print("=" * 80)
        
        failover_start = datetime.utcnow()
        
        try:
            # Step 1: Fence active server (prevent split-brain)
            print(f"[Coordinator] Step 1/4: Fencing active server...")
            await self._fence_active_server()
            
            # Step 2: Verify passive is ready
            print(f"[Coordinator] Step 2/4: Verifying passive server...")
            passive_health = self.passive.get_health_status()
            if not passive_health.is_healthy:
                raise Exception("Passive server is not healthy - FAILOVER ABORTED")
            
            # Step 3: Promote passive to active
            print(f"[Coordinator] Step 3/4: Promoting passive to active...")
            await self.passive.promote_to_active()
            
            # Step 4: Demote old active to passive (if reachable)
            print(f"[Coordinator] Step 4/4: Demoting old active...")
            try:
                await self.active.demote_to_passive()
            except Exception as e:
                print(f"[Coordinator] Could not demote old active: {e}")
            
            # Swap references
            self.active, self.passive = self.passive, self.active
            
            # Record failover
            failover_duration = (datetime.utcnow() - failover_start).total_seconds()
            self.failover_count += 1
            self.last_failover_time = datetime.utcnow()
            
            self.failover_history.append({
                'failover_id': self.failover_count,
                'reason': reason.value,
                'old_active': self.passive.server_id,  # Swapped
                'new_active': self.active.server_id,
                'duration_seconds': failover_duration,
                'timestamp': failover_start.isoformat()
            })
            
            print("=" * 80)
            print(f"✓ FAILOVER COMPLETE in {failover_duration:.2f}s")
            print(f"  New Active: {self.active.server_id} ({self.active.datacenter})")
            print(f"  New Passive: {self.passive.server_id} ({self.passive.datacenter})")
            print("=" * 80 + "\\n")
            
            # Alert operations team
            await self._send_failover_alert(reason, failover_duration)
            
        except Exception as e:
            print(f"[Coordinator] ❌ FAILOVER FAILED: {e}")
            # In production: Trigger manual intervention
            raise
    
    async def _fence_active_server(self):
        """
        Fence active server to prevent split-brain
        
        Methods:
        - Update load balancer to stop routing traffic
        - Revoke database write permissions
        - Kill process if necessary (STONITH)
        """
        # Simulate fencing
        self.active.is_processing_orders = False
        await asyncio.sleep(0.5)
    
    async def _send_failover_alert(self, reason: FailoverReason, duration: float):
        """Send critical alert to operations team"""
        alert = {
            'severity': 'CRITICAL',
            'event': 'FAILOVER',
            'reason': reason.value,
            'duration_seconds': duration,
            'new_active': self.active.server_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # In production: Send to PagerDuty, Slack, email
        print(f"[Alert] {alert}")
    
    def get_failover_statistics(self) -> dict:
        """Get failover statistics"""
        return {
            'total_failovers': self.failover_count,
            'last_failover': self.last_failover_time.isoformat() if self.last_failover_time else None,
            'current_active': self.active.server_id,
            'current_passive': self.passive.server_id,
            'failover_history': self.failover_history
        }


# Example: Failover scenario
async def failover_example():
    """Demonstrate active-passive failover"""
    
    # Setup servers
    server_nyc = TradingServer(
        server_id="SERVER-NYC-01",
        datacenter="NYC",
        initial_role=ServerRole.ACTIVE
    )
    
    server_chi = TradingServer(
        server_id="SERVER-CHI-01",
        datacenter="Chicago",
        initial_role=ServerRole.PASSIVE
    )
    
    # Start order processing on active
    await server_nyc.start_order_processing()
    
    # Setup coordinator
    coordinator = FailoverCoordinator(
        active_server=server_nyc,
        passive_server=server_chi,
        failover_threshold=3
    )
    
    # Start monitoring
    monitoring_task = asyncio.create_task(coordinator.monitor_health())
    
    # Simulate operations
    await asyncio.sleep(10)
    
    # Simulate failure
    print("\\n[Test] Simulating active server failure...\\n")
    server_nyc.is_healthy = False
    
    # Wait for failover
    await asyncio.sleep(20)
    
    # Print statistics
    stats = coordinator.get_failover_statistics()
    print("\\nFailover Statistics:")
    print(f"  Total failovers: {stats['total_failovers']}")
    print(f"  Current active: {stats['current_active']}")
    
    monitoring_task.cancel()

# asyncio.run(failover_example())
\`\`\`

---

## Database Replication

\`\`\`sql
-- =============================================================================
-- PostgreSQL Streaming Replication Setup
-- =============================================================================

-- PRIMARY SERVER CONFIGURATION
-- /etc/postgresql/14/main/postgresql.conf

-- Enable replication
wal_level = replica
max_wal_senders = 10
max_replication_slots = 10
wal_keep_size = 1GB

-- Archive mode (for point-in-time recovery)
archive_mode = on
archive_command = 'cp %p /archive/%f'

-- Synchronous replication (zero data loss)
synchronous_commit = on
synchronous_standby_names = 'standby1'

-- Connection settings
listen_addresses = '*'
max_connections = 200

-- =============================================================================
-- Create replication user
-- =============================================================================

CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'strong_password_here';

-- Grant permissions
GRANT CONNECT ON DATABASE trading TO replicator;

-- =============================================================================
-- Configure pg_hba.conf (allow replication connections)
-- =============================================================================

-- Add to /etc/postgresql/14/main/pg_hba.conf:
-- host replication replicator 10.0.0.0/8 md5

-- =============================================================================
-- REPLICA SERVER CONFIGURATION
-- =============================================================================

-- 1. Stop PostgreSQL on replica
-- sudo systemctl stop postgresql

-- 2. Remove existing data directory
-- sudo rm -rf /var/lib/postgresql/14/main

-- 3. Create base backup from primary
-- sudo -u postgres pg_basebackup -h primary_server -D /var/lib/postgresql/14/main -U replicator -P -v -R

-- 4. Start PostgreSQL
-- sudo systemctl start postgresql

-- =============================================================================
-- Monitoring Replication
-- =============================================================================

-- On PRIMARY: Check replication status
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    sync_state,
    EXTRACT(EPOCH FROM (now() - backend_start)) AS connection_age_seconds,
    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS replication_lag_seconds
FROM pg_stat_replication;

-- On REPLICA: Check replication lag
SELECT
    now() - pg_last_xact_replay_timestamp() AS replication_lag,
    pg_is_in_recovery() AS is_replica;

-- On REPLICA: Check replay progress
SELECT
    pg_last_wal_receive_lsn() AS receive_lsn,
    pg_last_wal_replay_lsn() AS replay_lsn,
    pg_last_wal_replay_lsn() = pg_last_wal_receive_lsn() AS replay_caught_up;

-- =============================================================================
-- Failover: Promote Replica to Primary
-- =============================================================================

-- On REPLICA: Promote to primary
SELECT pg_promote();

-- Or use command line:
-- sudo -u postgres pg_ctl promote -D /var/lib/postgresql/14/main

-- Verify promotion
SELECT pg_is_in_recovery();  -- Should return 'false'

-- =============================================================================
-- Automatic Failover with Patroni
-- =============================================================================

-- Patroni YAML configuration
"""
scope: trading_cluster
name: server_nyc_01

restapi:
  listen: 0.0.0.0:8008
  connect_address: 10.0.1.10:8008

etcd:
  host: 10.0.1.5:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        max_connections: 200
        max_wal_senders: 10

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 10.0.1.10:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  authentication:
    replication:
      username: replicator
      password: strong_password
    superuser:
      username: postgres
      password: admin_password
"""
\`\`\`

---

## Geographic Disaster Recovery

\`\`\`python
"""
Multi-Region DR with Active-Active

Setup:
- Region 1 (US-East): Primary trading
- Region 2 (US-West): Secondary trading
- Region 3 (EU): Tertiary (compliance)

All regions active, database conflict resolution via Galera/CockroachDB
"""

from enum import Enum
from typing import Dict, List

class Region(Enum):
    US_EAST = "US_EAST"
    US_WEST = "US_WEST"
    EU_WEST = "EU_WEST"

class GeoDRManager:
    """
    Geographic disaster recovery manager
    
    Features:
    - Multi-region active-active
    - Intelligent routing (lowest latency)
    - Automatic region failover
    - Data consistency across regions
    """
    
    def __init__(self):
        self.regions = {
            Region.US_EAST: {
                'datacenter': 'NYC',
                'status': 'ACTIVE',
                'latency_ms': 0,
                'capacity_pct': 100
            },
            Region.US_WEST: {
                'datacenter': 'San Francisco',
                'status': 'ACTIVE',
                'latency_ms': 0,
                'capacity_pct': 100
            },
            Region.EU_WEST: {
                'datacenter': 'London',
                'status': 'ACTIVE',
                'latency_ms': 0,
                'capacity_pct': 50  # Compliance only
            }
        }
    
    def route_order(self, client_location: str) -> Region:
        """
        Route order to nearest healthy region
        
        Logic:
        1. Calculate latency to each region
        2. Filter healthy regions
        3. Route to lowest latency
        """
        # Simplified: In production, use GeoDNS or Anycast
        if client_location.startswith('US'):
            return Region.US_EAST
        elif client_location.startswith('EU'):
            return Region.EU_WEST
        else:
            return Region.US_WEST
    
    def handle_region_failure(self, failed_region: Region):
        """Handle complete region failure"""
        print(f"[GeoDR] Region {failed_region.value} FAILED")
        
        # Mark region as down
        self.regions[failed_region]['status'] = 'DOWN'
        
        # Redistribute traffic to healthy regions
        healthy_regions = [
            r for r, info in self.regions.items()
            if info['status'] == 'ACTIVE'
        ]
        
        print(f"[GeoDR] Redirecting traffic to: {[r.value for r in healthy_regions]}")
\`\`\`

---

## DR Testing

\`\`\`python
"""
Disaster Recovery Testing (Chaos Engineering)
"""

import asyncio
from datetime import datetime

class DRTestSuite:
    """
    DR testing scenarios
    
    Test monthly:
    - Database failover
    - Complete datacenter failure
    - Network partition
    - Data corruption recovery
    """
    
    async def test_database_failover(self):
        """Test database failover (RTO target: <30 seconds)"""
        print("\\n" + "=" * 80)
        print("DR TEST: Database Failover")
        print("=" * 80)
        
        start = datetime.utcnow()
        
        # 1. Kill primary database
        print("[Test] Killing primary database...")
        await asyncio.sleep(1)
        
        # 2. Detect failure
        print("[Test] Failure detected...")
        await asyncio.sleep(5)
        
        # 3. Promote replica
        print("[Test] Promoting replica...")
        await asyncio.sleep(10)
        
        # 4. Update application
        print("[Test] Updating application config...")
        await asyncio.sleep(2)
        
        # 5. Resume trading
        print("[Test] Resuming trading...")
        
        duration = (datetime.utcnow() - start).total_seconds()
        
        print(f"\\n✓ Failover completed in {duration:.1f}s")
        print(f"  RTO target: 30s")
        print(f"  Result: {'PASS' if duration < 30 else 'FAIL'}")
    
    async def test_full_datacenter_failure(self):
        """Test complete datacenter failure (RTO target: <5 minutes)"""
        print("\\n" + "=" * 80)
        print("DR TEST: Complete Datacenter Failure")
        print("=" * 80)
        
        start = datetime.utcnow()
        
        # Simulate complete outage
        print("[Test] Datacenter offline...")
        await asyncio.sleep(30)
        
        duration = (datetime.utcnow() - start).total_seconds()
        print(f"\\n✓ Recovered in {duration/60:.1f} minutes")

# Run DR tests
# asyncio.run(DRTestSuite().test_database_failover())
\`\`\`

---

## Summary

**DR Strategy Comparison:**

| Strategy | RTO | RPO | Cost | Complexity | Use Case |
|----------|-----|-----|------|------------|----------|
| Backup/Restore | Hours | 1 day | Low | Low | Non-critical systems |
| Active-Passive | <5 min | <1 sec | Medium | Medium | Most trading systems |
| Active-Active | <1 sec | 0 | High | High | HFT, market making |
| Multi-Region | <1 sec | 0 | Very High | Very High | Global trading firms |

**Production DR Checklist:**
- ✅ Automated failover (<5 min RTO)
- ✅ Database streaming replication (<1 sec RPO)
- ✅ Geographic redundancy (multi-region)
- ✅ Split-brain prevention (fencing)
- ✅ Monthly DR testing
- ✅ Documented runbooks
- ✅ 24/7 on-call team
- ✅ Automated monitoring and alerts

**Next Section**: Module 14.14 - Production Deployment
`,
};
