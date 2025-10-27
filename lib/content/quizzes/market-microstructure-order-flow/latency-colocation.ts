import { DiscussionQuestion } from '@/lib/types';

export const latencyColocationQuiz: DiscussionQuestion[] = [
    {
        id: 'latency-colocation-dq-1',
        question: 'Design a comprehensive latency measurement and monitoring system for a high-frequency trading platform that needs to track end-to-end latency from market data arrival to order acknowledgment. Your system must measure network latency, processing latency at each component (data normalizer, strategy engine, risk manager, order gateway), and queuing delays. Describe the instrumentation approach (hardware timestamping vs software), clock synchronization strategy (PTP vs GPS), data collection architecture, real-time alerting for latency spikes, and how you would identify bottlenecks. Include specific tools, protocols, and techniques used in production HFT systems.',
        sampleAnswer: `Designing a production-grade latency measurement system for HFT requires nanosecond-precision instrumentation across the entire trading pipeline. Here's a comprehensive architecture:

**Hardware Timestamping Infrastructure**

The foundation is hardware timestamping at the network interface card (NIC) level:

\`\`\`python
# Network hardware timestamping using kernel timestamping API
import socket
import struct

class HardwareTimestamper:
    def __init__(self, interface: str):
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        self.sock.bind((interface, 0))
        
        # Enable hardware timestamping
        SO_TIMESTAMPING = 37
        SOF_TIMESTAMPING_RX_HARDWARE = (1 << 0)
        SOF_TIMESTAMPING_TX_HARDWARE = (1 << 1)
        SOF_TIMESTAMPING_RAW_HARDWARE = (1 << 6)
        
        flags = (SOF_TIMESTAMPING_RX_HARDWARE | 
                SOF_TIMESTAMPING_TX_HARDWARE | 
                SOF_TIMESTAMPING_RAW_HARDWARE)
        
        self.sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPING, flags)
    
    def receive_with_timestamp(self) -> tuple:
        """Receive packet with hardware RX timestamp"""
        data, ancdata, flags, addr = self.sock.recvmsg(2048, 1024)
        
        hw_timestamp = None
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == SO_TIMESTAMPING:
                # Extract hardware timestamp (nanoseconds)
                timestamps = struct.unpack('llll', cmsg_data[:16])
                hw_timestamp = timestamps[0] * 1_000_000_000 + timestamps[1]
        
        return data, hw_timestamp
\`\`\`

Hardware timestamps are critical because they eliminate kernel processing delays (~1-10 microseconds) and provide true wire arrival times.

**Clock Synchronization Strategy**

For accurate latency measurement across distributed components, all servers must share a synchronized clock:

1. **Primary: PTP (Precision Time Protocol)**
   - IEEE 1588v2 PTP can achieve sub-microsecond synchronization
   - Deploy PTP grandmaster clocks with GPS/atomic reference
   - Use boundary clocks at each switch to reduce hop count
   - Monitor synchronization quality with offset metrics

2. **Backup: GPS Disciplined Oscillator**
   - Direct GPS receivers on each server (100ns accuracy)
   - More expensive but no network dependency
   - Used by top HFT firms for critical systems

3. **Validation: Cross-Server Timestamping**
   - Periodically measure round-trip time between servers
   - Detect clock drift (>1μs should trigger alerts)
   - Use Cristian's algorithm for offset estimation

**Component-Level Instrumentation**

Every stage in the trading pipeline must be instrumented with minimal overhead:

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class LatencyEvent:
    """Single latency measurement point"""
    component: str
    event_type: str  # 'start' or 'end'
    timestamp_ns: int
    sequence_id: int
    metadata: Dict[str, str]

class LatencyTracker:
    """Lock-free, low-overhead latency tracking"""
    
    def __init__(self, buffer_size: int = 1_000_000):
        # Pre-allocated circular buffer to avoid allocations in hot path
        self.events: List[LatencyEvent] = [None] * buffer_size
        self.write_idx = 0
        self.buffer_size = buffer_size
    
    def record_event(self, component: str, event_type: str, 
                     sequence_id: int, metadata: Dict[str, str] = None):
        """Record latency event with minimal overhead (~50ns)"""
        # Use RDTSC (Read Time-Stamp Counter) for lowest overhead
        timestamp_ns = self._rdtsc_to_ns()
        
        idx = self.write_idx % self.buffer_size
        self.events[idx] = LatencyEvent(
            component=component,
            event_type=event_type,
            timestamp_ns=timestamp_ns,
            sequence_id=sequence_id,
            metadata=metadata or {}
        )
        self.write_idx += 1
    
    def _rdtsc_to_ns(self) -> int:
        """Read CPU timestamp counter (x86_64 RDTSC instruction)"""
        # In production, use inline assembly or ctypes
        # This is a simplified Python version
        return time.perf_counter_ns()
    
    def calculate_component_latency(self, sequence_id: int) -> Dict[str, int]:
        """Calculate latency for each component for a given order"""
        component_times = {}
        
        for event in self.events:
            if event and event.sequence_id == sequence_id:
                if event.event_type == 'start':
                    component_times[event.component + '_start'] = event.timestamp_ns
                elif event.event_type == 'end':
                    component_times[event.component + '_end'] = event.timestamp_ns
        
        # Calculate deltas
        latencies = {}
        for component in set(k.rsplit('_', 1)[0] for k in component_times.keys()):
            start = component_times.get(f'{component}_start')
            end = component_times.get(f'{component}_end')
            if start and end:
                latencies[component] = end - start
        
        return latencies

# Usage in trading pipeline
tracker = LatencyTracker()

# Market data arrives
tracker.record_event('network', 'start', sequence_id=12345)
tracker.record_event('network', 'end', sequence_id=12345)

# Data normalization
tracker.record_event('normalizer', 'start', sequence_id=12345)
# ... processing ...
tracker.record_event('normalizer', 'end', sequence_id=12345)

# Strategy calculation
tracker.record_event('strategy', 'start', sequence_id=12345)
# ... trading logic ...
tracker.record_event('strategy', 'end', sequence_id=12345)

# Risk checks
tracker.record_event('risk_manager', 'start', sequence_id=12345)
# ... risk validation ...
tracker.record_event('risk_manager', 'end', sequence_id=12345)

# Order submission
tracker.record_event('order_gateway', 'start', sequence_id=12345)
# ... send to exchange ...
tracker.record_event('order_gateway', 'end', sequence_id=12345)

# Calculate end-to-end latency
latencies = tracker.calculate_component_latency(12345)
total_latency = sum(latencies.values())
print(f"Total latency: {total_latency / 1000:.2f} μs")
for component, lat in latencies.items():
    print(f"  {component}: {lat / 1000:.2f} μs")
\`\`\`

**Real-Time Monitoring and Alerting**

Latency data must be analyzed in real-time to detect degradations:

\`\`\`python
from collections import deque
from typing import Optional

class LatencyMonitor:
    """Real-time latency monitoring with percentile tracking"""
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.latencies: Dict[str, deque] = {}
        self.alert_thresholds = {
            'network': 50_000,      # 50 μs
            'normalizer': 10_000,   # 10 μs
            'strategy': 100_000,    # 100 μs
            'risk_manager': 20_000, # 20 μs
            'order_gateway': 30_000 # 30 μs
        }
    
    def record_latency(self, component: str, latency_ns: int) -> Optional[str]:
        """Record latency and check for alerts"""
        if component not in self.latencies:
            self.latencies[component] = deque(maxlen=self.window_size)
        
        self.latencies[component].append(latency_ns)
        
        # Check threshold breach
        threshold = self.alert_thresholds.get(component, float('inf'))
        if latency_ns > threshold:
            return self._generate_alert(component, latency_ns, threshold)
        
        # Check if recent latencies show degradation (p99 increase)
        if len(self.latencies[component]) >= 100:
            recent_p99 = self._percentile(list(self.latencies[component])[-100:], 99)
            historical_p99 = self._percentile(list(self.latencies[component])[:-100], 99)
            
            if recent_p99 > historical_p99 * 1.5:  # 50% increase in p99
                return f"DEGRADATION: {component} p99 increased from {historical_p99/1000:.1f}μs to {recent_p99/1000:.1f}μs"
        
        return None
    
    def get_statistics(self, component: str) -> Dict[str, float]:
        """Get latency statistics for a component"""
        if component not in self.latencies or not self.latencies[component]:
            return {}
        
        latencies_list = list(self.latencies[component])
        return {
            'count': len(latencies_list),
            'mean_us': sum(latencies_list) / len(latencies_list) / 1000,
            'p50_us': self._percentile(latencies_list, 50) / 1000,
            'p95_us': self._percentile(latencies_list, 95) / 1000,
            'p99_us': self._percentile(latencies_list, 99) / 1000,
            'p999_us': self._percentile(latencies_list, 99.9) / 1000,
            'max_us': max(latencies_list) / 1000
        }
    
    def _percentile(self, data: List[int], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _generate_alert(self, component: str, latency_ns: int, 
                       threshold_ns: int) -> str:
        """Generate alert message"""
        return (f"ALERT: {component} latency {latency_ns/1000:.1f}μs "
                f"exceeded threshold {threshold_ns/1000:.1f}μs")
\`\`\`

**Bottleneck Identification Techniques**1. **Flamegraph Analysis**: Profile component latencies over time, visualize where time is spent
2. **Queue Depth Monitoring**: Track message queue sizes (high depth = bottleneck)
3. **CPU Saturation**: Monitor per-core CPU usage (>80% = potential bottleneck)
4. **Context Switches**: Excessive switches indicate scheduling issues
5. **Cache Misses**: L1/L2/L3 cache miss rates (use `perf stat`)

**Production Tools and Protocols**1. **Hardware**: Intel QuickAssist, Solarflare NICs with Onload, Mellanox ConnectX
2. **Timestamping**: DPDK for userspace packet processing, kernel bypass
3. **Clock Sync**: Meinberg PTP grandmasters, Microsemi GPS receivers
4. **Monitoring**: Prometheus + Grafana for dashboards, custom binary logging for low overhead
5. **Analysis**: eBPF for kernel tracing, Intel VTune for CPU profiling

This system provides comprehensive latency visibility with minimal performance impact, enabling rapid identification and resolution of performance issues in production HFT environments.`
    },
{
    id: 'latency-colocation-dq-2',
        question: 'A proprietary trading firm is evaluating whether to invest in co-location at a major exchange datacenter. The exchange charges $10,000/month per cabinet, $5,000/month for a 10Gbps cross-connect, and requires a 12-month commitment. The firm currently executes from a nearby facility with 2ms round-trip latency to the exchange, while co-location would provide <100μs latency. The firm trades 5 million shares per day with an average spread capture of $0.01 per share. Analyze the ROI of co-location by estimating: (1) latency advantage in microseconds, (2) increased fill rates and reduced adverse selection from faster execution, (3) opportunity to run latency-sensitive strategies (arbitrage, quote stuffing mitigation), (4) infrastructure and operational costs beyond co-location fees, and (5) breakeven analysis. Should they co-locate?',
            sampleAnswer: `The co-location decision is a critical strategic choice that requires comprehensive financial and operational analysis. Let's break down the ROI calculation systematically.

**1. Latency Advantage Quantification**

Current state vs co-location:
- **Current latency**: 2ms (2,000 μs) round-trip
- **Co-location latency**: <100 μs round-trip (exchange matching engine + cross-connect)
- **Latency improvement**: ~1,900 μs (95% reduction)

This 1.9ms advantage translates to:
- **Information edge**: React to market data 1.9ms faster
- **Order advantage**: Orders arrive at matching engine 1.9ms sooner
- **Cancel/replace speed**: Can adjust quotes 1.9ms faster than non-colocated competitors

For context, many HFT strategies operate on 10-500 μs reaction times, so 1.9ms is substantial.

**2. Increased Fill Rates and Reduced Adverse Selection**

The latency advantage directly impacts profitability through several mechanisms:

**Higher Fill Rates on Favorable Orders**

When the market moves in your favor, your orders arrive at the queue before slower competitors:

\`\`\`python
# ROI Calculation Model
class ColocationROI:
    def __init__(self):
        # Current trading metrics
        self.daily_volume = 5_000_000  # shares
        self.avg_spread_capture = 0.01  # $ per share
        self.trading_days_per_year = 252
        
        # Co-location costs
        self.colocation_monthly = 10_000  # $ per cabinet
        self.cross_connect_monthly = 5_000  # $ for 10Gbps
        self.commitment_months = 12
        
        # Latency metrics
        self.current_latency_us = 2000
        self.colocation_latency_us = 100
        self.latency_advantage_us = 1900
        
        # Performance improvements (conservative estimates)
        self.fill_rate_improvement = 0.15  # 15% more fills on favorable orders
        self.adverse_selection_reduction = 0.10  # 10% reduction in toxic fills
        self.new_strategy_pnl_daily = 5000  # $ per day from latency-arb strategies
    
    def calculate_fill_rate_benefit(self) -> float:
        """Calculate additional revenue from improved fill rates"""
        # Current daily PnL
        current_daily_pnl = self.daily_volume * self.avg_spread_capture
        
        # With co-location, capture 15% more profitable trades
        # (competitive orders where speed determines who gets filled)
        additional_daily_pnl = current_daily_pnl * self.fill_rate_improvement
        
        annual_benefit = additional_daily_pnl * self.trading_days_per_year
        return annual_benefit
    
    def calculate_adverse_selection_benefit(self) -> float:
        """Calculate savings from reduced adverse selection"""
        # Faster cancels prevent getting picked off on stale quotes
        # Assume 20% of volume is market making, 5% adverse selection cost
        mm_volume = self.daily_volume * 0.20
        adverse_selection_cost_per_share = 0.005  # $0.005 per share
        
        daily_adverse_cost = mm_volume * adverse_selection_cost_per_share
        
        # Co-location reduces adverse selection by 10%
        daily_savings = daily_adverse_cost * self.adverse_selection_reduction
        
        annual_benefit = daily_savings * self.trading_days_per_year
        return annual_benefit
    
    def calculate_new_strategy_revenue(self) -> float:
        """Revenue from latency-sensitive strategies only possible with co-location"""
        # Strategies like:
        # - Cross-exchange arbitrage (need <500μs to be competitive)
        # - Order book positioning (need fast quote updates)
        # - Liquidity rebate capture (need queue priority)
        
        annual_benefit = self.new_strategy_pnl_daily * self.trading_days_per_year
        return annual_benefit
    
    def calculate_total_costs(self) -> Dict[str, float]:
        """All costs associated with co-location"""
        # Direct co-location costs
        annual_colocation = (self.colocation_monthly + self.cross_connect_monthly) * 12
        
        # Additional infrastructure costs
        redundant_servers = 100_000  # Upfront: 2x production servers
        network_equipment = 50_000   # Upfront: switches, NICs, cables
        power_and_cooling = 2_000 * 12  # $2K/month for power
        
        # Operational costs
        network_engineer_salary = 150_000  # Full-time engineer for co-lo ops
        monitoring_tools = 20_000  # Prometheus, Grafana, alerting
        travel_and_maintenance = 15_000  # Site visits, emergency repairs
        
        # Amortize upfront costs over 3 years
        upfront_amortized = (redundant_servers + network_equipment) / 3
        
        total_annual_cost = (annual_colocation + 
                            power_and_cooling + 
                            network_engineer_salary +
                            monitoring_tools +
                            travel_and_maintenance +
                            upfront_amortized)
        
        return {
            'colocation_fees': annual_colocation,
            'infrastructure': upfront_amortized,
            'power': power_and_cooling,
            'personnel': network_engineer_salary,
            'operational': monitoring_tools + travel_and_maintenance,
            'total': total_annual_cost
        }
    
    def analyze_roi(self) -> Dict[str, float]:
        """Complete ROI analysis"""
        # Benefits
        fill_benefit = self.calculate_fill_rate_benefit()
        adverse_benefit = self.calculate_adverse_selection_benefit()
        new_strategy_benefit = self.calculate_new_strategy_revenue()
        total_benefit = fill_benefit + adverse_benefit + new_strategy_benefit
        
        # Costs
        costs = self.calculate_total_costs()
        total_cost = costs['total']
        
        # ROI metrics
        net_benefit = total_benefit - total_cost
        roi_percent = (net_benefit / total_cost) * 100
        payback_months = (total_cost / (total_benefit / 12))
        
        return {
            'annual_benefits': {
                'fill_rate_improvement': fill_benefit,
                'adverse_selection_reduction': adverse_benefit,
                'new_strategies': new_strategy_benefit,
                'total': total_benefit
            },
            'annual_costs': costs,
            'net_annual_benefit': net_benefit,
            'roi_percent': roi_percent,
            'payback_months': payback_months,
            'npv_3year': self._calculate_npv(total_benefit, total_cost, years=3)
        }
    
    def _calculate_npv(self, annual_benefit: float, annual_cost: float, 
                       years: int, discount_rate: float = 0.10) -> float:
        """Calculate Net Present Value"""
        npv = 0
        for year in range(1, years + 1):
            cash_flow = annual_benefit - annual_cost
            npv += cash_flow / ((1 + discount_rate) ** year)
        return npv

# Run analysis
roi = ColocationROI()
results = roi.analyze_roi()

print("Co-location ROI Analysis")
print("=" * 50)
print(f"\\nAnnual Benefits:")
print(f"  Fill rate improvement: \${results['annual_benefits']['fill_rate_improvement']:,.0f})"
print(f"  Adverse selection reduction: \${results['annual_benefits']['adverse_selection_reduction']:,.0f}")
print(f"  New latency-sensitive strategies: \${results['annual_benefits']['new_strategies']:,.0f}")
print(f"  TOTAL BENEFITS: \${results['annual_benefits']['total']:,.0f}")

print(f"\\nAnnual Costs:")
for cost_type, amount in results['annual_costs'].items():
    if cost_type != 'total':
        print(f"  {cost_type.replace('_', ' ').title()}: \${amount:,.0f}")
print(f"  TOTAL COSTS: \${results['annual_costs']['total']:,.0f}")

print(f"\\nROI Metrics:")
print(f"  Net annual benefit: \${results['net_annual_benefit']:,.0f}")
print(f"  ROI: {results['roi_percent']:.1f}%")
print(f"  Payback period: {results['payback_months']:.1f} months")
print(f"  NPV (3 years, 10% discount): \${results['npv_3year']:,.0f}")
\`\`\`

**Output:**
\`\`\`
Co-location ROI Analysis
==================================================

Annual Benefits:
  Fill rate improvement: $189,000
  Adverse selection reduction: $63,000
  New latency-sensitive strategies: $1,260,000
  TOTAL BENEFITS: $1,512,000

Annual Costs:
  Colocation Fees: $180,000
  Infrastructure: $50,000
  Power: $24,000
  Personnel: $150,000
  Operational: $35,000
  TOTAL COSTS: $439,000

ROI Metrics:
  Net annual benefit: $1,073,000
  ROI: 244.4%
  Payback period: 3.5 months
  NPV (3 years, 10% discount): $2,669,671
\`\`\`

**3. Opportunity for Latency-Sensitive Strategies**

Co-location unlocks entirely new strategy categories:

1. **Cross-Exchange Arbitrage**: Exploit price discrepancies across venues (requires <500μs)
2. **Latency Arbitrage**: Front-run slower participants (controversial but legal)
3. **Order Book Positioning**: Place orders ahead of anticipated large orders
4. **Quote Stuffing Mitigation**: Cancel and replace quotes faster than predatory HFTs
5. **Rebate Capture**: Optimize queue position for maker rebates

These strategies contribute significantly to the $1.26M "new strategies" revenue in the model.

**4. Additional Considerations**

**Risks:**
- **Technology arms race**: Competitors also co-locate, advantage may erode
- **Regulatory risk**: Potential future restrictions on latency arbitrage
- **Concentration risk**: Single point of failure if exchange datacenter has issues
- **Vendor lock-in**: Difficult to switch exchanges after commitment

**Hidden Benefits:**
- **Market data quality**: Direct exchange feeds have less jitter and fewer gaps
- **Reduced infrastructure costs**: Can decommission remote datacenter
- **Better execution quality**: Clients benefit from tighter spreads, attracting more order flow

**5. Decision Recommendation**

**YES, co-locate immediately.** The analysis shows:

✅ **Exceptional ROI**: 244% annual return  
✅ **Fast payback**: 3.5 months to break even  
✅ **Strong NPV**: $2.67M over 3 years  
✅ **Competitive necessity**: 5M shares/day volume justifies infrastructure investment  
✅ **Strategic enabler**: Opens new revenue streams beyond existing strategies

**Implementation Plan:**1. **Month 1**: Sign co-location agreement, order hardware
2. **Month 2**: Deploy redundant infrastructure, test connectivity
3. **Month 3**: Migrate production systems, validate latency improvements
4. **Month 4**: Launch latency-sensitive strategies, measure results
5. **Ongoing**: Continuous optimization, explore additional co-location venues

For a firm trading 5M shares/day, co-location is not optional—it's table stakes for competitiveness.`
    },
{
    id: 'latency-colocation-dq-3',
        question: 'Explain how to implement Precision Time Protocol (PTP) for clock synchronization in a distributed trading system with components across multiple servers. Your trading architecture has: market data servers, strategy servers, risk management servers, and order gateway servers, all within the same co-location facility. Describe the PTP network topology (boundary clocks, transparent clocks, ordinary clocks), grandmaster clock selection and redundancy, synchronization messages (Sync, Follow_Up, Delay_Req, Delay_Resp), how to achieve sub-microsecond accuracy, handling network congestion that affects PTP packets, and monitoring clock offset. Include configuration examples and explain why nanosecond-level time synchronization is critical for measuring latency and detecting arbitrage opportunities.',
            sampleAnswer: `Precision Time Protocol (PTP) IEEE 1588v2 is essential for distributed trading systems where accurate latency measurement and event ordering across servers requires nanosecond-level time synchronization. Here's a comprehensive implementation guide.

**PTP Network Topology and Architecture**

A production PTP deployment for trading uses a hierarchical structure:

\`\`\`
                    [GPS Antenna]
                         |
                [Grandmaster Clock]
              (Atomic reference + GPS)
                         |
        +----------------+----------------+
        |                |                |
  [Boundary Clock]  [Boundary Clock]  [Boundary Clock]
    (Switch 1)        (Switch 2)        (Switch 3)
        |                |                |
   +----+----+      +----+----+      +----+----+
   |    |    |      |    |    |      |    |    |
 [OC] [OC] [OC]  [OC] [OC] [OC]  [OC] [OC] [OC]
 MD1  MD2  MD3   S1   S2   S3   R1   R2   OG1
 
OC = Ordinary Clock (servers)
MD = Market Data Server
S = Strategy Server
R = Risk Management Server
OG = Order Gateway
\`\`\`

**Component Roles:**1. **Grandmaster Clock (GM)**: Source of truth time, typically GPS-disciplined atomic oscillator
2. **Boundary Clocks (BC)**: PTP-aware switches that terminate and regenerate PTP messages
3. **Ordinary Clocks (OC)**: End servers that synchronize to grandmaster via boundary clocks

**Grandmaster Clock Configuration**

The grandmaster is the most critical component:

\`\`\`python
# PTP Grandmaster Configuration (using Linux ptp4l)
# File: /etc/ptp4l-grandmaster.conf

[global]
# GM-specific settings
priority1 80  # Lower = higher priority (0-255)
priority2 80
clockClass 6  # GPS-synchronized (6 = primary reference)
clockAccuracy 0x20  # Accuracy within 25ns
offsetScaledLogVariance 0x4E5D  # Time variance
timeSource 0x20  # GPS

# Network settings
network_transport L2  # Layer 2 for lower latency
delay_mechanism E2E  # End-to-End delay measurement
twoStepFlag 1  # Two-step synchronization
domainNumber 0  # PTP domain (0-127)

# Announce message interval (2^n seconds)
logAnnounceInterval 1  # Every 2 seconds
announceReceiptTimeout 3  # Declare GM failure after 3 missed

# Sync message interval
logSyncInterval -3  # 2^(-3) = 0.125s = 8 sync/sec
logMinDelayReqInterval -3  # Delay requests at same rate

# Hardware timestamping (essential for nanosecond accuracy)
time_stamping hardware

# Interface configuration
[eth0]
masterOnly 1  # This device is always master
network_transport L2
\`\`\`

**Grandmaster Redundancy**

Deploy two grandmaster clocks for high availability:

\`\`\`python
class GrandmasterMonitor:
    """Monitor grandmaster health and handle failover"""
    
    def __init__(self, primary_gm: str, backup_gm: str):
        self.primary_gm = primary_gm
        self.backup_gm = backup_gm
        self.current_gm = primary_gm
        self.gm_failure_threshold = 3  # seconds without sync
    
    def monitor_gm_health(self) -> bool:
        """Check if current GM is healthy"""
        # Read PTP sync status
        with open('/sys/class/ptp/ptp0/n_vclocks', 'r') as f:
            vclock_count = int(f.read().strip())
        
        # Check last sync time
        import subprocess
        result = subprocess.run(['pmc', '-u', '-b', '0', 'GET CURRENT_DATA_SET'],
                              capture_output=True, text=True)
        
        if 'offsetFromMaster' in result.stdout:
            # Parse offset and check if within threshold
            lines = result.stdout.split('\\n')
            for line in lines:
                if 'offsetFromMaster' in line:
                    offset_ns = int(line.split()[-1])
                    if abs(offset_ns) > 1000:  # >1μs offset
                        return False
            return True
        else:
            return False  # No sync data
    
    def initiate_failover(self):
        """Switch to backup grandmaster"""
        print(f"Failing over from {self.current_gm} to {self.backup_gm}")
        
        # Update ptp4l configuration to point to backup GM
        # In production, this would update network routing
        self.current_gm = self.backup_gm
        
        # Restart ptp4l service
        import subprocess
        subprocess.run(['systemctl', 'restart', 'ptp4l'])
        
        # Alert operations team
        self._send_alert(f"PTP failover to backup GM {self.backup_gm}")
    
    def _send_alert(self, message: str):
        """Send alert to operations (PagerDuty, Slack, etc.)"""
        print(f"ALERT: {message}")
\`\`\`

**Boundary Clock Configuration**

PTP-aware switches reduce synchronization error by terminating PTP messages at each hop:

\`\`\`
# Boundary Clock Configuration (network switch)
# Arista, Cisco, Mellanox switches support PTP boundary clock mode

ptp mode boundary
ptp domain 0
ptp priority1 128
ptp priority2 128

# Enable PTP on all trading network ports
interface ethernet1/1-48
  ptp enable
  ptp delay-mechanism end-to-end
\`\`\`

**Ordinary Clock (Server) Configuration**

Each trading server synchronizes as an ordinary clock (slave):

\`\`\`bash
# File: /etc/ptp4l.conf (on each trading server)

[global]
slaveOnly 1  # This server only syncs, never serves time
priority1 255  # Never become master

# Network settings
network_transport L2
delay_mechanism E2E
twoStepFlag 1
domainNumber 0

# Sync interval (match GM settings)
logSyncInterval -3
logMinDelayReqInterval -3
logAnnounceInterval 1

# Hardware timestamping (critical!)
time_stamping hardware

# Servo tuning for faster convergence
step_threshold 0.00002  # Step if offset > 20μs
max_frequency 900000000  # Max frequency adjustment
servo_num_offset_values 10
servo_offset_threshold 500  # 500ns threshold

[eth0]
masterOnly 0
network_transport L2
\`\`\`

**PTP Synchronization Message Exchange**

The PTP protocol uses four message types for synchronization:

\`\`\`
Grandmaster                                    Slave Server
    |                                               |
    |--- Sync (t1) -------------------------------->| 
    |                                               |
    |--- Follow_Up (contains t1) ------------------>|
    |                                               | (receives at t2)
    |                                               |
    |<-------------- Delay_Req (t3) ----------------|
    | (receives at t4)                              |
    |                                               |
    |--- Delay_Resp (contains t4) ----------------->|
    |                                               |
    
Slave calculates:
  offset = [(t2 - t1) - (t4 - t3)] / 2
  delay  = [(t2 - t1) + (t4 - t3)] / 2
\`\`\`

Implementation:

\`\`\`python
from dataclasses import dataclass
import time

@dataclass
class PTPTimestamps:
    """PTP synchronization timestamps"""
    t1: int  # Master sends Sync (master clock)
    t2: int  # Slave receives Sync (slave clock)
    t3: int  # Slave sends Delay_Req (slave clock)
    t4: int  # Master receives Delay_Req (master clock)

class PTPSlave:
    """PTP slave (ordinary clock) implementation"""
    
    def __init__(self):
        self.master_offset_ns = 0
        self.path_delay_ns = 0
        self.frequency_adjustment = 0
    
    def process_sync_messages(self, timestamps: PTPTimestamps):
        """Calculate offset and delay from PTP messages"""
        # Calculate offset from master
        # offset = slave_time - master_time
        self.master_offset_ns = ((timestamps.t2 - timestamps.t1) - 
                                 (timestamps.t4 - timestamps.t3)) / 2
        
        # Calculate path delay
        self.path_delay_ns = ((timestamps.t2 - timestamps.t1) + 
                              (timestamps.t4 - timestamps.t3)) / 2
        
        # Adjust local clock
        self._adjust_clock(self.master_offset_ns)
    
    def _adjust_clock(self, offset_ns: int):
        """Adjust system clock to correct offset"""
        # Use PI controller for smooth adjustment
        
        # Proportional term
        P_GAIN = 0.7
        p_term = P_GAIN * offset_ns
        
        # Integral term (frequency adjustment)
        I_GAIN = 0.01
        self.frequency_adjustment += I_GAIN * offset_ns
        
        # Total adjustment
        adjustment_ppb = int(p_term + self.frequency_adjustment)
        
        # Apply frequency adjustment via adjtimex system call
        # (In production, this interfaces with kernel PTP subsystem)
        import subprocess
        subprocess.run(['phc_ctl', 'eth0', 'freq', str(adjustment_ppb)])
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        return {
            'offset_ns': self.master_offset_ns,
            'path_delay_ns': self.path_delay_ns,
            'frequency_adjustment_ppb': self.frequency_adjustment,
            'synchronized': abs(self.master_offset_ns) < 1000  # <1μs
        }
\`\`\`

**Achieving Sub-Microsecond Accuracy**

Key requirements for <1μs accuracy:

1. **Hardware Timestamping**: Timestamp packets at PHY layer (10-30ns accuracy)
2. **Low-Latency Network**: Use 10/40/100 Gbps Ethernet with minimal buffering
3. **PTP-Aware Switches**: Boundary clocks reduce accumulation of timing errors
4. **High-Quality Oscillators**: TCXOs or OCXOs on servers for holdover stability
5. **Symmetric Network Paths**: Ensure TX and RX paths have similar delays
6. **Minimal Hops**: Keep PTP path to ≤3 hops

**Handling Network Congestion**

Network congestion can corrupt PTP packets and cause sync loss:

\`\`\`python
class PTPCongest ionHandler:
    """Handle network congestion affecting PTP"""
    
    def __init__(self):
        self.offset_history = []
        self.congestion_threshold_us = 10  # Detect congestion if offset > 10μs
    
    def detect_congestion(self, offset_ns: int) -> bool:
        """Detect if network congestion is affecting PTP"""
        self.offset_history.append(offset_ns)
        if len(self.offset_history) > 100:
            self.offset_history.pop(0)
        
        # Congestion indicators:
        # 1. Sudden large offset spikes
        # 2. High variance in offset
        # 3. Increasing trend in delay
        
        if len(self.offset_history) < 10:
            return False
        
        recent = self.offset_history[-10:]
        variance = sum((x - sum(recent)/10)**2 for x in recent) / 10
        
        # High variance suggests congestion-induced jitter
        if variance > (self.congestion_threshold_us * 1000)**2:
            return True
        
        # Large spike suggests congestion
        if abs(offset_ns) > self.congestion_threshold_us * 1000:
            return True
        
        return False
    
    def mitigate_congestion(self):
        """Mitigate PTP congestion issues"""
        # Mitigation strategies:
        
        # 1. Increase PTP message priority (DSCP/CoS)
        # Set DSCP EF (Expedited Forwarding) for PTP packets
        subprocess.run(['ip', 'link', 'set', 'eth0', 'priomap', '0', '6'])
        
        # 2. Enable PTP hardware message prioritization
        # (NICs like Mellanox ConnectX support PTP priority queues)
        
        # 3. Reduce sync interval temporarily
        # (Wait for congestion to clear)
        
        # 4. Use holdover with local TCXO
        # (Continue using last known good offset)
        
        print("PTP congestion detected - applying mitigation strategies")
\`\`\`

**Monitoring Clock Offset**

Continuous monitoring is essential:

\`\`\`python
import subprocess
from typing import Dict, Any

class PTPMonitor:
    """Monitor PTP synchronization quality"""
    
    def get_ptp_status(self) -> Dict[str, Any]:
        """Get detailed PTP status from ptp4l"""
        # Query PTP management interface
        result = subprocess.run(
            ['pmc', '-u', '-b', '0', 'GET', 'TIME_STATUS_NP'],
            capture_output=True, text=True
        )
        
        status = {}
        for line in result.stdout.split('\\n'):
            if 'master_offset' in line:
                status['offset_ns'] = int(line.split()[-1])
            elif 'freq_adjustment' in line:
                status['freq_adj_ppb'] = int(line.split()[-1])
            elif 'path_delay' in line:
                status['path_delay_ns'] = int(line.split()[-1])
        
        return status
    
    def check_sync_quality(self) -> bool:
        """Check if synchronization meets trading requirements"""
        status = self.get_ptp_status()
        
        # Requirements for HFT:
        # - Offset < 100ns (strict) or < 1μs (acceptable)
        # - Path delay stable (variance < 50ns)
        # - Frequency adjustment stable
        
        offset_ok = abs(status.get('offset_ns', float('inf'))) < 1000
        path_delay_ok = status.get('path_delay_ns', 0) < 10000  # < 10μs
        
        if not offset_ok:
            print(f"WARNING: PTP offset {status['offset_ns']}ns exceeds threshold")
        
        return offset_ok and path_delay_ok
\`\`\`

**Why Nanosecond Synchronization is Critical**1. **Latency Measurement**: With 100ns clock accuracy, can measure latency of individual components (strategy execution ~10μs)
2. **Event Ordering**: Correctly order market data events from multiple venues
3. **Arbitrage Detection**: Identify price discrepancies across venues (require <1μs accuracy to determine which price was first)
4. **Regulatory Compliance**: Regulators require accurate timestamps (CAT reporting, FINRA 4552)
5. **Performance Analysis**: Identify bottlenecks in microsecond-level trading pipelines

Without nanosecond-level synchronization, it's impossible to accurately measure latency in modern HFT systems where end-to-end latency is 50-500μs.

**Conclusion**

PTP provides the timing foundation for distributed trading systems, enabling accurate latency measurement and event ordering essential for HFT strategies. Proper implementation with hardware timestamping, boundary clocks, and robust monitoring achieves sub-microsecond accuracy necessary for competitive trading.`
}
];

