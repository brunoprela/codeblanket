export const timestampManagement = {
  title: 'Timestamp Management and Clock Sync',
  id: 'timestamp-management',
  content: `
# Timestamp Management and Clock Sync

## Introduction

Accurate timestamps are critical for market data - they determine trade ordering, causality analysis, and regulatory compliance. Microsecond precision and synchronized clocks across systems are essential for high-frequency trading.

**Why Timestamps Matter:**
- **Trade ordering**: Determine which order arrived first (FIFO matching)
- **Causality**: Did quote update cause trade, or vice versa?
- **Latency measurement**: How long from exchange → strategy?
- **Compliance**: SEC CAT requires microsecond timestamps
- **Backtesting**: Replay historical data in correct sequence

**Industry Standards:**
- **Exchanges**: GPS-synchronized atomic clocks (< 1μs accuracy)
- **HFT firms**: PTP (Precision Time Protocol) for < 100ns sync
- **Institutional**: NTP (Network Time Protocol) for < 1ms sync
- **Retail**: System clocks (10-100ms accuracy, often sufficient)

This section covers NTP, PTP, timestamp precision, clock skew handling, and production time synchronization.

---

## Timestamp Precision Levels

\`\`\`python
from datetime import datetime, timezone
import time

# Precision levels
class TimestampPrecision:
    """Different timestamp precision levels"""
    
    @staticmethod
    def second_precision():
        """Second precision (1000ms granularity)"""
        return int(time.time())  # Unix epoch seconds
    
    @staticmethod
    def millisecond_precision():
        """Millisecond precision (1ms granularity)"""
        return int(time.time() * 1000)
    
    @staticmethod
    def microsecond_precision():
        """Microsecond precision (1μs granularity)"""
        return int(time.time() * 1_000_000)
    
    @staticmethod
    def nanosecond_precision():
        """Nanosecond precision (1ns granularity) - CPU clock only"""
        return time.time_ns()  # Python 3.7+
    
    @staticmethod
    def datetime_utc():
        """UTC datetime with microsecond precision"""
        return datetime.now(timezone.utc)

# Usage
print(f"Seconds: {TimestampPrecision.second_precision()}")
print(f"Milliseconds: {TimestampPrecision.millisecond_precision()}")
print(f"Microseconds: {TimestampPrecision.microsecond_precision()}")
print(f"Nanoseconds: {TimestampPrecision.nanosecond_precision()}")

# Precision requirements by use case:
# - Backtesting daily data: Second precision OK
# - Real-time trading (non-HFT): Millisecond precision
# - HFT market making: Microsecond precision REQUIRED
# - FPGA trading: Nanosecond precision
# - Regulatory (SEC CAT): Microsecond precision REQUIRED
\`\`\`

---

## NTP (Network Time Protocol)

\`\`\`python
import ntplib
from datetime import datetime
import time

class NTPSync:
    """Synchronize system clock with NTP servers"""
    
    def __init__(self, ntp_servers: list = None):
        self.ntp_servers = ntp_servers or [
            'time.google.com',
            'time.cloudflare.com',
            'pool.ntp.org'
        ]
        self.client = ntplib.NTPClient()
        self.offset = 0.0  # System clock offset from NTP server
    
    def sync(self) -> dict:
        """Query NTP server and calculate clock offset"""
        for server in self.ntp_servers:
            try:
                response = self.client.request(server, version=3, timeout=5)
                
                # Clock offset (positive = system clock ahead)
                self.offset = response.offset
                
                return {
                    'server': server,
                    'offset_ms': self.offset * 1000,
                    'delay_ms': response.delay * 1000,
                    'precision': 'millisecond',
                    'stratum': response.stratum,  # Distance from atomic clock
                    'synced': True
                }
            except Exception as e:
                print(f"NTP sync failed with {server}: {e}")
                continue
        
        return {'synced': False, 'error': 'All NTP servers failed'}
    
    def get_accurate_time(self) -> float:
        """Get NTP-corrected timestamp"""
        return time.time() - self.offset
    
    def monitor_drift(self):
        """Continuously monitor clock drift"""
        while True:
            result = self.sync()
            if result.get('synced'):
                offset_ms = result['offset_ms']
                if abs(offset_ms) > 100:  # > 100ms drift
                    print(f"WARNING: Large clock drift: {offset_ms:.2f}ms")
            
            time.sleep(3600)  # Check hourly

# Usage
ntp = NTPSync()
sync_result = ntp.sync()
print(f"Clock offset: {sync_result['offset_ms']:.3f}ms")
print(f"Network delay: {sync_result['delay_ms']:.3f}ms")

# Accuracy: NTP provides 1-10ms accuracy over internet
# For sub-millisecond accuracy, use PTP (see below)
\`\`\`

---

## PTP (Precision Time Protocol)

\`\`\`python
# PTP requires hardware support (PTP-capable network cards)
# Provides sub-microsecond accuracy (< 1μs typical, < 100ns achievable)

class PTPMonitor:
    """Monitor PTP synchronization status"""
    
    def __init__(self):
        self.ptp_device = '/dev/ptp0'  # Hardware PTP clock
    
    def check_ptp_status(self) -> dict:
        """Check PTP sync status (Linux ptp4l daemon)"""
        import subprocess
        
        try:
            # Query ptp4l daemon
            result = subprocess.run(
                ['pmc', '-u', '-b', 0, 'GET CURRENT_DATA_SET'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse output for offset and status
            output = result.stdout
            
            # Extract offset from master clock
            if 'offsetFromMaster' in output:
                # PTP is synchronized
                return {
                    'synced': True,
                    'protocol': 'PTP',
                    'accuracy': 'sub-microsecond',
                    'offset_ns': self._parse_offset(output)
                }
        except Exception as e:
            print(f"PTP check failed: {e}")
        
        return {'synced': False, 'protocol': 'PTP'}
    
    def _parse_offset(self, output: str) -> int:
        """Parse offset from PTP daemon output"""
        # Example: offsetFromMaster: 234 nanoseconds
        import re
        match = re.search(r'offsetFromMaster:\s*(-?\d+)', output)
        if match:
            return int(match.group(1))
        return 0

# PTP Architecture:
# 1. Grandmaster Clock: GPS-synchronized atomic clock
# 2. PTP Switch: Hardware-assisted timestamp forwarding
# 3. PTP NIC: Network card with hardware timestamping
# 4. Software daemon: ptp4l synchronizes system clock

# Accuracy comparison:
# - System clock (no sync): ±100ms-1s drift per day
# - NTP: ±1-10ms accuracy
# - PTP over internet: ±100μs accuracy  
# - PTP in datacenter: ±1μs accuracy
# - PTP with hardware support: ±100ns accuracy
# - GPS clock: ±10ns accuracy (atomic reference)
\`\`\`

---

## Handling Clock Skew

\`\`\`python
from datetime import datetime, timedelta
from typing import Dict

class ClockSkewDetector:
    """Detect and handle clock skew across systems"""
    
    def __init__(self, tolerance_ms: float = 10.0):
        self.tolerance_ms = tolerance_ms
        
        # Reference timestamps from different systems
        self.system_offsets: Dict[str, float] = {}
        
        # Skew alerts
        self.skew_alerts = []
    
    def register_system_offset(self, system_id: str, offset_ms: float):
        """Register known clock offset for a system"""
        self.system_offsets[system_id] = offset_ms
    
    def correct_timestamp(self, timestamp: datetime, 
                         system_id: str) -> datetime:
        """Correct timestamp from system with known offset"""
        if system_id in self.system_offsets:
            offset_ms = self.system_offsets[system_id]
            return timestamp - timedelta(milliseconds=offset_ms)
        return timestamp
    
    def detect_skew(self, ts1: datetime, ts2: datetime, 
                   label: str = "") -> bool:
        """Detect if timestamps show unexpected skew"""
        diff_ms = abs((ts2 - ts1).total_seconds() * 1000)
        
        if diff_ms > self.tolerance_ms:
            alert = {
                'timestamp': datetime.now(),
                'label': label,
                'diff_ms': diff_ms,
                'ts1': ts1,
                'ts2': ts2
            }
            self.skew_alerts.append(alert)
            print(f"Clock skew detected: {label} {diff_ms:.2f}ms")
            return True
        
        return False

# Example: Detect skew between exchange timestamp and local receive time
detector = ClockSkewDetector(tolerance_ms=50.0)

# Register known offset (NYSE clock 2ms fast)
detector.register_system_offset('NYSE', 2.0)

# Correct incoming timestamp
exchange_ts = datetime.now()
corrected_ts = detector.correct_timestamp(exchange_ts, 'NYSE')
print(f"Corrected: {exchange_ts} → {corrected_ts}")
\`\`\`

---

## Production Best Practices

1. **Use UTC everywhere** - Avoid timezone confusion
2. **Sync with NTP** - Minimum requirement for production
3. **Monitor drift** - Alert if clock drifts > 50ms
4. **Store exchange timestamps** - Don't rely on receive timestamps
5. **Microsecond precision** - Required for HFT and compliance
6. **Hardware timestamping** - Use NIC hardware timestamps when available
7. **PTP for HFT** - Essential for sub-millisecond strategies

---

## Compliance Requirements

**SEC CAT (Consolidated Audit Trail):**
- Microsecond timestamp precision REQUIRED
- Clock sync within 100ms of NIST atomic clock
- Daily drift reporting
- Penalties for non-compliance: $1M+ fines

Now you can properly manage timestamps and synchronize clocks for trading systems!
`,
};
