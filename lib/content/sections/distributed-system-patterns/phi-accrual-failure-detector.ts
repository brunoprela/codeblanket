/**
 * Phi Accrual Failure Detector Section
 */

export const phiaccrualfailuredetectorSection = {
  id: 'phi-accrual-failure-detector',
  title: 'Phi Accrual Failure Detector',
  content: `Phi Accrual Failure Detector is an adaptive failure detection algorithm that provides a continuous suspicion level instead of a binary "alive/dead" decision. It\'s used in production systems like Apache Cassandra and Akka to handle the challenging problem of distinguishing between slow nodes and dead nodes.

## What is Phi Accrual Failure Detector?

Traditional failure detectors give a **binary answer**: "Is this node alive or dead?"

**Problem**: Hard to choose the right timeout—too short causes false positives, too long causes slow detection.

**Phi Accrual Solution**: Instead of binary, give a **continuous suspicion level** (phi value):
- **Low phi** (0-5): Node is probably alive
- **Medium phi** (5-8): Node is suspicious
- **High phi** (>8): Node is probably dead

**Key Insight**: Let the application choose the threshold based on its requirements, rather than hardcoding a timeout.

\`\`\`
Traditional:
  Timeout = 5 seconds
  If no heartbeat for 5s → Dead (binary decision)

Phi Accrual:
  Calculate phi value based on heartbeat history
  phi = 3 → Probably alive
  phi = 7 → Suspicious
  phi = 12 → Probably dead
  Application decides threshold (e.g., phi > 8 → Dead)
\`\`\`

---

## Why Phi Accrual?

### **1. Adaptive to Network Conditions**

**Problem**: Network latency varies (Wi-Fi, cloud networks, intercontinental links).

\`\`\`
Fixed timeout (3 seconds):
  - Fast network: Works well
  - Slow network: Many false positives ❌

Phi Accrual:
  - Fast network: phi threshold reached quickly
  - Slow network: phi accounts for higher latency ✅
\`\`\`

**Adapts**: Learns from historical heartbeat patterns.

### **2. Handles Variable Latency**

Network latency isn't constant:

\`\`\`
Heartbeats arrive at:
  T=0s, 1s, 2s, 3s, 4s (regular) → Low phi
  T=5s, 5.5s, 6.2s, 7.1s (irregular) → Higher phi
  T=15s (missed multiple) → Very high phi
\`\`\`

Phi accrual considers:
- **Mean** heartbeat interval
- **Variance** (how much jitter)
- **Time since last heartbeat**

### **3. Flexible Thresholds**

Different operations can use different thresholds:

\`\`\`
Read operations: phi > 5 (mark suspected, try another node)
Write operations: phi > 10 (wait longer, critical for consistency)
Leader election: phi > 12 (very confident before declaring leader dead)
\`\`\`

**Benefit**: One failure detector, multiple use cases.

### **4. Quantifiable Confidence**

Phi value has probabilistic interpretation:

\`\`\`
phi = 1  → ~10% chance node is dead
phi = 2  → ~1% chance node is dead
phi = 3  → ~0.1% chance node is dead
phi = 8  → ~0.000000001% chance node is dead
\`\`\`

**Meaning**: phi = 8 means "I'm 99.999999% confident this node is dead."

---

## How Phi Accrual Works

### **Core Formula**

\`\`\`
phi = -log10(P(t_now - t_last | heartbeat_distribution))

Where:
  t_now = current time
  t_last = time of last heartbeat
  heartbeat_distribution = learned from history
\`\`\`

**Intuition**: 
- Small delay (expected) → Low phi
- Large delay (unexpected) → High phi
- Considers historical distribution (not just last interval)

### **Step-by-Step Calculation**

**1. Track Heartbeat Arrival Times**

\`\`\`
Heartbeats arrive at:
  T=0s, 1.1s, 2.0s, 2.9s, 4.2s, 5.1s, ...

Calculate intervals:
  [1.1s, 0.9s, 0.9s, 1.3s, 0.9s, ...]
\`\`\`

**2. Compute Statistics**

\`\`\`
mean_interval = 1.0s
stddev_interval = 0.2s
\`\`\`

**Keep**: Running mean and standard deviation (or use sliding window).

**3. Calculate Phi**

When checking if node is alive:

\`\`\`
time_since_last_heartbeat = current_time - last_heartbeat_time
expected_arrival_time = last_heartbeat_time + mean_interval

If time_since_last_heartbeat is:
  - Within 1 stddev of mean: phi ≈ 0-2 (normal)
  - 2-3 stddevs: phi ≈ 3-5 (slightly suspicious)
  - 4+ stddevs: phi ≈ 8+ (very suspicious)
\`\`\`

**Simplified Formula**:

\`\`\`
t = time_since_last_heartbeat
mean = mean_interval
stddev = stddev_interval

z_score = (t - mean) / stddev  // How many stddevs away
phi = -log10(1 - cumulative_normal_distribution (z_score))
\`\`\`

### **Example Calculation**

\`\`\`
Historical intervals: [1s, 1s, 1s, 1s, 1s]
mean = 1s, stddev = 0.05s (very consistent)

Scenario 1: 1.2s since last heartbeat
  z_score = (1.2 - 1) / 0.05 = 4
  P(alive) = very low (4 stddevs)
  phi ≈ 8

Scenario 2: 0.5s since last heartbeat
  z_score = (0.5 - 1) / 0.05 = -10
  P(alive) = very high (heartbeat arrived early)
  phi ≈ 0
\`\`\`

**Key Point**: Consistent heartbeats (low stddev) → High phi for deviations. Inconsistent heartbeats (high stddev) → Lower phi for same deviation.

---

## Implementation

### **Sliding Window**

Keep track of recent N heartbeat intervals:

\`\`\`
class PhiAccrualFailureDetector:
    def __init__(self, window_size=1000):
        self.intervals = []  // Recent intervals
        self.last_heartbeat_time = None
        self.window_size = window_size
    
    def heartbeat_received (self, timestamp):
        if self.last_heartbeat_time is not None:
            interval = timestamp - self.last_heartbeat_time
            self.intervals.append (interval)
            if len (self.intervals) > self.window_size:
                self.intervals.pop(0)  // Keep only recent
        self.last_heartbeat_time = timestamp
    
    def phi (self, current_time):
        if self.last_heartbeat_time is None:
            return 0.0  // No data yet
        
        time_since_last = current_time - self.last_heartbeat_time
        mean_interval = mean (self.intervals)
        stddev_interval = stddev (self.intervals)
        
        // Calculate phi using normal distribution
        z_score = (time_since_last - mean_interval) / stddev_interval
        prob = cumulative_normal (z_score)
        phi = -log10(1 - prob)
        
        return phi
\`\`\`

### **Exponentially Decaying Reservoir**

Instead of fixed window, use exponential decay to weight recent intervals more:

\`\`\`
new_mean = alpha * new_interval + (1 - alpha) * old_mean
new_variance = alpha * (new_interval - new_mean)^2 + (1 - alpha) * old_variance

Where alpha = 0.1 (weight factor)
\`\`\`

**Benefit**: Adapts faster to changes, no need to store all intervals.

### **Minimum Samples**

Don't calculate phi until enough samples:

\`\`\`
def phi (self, current_time):
    if len (self.intervals) < MIN_SAMPLES:  // e.g., 10
        return 0.0  // Not enough data, assume alive
    
    // ... calculate phi
\`\`\`

**Reason**: Need sufficient history for accurate statistics.

---

## Phi Threshold Selection

**Application-specific thresholds**:

### **Cassandra Default: phi = 8**

\`\`\`
if phi > 8:
    mark_node_suspected()
\`\`\`

**Reasoning**: ~0.000000001% false positive rate (very confident).

### **Lower Threshold (phi = 5)**

\`\`\`
For read operations:
    if phi > 5:
        skip_node()  // Try another replica
\`\`\`

**Reasoning**: Don't wait long for suspicious node, prioritize latency.

### **Higher Threshold (phi = 12)**

\`\`\`
For leader election:
    if phi > 12:
        trigger_election()
\`\`\`

**Reasoning**: Very high confidence before triggering expensive operation.

### **Dynamic Threshold**

Adjust based on cluster size or network conditions:

\`\`\`
if cluster_size > 100:
    threshold = 10  // Large cluster, tolerate more variance
else:
    threshold = 8   // Small cluster, detect quickly
\`\`\`

---

## Phi Accrual in Apache Cassandra

Cassandra uses phi accrual for gossip-based failure detection.

### **Configuration**

\`\`\`yaml
# cassandra.yaml
phi_convict_threshold: 8  # Default
\`\`\`

### **How It Works**

\`\`\`
1. Every second, each node gossips with 1-3 random nodes
2. Gossip includes heartbeat information
3. Each node tracks phi values for all other nodes
4. If phi > threshold:
   - Mark node as "suspected" (not dead immediately)
5. After additional confirmations:
   - Mark node as "down"
   - Stop routing requests to it
6. If node comes back:
   - phi drops, mark as "up"
\`\`\`

### **Suspected vs Down**

\`\`\`
phi > 8:  Suspected (still try to gossip)
phi > 8 for 30s + no gossip: Down (stop routing)
\`\`\`

**Reason**: Distinguish between temporary network issues and actual failures.

### **Monitoring**

Cassandra exposes phi values via JMX:

\`\`\`
nodetool describecluster
\`\`\`

Shows phi values for all nodes.

---

## Comparison with Fixed Timeout

### **Fixed Timeout**

\`\`\`
timeout = 3 seconds
if no_heartbeat_for(3s):
    mark_dead()
\`\`\`

**Pros**:
- Simple to implement
- Predictable behavior

**Cons**:
- Doesn't adapt to network conditions
- Hard to choose right timeout (too short = false positives, too long = slow detection)
- Same timeout for all nodes (ignores individual characteristics)

### **Phi Accrual**

\`\`\`
phi = calculate_phi (heartbeat_history)
if phi > threshold:
    mark_suspected()
\`\`\`

**Pros**:
- Adapts to network latency and jitter
- Different thresholds for different operations
- Continuous suspicion level (not binary)
- Learns from history

**Cons**:
- More complex to implement
- Requires maintaining statistics
- Needs warm-up period (collecting initial samples)

---

## Challenges and Solutions

### **1. Cold Start**

**Problem**: Not enough heartbeat history initially.

**Solution**:
- Start with low phi (assume alive)
- Use default mean/stddev until enough samples
- Typical: 10-20 samples before trusting phi

### **2. Network Jitter**

**Problem**: High variance in intervals → hard to detect failures.

**Solution**:
- Use larger threshold (phi > 10)
- Use percentile-based calculation (ignore outliers)
- Combine with other signals (gossip consensus)

### **3. Permanent Network Partition**

**Problem**: Partitioned nodes have high phi indefinitely.

**Solution**:
- After threshold time (e.g., 10 minutes), mark as "permanently down"
- Require manual intervention to re-add
- Use quorum to avoid split-brain

### **4. Clock Skew**

**Problem**: Node clocks differ, timestamps unreliable.

**Solution**:
- Use local clock only (don't rely on remote timestamps)
- Measure intervals using local clock
- Only compare relative times, not absolute

### **5. Correlated Failures**

**Problem**: Network issue affects all nodes → all have high phi.

**Solution**:
- Use consensus (multiple nodes must agree)
- Implement "jury" system (poll multiple nodes)
- Gradual degradation (mark suspected before down)

---

## Real-World Usage

### **Apache Cassandra**

- **Use**: Gossip-based failure detection
- **Threshold**: phi > 8 (default)
- **Benefit**: Handles variable network latency across data centers

### **Akka Cluster (Scala)**

- **Use**: Actor system failure detection
- **Threshold**: phi > 8 to 12 (configurable)
- **Benefit**: Adaptive to different deployment environments

### **Riak**

- **Use**: Node membership and failure detection
- **Similar approach**: Learns from heartbeat patterns

---

## Interview Tips

### **Key Concepts to Explain**1. **What is phi accrual**: Continuous suspicion level instead of binary alive/dead
2. **Why better than fixed timeout**: Adapts to network conditions, flexible thresholds
3. **How it works**: Learn mean/stddev from heartbeat history, calculate phi based on deviation
4. **Phi interpretation**: phi = 8 means ~99.999999% confidence node is dead
5. **Real-world**: Cassandra (gossip), Akka (cluster membership)

### **Common Interview Questions**

**Q: What problem does phi accrual solve that fixed timeout doesn't?**
A: "Fixed timeout doesn't adapt to network conditions. If you set timeout=3s: fast network works fine, but slow network causes false positives (temporary delays declared as failures). Phi accrual learns from historical heartbeat patterns—if heartbeats usually arrive in 1s±0.1s (low variance), a 2s delay is very suspicious (high phi). If heartbeats are irregular (high variance), a 2s delay is normal (low phi). It adapts automatically without manual tuning."

**Q: How do you choose the phi threshold?**
A: "Depends on operation criticality and acceptable false positive rate. phi=8 (Cassandra default) gives ~99.999999% confidence—good for marking nodes down. Lower threshold (phi=5) for reads where you just want to try another node quickly. Higher threshold (phi=12) for critical operations like leader election. You can also have multiple thresholds: phi>5 for 'suspected' (route around), phi>8 for 'down' (mark as failed)."

**Q: What happens if there's not enough heartbeat history?**
A: "Cold start problem. Initially, not enough data to calculate accurate mean/stddev. Solution: (1) Start with phi=0 (assume alive). (2) Use default parameters (e.g., mean=1s, stddev=0.5s) until enough samples. (3) Require minimum samples (10-20) before trusting phi. (4) Bootstrap with expected values based on network type. After warm-up period (30-60s), phi becomes accurate."

**Q: Can phi accrual distinguish between slow node and dead node?**
A: "Not perfectly, but better than fixed timeout. Slow node: heartbeats arrive but delayed → phi rises gradually, may cross threshold if consistently slow. Dead node: heartbeats stop → phi rises sharply. You can combine phi with other signals: (1) If phi>5 but node responds to other requests, it's slow (not dead). (2) Use multi-round detection: phi>8 for 30s → likely dead. (3) Consensus: multiple nodes must agree on high phi."

---

## Advanced Topics

### **Mathematical Foundation**

Phi accrual assumes heartbeat intervals follow a normal distribution.

\`\`\`
X ~ N(mean, stddev)  // Heartbeat intervals

P(node alive | t seconds since last heartbeat):
  = P(X > t)
  = 1 - CDF(t)  // Cumulative distribution function

phi = -log10(P(node alive))
\`\`\`

**Why log**: Maps probabilities to intuitive scale:
- P=0.1 (10% alive) → phi = 1
- P=0.01 (1% alive) → phi = 2
- P=0.000000001 → phi = 8

### **Non-Normal Distributions**

If heartbeat intervals don't follow normal distribution:

**Alternative**: Use empirical CDF (rank-based)

\`\`\`
intervals = [0.9, 1.0, 1.1, 0.95, 1.05, ...]
intervals_sorted = [0.9, 0.95, 1.0, 1.05, 1.1, ...]

t = 2.0  // Time since last heartbeat
rank = how many intervals < 2.0 → 100% (all intervals)
P(alive) = 0
phi = infinity
\`\`\`

**Benefit**: Works for any distribution.

### **Combination with Gossip**

Phi accrual + gossip = robust failure detection:

\`\`\`
1. Each node calculates phi for all other nodes
2. Gossip phi values (not just heartbeat)
3. Aggregate: avg_phi = average phi from multiple observers
4. Decide: if avg_phi > threshold, mark down
\`\`\`

**Benefit**: Reduces false positives from network partitions.

---

## Summary

Phi Accrual Failure Detector is a sophisticated, adaptive approach to failure detection:

1. **Core Idea**: Continuous suspicion level (phi) instead of binary alive/dead
2. **Benefits**: Adapts to network conditions, flexible thresholds, quantifiable confidence
3. **How It Works**: Learn mean/stddev from heartbeat history, calculate phi based on time since last heartbeat
4. **Interpretation**: phi=8 means 99.999999% confidence node is dead
5. **Real-World**: Cassandra (gossip), Akka (cluster), Riak (membership)
6. **Threshold**: Application-specific (phi=5 for reads, phi=8 typical, phi=12 for critical ops)
7. **Trade-offs**: More complex than fixed timeout, requires warm-up period, but much more adaptive

**Interview Focus**: Understand the problem it solves (adapting to network variance), how phi is calculated (based on historical distribution), threshold selection (application-specific), and real-world usage (Cassandra).
`,
};
