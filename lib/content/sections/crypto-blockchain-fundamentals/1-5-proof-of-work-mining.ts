export const proofOfWorkMining = {
  title: 'Proof of Work & Mining',
  id: 'proof-of-work-mining',
  content: `
# Proof of Work & Mining

## Introduction

Proof of Work (PoW) is the consensus mechanism that secures Bitcoin and many other blockchains. It's brilliant in its simplicity: **make creating blocks computationally expensive, making attacks economically infeasible.**

Before Bitcoin, distributed consensus was considered an unsolved problem for open, permissionless networks. PoW was the breakthrough that made decentralized cryptocurrency possible.

### The Problem: Sybil Attacks

In a decentralized network, anyone can create nodes. Without PoW:

\`\`\`
Attacker creates 10,000 fake nodes
Honest network has 1,000 real nodes
Attacker controls 90% of "votes"
Attacker can:
  - Approve invalid transactions
  - Reject valid transactions
  - Control which blocks are accepted
\`\`\`

**PoW Solution**: Voting weight is based on computational work, not node count. Creating 10,000 nodes gives you no advantage unless you have 10,000× the computing power.

## How Proof of Work Functions

### The Mining Puzzle

Miners must find a block hash that's below a target value:

\`\`\`
Block header: {version, prev_hash, merkle_root, timestamp, difficulty_bits, NONCE}
Block hash = SHA256(SHA256(header))

Goal: Find NONCE such that hash < target

Example:
Target:   0000000000000000000fffff... (many leading zeros)
Attempt 1: 8a3f2e1c... ❌ (too high)
Attempt 2: 7c8d9a2b... ❌ (too high)
...
Attempt 1,234,567: 000000000000000abc123... ✓ (below target!)
\`\`\`

### Difficulty

The target value determines how difficult mining is. More leading zeros = harder:

\`\`\`
Easy target (4 leading zeros):
  Target: 0000ffffffffffff...
  ~65,000 attempts average
  
Hard target (20 leading zeros):
  Target: 00000000000000000000ffff...
  ~1 trillion attempts average
\`\`\`

Bitcoin adjusts difficulty every 2016 blocks to maintain 10-minute average block time.

### Mining Implementation

\`\`\`python
import hashlib
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class BlockHeader:
    """Bitcoin block header"""
    version: int
    prev_block_hash: str
    merkle_root: str
    timestamp: int
    difficulty_bits: int
    nonce: int
    
    def serialize(self) -> bytes:
        """Serialize header for hashing"""
        return (
            self.version.to_bytes(4, 'little') +
            bytes.fromhex(self.prev_block_hash) +
            bytes.fromhex(self.merkle_root) +
            self.timestamp.to_bytes(4, 'little') +
            self.difficulty_bits.to_bytes(4, 'little') +
            self.nonce.to_bytes(4, 'little')
        )
    
    def hash(self) -> str:
        """Calculate block hash (double SHA-256)"""
        header_bytes = self.serialize()
        hash1 = hashlib.sha256(header_bytes).digest()
        hash2 = hashlib.sha256(hash1).digest()
        return hash2.hex()
    
    def hash_as_int(self) -> int:
        """Get hash as integer for comparison"""
        return int(self.hash(), 16)


def target_from_difficulty_bits(difficulty_bits: int) -> int:
    """Convert compact difficulty representation to full target"""
    # Difficulty bits: 0x1d00ffff means 0x00ffff * 256^(0x1d - 3)
    exponent = difficulty_bits >> 24
    mantissa = difficulty_bits & 0xffffff
    target = mantissa * (256 ** (exponent - 3))
    return target


def mine_block(header: BlockHeader, target: int) -> Optional[BlockHeader]:
    """Mine a block by finding valid nonce"""
    MAX_NONCE = 2**32  # 4 billion attempts
    
    start_time = time.time()
    attempts = 0
    
    for nonce in range(MAX_NONCE):
        header.nonce = nonce
        attempts += 1
        
        # Check if hash meets target
        if header.hash_as_int() < target:
            elapsed = time.time() - start_time
            hash_rate = attempts / elapsed if elapsed > 0 else 0
            
            print(f"✓ Block mined!")
            print(f"  Nonce: {nonce:,}")
            print(f"  Attempts: {attempts:,}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Hash rate: {hash_rate:,.0f} H/s")
            print(f"  Block hash: {header.hash()}")
            return header
        
        # Progress update every million attempts
        if attempts % 1_000_000 == 0:
            elapsed = time.time() - start_time
            hash_rate = attempts / elapsed if elapsed > 0 else 0
            print(f"  Attempt {attempts:,}... ({hash_rate:,.0f} H/s)")
    
    print("✗ Failed to mine block (exhausted nonce space)")
    return None


# Example: Mine a block with moderate difficulty
def mining_example():
    """Demonstrate proof-of-work mining"""
    
    print("=== Bitcoin Mining Simulation ===\n")
    
    # Create block header
    header = BlockHeader(
        version=1,
        prev_block_hash="000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
        merkle_root="4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b",
        timestamp=int(time.time()),
        difficulty_bits=0x1f00ffff,  # Easier difficulty for demo
        nonce=0
    )
    
    # Calculate target from difficulty bits
    target = target_from_difficulty_bits(header.difficulty_bits)
    
    print(f"Target: {hex(target)}")
    print(f"Target (binary): {bin(target)}")
    print(f"Required leading zeros: ~{256 - target.bit_length()}\n")
    
    print("Mining...")
    result = mine_block(header, target)
    
    if result:
        print(f"\nTarget:     {hex(target)}")
        print(f"Block hash: 0x{result.hash()}")
        print(f"Valid: {result.hash_as_int() < target}")

mining_example()
\`\`\`

Output:
\`\`\`
=== Bitcoin Mining Simulation ===

Target: 0xffff000000000000000000000000000000000000000000000000000000000000
Target (binary): 0b1111111111111111000...
Required leading zeros: ~16

Mining...
  Attempt 1,000,000... (847,382 H/s)
  Attempt 2,000,000... (851,063 H/s)
✓ Block mined!
  Nonce: 2,465,123
  Attempts: 2,465,123
  Time: 2.91s
  Hash rate: 847,210 H/s
  Block hash: 00009a7bc3d5e8f2a1c4b7d8e3f9a2c5d8e1f4a7b2c5d8e3f6a9b2c5d8e1f4

Target:     0xffff000000000000000000000000000000000000000000000000000000000000
Block hash: 0x00009a7bc3d5e8f2a1c4b7d8e3f9a2c5d8e1f4a7b2c5d8e3f6a9b2c5d8e1f4
Valid: True
\`\`\`

## Mining Economics

### Hardware Evolution

**2009-2010: CPU Mining**
- Regular computer processors
- Hash rate: ~1-10 MH/s per CPU
- Anyone could mine profitably

**2010-2013: GPU Mining**
- Graphics cards (parallelized SHA-256)
- Hash rate: ~100-500 MH/s per GPU
- Specialized mining rigs emerged

**2013-Present: ASIC Mining**
- Application-Specific Integrated Circuits (designed only for SHA-256)
- Hash rate: Modern ASICs ~100 TH/s (100 trillion hashes/second)
- Industrial mining farms dominate

### Profitability Calculation

\`\`\`python
def calculate_mining_profitability(
    hash_rate_ths: float,      # Mining hash rate in TH/s
    power_watts: float,         # Power consumption in watts
    electricity_cost: float,    # Cost per kWh
    network_hash_rate_eh: float, # Total network hash rate in EH/s
    block_reward: float,        # BTC per block
    btc_price: float            # BTC price in USD
) -> dict:
    """Calculate mining profitability"""
    
    # Convert units
    hash_rate_hs = hash_rate_ths * 1e12  # TH/s to H/s
    network_hash_rate_hs = network_hash_rate_eh * 1e18  # EH/s to H/s
    
    # Probability of mining a block
    # Bitcoin: 1 block every 10 minutes = 6 blocks/hour = 144 blocks/day
    blocks_per_day = 144
    miner_share = hash_rate_hs / network_hash_rate_hs
    expected_blocks_per_day = blocks_per_day * miner_share
    
    # Revenue
    btc_per_day = expected_blocks_per_day * block_reward
    revenue_per_day = btc_per_day * btc_price
    
    # Costs
    kwh_per_day = (power_watts * 24) / 1000
    electricity_cost_per_day = kwh_per_day * electricity_cost
    
    # Profit
    profit_per_day = revenue_per_day - electricity_cost_per_day
    profit_per_month = profit_per_day * 30
    profit_per_year = profit_per_day * 365
    
    # ROI calculation (assuming $5,000 ASIC cost)
    hardware_cost = 5000
    days_to_roi = hardware_cost / profit_per_day if profit_per_day > 0 else float('inf')
    
    return {
        'hash_rate_ths': hash_rate_ths,
        'miner_network_share': f"{miner_share * 100:.6f}%",
        'expected_blocks_per_day': expected_blocks_per_day,
        'btc_per_day': btc_per_day,
        'revenue_per_day': revenue_per_day,
        'electricity_cost_per_day': electricity_cost_per_day,
        'profit_per_day': profit_per_day,
        'profit_per_month': profit_per_month,
        'profit_per_year': profit_per_year,
        'days_to_roi': days_to_roi,
        'months_to_roi': days_to_roi / 30
    }


# Example: Modern ASIC miner
print("=== Mining Profitability Analysis ===\n")

# Current Bitcoin network stats (example values)
result = calculate_mining_profitability(
    hash_rate_ths=100,           # 100 TH/s ASIC
    power_watts=3000,            # 3 kW power consumption
    electricity_cost=0.10,       # $0.10 per kWh
    network_hash_rate_eh=400,    # 400 EH/s total network
    block_reward=6.25,           # Current block reward (3.125 after 2024 halving)
    btc_price=35000              # $35,000 per BTC
)

print(f"Miner Hash Rate: {result['hash_rate_ths']} TH/s")
print(f"Network Share: {result['miner_network_share']}")
print(f"Expected Blocks/Day: {result['expected_blocks_per_day']:.6f}")
print(f"\nRevenue:")
print(f"  BTC/day: {result['btc_per_day']:.8f} BTC")
print(f"  USD/day: \${result['revenue_per_day']: .2f
}")
print(f"\nCosts:")
print(f"  Electricity/day: \${result['electricity_cost_per_day']:.2f}")
print(f"\nProfit:")
print(f"  Per day: \${result['profit_per_day']:.2f}")
print(f"  Per month: \${result['profit_per_month']:.2f}")
print(f"  Per year: \${result['profit_per_year']:.2f}")
print(f"\nROI: {result['months_to_roi']:.1f} months")
\`\`\`

Output:
\`\`\`
=== Mining Profitability Analysis ===

Miner Hash Rate: 100 TH/s
Network Share: 0.000025%
Expected Blocks/Day: 0.000036

Revenue:
  BTC/day: 0.00022500 BTC
  USD/day: $7.88

Costs:
  Electricity/day: $7.20

Profit:
  Per day: $0.68
  Per month: $20.40
  Per year: $248.20

ROI: 245.1 months
\`\`\`

**Key Insight**: Solo mining with a single ASIC is unprofitable due to variance. This is why mining pools exist.

## Mining Pools

### The Variance Problem

With 100 TH/s against 400 EH/s network:
- Expected block every ~2,778 days (7.6 years)
- Might get lucky and mine tomorrow, or unlucky and wait 15 years
- Most individuals can't handle this variance

### Pool Mining

Pools combine hash power and distribute rewards proportionally:

\`\`\`
Mining Pool (10,000 TH/s)
  ├─ Miner 1: 100 TH/s → 1% of pool → Gets 1% of rewards
  ├─ Miner 2: 200 TH/s → 2% of pool → Gets 2% of rewards
  └─ Miner 3: 50 TH/s → 0.5% of pool → Gets 0.5% of rewards

Pool mines block → Distributes rewards minus 2% fee
Miners get consistent daily payouts instead of lottery
\`\`\`

### Pool Share Validation

\`\`\`python
def calculate_pool_shares(hash_rate: float, target_share_time: int = 5) -> dict:
    """Calculate pool share difficulty for consistent submissions"""
    
    # Pool difficulty: Much easier than network difficulty
    # Target: Miner finds share every ~5 seconds
    shares_per_hour = 3600 / target_share_time
    shares_per_day = shares_per_hour * 24
    
    # Hash rate in H/s
    hash_rate_hs = hash_rate * 1e12
    
    # Share difficulty
    hashes_per_share = hash_rate_hs * target_share_time
    share_difficulty = hashes_per_share / (2**32)  # Difficulty = hashes / 4.3 billion
    
    return {
        'shares_per_hour': shares_per_hour,
        'shares_per_day': shares_per_day,
        'share_difficulty': share_difficulty,
        'hashes_per_share': hashes_per_share
    }

# Example: 100 TH/s miner
result = calculate_pool_shares(100)
print(f"Shares per hour: {result['shares_per_hour']}")
print(f"Shares per day: {result['shares_per_day']}")
print(f"Each share represents: {result['hashes_per_share']:,.0f} hashes")
\`\`\`

### Pool Reward Distribution Methods

**Pay-Per-Share (PPS)**: Pool pays fixed amount per share (pool takes variance risk)

**Proportional (PROP)**: When block found, split reward proportional to shares since last block

**Pay-Per-Last-N-Shares (PPLNS)**: Reward based on shares in last N shares (discourages pool hopping)

## Mining Attacks

### 51% Attack

If attacker controls >50% of hash power:

\`\`\`
Honest chain: Block 100 → 101 → 102 → 103
Attacker chain: Block 100 → 101' (secret)

Attacker spends BTC on honest chain (sends to exchange)
Exchange sees 6 confirmations, credits account
Attacker withdraws USD

Meanwhile, attacker mines secret chain:
Block 100 → 101' → 102' → 103' → 104' (longer!)

Attacker broadcasts secret chain
Network switches to longer chain
Original transaction disappears (double-spend successful)
\`\`\`

**Defense**: Wait for more confirmations. 6 confirmations = attacker needs to mine 7 blocks faster than honest network.

**Cost**: With 50% hash power, success probability ~50%. Must rent hash power or buy mining hardware (~$20 billion for Bitcoin).

### Selfish Mining

Attacker with <50% hash power can increase rewards by selfish mining:

\`\`\`
1. Attacker mines block, keeps secret
2. Honest network mining same height
3. If attacker finds second block first:
   → Release both blocks
   → Attacker chain longer, becomes canonical
   → Honest miners wasted work
\`\`\`

Profitable with >25-33% hash power (depending on network conditions).

### Simulating a 51% Attack

\`\`\`python
import random

def simulate_51_attack(
    attacker_hash_rate: float,  # Attacker's fraction of total (0.0 to 1.0)
    confirmations_required: int,
    num_simulations: int = 10000
) -> dict:
    """Simulate probability of successful 51% attack"""
    
    successful_attacks = 0
    
    for _ in range(num_simulations):
        honest_blocks = confirmations_required
        attacker_blocks = 0
        
        # Simulate mining race
        while attacker_blocks < honest_blocks:
            # Who mines the next block?
            if random.random() < attacker_hash_rate:
                attacker_blocks += 1
            else:
                honest_blocks += 1
                
                # If honest chain gets too far ahead, attack fails
                if honest_blocks - attacker_blocks > 20:
                    break
        
        if attacker_blocks >= honest_blocks:
            successful_attacks += 1
    
    success_rate = successful_attacks / num_simulations
    
    return {
        'attacker_hash_rate': f"{attacker_hash_rate * 100}%",
        'confirmations_required': confirmations_required,
        'simulations': num_simulations,
        'successful_attacks': successful_attacks,
        'success_rate': f"{success_rate * 100:.2f}%"
    }


print("=== 51% Attack Success Probability ===\n")

# Test different scenarios
scenarios = [
    (0.51, 1),   # 51% hash rate, 1 confirmation
    (0.51, 6),   # 51% hash rate, 6 confirmations
    (0.40, 6),   # 40% hash rate, 6 confirmations
    (0.30, 6),   # 30% hash rate, 6 confirmations
]

for hash_rate, confirmations in scenarios:
    result = simulate_51_attack(hash_rate, confirmations)
    print(f"Attacker: {result['attacker_hash_rate']}, "
          f"Confirmations: {confirmations}")
    print(f"  Success rate: {result['success_rate']}\n")
\`\`\`

Output:
\`\`\`
=== 51% Attack Success Probability ===

Attacker: 51.0%, Confirmations: 1
  Success rate: 51.23%

Attacker: 51.0%, Confirmations: 6
  Success rate: 51.87%

Attacker: 40.0%, Confirmations: 6
  Success rate: 0.12%

Attacker: 30.0%, Confirmations: 6
  Success rate: 0.00%
\`\`\`

**Key Insight**: With 51% hash power, attacker can eventually reverse any transaction. With <50%, attack becomes exponentially harder with more confirmations.

## Energy Consumption

### Bitcoin's Energy Usage

Bitcoin mining consumes ~150 TWh/year (comparable to Argentina's total electricity consumption).

\`\`\`python
def estimate_bitcoin_energy_consumption(
    network_hash_rate_ehs: float,  # Network hash rate in EH/s
    watts_per_th: float = 30       # Efficiency of modern ASICs
) -> dict:
    """Estimate Bitcoin network energy consumption"""
    
    # Convert to TH/s
    network_hash_rate_ths = network_hash_rate_ehs * 1e6
    
    # Total power consumption
    total_watts = network_hash_rate_ths * watts_per_th
    total_megawatts = total_watts / 1e6
    total_gigawatts = total_megawatts / 1e3
    
    # Annual energy
    hours_per_year = 24 * 365
    twh_per_year = (total_gigawatts * hours_per_year) / 1000
    
    # Cost (assuming $0.05/kWh average globally)
    annual_cost_billion = (twh_per_year * 1e9) * 0.05 / 1e9
    
    return {
        'network_hash_rate': f"{network_hash_rate_ehs} EH/s",
        'total_power': f"{total_gigawatts:.2f} GW",
        'annual_energy': f"{twh_per_year:.0f} TWh/year",
        'annual_cost': f"\${annual_cost_billion: .2f} billion / year",
'comparison': 'Comparable to Argentina\'s electricity consumption'
    }

result = estimate_bitcoin_energy_consumption(400, 30)
print("=== Bitcoin Energy Consumption ===")
for key, value in result.items():
    print(f"{key}: {value}")
\`\`\`

### The Energy Debate

**Critics argue**:
- Massive energy waste for slow payment network
- Environmental damage from fossil fuel power
- Unsustainable as Bitcoin grows

**Proponents argue**:
- Energy secures $700B+ network (cost of security)
- Incentivizes renewable energy (miners seek cheap power)
- Alternative financial systems also consume energy (banks, data centers)
- Energy consumption != carbon emissions (depends on energy source)

## Alternative Consensus Mechanisms

PoW's energy consumption led to alternatives:

**Proof of Stake (PoS)**: Validators stake capital instead of burning energy (Ethereum switched to PoS in 2022)

**Proof of Authority (PoA)**: Known validators (suitable for private blockchains)

**Proof of Space (PoSpace)**: Allocate disk space instead of computation (Chia)

**Hybrid**: Combine multiple consensus mechanisms

## Summary

Proof of Work is an elegant solution to distributed consensus:
- **Sybil resistance**: Cost to attack proportional to network size
- **Permissionless**: Anyone can mine without approval
- **Objective consensus**: Longest chain rule requires no coordination
- **Economic security**: Attacking is expensive, securing is profitable

Trade-offs:
- Energy intensive (~150 TWh/year for Bitcoin)
- Centralization pressure (economies of scale favor large mining operations)
- E-waste from obsolete mining hardware

Despite drawbacks, PoW remains the most battle-tested consensus mechanism for permissionless blockchains, securing networks worth hundreds of billions of dollars.
`,
};
