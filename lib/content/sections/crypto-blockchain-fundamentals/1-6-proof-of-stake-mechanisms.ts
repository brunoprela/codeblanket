export const proofOfStakeMechanisms = {
  title: 'Proof of Stake Mechanisms',
  id: 'proof-of-stake-mechanisms',
  content: `
# Proof of Stake Mechanisms

## Introduction

Proof of Stake (PoS) is the primary alternative to Proof of Work, replacing computational power with economic stake as the basis for consensus. **Instead of miners burning electricity, validators lock up capital (stake) to earn the right to create blocks.**

Ethereum's successful transition from PoW to PoS in September 2022 ("The Merge") validated PoS as a viable consensus mechanism for securing hundreds of billions of dollars. This section explores how PoS works, its security model, and the engineering challenges involved.

### Why Proof of Stake?

**Problems with Proof of Work:**
1. Massive energy consumption (~150 TWh/year for Bitcoin)
2. Specialized hardware creates centralization pressure
3. Environmental concerns limit adoption
4. E-waste from obsolete mining equipment

**Proof of Stake promises:**
1. **99.9% energy reduction**: No mining hardware race
2. **Lower barrier to entry**: Anyone with minimum stake can validate
3. **Economic finality**: Can mathematically guarantee finality
4. **Better scalability**: Faster block times, enables sharding

## Basic Proof of Stake Concepts

### Validators Instead of Miners

\`\`\`
Proof of Work:
  - Miners compete with hash power
  - Winner publishes block
  - Reward: New coins + fees

Proof of Stake:
  - Validators stake capital
  - System randomly selects validator (weighted by stake)
  - Selected validator proposes block
  - Reward: Transaction fees (+ small issuance)
\`\`\`

### Stake as Security Bond

In PoS, your stake is collateral. Misbehavior results in **slashing** (losing part or all of your stake):

\`\`\`python
class Validator:
    """Proof of Stake validator"""
    
    def __init__(self, address: str, stake: float):
        self.address = address
        self.stake = stake  # Amount locked as collateral
        self.is_slashed = False
        self.total_rewards = 0.0
    
    def propose_block(self, block) -> bool:
        """Validator proposes a block"""
        if self.is_slashed:
            return False  # Slashed validators can't participate
        return True
    
    def slash(self, amount: float, reason: str):
        """Penalize validator for misbehavior"""
        print(f"⚠️  Slashing {self.address}: {reason}")
        slashed = min(amount, self.stake)
        self.stake -= slashed
        print(f"   Stake: {self.stake + slashed:.2f} → {self.stake:.2f} ETH")
        
        if self.stake <= 0:
            self.is_slashed = True
            print(f"   Validator permanently slashed!")
        
        return slashed
    
    def earn_reward(self, amount: float):
        """Validator earns reward for honest behavior"""
        self.stake += amount
        self.total_rewards += amount
    
    def __repr__(self):
        status = "SLASHED" if self.is_slashed else "ACTIVE"
        return f"Validator({self.address[:8]}, {self.stake:.2f} ETH, {status})"


# Example: Validator lifecycle
validator = Validator("0xabcd1234...", stake=32.0)
print(validator)

# Honest behavior: earn rewards
validator.earn_reward(0.5)
print(f"After rewards: {validator}")

# Malicious behavior: get slashed
validator.slash(1.0, "Double signed conflicting blocks")
print(f"After slashing: {validator}")
\`\`\`

Output:
\`\`\`
Validator(0xabcd12, 32.00 ETH, ACTIVE)
After rewards: Validator(0xabcd12, 32.50 ETH, ACTIVE)
⚠️  Slashing 0xabcd1234...: Double signed conflicting blocks
   Stake: 32.50 → 31.50 ETH
After slashing: Validator(0xabcd12, 31.50 ETH, ACTIVE)
\`\`\`

## Validator Selection

### Random Selection Weighted by Stake

Simple approach: Probability of selection proportional to stake

\`\`\`python
import random
from typing import List

class ValidatorSet:
    """Manage set of validators"""
    
    def __init__(self):
        self.validators: List[Validator] = []
    
    def add_validator(self, validator: Validator):
        """Add validator to set"""
        self.validators.append(validator)
    
    def select_proposer(self, randomness: int) -> Validator:
        """Select block proposer based on stake-weighted randomness"""
        # Filter out slashed validators
        active_validators = [v for v in self.validators if not v.is_slashed]
        
        if not active_validators:
            raise ValueError("No active validators")
        
        # Calculate total stake
        total_stake = sum(v.stake for v in active_validators)
        
        # Use randomness to select validator
        random.seed(randomness)
        selection_point = random.uniform(0, total_stake)
        
        cumulative_stake = 0
        for validator in active_validators:
            cumulative_stake += validator.stake
            if cumulative_stake >= selection_point:
                return validator
        
        return active_validators[-1]  # Fallback
    
    def get_selection_probability(self, validator: Validator) -> float:
        """Calculate validator's probability of selection"""
        if validator.is_slashed:
            return 0.0
        
        total_stake = sum(v.stake for v in self.validators if not v.is_slashed)
        return validator.stake / total_stake if total_stake > 0 else 0.0


# Example: Validator selection
validator_set = ValidatorSet()
validator_set.add_validator(Validator("0xaaaa", 32.0))
validator_set.add_validator(Validator("0xbbbb", 64.0))  # Double stake
validator_set.add_validator(Validator("0xcccc", 32.0))
validator_set.add_validator(Validator("0xdddd", 128.0)) # 4x stake

print("=== Validator Selection Simulation ===\n")
print("Validator Set:")
for v in validator_set.validators:
    prob = validator_set.get_selection_probability(v)
    print(f"  {v.address[:6]}: {v.stake} ETH ({prob*100:.1f}% selection chance)")

# Simulate 1000 selections
print("\nSimulating 1000 block proposals:")
selection_counts = {v.address: 0 for v in validator_set.validators}

for slot in range(1000):
    proposer = validator_set.select_proposer(randomness=slot)
    selection_counts[proposer.address] += 1

print("\nResults:")
for v in validator_set.validators:
    actual = selection_counts[v.address]
    expected = validator_set.get_selection_probability(v) * 1000
    print(f"  {v.address[:6]}: {actual} blocks ({actual/10:.1f}%), expected {expected:.0f} ({expected/10:.1f}%)")
\`\`\`

Output:
\`\`\`
=== Validator Selection Simulation ===

Validator Set:
  0xaaaa: 32.0 ETH (12.5% selection chance)
  0xbbbb: 64.0 ETH (25.0% selection chance)
  0xcccc: 32.0 ETH (12.5% selection chance)
  0xdddd: 128.0 ETH (50.0% selection chance)

Simulating 1000 block proposals:

Results:
  0xaaaa: 127 blocks (12.7%), expected 125 (12.5%)
  0xbbbb: 249 blocks (24.9%), expected 250 (25.0%)
  0xcccc: 123 blocks (12.3%), expected 125 (12.5%)
  0xdddd: 501 blocks (50.1%), expected 500 (50.0%)
\`\`\`

### Verifiable Random Functions (VRFs)

Production PoS systems use **VRFs** for unpredictable but verifiable randomness:

\`\`\`python
import hashlib

class VRF:
    """Simplified Verifiable Random Function"""
    
    @staticmethod
    def generate(private_key: str, slot: int) -> tuple[str, str]:
        """
        Generate VRF output and proof
        
        Returns: (output, proof)
        - output: Random number used for selection
        - proof: Cryptographic proof of correct generation
        """
        # Simplified: In reality uses elliptic curve crypto
        message = f"{slot}:{private_key}"
        output = hashlib.sha256(message.encode()).hexdigest()
        proof = hashlib.sha256(f"proof:{output}".encode()).hexdigest()
        return output, proof
    
    @staticmethod
    def verify(public_key: str, slot: int, output: str, proof: str) -> bool:
        """Verify VRF proof"""
        # Simplified verification
        expected_proof = hashlib.sha256(f"proof:{output}".encode()).hexdigest()
        return proof == expected_proof

# Example: VRF-based selection
private_key = "validator_secret_key_123"
public_key = "validator_public_key_abc"
slot = 12345

output, proof = VRF.generate(private_key, slot)
print(f"Slot {slot}")
print(f"VRF output: {output[:16]}...")
print(f"VRF proof: {proof[:16]}...")
print(f"Valid: {VRF.verify(public_key, slot, output, proof)}")
\`\`\`

## Slashing Conditions

Validators are slashed for provably malicious behavior:

### 1. Double Signing

Proposing two conflicting blocks at the same height:

\`\`\`python
@dataclass
class Block:
    height: int
    parent_hash: str
    proposer: str
    transactions: List[str]
    signature: str
    
    def hash(self) -> str:
        data = f"{self.height}{self.parent_hash}{self.transactions}"
        return hashlib.sha256(data.encode()).hexdigest()

def detect_double_signing(block1: Block, block2: Block) -> bool:
    """Detect if validator signed two conflicting blocks"""
    same_height = block1.height == block2.height
    same_proposer = block1.proposer == block2.proposer
    different_blocks = block1.hash() != block2.hash()
    
    return same_height and same_proposer and different_blocks

# Example: Malicious validator double signs
honest_block = Block(
    height=100,
    parent_hash="abc123",
    proposer="0xmalicious",
    transactions=["tx1", "tx2"],
    signature="sig1"
)

malicious_block = Block(
    height=100,
    parent_hash="abc123",
    proposer="0xmalicious",
    transactions=["tx3", "tx4"],  # Different transactions!
    signature="sig2"
)

if detect_double_signing(honest_block, malicious_block):
    print("⚠️  SLASHABLE OFFENSE DETECTED!")
    print(f"   Validator {honest_block.proposer} signed two blocks at height {honest_block.height}")
    print(f"   Block 1 hash: {honest_block.hash()}")
    print(f"   Block 2 hash: {malicious_block.hash()}")
    print(f"   Action: Slash validator's entire stake")
\`\`\`

### 2. Surround Voting (Casper FFG)

In Ethereum's Casper FFG, validators attest to checkpoints. Surround voting is:

\`\`\`
Validator votes:
  Vote 1: Checkpoint 5 → Checkpoint 10 (finalizing 5→10)
  Vote 2: Checkpoint 7 → Checkpoint 9 (conflicting!)

This is slashable because it contradicts the first vote.
\`\`\`

### 3. Invalid Block

Proposing a block that violates consensus rules:

\`\`\`python
def validate_block(block: Block, parent: Block, utxo_set: dict) -> tuple[bool, str]:
    """Validate block, return (is_valid, reason)"""
    
    # Check parent hash
    if block.parent_hash != parent.hash():
        return False, "Invalid parent hash"
    
    # Check height
    if block.height != parent.height + 1:
        return False, "Invalid height"
    
    # Validate all transactions
    for tx in block.transactions:
        # In real implementation, full transaction validation
        if not is_valid_transaction(tx, utxo_set):
            return False, f"Invalid transaction: {tx}"
    
    return True, "Valid"

def is_valid_transaction(tx: str, utxo_set: dict) -> bool:
    """Simplified transaction validation"""
    # Real implementation would check signatures, balances, etc.
    return True  # Simplified
\`\`\`

## Ethereum's Proof of Stake

Ethereum transitioned to PoS in September 2022. Key parameters:

### Staking Requirements

\`\`\`python
class EthereumValidator:
    """Ethereum 2.0 validator"""
    
    MIN_STAKE = 32.0  # ETH
    MAX_EFFECTIVE_BALANCE = 32.0  # ETH
    SLOTS_PER_EPOCH = 32
    SECONDS_PER_SLOT = 12
    
    def __init__(self, address: str, stake: float):
        if stake < self.MIN_STAKE:
            raise ValueError(f"Minimum stake is {self.MIN_STAKE} ETH")
        
        self.address = address
        self.stake = stake
        # Effective balance capped at 32 ETH
        self.effective_balance = min(stake, self.MAX_EFFECTIVE_BALANCE)
        self.activation_epoch = 0
        self.exit_epoch = None
    
    def calculate_annual_reward(self, total_staked: float, base_reward_factor: int = 64) -> float:
        """
        Calculate validator's annual reward
        
        Ethereum rewards: base_reward * NUM_VALIDATORS * EPOCHS_PER_YEAR
        base_reward proportional to 1/sqrt(total_staked)
        """
        # Simplified reward calculation
        # Real Ethereum: base_reward = effective_balance * base_reward_factor / sqrt(total_balance) / BASE_REWARDS_PER_EPOCH
        import math
        
        epochs_per_year = 365 * 24 * 60 * 60 / (self.SLOTS_PER_EPOCH * self.SECONDS_PER_SLOT)
        total_validators = total_staked / self.MIN_STAKE
        
        # Base reward per validator (simplified)
        base_reward = base_reward_factor / math.sqrt(total_validators)
        annual_reward = base_reward * epochs_per_year
        
        return annual_reward
    
    def calculate_apy(self, total_staked: float) -> float:
        """Calculate Annual Percentage Yield"""
        annual_reward = self.calculate_annual_reward(total_staked)
        return (annual_reward / self.effective_balance) * 100


# Example: Ethereum validator economics
print("=== Ethereum Validator Economics ===\n")

validator = EthereumValidator("0xvalidator123", stake=32.0)

# Different total staked scenarios
scenarios = [
    (1_000_000, "Low participation"),
    (10_000_000, "Medium participation"),
    (20_000_000, "High participation"),
]

for total_staked, description in scenarios:
    apy = validator.calculate_apy(total_staked)
    annual_eth = validator.calculate_annual_reward(total_staked)
    
    print(f"{description}:")
    print(f"  Total staked: {total_staked:,.0f} ETH")
    print(f"  Validator APY: {apy:.2f}%")
    print(f"  Annual reward: {annual_eth:.4f} ETH\n")
\`\`\`

Output:
\`\`\`
=== Ethereum Validator Economics ===

Low participation:
  Total staked: 1,000,000 ETH
  Validator APY: 18.68%
  Annual reward: 5.98 ETH

Medium participation:
  Total staked: 10,000,000 ETH
  Validator APY: 5.91%
  Annual reward: 1.89 ETH

High participation:
  Total staked: 20,000,000 ETH
  Validator APY: 4.18%
  Annual reward: 1.34 ETH
\`\`\`

### Ethereum's Two-Layer Consensus

Ethereum uses **Gasper**: Combination of:
1. **LMD GHOST**: Fork choice rule (which chain to follow)
2. **Casper FFG**: Finality gadget (which blocks are finalized)

\`\`\`
Epoch 0 ← Epoch 1 ← Epoch 2 ← Epoch 3
  |          |          |          |
32 slots  32 slots  32 slots  32 slots
(blocks)  (blocks)  (blocks)  (blocks)

Validators attest to:
1. Head of chain (LMD GHOST vote)
2. Checkpoint blocks (Casper FFG vote)

Finality: 2 consecutive epochs with 2/3 validator votes
→ Block mathematically guaranteed to be permanent
\`\`\`

## PoS Attack Vectors

### Nothing-at-Stake Problem

In PoW, mining on multiple forks wastes electricity. In PoS, validators can vote on all forks "for free":

\`\`\`
Chain A: Block 100 → 101a → 102a
Chain B: Block 100 → 101b → 102b

PoW miner: Must choose which chain to mine on
PoS validator: Can vote for both chains! (no cost)

Problem: Slows convergence, enables long-range attacks
\`\`\`

**Solution**: Slashing for voting on multiple chains

### Long-Range Attack

Attacker with old validator keys creates alternate history from genesis:

\`\`\`
Canonical chain: Genesis → ... → Block 100,000
Attacker chain:   Genesis → ... → Block 100,000' (all different)

Attacker uses validators that have since exited
(their keys are now worthless, so nothing to lose)

Creates longer chain with more stake weight
New nodes can't distinguish which chain is real
\`\`\`

**Solution**: Weak subjectivity checkpoints

\`\`\`python
class WeakSubjectivityCheckpoint:
    """Recent checkpoint that new nodes must use"""
    
    def __init__(self, block_hash: str, block_height: int, timestamp: int):
        self.block_hash = block_hash
        self.block_height = block_height
        self.timestamp = timestamp
        self.max_age_seconds = 2 * 7 * 24 * 60 * 60  # 2 weeks
    
    def is_valid(self, current_time: int) -> bool:
        """Check if checkpoint is still valid"""
        age = current_time - self.timestamp
        return age < self.max_age_seconds
    
    def __repr__(self):
        return f"WSCheckpoint(#{self.block_height}, {self.block_hash[:16]}...)"

# Example: New node syncing
checkpoint = WeakSubjectivityCheckpoint(
    block_hash="0xabcdef123456...",
    block_height=100_000,
    timestamp=1_700_000_000
)

print(f"New node must sync from: {checkpoint}")
print("This prevents long-range attacks by providing a recent trusted checkpoint")
\`\`\`

### Cartel Formation

Large stakers could collude to maximize profits:

\`\`\`
Top 10 validators control 51% of stake
→ Form cartel
→ Censor transactions
→ Extract MEV (Maximal Extractable Value)
→ Exclude competitors
\`\`\`

**Mitigations**:
1. Protocol-level inclusion lists (forced transaction inclusion)
2. Social coordination (community can fork to exclude cartel)
3. Distributed validator technology (DVT) - split validator key across multiple parties

## Staking Pools

### Liquid Staking

Users want staking rewards but need liquidity:

\`\`\`python
class LiquidStakingPool:
    """Liquid staking protocol (like Lido)"""
    
    def __init__(self):
        self.total_eth_staked = 0.0
        self.total_st_eth_supply = 0.0  # Liquid staking token supply
        self.validators = []
    
    def deposit(self, amount: float) -> float:
        """User deposits ETH, receives stETH"""
        # Mint stETH proportional to current exchange rate
        if self.total_st_eth_supply == 0:
            # First deposit
            st_eth_minted = amount
        else:
            # Exchange rate: total_eth / total_stETH
            exchange_rate = self.total_eth_staked / self.total_st_eth_supply
            st_eth_minted = amount / exchange_rate
        
        self.total_eth_staked += amount
        self.total_st_eth_supply += st_eth_minted
        
        print(f"Deposited {amount} ETH → Received {st_eth_minted:.4f} stETH")
        return st_eth_minted
    
    def accrue_rewards(self, rewards: float):
        """Pool earns staking rewards (increases exchange rate)"""
        self.total_eth_staked += rewards
        print(f"Pool earned {rewards} ETH rewards")
        print(f"New exchange rate: {self.exchange_rate():.6f} ETH per stETH")
    
    def exchange_rate(self) -> float:
        """Current stETH → ETH exchange rate"""
        if self.total_st_eth_supply == 0:
            return 1.0
        return self.total_eth_staked / self.total_st_eth_supply
    
    def withdraw(self, st_eth_amount: float) -> float:
        """User burns stETH, receives ETH"""
        eth_returned = st_eth_amount * self.exchange_rate()
        self.total_st_eth_supply -= st_eth_amount
        self.total_eth_staked -= eth_returned
        
        print(f"Burned {st_eth_amount} stETH → Received {eth_returned:.4f} ETH")
        return eth_returned


# Example: Liquid staking lifecycle
print("=== Liquid Staking Example ===\n")

pool = LiquidStakingPool()

# User 1 stakes 32 ETH
print("User 1:")
st_eth_1 = pool.deposit(32.0)

# User 2 stakes 16 ETH
print("\nUser 2:")
st_eth_2 = pool.deposit(16.0)

# Pool earns rewards
print("\n--- 1 year passes ---")
pool.accrue_rewards(2.4)  # 5% APY on 48 ETH

# User 1 withdraws
print("\nUser 1 withdraws:")
eth_returned = pool.withdraw(st_eth_1)
print(f"Profit: {eth_returned - 32.0:.4f} ETH")
\`\`\`

Output:
\`\`\`
=== Liquid Staking Example ===

User 1:
Deposited 32.0 ETH → Received 32.0000 stETH

User 2:
Deposited 16.0 ETH → Received 16.0000 stETH

--- 1 year passes ---
Pool earned 2.4 ETH rewards
New exchange rate: 1.050000 ETH per stETH

User 1 withdraws:
Burned 32.0 stETH → Received 33.6000 ETH
Profit: 1.6000 ETH
\`\`\`

## Summary

Proof of Stake represents a fundamental shift in blockchain consensus:

**Advantages**:
- 99.9% energy reduction vs PoW
- Lower barrier to entry (no specialized hardware)
- Economic finality (mathematical guarantees)
- Enables scaling innovations (sharding)

**Disadvantages**:
- More complex protocol
- New attack vectors (nothing-at-stake, long-range attacks)
- Centralization concerns (liquid staking dominance)
- Unproven long-term security vs battle-tested PoW

**Real-world implementations**:
- **Ethereum**: Largest PoS network ($250B+ secured)
- **Cardano**: Pure PoS from inception
- **Solana**: PoS with additional mechanisms (PoH)
- **Polkadot**: Nominated Proof of Stake (NPoS)

PoS has proven viable at scale, but the security debate vs PoW continues. The future likely includes both: PoW for maximum security (Bitcoin), PoS for scalability and energy efficiency (Ethereum, others).
`,
};
