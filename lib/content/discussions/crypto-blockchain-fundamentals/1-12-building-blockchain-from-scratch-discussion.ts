export const buildingBlockchainFromScratchDiscussion = [
  {
    id: 1,
    question:
      "What's the minimum viable set of features for a functioning blockchain?",
    answer:
      'MVP blockchain needs: (1) Block structure (prev hash, timestamp, data, nonce), (2) Chain validation (verify hashes link correctly), (3) Proof-of-work (difficulty-adjusted mining), (4) Consensus rule (longest chain), (5) Transaction model (UTXO or account), (6) Digital signatures (authorize transactions), (7) P2P networking (gossip protocol). Optional for MVP: Smart contracts, complex scripting, merkle trees, advanced consensus. Can build working blockchain in ~500 lines of Python. Real blockchains add: Security hardening, optimization, DoS protection, advanced features. Learning: Start simple, add complexity incrementally.',
  },
  {
    id: 2,
    question:
      'If you were designing a new blockchain today, what lessons from Bitcoin/Ethereum would you incorporate or avoid?',
    answer:
      'Keep from Bitcoin: (1) UTXO model for payments (privacy), (2) Simple scripting (security), (3) Proof-of-work proven but energy-intensive. Keep from Ethereum: (1) Account model for smart contracts, (2) Turing-complete VM (enables dApps), (3) Gas model (prevents DoS). Improve: (1) Start with PoS (energy efficiency), (2) Built-in privacy (Zcash-style), (3) Modular architecture (easy upgrades), (4) Better scalability (sharding from start), (5) Formal verification tools, (6) Account abstraction (better UX). Avoid: (1) Over-complexity (Ethereum gas is complex), (2) Immutable bugs (have upgrade path), (3) Centralization (even with PoS). Trade-offs: Simplicity vs features, security vs innovation.',
  },
  {
    id: 3,
    question:
      "Walk through the attack vectors you'd need to defend against when launching a new blockchain.",
    answer:
      'Attack vectors to defend: (1) 51% attack—ensure sufficient decentralization, PoS slashing, (2) Double-spend—require confirmations, finality gadget, (3) Sybil attack—PoW/PoS resistance, (4) Eclipse attack—diverse peer selection, (5) Selfish mining—protocol-level defense, (6) Long-range attack (PoS)—weak subjectivity, (7) Nothing-at-stake—slashing, (8) Smart contract bugs—audits, formal verification, (9) DoS—gas limits, rate limiting, (10) Network partition—prioritize liveness or safety. Defense strategy: Testnet for months, bug bounties ($500K+), security audits (3+ firms), start with small economic value, gradual rollout. Most new chains fail due to: Insufficient testing, premature launch, unforeseen attack vectors.',
  },
];
