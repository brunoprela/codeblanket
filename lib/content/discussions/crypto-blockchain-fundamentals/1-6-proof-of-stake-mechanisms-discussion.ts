export const proofOfStakeMechanismsDiscussion = [
  {
    id: 1,
    question: 'Compare security of PoW vs PoS for a $100B blockchain.',
    answer:
      "PoW: Attack costs physical resources (~$10B+ equipment). External to chain. PoS: Attack costs stake (~$50B+ for 51%). Internal to chain. PoW advantage: Cannot reuse attack (energy spent). PoS advantage: Slashing destroys attacker's stake. PoW risk: Mining centralization. PoS risk: Wealth concentration. Both secure at scale, different trust models.",
  },
  {
    id: 2,
    question:
      "Explain the 'nothing-at-stake' problem and how slashing solves it.",
    answer:
      'Nothing-at-stake: Without cost, validators could vote on ALL forks simultaneously, preventing consensus. PoW prevents this with energy cost. PoS solution: Slashing—if validator votes on conflicting blocks, loses entire stake. Makes attacking expensive, forces validators to choose one fork. Criticism: Requires detecting violations on-chain, complex implementation.',
  },
  {
    id: 3,
    question: 'Why does Ethereum PoS require 32 ETH minimum stake?',
    answer:
      '32 ETH (~$60K) balances: (1) Sufficient economic deterrent (slashing hurts), (2) Enables ~500K validators (decentralization), (3) Manageable validation overhead. Lower stake: Too many validators, network overhead. Higher stake: Too few validators, centralization. Can pool stake via services, but introduces trust assumptions.',
  },
];
