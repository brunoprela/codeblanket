export const proofOfWorkMiningDiscussion = [
  {
    id: 1,
    question:
      "Is Bitcoin mining 'wasteful'? Analyze the energy consumption vs security trade-off.",
    answer:
      'Bitcoin uses ~200 TWh/year (0.5% global electricity). Critics: waste, environmental damage. Defenders: (1) Secures $600B network, (2) Uses stranded/renewable energy, (3) Incentivizes renewable development. Energy per transaction misleading—security is per block, not per transaction. Alternative: Proof-of-Stake uses 99.9% less energy but different security model. Trade-off: Physical energy waste vs economic security guarantees.',
  },
  {
    id: 2,
    question:
      "How does mining centralization threaten Bitcoin's security model?",
    answer:
      'Top 4 mining pools control >60% hash power. Risks: (1) Pool collusion for 51% attack, (2) Government coercion of pools, (3) Single points of failure. Mitigations: (1) Miners can switch pools instantly, (2) Economic disincentives (attack destroys Bitcoin value), (3) Geographic distribution of hash power. China mining ban (2021) showed resilience—hash power migrated globally.',
  },
  {
    id: 3,
    question:
      "Explain the 'nothing at stake' problem that would exist if Bitcoin had no mining cost.",
    answer:
      'Without mining cost, miners could mine on ALL competing forks simultaneously (no penalty). This would: (1) Prevent consensus (every fork equally valid), (2) Enable easy 51% attacks (no resource cost), (3) Make double-spending trivial. Proof-of-Work solves this: mining costs energy, forcing miners to choose one fork, creating economic consensus. Proof-of-Stake different approach: slashing penalties replace energy cost.',
  },
];
