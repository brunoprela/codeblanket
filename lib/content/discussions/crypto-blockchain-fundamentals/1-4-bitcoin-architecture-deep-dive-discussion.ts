export const bitcoinArchitectureDeepDiveDiscussion = [
  {
    id: 1,
    question:
      'Why did Bitcoin choose the UTXO model over an account-based model? Analyze the trade-offs in terms of privacy, scalability, and smart contract capability. If you were designing Bitcoin today, would you still choose UTXO?',
    answer:
      "Bitcoin's UTXO model provides superior privacy and parallelism at the cost of complexity. Privacy: Each output is independent, enabling address rotation without linking transactions. Account model links all transactions to one address. Scalability: UTXOs can be processed in parallel (no global account state), while account model requires sequential processing to prevent race conditions. Smart contracts: Account model (Ethereum) enables complex state machines; UTXO limits programmability. Today's choice: Depends on use case. For digital cash prioritizing privacy: UTXO. For smart contract platform: Account model. Hybrid possible: Use UTXO for value transfer, account model for contracts (like Bitcoin with Rootstock).",
  },
  {
    id: 2,
    question:
      "Explain Bitcoin Script's limitations (no loops, limited opcodes) and why these are security features, not bugs. Could Bitcoin add Turing-complete scripting? What would be the consequences?",
    answer:
      "Bitcoin Script limitations are deliberate security features. No loops prevents: (1) Infinite execution DoS attacks, (2) Unpredictable resource consumption, (3) Non-deterministic execution times. Limited opcodes ensures: (1) Every script terminates, (2) Validation is fast and predictable, (3) No state changes beyond transaction. Adding Turing completeness (like Ethereum) would introduce: (1) Halting problem (can't determine if script terminates), (2) Gas system needed (complex), (3) Slower validation, (4) More attack surface. Consequence: Bitcoin would need massive redesign, losing its primary advantage (simple, predictable validation). Ethereum pays this cost with higher complexity and gas mechanics. Bitcoin's choice: Simplicity and security over programmability. Right for digital gold, wrong for dApp platform.",
  },
  {
    id: 3,
    question:
      "Bitcoin's difficulty adjusts every 2016 blocks. What happens if 90% of hash power suddenly disappears? Walk through the recovery scenario and explain why this is an existential threat.",
    answer:
      'If 90% hash power disappears: (1) Remaining 10% takes 10× longer per block (100 minutes vs 10 minutes), (2) 2016 blocks takes ~4 months instead of 2 weeks, (3) Network essentially frozen for 4 months until difficulty adjusts. Recovery: After 2016 blocks, difficulty drops to 1/10th, returns to 10-minute blocks. But 4-month freeze would: (1) Destroy user confidence, (2) Make Bitcoin unusable for payments, (3) Allow competing chain to take market share. Real scenario: If China banned mining (60% hash power): 2016 blocks would take ~5 weeks, not 4 months, but still severe. Mitigation: Emergency hard fork to adjust difficulty sooner, but requires coordination and violates immutability principle. This is why decentralized hash power matters—single point of failure for hash rate is existential risk.',
  },
];
