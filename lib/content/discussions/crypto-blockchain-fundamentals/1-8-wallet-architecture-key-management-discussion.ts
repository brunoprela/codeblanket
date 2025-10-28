export const walletArchitectureKeyManagementDiscussion = [
  {
    id: 1,
    question:
      'Explain the security trade-offs between hot wallets, hardware wallets, and paper wallets.',
    answer:
      'Hot wallet: High convenience, high risk (malware/hacks). Hardware wallet: Medium convenience, low risk (air-gapped signing). Paper wallet: Low convenience, medium risk (physical theft/damage). Best practice: Tiered approach—hot wallet for daily spending (<$1K), hardware wallet for medium amounts ($1K-$100K), multi-sig cold storage for large amounts (>$100K). Most funds lost to convenience (using hot wallets) not security breaches.',
  },
  {
    id: 2,
    question:
      'How do HD wallets derive child keys from master seed? Why is this better than random key generation?',
    answer:
      'HD wallets use BIP32: Master seed → Master key → Child keys via one-way function. Each child key cryptographically derived but appears random. Advantages: (1) Single backup (12-word phrase), (2) Infinite addresses from one seed, (3) Can derive public keys without private keys (watch-only wallets), (4) Organized key structure. vs Random generation: Need backup for EVERY key. Lose one backup = lose those funds. HD wallet: Lose seed = lose everything, but only one thing to backup.',
  },
  {
    id: 3,
    question:
      'Design a multi-sig setup for a $100M crypto treasury. What M-of-N would you choose and why?',
    answer:
      'Recommended: 3-of-5 multi-sig. 5 key holders: (1) CEO, (2) CFO, (3) CTO, (4) Board member, (5) Legal counsel. Requires 3 signatures for any transaction. Reasoning: 2-of-3 too risky—single malicious actor + one compromised key = theft. 4-of-7 too slow—hard to coordinate 4 people. 3-of-5 balances security (need compromise 3 people) and availability (can lose 2 keys and still access funds). Geographic distribution: Keys in different jurisdictions. Hardware wallets for each signer. Additional: Time-locks for large transactions, alerts on all signing activity.',
  },
];
