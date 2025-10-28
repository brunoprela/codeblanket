export const nodeTypesNetworkArchitectureDiscussion = [
  {
    id: 1,
    question:
      'Explain the security trade-offs of running an SPV client vs full node.',
    answer:
      "SPV (Simplified Payment Verification): Downloads block headers only (~80B vs ~1MB per block). Verifies transactions via Merkle proofs. Advantages: Lightweight (64MB vs 500GB), works on mobile. Disadvantages: (1) Trusts longest chain (can't validate all transactions), (2) Vulnerable to eclipse attack (malicious nodes feed fake chain), (3) Privacy leak (requests specific transactions from full nodes). Full node: Validates everything, trustless, maximum security. Trade-off: SPV for convenience, full node for security. Most users use SPV (mobile wallets), exchanges/merchants run full nodes.",
  },
  {
    id: 2,
    question:
      "What is an eclipse attack and how can it compromise Bitcoin's security model?",
    answer:
      "Eclipse attack: Attacker controls all of victim's peer connections, isolating them from honest network. Execution: (1) Sybil attack—create many nodes, (2) Target victim during restart/new connections, (3) Fill victim's peer slots with attacker nodes. Consequences: Victim sees attacker's fake chain, accepts invalid transactions, double-spend victim, deny service (never see real chain). Defense: (1) Diverse peer selection (geographic, AS diversity), (2) Maintain some long-lived connections, (3) Outbound connections to prevent incoming floods, (4) Checkpoint recent blocks. Real-world: Difficult but possible, especially for newly-joining nodes.",
  },
  {
    id: 3,
    question:
      'Design a strategy for running a Bitcoin full node with 99.9% uptime on limited budget.',
    answer:
      "Budget setup (~$100/month): Hardware: VPS with 8GB RAM, 1TB SSD (AWS, DigitalOcean, Hetzner ~$50/month). Software: Bitcoin Core pruned mode (keeps only recent ~550GB), systematic updates. Uptime strategy: (1) Monitoring (Grafana + Prometheus), (2) Auto-restart on crash, (3) Multiple DNS seeds, (4) Geographic redundancy (backup node in different region), (5) DDoS protection via Cloudflare/firewall. Cost optimization: Prune mode (550GB vs 500GB+ full), bandwidth limits, S3 backup of chainstate. Trade-off: Pruned node can't serve full historical data but validates everything. For personal use or payment processing, pruned mode sufficient.",
  },
];
