import { MultipleChoiceQuestion } from '@/lib/types';

export const buildingBlockchainFromScratchMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bbfs-mc-1',
      question:
        'What is the minimum data each block must contain to form a valid blockchain?',
      options: [
        'Just the transactions',
        'Previous block hash, timestamp, transactions, nonce',
        'Only the block number',
        'Previous hash is sufficient',
      ],
      correctAnswer: 1,
      explanation:
        'Minimum block structure: (1) Previous block hash—links to chain, enables tamper detection, (2) Timestamp—proof of chronology, difficulty adjustment, (3) Transactions/data—actual content, (4) Nonce—proof-of-work solution. Optional but common: Merkle root (instead of full transactions), difficulty target, block reward. Why each matters: Prev hash—change any block invalidates all subsequent blocks. Timestamp—required for difficulty adjustment. Nonce—proves work was done. Without these, not secure blockchain—just linked list. Real Bitcoin block header: 80 bytes containing these fields.',
    },
    {
      id: 'bbfs-mc-2',
      question:
        'In a proof-of-work blockchain, what happens if you change data in an old block?',
      options: [
        'Nothing—blocks are independent',
        'That block becomes invalid but others are fine',
        'That block AND all subsequent blocks become invalid',
        'The blockchain auto-corrects',
      ],
      correctAnswer: 2,
      explanation:
        "Changing old block invalidates entire chain from that point forward. Why: Each block contains hash of previous block. Change block N → hash of block N changes → block N+1 previous hash field no longer matches → block N+1 invalid → block N+2 references invalid block N+1 → entire chain invalid. Attacker must re-mine changed block AND all subsequent blocks. Cost increases exponentially with depth. This is blockchain's tamper-evidence property. Real scenario: To change 6-block-deep transaction, must re-mine 6 blocks faster than honest network mines 1 block—requires >50% hash power.",
    },
    {
      id: 'bbfs-mc-3',
      question:
        'Why do simple blockchains use proof-of-work instead of just having nodes vote on valid blocks?',
      options: [
        'PoW is faster',
        'PoW is cheaper',
        'Voting vulnerable to Sybil attacks (attacker creates many fake nodes)',
        'PoW uses less electricity',
      ],
      correctAnswer: 2,
      explanation:
        'Voting fails in permissionless systems due to Sybil attack: Attacker creates thousands of fake nodes, controls majority vote, can approve invalid blocks. PoW solves this: Vote weight tied to computational work (hard to fake), not node count (easy to fake). One CPU = one vote would fail (create 10K virtual machines). One hash = one vote works (must actually compute). This is why PoW burns energy—makes vote weight scarce and unfakeable. Alternative: PoS uses staked capital (also hard to fake) instead of computation. Trade-off: PoW energy waste vs Sybil resistance. Permissioned blockchains can use voting (known participants).',
    },
    {
      id: 'bbfs-mc-4',
      question:
        'What does "longest chain rule" mean and why do blockchains use it?',
      options: [
        'Literal length in bytes',
        'Most transactions win',
        'Chain with most cumulative proof-of-work wins',
        'Oldest chain wins',
      ],
      correctAnswer: 2,
      explanation:
        'Longest chain = most cumulative work (technically "heaviest" not "longest"). When fork occurs: (1) Network temporarily has two valid chains, (2) Miners build on their preferred chain, (3) Eventually one chain grows longer (more PoW), (4) All nodes switch to longest chain. Why: (1) Objective rule (no coordination needed), (2) Represents majority hash power, (3) Economic incentive (miners on shorter chain lose rewards). Not literal length—a 100-block chain with low difficulty loses to 50-block chain with high difficulty. Ethereum slight variation: GHOST protocol considers uncle blocks. Rule ensures eventual consensus without central authority.',
    },
    {
      id: 'bbfs-mc-5',
      question: 'Why is a "genesis block" special in blockchain?',
      options: [
        'It has the most transactions',
        "It's the first block with no previous block to reference",
        'It can be modified later',
        'It contains all the cryptocurrency',
      ],
      correctAnswer: 1,
      explanation:
        'Genesis block: First block in chain, hardcoded in software, has no previous block hash (or references all zeros). Special properties: (1) Not mined normally—created by founders, (2) Contains initial state/distribution, (3) Immutable—changing genesis requires new blockchain, (4) Embedded in all nodes (trust anchor). Bitcoin genesis (Jan 3, 2009): Contains famous headline "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"—proves block created after that date. Genesis defines blockchain identity—different genesis = different blockchain. All validation traces back to genesis.',
    },
  ];
