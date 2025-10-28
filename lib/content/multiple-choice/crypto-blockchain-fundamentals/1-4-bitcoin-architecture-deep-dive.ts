import { MultipleChoiceQuestion } from '@/lib/types';

export const bitcoinArchitectureDeepDiveMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bad-mc-1',
      question:
        "In Bitcoin's UTXO model, what happens to a UTXO once it is used as an input in a transaction?",
      options: [
        'It is marked as spent but remains in the UTXO set',
        'It is completely removed from the UTXO set',
        'It is locked for 6 confirmations before removal',
        'It is split into smaller UTXOs automatically',
      ],
      correctAnswer: 1,
      explanation:
        'UTXOs are completely removed from the UTXO set once spent. Bitcoin tracks UTXOs (Unspent Transaction Outputs), not spent outputs. When transaction references UTXO as input, validators check it exists in UTXO set and hasn\'t been spent. After transaction confirms, UTXO is removed. New outputs become new UTXOs. This enables fast double-spend checking: simply verify UTXO exists. If same UTXO appears in two transactions, second is invalid (double-spend attempt). Account model different: deducts balance, no "removal." Bitcoin\'s approach trades storage efficiency (must store all UTXOs) for validation speed and parallelism.',
    },
    {
      id: 'bad-mc-2',
      question: "What is the purpose of Bitcoin Script's OP_CHECKSIG opcode?",
      options: [
        'To verify the transaction signature matches the public key',
        'To check if two signatures are identical',
        'To generate a new signature for the transaction',
        'To validate the transaction format',
      ],
      correctAnswer: 0,
      explanation:
        "OP_CHECKSIG verifies ECDSA signature matches public key for transaction. Execution: (1) Pop signature and pubkey from stack, (2) Hash transaction (excluding signatures), (3) Verify signature using ECDSA verify algorithm, (4) Push TRUE if valid, FALSE if invalid. This is how Bitcoin proves authorization: only private key owner can create valid signature. Combined with OP_HASH160 in P2PKH: OP_DUP OP_HASH160 <hash> OP_EQUALVERIFY OP_CHECKSIG ensures: (1) Public key hashes to expected value, (2) Signature proves private key ownership. Without OP_CHECKSIG, anyone could spend anyone's Bitcoin. This is core security primitive.",
    },
    {
      id: 'bad-mc-3',
      question:
        'Why does Bitcoin adjust mining difficulty every 2016 blocks instead of after every block?',
      options: [
        '2016 blocks is the minimum time needed to propagate difficulty changes globally',
        'Adjusting every block would be too computationally expensive',
        'It provides a stable measurement period to avoid manipulation and noise',
        'The 2016 number has cryptographic significance for hash functions',
      ],
      correctAnswer: 2,
      explanation:
        '2016 blocks (~2 weeks) provides stable measurement, preventing manipulation. If adjusted every block: (1) Network variance could cause wild swings (lucky blocks would increase difficulty unnecessarily), (2) Miners could manipulate timestamps to game difficulty, (3) No stable metric for actual hash rate. 2 weeks is long enough to: Average out randomness, Resist manipulation attempts, Provide meaningful hash rate signal. But short enough to: Adapt to real changes, Not freeze network if hash rate drops. Real example: 2013 ASIC introduction increased hash rate 100× in weeks; 2-week adjustment handled it. Answer A wrong: Difficulty changes propagate instantly (in block header). Answer B wrong: Calculation is trivial. Answer D wrong: 2016 chosen for practical reasons, not cryptographic.',
    },
    {
      id: 'bad-mc-4',
      question: 'What is an "orphan block" in Bitcoin?',
      options: [
        'A block with no transactions except the coinbase',
        'A valid block that was mined but not included in the longest chain',
        'A block from a non-mining node',
        'A block with invalid transactions',
      ],
      correctAnswer: 1,
      explanation:
        'Orphan block: Valid block mined but abandoned because another block at same height won the race. Scenario: (1) Miner A finds block X at height 100, (2) Miner B finds block Y at height 100, (3) Both broadcast, network splits, (4) Miner C finds block at height 101 building on X, (5) Longest chain is now X→C, Y is orphaned. Consequences: (1) Miner B loses block reward (12.5 BTC + fees wasted), (2) Transactions in Y might not be in X (must be re-mined), (3) Anyone who accepted Y must reorganize. Orphan rate: ~0.5-1% of blocks. Caused by: Network latency, simultaneous mining, selfish mining attacks. This is why 10-minute block time exists: gives propagation time, reduces orphan rate. 10-second blocks would have 50%+ orphan rate.',
    },
    {
      id: 'bad-mc-5',
      question:
        'In Bitcoin Script, why are loops (OP_LOOP, OP_WHILE) not allowed?',
      options: [
        "Loops would make Bitcoin's blockchain too large",
        'Loops could create non-terminating scripts that DOS the network',
        'Loops are unnecessary for payment transactions',
        'Satoshi forgot to implement them',
      ],
      correctAnswer: 1,
      explanation:
        "Loops enable non-terminating scripts: while(true){} would force validators to execute forever, DOSing validation. Bitcoin must validate ALL transactions in ALL blocks, so malicious loop would halt the network. Without loops: Every script guaranteed to terminate (bounded by script size and opcode count). Max execution: O(script_size). With loops: Halting problem unsolvable—can't determine if script terminates before running it. Ethereum solution: Gas limits—charge per opcode, terminate after gas exhausted. But this adds complexity Bitcoin avoids. Trade-off: Bitcoin sacrifices Turing completeness for security and simplicity. Can't do complex smart contracts, but guaranteed fast validation. Correct for digital cash, wrong for computation platform.",
    },
  ];
