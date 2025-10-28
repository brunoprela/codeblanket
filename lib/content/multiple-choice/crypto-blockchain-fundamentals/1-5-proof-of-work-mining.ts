import { MultipleChoiceQuestion } from '@/lib/types';

export const proofOfWorkMiningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pow-mc-1',
    question: 'What is the primary purpose of proof-of-work in Bitcoin?',
    options: [
      'To verify transaction signatures',
      'To make block creation costly and prevent spam/attacks',
      'To distribute Bitcoin fairly to miners',
      'To compress blockchain data',
    ],
    correctAnswer: 1,
    explanation:
      "PoW makes block creation expensive (electricity + hardware), preventing spam and 51% attacks. Cost to attack > potential profit. Block reward is incentive, not primary purpose. PoW doesn't verify signatures (cryptography does) or compress data.",
  },
  {
    id: 'pow-mc-2',
    question: 'Why does Bitcoin adjust difficulty every 2016 blocks?',
    options: [
      'To keep block time at 10 minutes despite hash rate changes',
      'To make mining progressively harder',
      'To prevent memory overflow',
      '2016 is cryptographically significant',
    ],
    correctAnswer: 0,
    explanation:
      'Difficulty adjusts to maintain 10-minute blocks. If hash rate doubles, difficulty doubles to compensate. Formula: new_diff = old_diff × (target_time / actual_time). This keeps issuance predictable and gives time for block propagation.',
  },
  {
    id: 'pow-mc-3',
    question: 'What is a "51% attack" in Bitcoin?',
    options: [
      'Stealing 51% of all Bitcoin',
      'Controlling 51%+ hash power to rewrite blockchain history',
      'Hacking 51% of nodes',
      'Mining 51% of blocks',
    ],
    correctAnswer: 1,
    explanation:
      '51% attack: Control majority hash power to: (1) Double-spend by rewriting history, (2) Censor transactions, (3) Prevent confirmations. Cannot: Steal from addresses, create invalid transactions, change protocol rules. Cost: $10-20B+ for equipment. Economic deterrent: attacking Bitcoin destroys its value, making attack unprofitable.',
  },
  {
    id: 'pow-mc-4',
    question: 'Why did Bitcoin mining transition from CPUs to ASICs?',
    options: [
      'ASICs are required by Bitcoin protocol',
      'Economic competition drove specialization',
      'CPUs were banned for security',
      'ASICs make blocks smaller',
    ],
    correctAnswer: 1,
    explanation:
      'Economic competition drove specialization: CPU→GPU (100× faster)→FPGA→ASIC (1000× faster). Each improvement made previous hardware unprofitable. ASICs are chips designed ONLY for SHA-256, nothing else, maximizing efficiency. Result: Mining industrialized. Amateur mining impossible. Trade-off: Security increases (higher hash rate) but centralization risk (few can afford ASICs).',
  },
  {
    id: 'pow-mc-5',
    question: 'What happens if 90% of Bitcoin miners suddenly stop mining?',
    options: [
      'Bitcoin stops working permanently',
      'Remaining miners take 10× longer per block until difficulty adjusts',
      'Difficulty immediately drops to compensate',
      'Blocks continue at 10 minutes automatically',
    ],
    correctAnswer: 1,
    explanation:
      'If 90% miners stop: Remaining 10% takes 10× longer (100 min per block instead of 10). Must wait 2016 blocks for difficulty adjustment—would take ~4 months at 100 min/block. After adjustment, difficulty drops 10×, returns to 10 min. This is existential risk: 4-month freeze would destroy Bitcoin. Real scenario (2021): China ban reduced hash rate 50%, recovery took ~2 months (less severe).',
  },
];
