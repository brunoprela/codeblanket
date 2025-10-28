import { MultipleChoiceQuestion } from '@/lib/types';

export const proofOfStakeMechanismsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pos-mc-1',
    question: 'What is "slashing" in Proof-of-Stake?',
    options: [
      'Reducing block rewards over time',
      'Penalizing validators by destroying their stake for dishonest behavior',
      'Cutting transaction fees',
      'Splitting validators into shards',
    ],
    correctAnswer: 1,
    explanation:
      'Slashing destroys validator stake for provably malicious behavior: double-voting, surround voting, going offline. Example: Validator stakes 32 ETH, votes on conflicting blocks, loses portion or all of stake. This makes PoS attacks expensive—51% attack requires owning AND losing 51% of staked capital. Replaces PoW energy cost with economic cost.',
  },
  {
    id: 'pos-mc-2',
    question: 'What is the "nothing-at-stake" problem?',
    options: [
      'Validators with no stake cannot participate',
      'Without cost, validators could validate all forks simultaneously',
      "Staking doesn't generate rewards",
      'Validators need minimum 32 ETH',
    ],
    correctAnswer: 1,
    explanation:
      "Nothing-at-stake: In naive PoS, validators could vote on ALL competing forks (no cost like PoW energy). This prevents consensus—every fork equally valid. Solution: Slashing penalties for voting on conflicting chains. Now validators MUST choose one fork or lose stake. PoW naturally prevents this through energy cost—can't mine all forks simultaneously.",
  },
  {
    id: 'pos-mc-3',
    question: 'How does Ethereum achieve "finality" in PoS?',
    options: [
      'After 6 confirmations like Bitcoin',
      'After 2 epochs (~13 minutes) using Casper finality gadget',
      'Immediately upon block creation',
      'Never—remains probabilistic like PoW',
    ],
    correctAnswer: 1,
    explanation:
      "Ethereum uses Casper FFG (finality gadget): After 2 epochs of attestations, block becomes finalized and cannot be reverted. Requires 2/3 of validators to attest. Once finalized, reversing requires destroying 1/3+ of staked ETH through slashing. This is absolute finality, unlike Bitcoin's probabilistic finality. Trade-off: Requires honest 2/3 majority vs Bitcoin's 51%.",
  },
  {
    id: 'pos-mc-4',
    question: 'What is a "long-range attack" in PoS?',
    options: [
      'Attacking from far away geographically',
      'Using old validator keys to rewrite history from genesis',
      'Attacking future blocks',
      'Staking for long time periods',
    ],
    correctAnswer: 1,
    explanation:
      "Long-range attack: Attacker obtains old validator keys (bought/hacked from validators who exited), rewrites blockchain history from point where they had stake. Harder in PoW (needs historical hash power). PoS defense: Weak subjectivity—new nodes checkpoint recent block, won't accept chains forking before checkpoint. Trade-off: Requires some trust in checkpoint, not pure trustless like PoW.",
  },
  {
    id: 'pos-mc-5',
    question: 'Why is PoS more energy-efficient than PoW?',
    options: [
      'PoS uses better algorithms',
      "PoS doesn't require cryptographic hashing",
      'PoS replaces energy cost with economic stake, no computation race',
      'PoS has fewer validators',
    ],
    correctAnswer: 2,
    explanation:
      "PoS ~99.95% more energy-efficient because: No mining race—validators selected by stake, not computation. Don't need to try trillions of hashes. Validation requires minimal computation (verify signatures/blocks). PoW: Must continuously compute hashes (200+ TWh/year). PoS: Run validator software (0.01 TWh/year). Trade-off: PoS sacrifices physical security guarantees for capital efficiency.",
  },
];
