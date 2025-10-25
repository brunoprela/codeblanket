import { MultipleChoiceQuestion } from '@/lib/types';

export const combinatoricsCountingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cc-mc-1',
    question:
      'How many 5-card poker hands contain exactly one pair (two cards of same rank, three other cards of different ranks)?',
    options: ['1,098,240', '1,302,540', '2,598,960', '10,200'],
    correctAnswer: 0,
    explanation:
      'Choose rank for pair (13 choices), choose 2 suits for pair (C(4,2)=6), choose 3 ranks for other cards (C(12,3)=220), choose suit for each (4³=64). Total: 13 × 6 × 220 × 64 = 1,098,240. This is the second most common poker hand after high card.',
  },
  {
    id: 'cc-mc-2',
    question:
      'From point (0,0) to point (5,3) on a grid, moving only right (R) or up (U), how many shortest paths exist?',
    options: ['15', '35', '56', '120'],
    correctAnswer: 2,
    explanation:
      'Need 5 R moves and 3 U moves (total 8 moves). Choose which 3 positions (out of 8) are U moves: C(8,3) = 56. Equivalently C(8,5) for R positions. This is a classic application of combinations to grid paths.',
  },
  {
    id: 'cc-mc-3',
    question:
      'Six people sit around a circular table. How many distinct seating arrangements exist (considering rotations as same)?',
    options: ['120', '360', '720', '5,040'],
    correctAnswer: 0,
    explanation:
      "For circular arrangements, fix one person's position to eliminate rotational counting. Then arrange remaining 5 people: 5! = 120. If reflections were also considered same, divide by 2: 5!/2 = 60. But the question only mentions rotations, so answer is 120.",
  },
  {
    id: 'cc-mc-4',
    question:
      'How many ways can you distribute 10 identical coins into 3 distinct piggy banks (banks can be empty)?',
    options: ['30', '66', '120', '1,000'],
    correctAnswer: 1,
    explanation:
      'Stars and bars: n=10 identical objects, k=3 distinct bins. Formula: C(n+k-1, k-1) = C(10+3-1, 3-1) = C(12, 2) = 66. This counts all non-negative integer solutions to x₁ + x₂ + x₃ = 10.',
  },
  {
    id: 'cc-mc-5',
    question:
      'For n=4, how many derangements exist (permutations where no element stays in its original position)?',
    options: ['6', '9', '12', '18'],
    correctAnswer: 1,
    explanation:
      'D(4) = 4! × Σ((-1)^k/k!) for k=0 to 4 = 24 × (1 - 1 + 1/2 - 1/6 + 1/24) = 24 × 9/24 = 9. Alternatively, D(4) = 4!/e rounded = 24/2.718 ≈ 8.84 ≈ 9. Derangements count permutations with no fixed points.',
  },
];
