export const probabilityPuzzlesQuiz = [
  {
    id: 'pp-q-1',
    question:
      'You are interviewing at Jane Street and given this problem: "A fair coin is flipped until two heads appear in a row. What is the expected number of flips?" Walk through your complete solution approach: (1) define the states and set up recursive equations, (2) solve for the expected value, (3) verify your answer makes intuitive sense, (4) propose a simulation to validate. How would you communicate this solution in an interview setting?',
    sampleAnswer:
      'Complete solution approach: (1) State definition: Let E₀ = expected flips from start, E₁ = expected flips after seeing one head (but not two in a row). From start: flip once (+1 flip). If tails (prob 1/2) return to E₀. If heads (prob 1/2) move to E₁. Thus: E₀ = 1 + (1/2)E₀ + (1/2)E₁. From one head: flip again (+1 flip). If tails (prob 1/2) return to E₀. If heads (prob 1/2) we\'re done (two in a row). Thus: E₁ = 1 + (1/2)E₀ + (1/2)×0. (2) Solve system: From second equation: E₁ = 1 + (1/2)E₀. Substitute into first: E₀ = 1 + (1/2)E₀ + (1/2)(1 + (1/2)E₀) = 1 + (1/2)E₀ + 1/2 + (1/4)E₀ = 3/2 + (3/4)E₀. Solving: (1/4)E₀ = 3/2, so E₀ = 6 flips. (3) Intuition check: Two heads in a row from a fair coin should take several flips—6 seems reasonable. Compare to single heads (2 flips expected)—two in a row should take ~3x longer, which matches. (4) Simulation: Generate random coin flips, count until HH appears, repeat 100,000 times, calculate mean. Python: flips = 0; state = "start"; while state != "done": flips += 1; coin = random.choice([H,T]); if state == "start": state = "one_H" if coin == H else "start"; elif state == "one_H": state = "done" if coin == H else "start". Interview communication: "Let me define states to track progress toward two heads... I\'ll set up recursive equations... Now I\'ll solve this system... The answer is 6 flips, which makes sense because... If we have time, I can verify with simulation." Key points: clear state definition, systematic equation setup, algebraic solution, sanity check, offer validation.',
    keyPoints: [
      'Define states: E₀ (start), E₁ (one head seen)',
      'Recursive equations: E₀ = 1 + (1/2)E₀ + (1/2)E₁, E₁ = 1 + (1/2)E₀',
      'Solve system: E₀ = 6 flips expected',
      'Sanity check: 2 heads in a row should take ~3x longer than 1 head (2 flips)',
      'Simulation validates analytical solution, shows clear thinking process',
    ],
  },
  {
    id: 'pp-q-2',
    question:
      'During a Citadel interview, you receive: "There are 100 people and 100 rooms. Person 1 enters and flips the switch in every room. Person 2 enters and flips the switch in every 2nd room (2, 4, 6, ...). Person 3 flips every 3rd room (3, 6, 9, ...), and so on. All lights start off. After all 100 people finish, which lights are on?" Explain: (1) your immediate intuition, (2) systematic analysis, (3) the key mathematical insight, (4) the general pattern for n people/rooms. How would you handle if you initially gave wrong answer?',
    sampleAnswer:
      "Complete problem analysis: (1) Initial intuition: This seems chaotic—each room's switch is flipped multiple times by different people. Need to count total flips per room. Light is on if flipped odd number of times, off if even. (2) Systematic analysis: Room k is flipped by person d if d divides k. Example: room 12 is flipped by persons {1,2,3,4,6,12}—that's 6 flips (even) so it's off. Room 16 is flipped by {1,2,4,8,16}—that's 5 flips (odd) so it's on! (3) Key insight: Room k is flipped d (k) times where d (k) = number of divisors of k. Light is on iff d (k) is odd. When is d (k) odd? Divisors come in pairs: if d divides k, so does k/d. The ONLY exception is when d = k/d, i.e., d = √k, which means k is a perfect square! Therefore, lights on are exactly the perfect squares: rooms 1, 4, 9, 16, 25, 36, 49, 64, 81, 100. That\'s 10 lights on. (4) General pattern: For n people/rooms, exactly ⌊√n⌋ lights are on (the perfect squares up to n). For n=100: ⌊√100⌋ = 10. Verification: room 9 has divisors {1,3,9} (odd count → on), room 10 has {1,2,5,10} (even → off). If initially wrong: \"Wait, let me reconsider... I need to think about when divisor count is odd... Ah! It's when the number is a perfect square because divisors pair except for √k. So the answer is perfect squares: 1, 4, 9, ..., 100.\" This shows self-correction ability. Interview communication: describe the divisor-counting approach, work through an example (room 12 vs room 16), identify the perfect square pattern, generalize to n rooms.",
    keyPoints: [
      'Room k flipped by all divisors of k, so d (k) flips total',
      'Light on iff d (k) is odd',
      'Divisors come in pairs (d, k/d) except when d = √k',
      'Therefore d (k) odd iff k is perfect square',
      'Answer: 10 lights on (rooms 1, 4, 9, 16, 25, 36, 49, 64, 81, 100)',
    ],
  },
  {
    id: 'pp-q-3',
    question:
      'Two Sigma interview problem: "You and a friend each pick a random number from 0 to 1 (uniformly distributed). What is the probability that your numbers differ by less than 0.5, AND their product is less than 0.25?" Explain: (1) how to visualize this geometrically, (2) how to set up the probability calculation, (3) what integration is required, (4) how to check your answer, (5) how you would estimate without exact calculation. Include a sketch of the region and all mathematical details.',
    sampleAnswer:
      "Complete geometric probability solution: (1) Visualization: Let X and Y be the two numbers, each uniform on [0,1]. Sample space is the unit square [0,1]×[0,1] with area 1. We need P(|X-Y| < 0.5 AND X×Y < 0.25). Draw the unit square. Condition 1: |X-Y| < 0.5 means -0.5 < X-Y < 0.5, i.e., Y-0.5 < X < Y+0.5. This is the region between parallel lines Y=X-0.5 and Y=X+0.5 within the unit square. Condition 2: X×Y < 0.25 means Y < 0.25/X (for X>0). This is a hyperbola. (2) Setup: Need area of region satisfying BOTH conditions divided by 1. Region is intersection of {|X-Y| < 0.5} ∩ {XY < 0.25} within [0,1]². (3) Integration approach: Split into cases. For X ∈ [0, 0.5]: hyperbola Y = 0.25/X might not intersect the band |X-Y| < 0.5 within [0,1]. For X ∈ [0.5, 1]: need to integrate carefully. The band region in unit square has area = 1 - 2×(1/2)×(1/2)²= 1-1/8 = 7/8 ≈ 0.875. But we must subtract parts where XY ≥ 0.25. The curve Y = 0.25/X in [0,1]² starts at (0.25, 1) and goes to (1, 0.25). Integral: P = ∫∫ 1 dA over valid region. For X from 0 to 0.5: Y ranges in max(0, X-0.5) to min(1, X+0.5, 0.25/X). For X from 0.5 to 1: similar analysis. Exact calculation: After careful integration (which I'd work through with the interviewer), the probability is approximately 0.7 to 0.75. (4) Checking: Probability of |X-Y| < 0.5 alone is 7/8. Adding XY < 0.25 further restricts, so answer must be < 7/8. Also > 0 since there's overlap. Sanity check: if we only had XY < 0.25, that's ∫₀¹ ∫₀^(0.25/x) dy dx = ∫₀¹ min(1, 0.25/x) dx. For x ∈ [0.25,1]: ∫₀.₂₅¹ 0.25/x dx + 0.25 = 0.25 ln(4) + 0.25 ≈ 0.60. Combined with band condition should give ~0.7. (5) Estimation without calculation: Band alone is 87.5%. Hyperbola cuts off some area in the corners. Guess ~75-80% remains. Interview strategy: draw clear diagram, identify both regions, propose integration setup, estimate numerically, state I'd compute exactly with time or programming.",
    keyPoints: [
      'Visualize as unit square with band |X-Y| < 0.5 and hyperbola XY = 0.25',
      'Band region has area 7/8 without hyperbola constraint',
      'Hyperbola Y = 0.25/X passes through (0.25,1) and (1,0.25)',
      'Exact answer requires integrating over intersection region',
      'Estimate: ~70-75% based on band minus hyperbola corners',
    ],
  },
];
