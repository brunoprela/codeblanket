/**
 * Multiple choice questions for Prompt Evaluation & A/B Testing section
 */

export const promptEvaluationABTestingMultipleChoice = [
  {
    id: 'prompt-ab-mc-1',
    question:
      'You run an A/B test with Prompt A (control) and Prompt B (treatment). After collecting 1000 samples per variant, Prompt B shows 8% improvement with p-value of 0.03. What is the MOST appropriate action?',
    options: [
      'Immediately deploy Prompt B since p<0.05',
      'Continue testing because 8% seems too good to be true',
      'Check for other factors (time of day effects, user segments) and validate the improvement is real before deploying',
      'Discard the results and restart the test from scratch',
    ],
    correctAnswer: 2,
    explanation:
      "Option C is correct. While p<0.05 indicates statistical significance, you should validate before deploying: (1) Check for confounds: Did variants run at different times? Different user segments? Any technical issues? (2) Validate the improvement makes sense: Is 8% plausible given the prompt change? (3) Check key metrics: Does improvement hold across user segments? Any negative effects on other metrics (latency, cost)? (4) Look for Simpson's paradox: Could segment-level effects differ from aggregate? Only after validation should you deploy. Option A (immediate deployment) risks shipping a fluke or confounded result. Option B (continue indefinitely) wastes opportunity if result is real. Option D (restart) throws away good data unnecessarily. Best practice: Trust but verify—significance is necessary but not sufficient for deployment.",
  },
  {
    id: 'prompt-ab-mc-2',
    question:
      "You're using a multi-armed bandit to test 4 prompt variants. After 1000 users, Variant A gets 60% traffic, B gets 25%, C gets 10%, D gets 5%. What does this traffic allocation pattern indicate?",
    options: [
      'The bandit algorithm is broken—it should give equal traffic to all variants',
      'Variant A is performing best, B is second-best, C and D are performing poorly',
      'The test is complete and you should deploy Variant A',
      'You need to increase the exploration parameter to give other variants more chances',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. Multi-armed bandits dynamically allocate more traffic to better-performing variants while still exploring others. The 60-25-10-5 split indicates: Variant A is winning (gets most traffic), Variant B is second-place (decent traffic), Variants C and D are underperforming (minimal traffic). This is EXPECTED behavior, not a bug (Option A wrong). The algorithm balances exploitation (send traffic to known-good variant) with exploration (still try others in case early data was misleading). The test is NOT complete (Option C wrong)—you continue until confident A is truly best or until you hit sample size goal. You might increase exploration if you want more data on C/D (Option D), but current allocation is working as designed. Key insight: Unequal traffic is the GOAL of bandits—it minimizes opportunity cost by avoiding bad variants.',
  },
  {
    id: 'prompt-ab-mc-3',
    question:
      'Your A/B test comparing two prompts shows Prompt B is better (p=0.02) after 5 days. However, your boss asks "How do we know it will stay better?" What analysis should you do to address this concern?',
    options: [
      'The p-value already accounts for long-term stability, so no additional analysis is needed',
      'Analyze if the improvement is consistent across different time periods, user segments, and query types',
      'Run a second identical A/B test to confirm the results',
      'Deploy B and monitor for degradation',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. Statistical significance (p=0.02) tells you the difference is unlikely due to chance, but NOT that it's stable over time or across segments. Check: (1) Temporal stability: Is B better on Day 1, Day 2, ..., Day 5 consistently? Or did it spike on one day? (2) Segment analysis: Is B better for power users? New users? Different query types? (3) Novelty effects: Do users prefer B initially but regress to A over time? Plot daily/hourly metrics to visualize trends. If B is consistently better across time/segments, you can confidently deploy. If improvement varies wildly, investigate why. Option A is wrong—p-value only addresses sampling uncertainty, not stability. Option C (second test) is redundant if first test is well-designed. Option D (deploy and monitor) is risky—better to analyze existing data first.",
  },
  {
    id: 'prompt-ab-mc-4',
    question:
      'You want to test 3 prompt variants and have 300 users available. Your colleague suggests: Day 1: Test A vs B (150 each). Day 2: Test winner vs C (150 each). What is the MAIN problem with this approach?',
    options: [
      'Sequential testing requires more total users than simultaneous testing',
      'The winner of A vs B might win by luck, then lose to C, giving you the wrong final winner',
      'You cannot do statistical testing with only 150 users per variant',
      'This approach violates the assumptions of A/B testing',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. Sequential tournaments can select suboptimal winners due to transitivity violations. Example: True performance: A=70%, B=72%, C=75%. Day 1: A vs B with 150 each. Due to randomness, A might get lucky and appear to win (say A:52%, B:48% observed). Day 2: A vs C. C wins. Final choice: C. This outcome is correct, but you got lucky. If B had won Day 1 (as it \"should\"), you'd test B vs C and still pick C. However, if the true rankings were C=75%, A=71%, B=73%, this approach might miss that B is better than A. More problematic: if A=73%, B=70%, C=72%. A might beat B (correct), then A beats C (correct), so you pick A. But B vs C wasn't tested! B might beat C. You've found a local maximum, not global. Better approach: Test all three simultaneously (100 each), or use MAB. Option A is wrong—sequential can be more efficient. Option C is wrong—150 users can be sufficient depending on effect size.",
  },
  {
    id: 'prompt-ab-mc-5',
    question:
      'Your A/B test shows Prompt B has better quality scores but worse latency (1.2s vs 0.8s). How should you make the final decision?',
    options: [
      'Always choose quality over latency',
      'Always choose latency over quality',
      'Measure combined impact on user satisfaction or business metrics (engagement, conversion)',
      'Run a new test to find a prompt that wins on both dimensions',
    ],
    correctAnswer: 2,
    explanation:
      "Option C is correct. Quality vs latency requires measuring the actual business impact. Approach: (1) Look at user behavior: Does the 50% latency increase (0.8s→1.2s) hurt engagement (bounce rate, session length)? Does better quality improve outcomes (task completion, satisfaction ratings)? (2) Segment users: Power users might tolerate higher latency for quality, Casual users might prefer speed. (3) Measure north star metric: E-commerce: Does B increase conversion despite slower response? SaaS: Does B improve user retention? (4) Calculate trade-off: If B improves conversion by 5% but increases latency 50%, net impact is likely positive. If latency kills engagement, quality doesn't matter. Options A and B are wrong—neither dimension is universally more important. Option D (find better prompt) is ideal but may not be feasible. The right choice depends on your specific user base and business model. Always optimize for business outcomes, not proxy metrics.",
  },
];
