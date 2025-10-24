/**
 * Quiz questions for Prompt Evaluation & A/B Testing section
 */

export const promptEvaluationABTestingQuiz = [
  {
    id: 'prompt-ab-q-1',
    question:
      'You run an A/B test comparing two prompts with 500 users each. Prompt A: 72% success rate. Prompt B: 76% success rate. A colleague says "B is clearly better, let\'s ship it!" Calculate if this difference is statistically significant and explain your decision process for whether to deploy Prompt B.',
    hint: 'Need to check if the 4% difference is due to real improvement or just random chance. Use statistical significance testing.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Need statistical significance test, not just comparing percentages',
      'With 500 users each, 4% difference (76% vs 72%) is NOT significant (p=0.075)',
      'Should continue testing with more users or set higher bar for deployment',
      'Same absolute difference can be significant or not depending on sample size',
    ],
  },
  {
    id: 'prompt-ab-q-2',
    question:
      'You want to test 5 different prompt variations. Your colleague suggests running a single A/B/C/D/E test splitting traffic 5 ways (20% each). You have 1000 daily users. Is this a good strategy? Design a better approach using either sequential testing or multi-armed bandits.',
    hint: 'Testing 5 variants simultaneously with 200 users each gives little power. Consider alternative strategies that allocate traffic dynamically.',
    sampleAnswer:
      "**Problems with Uniform 5-Way Split:** (1) Each variant gets only 200 users/day → takes 5+ days to reach significance, (2) Bad variants keep getting 20% traffic even after we know they're bad, (3) Multiple comparisons problem (comparing 5 variants increases false positive risk). **Better Approach: Multi-Armed Bandit (Best)** **Setup:** Start with equal allocation (20% each), After each 100 users, recalculate each variant's performance, Shift traffic toward better performers using epsilon-greedy or UCB algorithm. **Example (Epsilon-Greedy with ε=0.2):** Day 1: Equal split (20% each), After 200 users/variant: Variant A: 75%, Variant B: 78%, Variant C: 71%, Variant D: 76%, Variant E: 69%, Day 2+: 80% traffic to best performer (B), 20% exploration to others, After 500 more users: B consistently best → 90% to B, 10% to others. **Result:** Variant B gets most traffic (80-90%), reducing opportunity cost of testing inferior variants, Still explore others in case of early flukes, Reach decision faster (2-3 days vs 5+ days), Users get better experience during test (more see good variant). **Alternative: Sequential Tournament:** Day 1-2: Test A vs B (500 each), pick winner, Day 3-4: Winner vs C (500 each), pick winner, Day 5-6: Winner vs D (500 each), etc. **Pros:** Simple, clear comparisons, full power for each test. **Cons:** Takes longer (10+ days), early variant has advantage. **Recommendation:** Use MAB for best balance of speed, user experience, and statistical rigor. Modern platforms (Optimizely, Split, etc.) support MAB out of the box.",
    keyPoints: [
      'Testing 5 variants equally is inefficient (200 users each, slow convergence)',
      'Multi-armed bandits dynamically allocate more traffic to better variants',
      'Reduces opportunity cost: users see good variants more, bad variants less',
      'Reaches decision faster while maintaining statistical validity',
    ],
  },
  {
    id: 'prompt-ab-q-3',
    question:
      'Your A/B test shows Prompt B has 10% higher quality but 50% higher cost per request. Prompt A: $0.02/request, 75% quality. Prompt B: $0.03/request, 82.5% quality. Your product has thin margins. Make a recommendation with financial analysis of when each prompt makes sense.',
    hint: 'Need to analyze cost vs quality trade-off in context of business model and user value.',
    sampleAnswer:
      "**Financial Analysis:** **Scenario 1: Freemium SaaS (Most Users Free)** Monthly users: 100K, Free users: 90K (no direct revenue), Paid users: 10K @ $20/month = $200K revenue. Cost with Prompt A: 100K × $0.02 = $2K/month, Cost with Prompt B: 100K × $0.03 = $3K/month, Extra cost: $1K/month. Quality impact: 7.5% better experience with B, Assume conversion improves: 10% → 11% (+10% relative) from better quality, New paid users: 100K × 0.11 = 11K, New revenue: 11K × $20 = $220K (+$20K), ROI: Spend extra $1K to gain $20K → 20x ROI. **Recommendation for Freemium: Use Prompt B.** Better quality improves conversion, pays for itself 20x over. **Scenario 2: High-Volume, Low-Margin Product** Monthly requests: 10M, Revenue per request: $0.025 (thin margins), Current profit: $0.005/request with Prompt A. Cost increase: $0.01/request for Prompt B, New profit: $0.025 - $0.03 = -$0.005 (LOSING money!). Even if quality → 10% more usage: New requests: 11M, Still losing: 11M × (-$0.005) = -$55K loss vs $50K profit. **Recommendation for High-Volume/Low-Margin: Use Prompt A.** Can't afford 50% cost increase with thin margins. **Hybrid Strategy (Best for Most):** Use Prompt B for: New users (acquisition, first impression matters), Paid/premium users (they expect quality), Complex queries (quality matters more). Use Prompt A for: Free users after onboarding, Simple queries (quality difference less noticeable), High-volume automated API calls. **Example Split:** 30% of requests use Prompt B (new users, paid, complex), 70% use Prompt A. Cost: 0.3×$0.03 + 0.7×$0.02 = $0.023/request (15% increase), Quality: Weighted avg ≈ 77.25%, Get 60% of quality improvement at 30% of cost increase.",
    keyPoints: [
      'Cost-quality trade-off depends on business model and margins',
      'Freemium: quality drives conversion → use better prompt despite cost',
      "Thin margins: can't afford 50% cost increase → use cheaper prompt",
      'Hybrid strategy: use expensive prompt for high-value scenarios (new users, paid tiers)',
    ],
  },
];
