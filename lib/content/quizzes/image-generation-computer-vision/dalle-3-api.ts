/**
 * Quiz questions for DALL-E 3 API section
 */

export const dalle3apiQuiz = [
  {
    id: 'igcv-dalle3-q-1',
    question:
      'DALL-E 3 automatically rewrites prompts (revised_prompt). How should you use this feature in production, and when might you want to be more prescriptive to minimize rewrites?',
    hint: 'Consider when AI enhancement helps vs. when you need exact control.',
    sampleAnswer:
      'The revised_prompt feature is double-edged: **Benefits** - DALL-E 3 adds helpful details like camera angles, lighting, style that improve results. A prompt like "a cat" becomes "a close-up photograph of a domestic short-haired tabby cat with green eyes..." producing much better images. **Production strategy**: 1) For initial exploration/ideation, allow rewrites to discover what works. 2) Log all revised_prompts and analyze patterns - you\'ll learn what details DALL-E adds. 3) For production templates, incorporate successful patterns directly: instead of "logo", write "minimalist professional logo design on white background, vector style, clean geometric shapes". **When to minimize rewrites**: 1) Specific requirements - "product photo with exact blue (#0066CC) background" where added details might conflict. 2) Consistency - generating variations where you need same style. 3) Text in images - be very specific: "poster with text \'SALE\' in bold red letters" to prevent text changes. 4) A/B testing - need identical conditions. **Best practice**: Start with detailed prompts that leave less room for interpretation. The more specific you are, the less DALL-E needs to add.',
    keyPoints: [
      'Revised prompts add helpful details but reduce control',
      'Log and analyze revised prompts to learn patterns',
      'Be specific in prompts to minimize unwanted additions',
      'Use detailed prompts for consistency and specific requirements',
    ],
  },
  {
    id: 'igcv-dalle3-q-2',
    question:
      'Explain the cost-benefit trade-offs between standard and HD quality in DALL-E 3. In what scenarios does HD quality provide sufficient value to justify 2x the cost?',
    hint: 'Think about use cases and quality requirements.',
    sampleAnswer:
      'HD vs Standard trade-off analysis: **Cost difference** - $0.04 vs $0.08 for square images (100% increase), $0.08 vs $0.12 for rectangular (50% increase). **Quality improvement** - HD provides better detail fidelity, sharper edges, more accurate fine features, but NOT higher resolution (both are 1024×1024). **Scenarios where HD justifies 2x cost**: 1) **Client-facing/marketing** - Brand reputation and professional appearance justify premium. One bad image costs more than HD generation. 2) **Print materials** - Extra detail helps when images are enlarged or closely examined. 3) **Small text in images** - HD significantly improves text legibility. 4) **Detailed subjects** - Intricate patterns, mechanical details, architectural features benefit from HD. 5) **Final deliverables** - When this is the end product, not iteration. **Use standard when**: 1) **Prototyping/iteration** - Exploring ideas, generating variations. 2) **Social media** - Compression and small display sizes negate HD benefits. 3) **Internal use** - Presentations, documentation where quality bar is lower. 4) **High volume** - Generating 1000s of images where cost compounds. **Hybrid approach**: Generate with standard, use HD for best results. Or A/B test to see if users/metrics show difference.',
    keyPoints: [
      'HD is 2x cost for better detail, same resolution',
      'Justify HD for client-facing, print, text, detailed subjects',
      'Use standard for prototyping, social media, high volume',
      'Hybrid: standard for iteration, HD for finals',
    ],
  },
  {
    id: 'igcv-dalle3-q-3',
    question:
      'Design a caching strategy for a DALL-E 3-based application that balances cost savings with freshness of results. What would you cache and for how long?',
    hint: 'Consider identical prompts, similar prompts, and use case patterns.',
    sampleAnswer:
      'Comprehensive caching strategy: **Level 1: Exact Match Cache** - Cache identical prompts indefinitely since DALL-E 3 is deterministic for same input (mostly). Hash prompt+parameters as key. Store: URL, revised_prompt, metadata. **TTL**: 30 days for active use, migrate to cold storage after. **Expected hit rate**: 15-30% in steady state. **Cost savings**: Avoid $0.04-$0.08 per hit. For 10K requests/day with 20% hit rate = $60-$120/day saved. **Level 2: Similar Prompt Suggestions** - When exact miss, compute semantic similarity (embeddings) to suggest existing images: "You generated something very similar yesterday - reuse that?" Don\'t auto-serve (might not be exact match) but save user a generation. **Level 3: Template Caching** - For applications with templates (e.g., "product on white background"), cache successful examples and offer as starting points. **What NOT to cache**: 1) User-specific generations with personal data. 2) Time-sensitive content ("today\'s news"). 3) Failed generations or low-quality results. **Cache invalidation**: 1) Explicit user refresh. 2) DALL-E model updates. 3) Monthly purge of unused entries. **Storage**: S3 for images, Redis for metadata/URLs. **Privacy**: Hash user IDs in cache keys, encrypt cached data. **Implementation**: Middleware checks cache before API call, updates after successful generation.',
    keyPoints: [
      'Exact match cache: hash of prompt+params, 30-day TTL',
      'Can save $60-120/day at scale (20% hit rate)',
      'Similar prompt suggestions for near-misses',
      'Template caching for common patterns',
    ],
  },
];
