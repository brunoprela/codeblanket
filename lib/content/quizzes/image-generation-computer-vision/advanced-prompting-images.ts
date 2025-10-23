/**
 * Quiz questions for Advanced Prompting for Images section
 */

export const advancedpromptingimagesQuiz = [
  {
    id: 'igcv-advprompt-q-1',
    question:
      'Explain the importance of prompt structure order (Subject → Action → Context → Style → Technical → Quality). Does this order actually matter, and how would you test it?',
    hint: 'Consider how models process tokens and attention mechanisms.',
    sampleAnswer:
      'Prompt order significance: **Why order matters** - Transformers process text sequentially and use attention to weight token importance. Earlier tokens often receive more attention in deciding overall image composition. Elements mentioned first tend to be more prominent. **Empirical evidence**: Tests show subject-first prompts ("a cat sitting on a chair") produce more focused compositions than reversed ("on a chair, there is a cat sitting"). **Optimal structure**: **Subject first** (what) - Establishes primary focus. "a businesswoman" **Action** (doing what) - Defines pose/activity. "presenting to a group" **Context** (where/when) - Sets environment. "in modern conference room" **Style** (aesthetic) - Artistic direction. "professional photography" **Technical** (camera/lighting) - Image characteristics. "50mm lens, natural lighting" **Quality modifiers** (enhancement) - "sharp focus, high resolution". **Testing methodology**: 1) Generate same concept with different orders, same seed. 2) Survey users on which is "more accurate to description". 3) Measure how often key elements appear. 4) Use CLIP score to measure prompt-image alignment. **Practical impact**: Order can mean difference between "portrait of woman with hat" (woman prominent, wearing hat) vs "hat on woman" (hat prominent, woman background). **Production advice**: Template your prompts with consistent structure. Let users describe subject freely but structure it properly before sending to model.',
    keyPoints: [
      'Earlier tokens tend to receive more attention',
      'Subject-first produces more focused compositions',
      'Consistent structure: Subject → Action → Context → Style → Technical → Quality',
      'Test with A/B comparisons and CLIP scores',
    ],
  },
  {
    id: 'igcv-advprompt-q-2',
    question:
      'Design a negative prompt strategy that maximizes quality without being overly restrictive. How would you balance avoiding unwanted elements with allowing creative freedom?',
    hint: 'Think about universal quality terms vs. specific content exclusions.',
    sampleAnswer:
      'Balanced negative prompt strategy: **Core principle** - Negative prompts should prevent common failures without overly constraining the creative space. **Three-tier approach**: **Tier 1: Universal quality negatives (always include)** - "blurry, low quality, low resolution, pixelated, jpeg artifacts, compression artifacts, grainy, noisy, distorted". These have no downside - you never want these. **Tier 2: Common failure modes (context-dependent)** - For people/portraits: "bad anatomy, deformed, disfigured, extra limbs, bad hands, bad proportions". For objects: "cropped, cut off, bad framing, tilted". These prevent common AI failures. **Tier 3: Content-specific exclusions (use sparingly)** - Only exclude what would truly ruin the image. If generating "photorealistic portrait", exclude "cartoon, anime, drawing". But don\'t blanket-exclude many styles or you limit variety. **What to avoid**: 1) **Contradictory negatives** - Don\'t exclude "dark" and "bright" simultaneously. 2) **Over-specification** - Long lists (50+ terms) can harm generation quality. 3) **Style conflicts** - If prompt says "oil painting", don\'t also negate "painting". **Testing approach**: Start with minimal negatives, gradually add more while comparing results. Find the sweet spot where quality improves without becoming repetitive. **Production template**: Base negative: quality terms (Tier 1) + anatomical terms (if people) + specific exclusions for use case. Example photorealistic portrait negative: "blurry, low quality, bad anatomy, bad face, asymmetric, cartoon, anime, 3d render".',
    keyPoints: [
      'Always include universal quality negatives',
      'Add common failure modes based on content type',
      'Use specific exclusions sparingly',
      'Avoid overly long lists that constrain creativity',
    ],
  },
  {
    id: 'igcv-advprompt-q-3',
    question:
      'How would you implement a prompt weighting system in production that allows non-technical users to emphasize elements without understanding the (keyword:weight) syntax?',
    hint: 'Think about UI/UX and translating user intent to technical weights.',
    sampleAnswer:
      'User-friendly weighting system design: **UI approach** - Instead of exposing technical syntax, provide intuitive controls: **Visual importance slider** - User sees their prompt elements as chips: [cat] [hat] [garden]. Each has a slider: "Normal | Emphasized | Critical" that maps to weights: 1.0, 1.3, 1.6. **Natural language modifiers** - Let users type naturally: "Make the hat more prominent" parses to increase hat weight. Or tags: "a cat [wearing a #important red hat] in garden" where #important = 1.4x. **Template-based** - Pre-built templates: "Product focus: highlights main item, blurs background" = (product:1.5), (background:0.7). **Smart detection** - Analyze prompt to identify key nouns, offer: "Which is most important? [cat] [hat] [garden]" - User clicks, you weight accordingly. **Backend implementation**: 1) Parse user-friendly input. 2) Map to technical weights. 3) Construct proper weighted prompt: "(red hat:1.4)" internally. 4) Log both versions for debugging. **Validation**: Prevent conflicts - if user emphasizes everything, normalize weights. Cap at 1.8 max (beyond that diminishes returns). **Testing methodology**: A/B test: let group A use raw weights, group B use UI. Measure: success rate, time to desired result, satisfaction. **Production considerations**: Store both user intent and technical prompt for reproducibility. Allow advanced users to toggle "expert mode" with direct weight control. **Error handling**: If weight combination produces poor results, suggest adjustments: "Making everything prominent reduces contrast - try emphasizing one key element".',
    keyPoints: [
      'Hide technical syntax behind intuitive UI controls',
      'Use sliders, tags, or natural language for emphasis',
      'Auto-detect key elements and offer importance selection',
      'Store both user intent and technical weights',
    ],
  },
];
