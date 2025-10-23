/**
 * Quiz questions for Image-to-Image Generation section
 */

export const imagetoimagegenerationQuiz = [
  {
    id: 'igcv-img2img-q-1',
    question:
      'Explain how the strength parameter in img2img affects the balance between preserving the input image structure and allowing creative changes. How would you choose the optimal strength value for different use cases?',
    hint: 'Consider the noise amount and its impact on generation.',
    sampleAnswer:
      'The strength parameter controls how much noise is added to the input image before denoising, fundamentally determining the creativity vs fidelity trade-off. **How it works**: strength=0.3 adds minimal noise, preserving 70% of original structure. strength=0.8 adds substantial noise, only loosely following input. **Use case guidelines**: **Subtle enhancement (0.2-0.3)**: Color grading, lighting adjustment, minor style tweaks. Input structure fully preserved, only surface changes. Use for: photo enhancement, color correction. **Style transfer (0.4-0.6)**: Convert photo to painting while keeping composition. Balanced transformation. Use for: artistic conversion, material changes, time-of-day shifts. **Creative reimagining (0.6-0.8)**: Major changes, loose interpretation. Use for: concept exploration, dramatic transformations. **Optimal selection methodology**: 1) Start at 0.5 (middle ground). 2) If result too similar to input, increase by 0.1. 3) If result loses key features, decrease by 0.1. 4) Generate batch with [0.3, 0.4, 0.5, 0.6, 0.7] to find sweet spot. **Production strategy**: For automated systems, use conservative values (0.3-0.5) to avoid unexpected results. For creative tools, let users control with real-time preview. **Key insight**: Same prompt, different strengths can produce vastly different results, so strength is as important as the prompt itself.',
    keyPoints: [
      'Strength = amount of noise added before denoising',
      'Low (0.2-0.4): preserve structure, subtle changes',
      'Medium (0.4-0.6): balanced style transfer',
      'High (0.6-0.8): creative freedom, loose interpretation',
    ],
  },
  {
    id: 'igcv-img2img-q-2',
    question:
      'Design a production img2img workflow for an e-commerce platform that needs to generate product variations (different colors, backgrounds, styles). What are the key challenges and how would you address them?',
    hint: 'Consider consistency, quality control, and scale.',
    sampleAnswer:
      'Production e-commerce img2img workflow: **Architecture**: 1) Input: Original product photo (white background, clean). 2) Variation pipeline: Color changes, background replacement, lifestyle context. 3) Quality control: Automated + human review. 4) Output: Multiple variation images. **Key challenges and solutions**: **Challenge 1: Consistency** - Need variations to look like same product. Solution: Use low strength (0.25-0.35) to preserve product features. Add "same product, exact same item" to prompt. Use same seed across variations when possible. Generate batch, filter for consistency. **Challenge 2: Color accuracy** - Customer expects "red shirt" to be actual red. Solution: Use color codes in prompts "(#FF0000 red)". Generate multiple candidates, use computer vision to verify color accuracy. Consider fine-tuning model on your product categories. **Challenge 3: Background realism** - Lifestyle contexts must look natural. Solution: Use ControlNet with depth maps to maintain product placement. Prompt with specific backgrounds: "product on wooden table in bright kitchen" vs vague "nice background". **Challenge 4: Quality control at scale** - Can\'t manually review 10,000 variations. Solution: Automated checks: 1) CLIP score for prompt adherence. 2) Object detection to verify product present. 3) Image quality metrics. 4) Flag outliers for human review. **Challenge 5: Cost and speed** - Need variations quickly and economically. Solution: Cache common variations. Batch process during off-hours. Use SD locally vs API. **Complete workflow**: Upload product → Extract product mask → Generate variations (colors: strength=0.3, backgrounds: strength=0.4, contexts: strength=0.5) → Quality filter → Human spot-check → Deploy to site. **Metrics to track**: Variation success rate, customer engagement with variations, return rates for variation purchases.',
    keyPoints: [
      'Low strength (0.25-0.35) for consistency',
      'Automated quality control at scale',
      'Different strengths for different variation types',
      'Batch processing and caching for efficiency',
    ],
  },
  {
    id: 'igcv-img2img-q-3',
    question:
      'Compare img2img with ControlNet for achieving precise structural control. In what scenarios would you choose one over the other?',
    hint: 'Think about the type and precision of control needed.',
    sampleAnswer:
      "img2img vs ControlNet comparison: **img2img control mechanism**: Uses input image directly as starting point. Control is implicit through the image itself. Strength parameter determines influence. Simpler, more intuitive. **ControlNet control mechanism**: Uses preprocessed control images (edges, pose, depth) as explicit guidance throughout generation. Condition signal maintained across all steps. More precise but requires preprocessing. **Choose img2img when**: 1) **Simple transformations** - Style transfer, color changes, minor modifications. Don't need pixel-perfect structure. 2) **No preprocessing available** - Don't have time/ability to generate control images. 3) **Creative interpretation desired** - Want AI to have some flexibility in composition. 4) **Speed critical** - No preprocessing overhead. 5) **Example**: Converting photos to paintings, changing time of day, material replacement. **Choose ControlNet when**: 1) **Precise structure required** - Need exact pose, exact composition, specific layout. 2) **Complex structural constraints** - Multiple elements must maintain specific relationships. 3) **Consistency across generations** - Same control image = same structure with different styles. 4) **Example**: Character in specific pose, architectural rendering with exact layout, maintaining photo composition while changing everything else. **Hybrid approach**: Use ControlNet for structure + img2img for fine-tuning. Or ControlNet at high strength + low img2img strength for precise control with minor adjustments. **Production decision tree**: Is structure control critical? Yes → ControlNet. Is preprocessing acceptable? No → img2img. Need creative freedom? Yes → img2img. **Real example**: E-commerce: img2img for product variations. Game development: ControlNet for character poses. Architecture: ControlNet for building layouts. Photo enhancement: img2img for style/quality.",
    keyPoints: [
      'img2img: implicit control via input image, simpler',
      'ControlNet: explicit control via preprocessed signals, more precise',
      'img2img for style transfer and creative changes',
      'ControlNet for structural precision and consistency',
    ],
  },
];
