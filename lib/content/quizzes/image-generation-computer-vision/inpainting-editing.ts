/**
 * Quiz questions for Inpainting & Editing section
 */

export const inpaintingeditingQuiz = [
  {
    id: 'igcv-inpaint-q-1',
    question:
      'Describe the key considerations for creating high-quality inpainting masks. Why does mask quality directly impact result quality?',
    hint: 'Think about feathering, precision, and blending.',
    sampleAnswer:
      'Mask quality is critical because it defines the boundary between preserved and generated content: **Feathering/blur (5-15 pixels)** - Soft edges create seamless blends. Hard edges create visible seams. The feather zone allows gradual transition where inpainted content blends with original. **Precision** - Mask should slightly expand beyond target object (5-10px) to ensure complete coverage. Under-masking leaves artifacts, over-masking regenerates too much. **Consistency** - Mask edges should follow natural boundaries (object edges, color transitions) rather than arbitrary lines. **Multiple passes** - For complex removals, iterate: rough mask first, examine result, refine mask for problem areas. **Validation** - View mask overlay on image before generation to catch issues. **Tools matter** - Brush-based masking for organic shapes, polygon/selection for geometric objects. **Common mistakes**: 1) No feathering = visible seams. 2) Too tight around object = artifacts remain. 3) Irregular edges = unnatural results. **Pro tip**: For critical work, create mask at 2x resolution then downsample for smoother edges.',
    keyPoints: [
      'Feathering edges (5-15px) critical for seamless blending',
      'Slightly expand mask beyond object boundaries',
      'Follow natural boundaries, not arbitrary lines',
      'Iterate and refine for complex cases',
    ],
  },
  {
    id: 'igcv-inpaint-q-2',
    question:
      'Design an inpainting workflow for e-commerce: automatically removing backgrounds from product photos and replacing with solid colors or scenes. What are the challenges?',
    hint: 'Consider automation, consistency, edge quality, and scale.',
    sampleAnswer:
      'Automated e-commerce inpainting workflow: **Pipeline**: 1) Object detection/segmentation (Segment Anything Model) to identify product. 2) Auto-generate mask around product with feathering. 3) Inpaint background with desired color/scene. 4) Quality validation. 5) Post-process edges if needed. **Key challenges**: **Challenge 1: Edge quality** - Product edges must be clean and natural-looking. Solution: Use high-quality segmentation model. Add dilate/erode operations to refine mask edges. Fine-tune mask feathering per product category. **Challenge 2: Consistency** - All products need similar background treatment. Solution: Template backgrounds with exact prompts: "solid #FFFFFF background, professional product photography, clean, no shadows". Lock seed for consistent style. **Challenge 3: Prompt matching** - Inpainted background must match product style. Solution: Analyze product (casual vs formal, modern vs vintage) and adapt background prompt. "Lifestyle kitchen scene, modern style" for modern products. **Challenge 4: Scale** - Processing thousands of products efficiently. Solution: Batch processing with GPU workers. Cache common background types. Parallel preprocessing. **Challenge 5: Quality control** - Can\'t manually review all outputs. Solution: Automated checks: edge smoothness metrics, color consistency, CLIP score for "professional product photo", flag outliers for human review. **Advanced: Multi-stage** - Stage 1: Remove background (inpaint with solid color). Stage 2: Add lifestyle context around clean product.',
    keyPoints: [
      'Auto-segment with SAM, refine masks per category',
      'Template backgrounds for consistency',
      'Automated quality checks at scale',
      'Multi-stage for complex backgrounds',
    ],
  },
  {
    id: 'igcv-inpaint-q-3',
    question:
      'Compare inpainting with image editing tools like Photoshop for object removal. When does AI inpainting excel and when should you use traditional tools?',
    hint: 'Think about complexity, control, learning curve, and cost.',
    sampleAnswer:
      "Inpainting vs Traditional Tools: **AI Inpainting strengths**: 1) **Complex textures** - AI generates realistic continuation of complex backgrounds (foliage, crowds, patterns) that would take hours manually. 2) **Speed** - Removes object in 30 seconds vs 30 minutes of cloning/healing. 3) **Large areas** - Handles large removals better than clone stamp. 4) **Learning curve** - Less skill needed than Photoshop mastery. 5) **Example excel**: Removing person from beach scene with sand, water, people in background - AI generates plausible continuation. **Traditional tools strengths**: 1) **Precision control** - Exact pixel-level control for critical work. 2) **Predictability** - Know exactly what you'll get. AI can surprise (good or bad). 3) **Small details** - Removing tiny blemish: clone stamp faster than AI setup. 4) **No artifacts** - Manual work won't introduce AI artifacts. 5) **Example excel**: Removing dust spots, adjusting product colors, precise edge work. **Hybrid approach (best)**: AI inpainting for initial removal → Photoshop for refinement. Use AI for 80% of work, manual tools for final 20% polish. **When to use what**: Simple/small removals (dust, blemishes): traditional tools. Complex/large removals (people, objects in complex backgrounds): AI inpainting. Critical/client work: AI + manual refinement. **Cost consideration**: AI inpainting costs $0.01-0.04 per image. Photoshop requires subscription + time investment. For high volume, AI is vastly cheaper.",
    keyPoints: [
      'AI excels at complex textures and large removals',
      'Traditional tools better for precision and predictability',
      'Hybrid approach combines AI speed with manual precision',
      'Choose based on complexity, volume, and quality requirements',
    ],
  },
];
