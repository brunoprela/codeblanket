/**
 * Quiz questions for ControlNet & Conditioning section
 */

export const controlnetconditioningQuiz = [
  {
    id: 'igcv-cn-q-1',
    question:
      'Explain why ControlNet provides better structural control than img2img. What is fundamentally different about how the control signal is applied?',
    hint: 'Think about when and how the control information influences generation.',
    sampleAnswer:
      'ControlNet vs img2img fundamental difference: **img2img control timing**: Control applied only at START - input image is encoded to latent, noise added, then denoising begins. The original image influence weakens with each step as noise is removed. At high strength (lots of noise), much of the original structure can be lost. Control is indirect and diminishes. **ControlNet control timing**: Control applied at EVERY STEP throughout entire generation process. At each denoising step, the UNet receives both the noisy latent AND the control signal (edges, pose, depth). The control continuously guides generation at all timesteps. Control is direct and persistent. **Architectural difference**: ControlNet adds trainable copies of UNet encoder blocks that process control images. These inject conditioning at multiple layers throughout the network. This is fundamentally different from just providing initial latent state. **Practical implications**: With img2img: "Generate character in this pose" might drift from pose as generation progresses. With ControlNet+pose: Character locked to exact pose skeleton throughout. **Analogy**: img2img is like showing artist a reference once before they paint. ControlNet is like projecting the reference onto canvas throughout entire painting process - artist constantly sees and follows it. **Best practices**: Use img2img when you want inspired-by relationship. Use ControlNet when you need enforced-by relationship. **Combination**: Can use both - ControlNet for structure (pose/composition), img2img starting image for style/colors. This provides multi-level control.',
    keyPoints: [
      'img2img: control only at start, weakens during generation',
      'ControlNet: control at every step throughout process',
      'ControlNet adds conditioning at multiple network layers',
      'Result: ControlNet provides pixel-perfect structural control',
    ],
  },
  {
    id: 'igcv-cn-q-2',
    question:
      'Design a production system using multiple ControlNets simultaneously (e.g., pose + depth + canny). How would you manage the computational cost and balance the control strengths?',
    hint: 'Consider VRAM, speed, and control conflicts.',
    sampleAnswer:
      "Multi-ControlNet production system design: **Computational challenges**: Each ControlNet adds ~1-2GB VRAM and ~20% generation time. Three ControlNets: +3-6GB VRAM, ~60% slower. Critical for resource planning. **Architecture approach**: 1) **Preprocessing pipeline**: Parallel preprocess control images (pose detection, depth estimation, edge detection) on CPU while GPU generates. Amortizes preprocessing cost. 2) **Smart caching**: Cache preprocessed controls for similar images. Pose skeleton doesn't change for same person. 3) **Selective application**: Not every generation needs all controls. Decision logic: portrait → pose+depth, architecture → edges+depth, art → edges only. **Balancing control strengths**: Competing controls can conflict. Guidelines: 1) **Primary control: 1.0-1.2 strength** - The most important structural element. For character images: pose is primary (1.0). 2) **Secondary controls: 0.5-0.8** - Supporting structure. Depth adds realism (0.6). 3) **Tertiary controls: 0.3-0.5** - Subtle guidance. Edges for fine details (0.4). **Conflict resolution**: Pose says arm here, depth says something in front. Solution: Weight pose higher for body structure, depth for environmental context. Test with different weight combinations, log successful patterns. **Production optimization strategies**: 1) **Model loading**: Load all ControlNets once at worker startup, not per generation. 2) **Batch processing**: Process multiple images with same control types together. 3) **Tiered service**: Basic tier uses 1 ControlNet (fast), premium uses multiple (slower but better). 4) **Dynamic selection**: Let AI decide which controls needed based on prompt analysis. **Monitoring**: Track VRAM usage, generation time by control combinations. Alert if approaching limits. **Example production config**: Character generation: pose (1.0) + depth (0.6) + edges (0.3). Architectural visualization: edges (1.2) + depth (0.8). Product photos: depth (1.0) + edges (0.5).",
    keyPoints: [
      'Each ControlNet adds ~1-2GB VRAM and ~20% time',
      'Weight primary control highest, secondary/tertiary lower',
      'Cache preprocessed controls for reuse',
      'Use selective application based on use case',
    ],
  },
  {
    id: 'igcv-cn-q-3',
    question:
      'You need to generate consistent character poses across hundreds of images. Design a system using ControlNet that ensures consistency while allowing variation in everything else. What are the gotchas?',
    hint: 'Consider pose extraction, storage, and variation control.',
    sampleAnswer:
      'Consistent character pose system: **Architecture**: 1) **Pose library**: Store canonical pose skeletons (standing, sitting, running, etc.) as OpenPose format. 2) **Pose selector**: User/system chooses pose from library. 3) **ControlNet generator**: Apply pose control with high strength (1.2-1.5) for consistency. 4) **Variation in other aspects**: Different characters, clothing, backgrounds via prompt. **Implementation steps**: 1) **Build pose library**: Extract poses from reference images OR generate synthetically OR use motion capture data. Store as JSON (OpenPose keypoint format). 2) **Pose preprocessing**: Convert stored poses to ControlNet input images (visual skeleton). Cache these - same pose skeleton reused across generations. 3) **Generation with variation**: \`controlnet.generate (image=pose_skeleton, prompt="[varied character description]", conditioning_scale=1.3)\`. High scale locks pose, prompt varies appearance. **Key gotchas and solutions**: **Gotcha 1: Pose drift** - Even with ControlNet, very complex poses may drift slightly. Solution: Use strength 1.2-1.5 (higher than normal). Validate output pose with detection. **Gotcha 2: Scale/proportion differences** - Different character sizes don\'t fit same pose. Solution: Store poses at multiple scales OR detect character size from prompt, scale pose accordingly. **Gotcha 3: Clothing/props interfering** - Baggy clothing obscures pose. Solution: Negative prompt: "obscured pose, unclear body position". Or generate nude mannequin first, clothe in post-processing. **Gotcha 4: Pose believability** - Some poses unnatural for certain actions. Solution: Tag poses with appropriate contexts. "Running" pose for action scenes, not formal portraits. **Gotcha 5: Batch consistency** - Need exact same pose skeleton used. Solution: Hash pose skeleton, verify in post-generation that same hash used. **Quality validation**: 1) Extract pose from generated image with OpenPose. 2) Compare to input skeleton (compute keypoint distance). 3) Alert if average distance > threshold. 4) Allow some tolerance for natural variation. **Production metrics**: Track pose adherence rate, generation success rate by pose type, user satisfaction. Iterate on problematic poses.',
    keyPoints: [
      'Store canonical pose skeletons in library',
      'Use high conditioning scale (1.2-1.5) for consistency',
      'Validate output poses match input skeletons',
      'Handle scale, clothing, and believability gotchas',
    ],
  },
];
