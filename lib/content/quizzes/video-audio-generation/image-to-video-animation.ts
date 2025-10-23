export const imageToVideoAnimationQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Explain how the motion_bucket_id parameter in Stable Video Diffusion controls animation intensity, and design a system that automatically determines the optimal motion level for different types of images (portraits, landscapes, products, abstract art). What metrics would you use to evaluate if the chosen motion level is appropriate?',
    sampleAnswer:
      'The motion_bucket_id (0-255) controls how much movement is introduced: low values (0-40) for subtle motion like breathing or gentle sway, medium (80-127) for natural scene animation, high (180-255) for dramatic effects. An automatic system would analyze the input image to determine content type using image classification, then apply heuristics: portraits get 20-60 (subtle, prevent uncanny valley), landscapes get 100-150 (natural environmental motion), products get 40-80 (gentle rotation/presentation), abstract art gets 120-200 (artistic interpretation allows more freedom). Evaluation metrics: temporal consistency (SSIM between adjacent frames >0.85 for portraits, >0.75 for landscapes), motion amount (optical flow magnitude), artifact detection (sudden changes, morphing), and user feedback. The system would generate variants at different motion levels, evaluate each, and select the optimal one based on weighted scoring of these metrics.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Compare the image-to-video approach with text-to-video for creating product demonstration videos. Design a complete pipeline that starts with product photography, animates them effectively, and handles common challenges like maintaining product integrity while adding engaging motion.',
    sampleAnswer:
      "Image-to-video is superior for product demos because: guaranteed product accuracy (critical for e-commerce), cost-effectiveness (reuse existing product photos), consistency across catalog, and faster iteration. Pipeline: 1) Preprocess images (remove background, center product, optimize resolution), 2) Classify product type (fashion, electronics, food, etc.) to determine animation style, 3) Generate with appropriate motion_bucket_id (60-100 for most products), 4) Post-process (stabilize, enhance, add subtle zoom), 5) Quality check (ensure product details remain clear, no distortion), 6) Composite onto branded background or lifestyle scene. Common challenges: preventing product distortion (use lower motion values, add product mask for protection), maintaining text/logos readable (detect and stabilize text regions), keeping proportions correct (geometric consistency checks), avoiding unrealistic physics (products floating). Solution: hybrid approach using both image conditioning and text prompts like 'professional product photography, studio lighting, gentle rotation' to guide motion while constraining it appropriately.",
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      "Design a quality control system for image-to-video generation that automatically detects and handles common failure modes: excessive morphing, artifacts, temporal inconsistencies, and motion that doesn't match the content. How would you implement an automatic retry mechanism with parameter adjustment?",
    sampleAnswer:
      'Quality control system with multiple validation layers: 1) Temporal consistency check using SSIM/LPIPS between adjacent frames (threshold: >0.80), 2) Morphing detection via structural similarity of detected objects across frames, 3) Artifact detection using anomaly detection on frame differences, 4) Motion appropriateness by comparing expected vs actual optical flow for content type, 5) Semantic consistency using CLIP embeddings (cosine similarity >0.90 across all frames). Implementation: generate video, run all checks, if any fail: identify failure type and adjust parameters accordingly. For excessive morphing: reduce motion_bucket_id by 30%, increase temporal weight. For artifacts: change seed, adjust noise level. For inconsistency: reduce motion, enable temporal smoothing. For inappropriate motion: adjust motion_bucket_id based on content classification. Automatic retry logic: max 3 attempts, each with progressively more conservative parameters, final fallback to minimal motion (motion_bucket_id=30). Track which parameter adjustments resolve which failure types to improve system over time. Cache successful configurations per image type to optimize future generations.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
