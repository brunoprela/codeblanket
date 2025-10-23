export const videoEditingWithAIQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Explain the challenges of maintaining temporal consistency when applying AI-powered style transfer or object removal to videos. How would you implement a production system that processes a 2-minute video while ensuring no flickering or artifacts appear? What are the computational trade-offs?',
    sampleAnswer:
      "Temporal consistency in video editing is challenging because processing frames independently causes flickering - style/edits vary slightly frame-to-frame. Solutions: 1) Temporal coherence loss that penalizes differences between adjacent frames during processing, 2) Optical flow guidance to warp previous frame's result and blend with current, 3) Recurrent processing where model maintains hidden state across frames, 4) Multi-frame context windows (process 5-10 frames together), 5) Post-processing temporal smoothing. For 2-minute video (3,600 frames at 30fps): process in overlapping batches of 30 frames (1 second), use GPU tiling to fit in memory, apply temporal filtering across batch boundaries, total time ~10-20 minutes on A100. Trade-offs: temporal consistency methods add 2-3x compute cost, larger context windows need more VRAM, real-time impossible for high quality. Production approach: process offline, show progress, offer preview at lower quality, cache results aggressively.",
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Design a video upscaling service that can handle 480p to 4K conversion while maintaining temporal consistency and managing GPU resources efficiently. How would you handle the massive computational requirements and provide progress feedback to users?',
    sampleAnswer:
      'Video upscaling from 480p to 4K is extremely compute-intensive (8.3x pixels, 120 frames for 4-second video). Architecture: 1) Split video into scenes using scene detection, 2) Process each scene independently (enables parallel processing), 3) Use tile-based processing to fit in GPU memory (divide each frame into 512x512 tiles with overlap), 4) Apply Real-ESRGAN or similar per-tile with temporal consistency constraints, 5) Blend tiles carefully to avoid seams, 6) Temporal smoothing pass across frames, 7) Reassemble scenes. GPU allocation: use 8x A100 GPUs in parallel, each handling different scenes/frames, estimate 1-2 minutes per second of input video. Progress feedback: track scene processing (Scene 1/5), frames within scene (Frame 45/120), overall percentage, estimated time remaining updated after each scene. Optimization strategies: preprocess at 720p first (faster preview), only upscale if user approves preview, cache intermediate results, use priority queue (pro users first), batch similar resolution videos together.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      'Compare optical flow-based video editing methods with deep learning approaches for tasks like frame interpolation and stabilization. When would you use each, and how might you combine them in a production pipeline for optimal results?',
    sampleAnswer:
      'Optical flow methods: calculate per-pixel motion vectors between frames, fast, predictable, works well for smooth motion, but struggles with occlusions, large motions, and complex scenes. Deep learning approaches: learn motion and appearance from data, handle occlusions better, produce more natural results, but slower and less predictable. Frame interpolation: optical flow good for high frame-rate conversion (30→60fps) where motion is small, DL (RIFE, FILM) better for slow-motion (30→240fps) with complex motion. Stabilization: optical flow sufficient for small shake removal, DL methods better for extreme shake or rolling shutter. Production pipeline combining both: 1) Use optical flow for initial motion estimation (fast, gets 80% of the way), 2) Apply DL refinement in areas where flow confidence is low (occlusions, motion boundaries), 3) Blend results using confidence maps, 4) Fall back to pure DL if flow quality is poor. This hybrid gives 5x speedup for easy cases while maintaining quality on hard cases.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
