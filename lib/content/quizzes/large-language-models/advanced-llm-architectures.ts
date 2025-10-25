export const advancedLLMArchitecturesQuiz = {
  title: 'Advanced LLM Architectures Discussion',
  id: 'advanced-llm-architectures-quiz',
  sectionId: 'advanced-llm-architectures',
  questions: [
    {
      id: 1,
      question:
        "Mixture of Experts (MoE) architectures enable massive parameter counts with manageable inference costs by activating only a subset of experts per token. Explain how MoE routing works and discuss the tradeoffs: parameter efficiency, training instability, load balancing challenges, and deployment complexity. Why hasn't MoE completely replaced dense models?",
      expectedAnswer:
        'Should cover: router learning to select K of N experts, sparse activation reducing compute, parameter count vs active parameters distinction, training challenges (routing collapse, expert specialization), load balancing auxiliary losses, memory requirements (all experts loaded), deployment complexity (distributed inference), communication overhead in multi-GPU settings, dense models being simpler and more stable, when MoE wins (massive scale, diverse tasks) vs dense (simpler deployment), and recent innovations like Mixtral showing MoE viability.',
    },
    {
      id: 2,
      question:
        'Long context models use various techniques to extend beyond the O(n²) attention limitation: sparse attention patterns, sliding windows, and alternative position encodings. Compare Longformer, BigBird, and ALiBi approaches. What are the tradeoffs in quality, speed, and implementation complexity? Can these techniques enable infinite context?',
      expectedAnswer:
        "Should discuss: full attention quadratic scaling being prohibitive, Longformer\'s local + global attention patterns, BigBird's random + window + global attention, ALiBi using attention biases instead of positional embeddings, quality degradation with approximations, speed improvements from sparsity, implementation complexity and hardware optimization, extrapolation to longer sequences than trained on, fundamental limits (attention dilution, memory constraints), and future directions in linear attention research.",
    },
    {
      id: 3,
      question:
        'Multimodal models like GPT-4V and Gemini integrate vision and language. Discuss the architectural challenges: aligning different modalities, training data requirements, and inference efficiency. How do vision transformers enable image understanding? What new capabilities and risks emerge from multimodal AI?',
      expectedAnswer:
        'Should cover: separate encoders for images and text, cross-modal attention mechanisms, contrastive pretraining (CLIP) for alignment, vision transformer (ViT) architecture treating images as token sequences, massive paired data requirements (image-caption pairs), computational cost of processing images, new capabilities (visual QA, diagram understanding, OCR-free document processing), risks (deep fakes, privacy via vision, combining modalities for harmful content), and future directions toward unified multimodal models.',
    },
  ],
};
