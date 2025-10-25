export const llmAlignmentRLHFMC = {
  title: 'LLM Alignment & RLHF Quiz',
  id: 'llm-alignment-rlhf-mc',
  sectionId: 'llm-alignment-rlhf',
  questions: [
    {
      id: 1,
      question:
        'What are the three main stages of RLHF (Reinforcement Learning from Human Feedback)?',
      options: [
        'Pretraining, fine-tuning, evaluation',
        'Supervised fine-tuning, reward modeling, RL optimization',
        'Data collection, model training, deployment',
        'Instruction tuning, preference learning, safety filtering',
      ],
      correctAnswer: 1,
      explanation:
        'RLHF consists of: (1) Supervised Fine-Tuning (SFT) on demonstration data, (2) training a Reward Model from human preference comparisons, and (3) RL optimization (typically PPO) to maximize the reward while staying close to the original model.',
    },
    {
      id: 2,
      question: 'What is "reward hacking" in the context of RLHF?',
      options: [
        'Hackers stealing reward models',
        'The model exploiting flaws in the reward model to get high reward without truly satisfying human preferences',
        'Increasing the reward signal during training',
        'Using multiple reward models simultaneously',
      ],
      correctAnswer: 1,
      explanation:
        'Reward hacking (or reward gaming) occurs when the model exploits weaknesses in the proxy reward model to achieve high scores without actually meeting human preferences. For example, generating overly verbose or sycophantic responses if the reward model favors them.',
    },
    {
      id: 3,
      question:
        'Why is a KL divergence penalty used during RL optimization in RLHF?',
      options: [
        'To improve training stability',
        'To prevent the model from diverging too far from the original pretrained/SFT model',
        'To increase generation diversity',
        'To reduce computational costs',
      ],
      correctAnswer: 1,
      explanation:
        'The KL penalty prevents the model from moving too far from its initialization (the SFT model), which could cause mode collapse (only generating certain types of responses) or catastrophic forgetting of language modeling capabilities.',
    },
    {
      id: 4,
      question: "What is Constitutional AI (Anthropic\'s approach)?",
      options: [
        'Using RLHF with constitutional law experts',
        'Training models to critique and revise their outputs based on a set of principles',
        'Hardcoding constitutional rules into the model',
        'Only using data from legal documents',
      ],
      correctAnswer: 1,
      explanation:
        'Constitutional AI uses AI-generated feedback based on constitutional principles. The model critiques its own outputs for alignment with specified principles, then revises them. This is more scalable than human feedback for every query.',
    },
    {
      id: 5,
      question: 'What alignment problem does "sycophancy" refer to?',
      options: [
        'Models refusing to answer questions',
        'Models agreeing with user statements even when they are incorrect',
        'Models being too verbose',
        'Models generating harmful content',
      ],
      correctAnswer: 1,
      explanation:
        'Sycophancy is when RLHF-trained models learn to agree with users or tell them what they want to hear (because human labelers may prefer agreeable responses), even when the user is factually wrong. This can reduce model usefulness.',
    },
  ],
};
