export const promptEngineeringMC = {
  title: 'Prompt Engineering Quiz',
  id: 'prompt-engineering-mc',
  sectionId: 'prompt-engineering',
  questions: [
    {
      id: 1,
      question: 'What is chain-of-thought (CoT) prompting?',
      options: [
        'Providing multiple sequential prompts',
        'Showing step-by-step reasoning in examples to elicit similar reasoning',
        'Chaining multiple model calls together',
        'Using thought tokens during training',
      ],
      correctAnswer: 1,
      explanation:
        'Chain-of-thought prompting includes explicit reasoning steps in few-shot examples (or asks the model to "think step by step" in zero-shot). This dramatically improves performance on reasoning tasks by showing the model how to break down problems.',
    },
    {
      id: 2,
      question:
        'In few-shot prompting, what is the typical optimal number of examples?',
      options: [
        '1 example is always sufficient',
        '3-10 examples typically work best',
        '50+ examples are always better',
        'More examples always improve performance linearly',
      ],
      correctAnswer: 1,
      explanation:
        'Typically 3-10 examples provide good performance, with diminishing returns beyond that. Too many examples waste context window and can hurt performance. The quality and representativeness of examples matters more than quantity.',
    },
    {
      id: 3,
      question: 'What does "self-consistency" in prompting refer to?',
      options: [
        'Ensuring prompts are grammatically correct',
        'Generating multiple reasoning paths and taking the majority answer',
        'Using the same prompt format across all queries',
        "Making sure the model doesn't contradict itself",
      ],
      correctAnswer: 1,
      explanation:
        'Self-consistency generates multiple chain-of-thought reasoning paths (with different random seeds) and takes the most common final answer. This can significantly improve accuracy by marginalizing over different reasoning approaches.',
    },
    {
      id: 4,
      question:
        'What is the primary limitation of zero-shot prompting compared to few-shot prompting?',
      options: [
        'It cannot work at all',
        'It requires larger models and may miss task nuances that examples would clarify',
        'It is more expensive',
        'It is slower',
      ],
      correctAnswer: 1,
      explanation:
        "Zero-shot prompting relies entirely on the model's pretraining and instruction- following capabilities.While it works surprisingly well with large, instruction- tuned models, few-shot examples can clarify task format, style, and edge cases.",
    },
    {
      id: 5,
      question:
        'Why does role prompting (e.g., "You are an expert Python programmer") often improve outputs?',
      options: [
        'It makes the model more polite',
        'It activates relevant patterns from training data with similar contexts',
        "It increases the model's actual capabilities",
        'It reduces hallucinations',
      ],
      correctAnswer: 1,
      explanation:
        "Role prompting likely activates patterns learned from similar contexts in training data (e.g., code tutorials, expert forums). It sets expectations for tone, style, and depth. However, it doesn't give the model fundamentally new capabilities.",
    },
  ],
};
