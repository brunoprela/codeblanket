export const llmEvaluationSafetyQuiz = {
  title: 'LLM Evaluation & Safety Discussion',
  id: 'llm-evaluation-safety-quiz',
  sectionId: 'llm-evaluation-safety',
  questions: [
    {
      id: 1,
      question:
        'Evaluating LLM capabilities is challenging due to their open-ended nature. Discuss the limitations of standard NLP metrics (BLEU, ROUGE, perplexity) for evaluating modern LLMs. How have benchmarks evolved from narrow tasks to comprehensive evaluations like MMLU and Big-Bench? What role does human evaluation play, and how do you scale it cost-effectively?',
      expectedAnswer:
        "Should cover: token-level metrics missing semantic quality, perplexity correlating poorly with usability, benchmark saturation and gaming, MMLU testing diverse knowledge domains, Big-Bench's challenging tasks, emergent capabilities not captured by standard metrics, human evaluation as gold standard but expensive and slow, LLM- as - judge using strong models to evaluate, pairwise comparison reducing bias, arena - style evaluation(Chatbot Arena), contamination issues with public benchmarks, and need for continual new evaluation paradigms.",
    },
    {
      id: 2,
      question:
        'AI safety encompasses many concerns: harmful content generation, bias and fairness, privacy and security, and existential risks. For LLM systems in production, discuss the safety layers you would implement: input filtering, output filtering, PII detection, bias mitigation, and monitoring. How do you balance safety with capability and user experience?',
      expectedAnswer:
        'Should discuss: defense in depth with multiple safety layers, pre-generation filtering of harmful requests, post-generation content filtering, PII detection and redaction strategies, bias evaluation across demographics, refusal behavior design (when to say no), overrefusal harming usability, adversarial testing (red teaming), user feedback loops, false positive tolerance tradeoffs, transparency about AI limitations, legal compliance requirements, ongoing monitoring and improvement, and ethical considerations in deployment decisions.',
    },
    {
      id: 3,
      question:
        'Discuss the phenomenon of jailbreaking—users bypassing safety guardrails through adversarial prompts. Why are LLMs vulnerable to jailbreaking? What techniques do attackers use (role-play, encoding, prompt injection)? How can we build more robust defenses? Is perfect defense possible?',
      expectedAnswer:
        'Should cover: safety training vs capability conflict, RLHF being shallow compared to pretraining, adversarial prompts exploiting model behavior, role-play bypassing safety personas, encoded/obfuscated inputs evading filters, prompt injection attacks mixing instructions and data, defenses including instruction hierarchy, input normalization, multiple classifiers, LLM-based detection of jailbreaks, cat-and-mouse dynamic between attackers and defenders, fundamental challenges in aligning powerful models, debate over whether robust alignment is achievable, and research directions in AI safety.',
    },
  ],
};
