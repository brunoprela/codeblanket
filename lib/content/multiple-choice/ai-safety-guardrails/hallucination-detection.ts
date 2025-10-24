/**
 * Multiple choice questions for Hallucination Detection section
 */

export const hallucinationdetectionMultipleChoice = [
  {
    id: 'halluc-det-mc-1',
    question:
      'An LLM generates: "According to a 2023 study by Smith et al., 80% of users prefer X." You search and cannot find this study. What is this?',
    options: [
      'A citation error (wrong year or author)',
      'A hallucinated citation (study does not exist)',
      'A paywalled study you cannot access',
      'A study that was retracted',
    ],
    correctAnswer: 1,
    explanation:
      'LLMs frequently hallucinate plausible-sounding citations that do not exist. This is a hallucinated citation. While options A, C, D are possible, option B is most likely given that LLMs are known to fabricate references when they lack real data.',
  },
  {
    id: 'halluc-det-mc-2',
    question:
      'You ask an LLM the same question 5 times (temperature=0.0). You get 5 different answers with different facts. What does this indicate?',
    options: [
      'The LLM is working correctly—diversity is good',
      'High likelihood of hallucination—inconsistent facts suggest uncertainty',
      'Temperature should be increased for consistency',
      'The question is ambiguous',
    ],
    correctAnswer: 1,
    explanation:
      "At temperature=0.0, the LLM should give consistent answers. Different facts across responses indicate the model is uncertain and likely hallucinating. Option A is wrong—factual inconsistency is bad. Option C is backwards (lower temperature = more consistent). Option D might be true but doesn't explain the inconsistency.",
  },
  {
    id: 'halluc-det-mc-3',
    question:
      'Your hallucination detector has 85% recall. What does this mean?',
    options: [
      '85% of responses contain hallucinations',
      '85% of hallucinations are detected',
      '85% of detected hallucinations are real (not false positives)',
      '85% of responses are hallucination-free',
    ],
    correctAnswer: 1,
    explanation:
      'Recall = True Positives / (True Positives + False Negatives) = Proportion of actual hallucinations that are detected. 85% recall means the detector catches 85% of hallucinations (misses 15%). Option C describes precision, not recall.',
  },
  {
    id: 'halluc-det-mc-4',
    question:
      'To reduce hallucinations, you add: "Only provide information you are certain is correct." Does this work?',
    options: [
      'Yes—LLMs will follow instructions to be more careful',
      'No—LLMs cannot reliably assess their own certainty',
      'Yes—but only for GPT-4 and newer models',
      'No—you must use a separate verification system',
    ],
    correctAnswer: 1,
    explanation:
      "LLMs cannot reliably assess their own certainty. They will still hallucinate confidently even with such instructions. Option D is also true (separate verification is needed), but B is the more direct answer to why the instruction alone doesn't work.",
  },
  {
    id: 'halluc-det-mc-5',
    question:
      'Your medical chatbot cannot tolerate false negatives (missed hallucinations). You should optimize for:',
    options: [
      'High precision (few false positives)',
      'High recall (catch all hallucinations)',
      'Fast response time',
      'Low cost',
    ],
    correctAnswer: 1,
    explanation:
      "High recall ensures you catch all hallucinations—critical for safety in medical contexts. This will increase false positives (blocking some correct info), but that's acceptable. False negatives (missed hallucinations) could harm patients. Options C and D are secondary to safety.",
  },
];
