/**
 * Multiple choice questions for PM Interview Preparation
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmInterviewPrepMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the CIRCLES framework used for in PM interviews?',
    options: [
      'Strategy questions (how would you grow X?)',
      'Product design questions (design a product for Y)',
      'Behavioral questions (tell me about a time when)',
      'Technical questions (how does X work?)',
    ],
    correctAnswer: 1,
    explanation:
      'CIRCLES framework is specifically for product design questions like "Design a product for X" or "Improve Y product." The framework: (C) Comprehend situation, (I) Identify customer, (R) Report customer needs, (C) Cut through prioritization, (L) List solutions, (E) Evaluate trade-offs, (S) Summarize recommendation. For strategy questions, use growth levers framework. For behavioral, use STAR framework. For technical, use architecture diagram approach. Each interview type has its own framework, and using the right one demonstrates preparation and structured thinking.',
  },
  {
    id: 'mc2',
    question: 'What is the STAR framework and when should you use it?',
    options: [
      'Strategy, Testing, Analysis, Results - for strategy questions',
      'Situation, Task, Action, Result - for behavioral questions',
      'Start, Think, Act, Review - for product design questions',
      'Survey, Target, Acquire, Retain - for growth questions',
    ],
    correctAnswer: 1,
    explanation:
      'STAR framework (Situation, Task, Action, Result) is for behavioral interview questions like "Tell me about a time you failed" or "Describe a conflict with engineering." Structure: (S) Situation: Set context in 30 seconds, (T) Task: What you were trying to achieve in 20 seconds, (A) Action: What you did specifically in 2 minutes, (R) Result: What happened, impact, learnings in 1 minute. Total: 3-4 minutes. Prepare 8-10 STAR stories covering: success, failure, conflict, leadership, data-driven decision, user research, technical challenge, prioritization. This framework demonstrates: self-awareness, accountability, learning ability, and communication skills.',
  },
  {
    id: 'mc3',
    question:
      'In a product design interview question, what should you do FIRST before proposing solutions?',
    options: [
      'Immediately propose 3-5 solutions to show creativity',
      'Ask clarifying questions about scope, user, and goals',
      'Discuss metrics and how to measure success',
      'Evaluate trade-offs of different approaches',
    ],
    correctAnswer: 1,
    explanation:
      'Always ask clarifying questions FIRST (2-3 minutes) before jumping to solutions. Ask about: (1) Platform (mobile app, web app, physical product?), (2) Target user (who specifically?), (3) Goal (acquire users, increase engagement, monetize?), (4) Constraints (timeline, resources, scope?). This shows thoughtfulness and prevents solving the wrong problem. Common mistake: Immediately saying "I\'d build X" without understanding context. Interviewers are testing your process, not just your final answer. Clarifying questions demonstrate: strategic thinking, customer focus, and structured approach. Only after clarifying should you identify users, define needs, and propose solutions.',
  },
  {
    id: 'mc4',
    question:
      'What is the recommended time allocation for answering a product design question in a PM interview?',
    options: [
      '5 minutes total (be very concise)',
      '30-45 minutes (be extremely thorough)',
      '15-20 minutes total using structured framework',
      'No time limit - take as long as you need',
    ],
    correctAnswer: 2,
    explanation:
      "15-20 minutes total is ideal for product design questions. Recommended breakdown: (1) Clarify 2-3 min, (2) Identify user 2 min, (3) Report needs 3 min, (4) List solutions 5 min, (5) Evaluate trade-offs 2 min, (6) Summarize with metrics 2 min. Going under 10 minutes seems rushed and lacking depth. Going over 25 minutes tests interviewer's patience and shows poor time management. Use a timer when practicing. Interviewers evaluate both your answer quality AND your ability to be concise. In real PM work, you need to present ideas efficiently to stakeholders. Practice timing until it becomes natural.",
  },
  {
    id: 'mc5',
    question:
      'According to the content, how many practice questions should you complete during a comprehensive 2-week interview prep?',
    options: [
      '5-10 questions (quality over quantity)',
      '15-20 questions (moderate practice)',
      '30+ questions across different types',
      '100+ questions (extensive practice)',
    ],
    correctAnswer: 2,
    explanation:
      'The 2-week prep plan recommends 30+ practice questions across different types: 10-15 product design questions, 8-10 strategy questions, 5-7 metrics questions, plus practicing behavioral STAR stories. Additionally: 2-3 full mock interviews with peers, company research for each target company, and preparing 8 STAR stories. You can\'t "wing" PM interviews - they require deliberate practice. Most candidates under-prepare, doing maybe 5-10 questions total. Doing 30+ puts you in top quartile of preparedness. This volume ensures: familiarity with question types, smooth use of frameworks, confidence during actual interviews, and ability to think on your feet.',
  },
];
