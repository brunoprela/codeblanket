/**
 * Multiple choice questions for PM Career Development
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmCareerDevelopmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the typical career progression path for Product Managers?',
    options: [
      'PM → Senior PM → VP Product (skip intermediate levels)',
      'Associate PM → PM → Senior PM → Lead/Principal PM → Director → VP Product',
      'PM → Engineering Manager → VP Product',
      'There is no standard career path in product management',
    ],
    correctAnswer: 1,
    explanation:
      'The typical PM career ladder: Associate PM (APM) 0-2 years → Product Manager (PM) 2-5 years → Senior PM 5-8 years → Lead/Principal PM 8-12 years → Group PM or Director 10+ years → VP Product/CPO 15+ years. At Senior PM level, the path splits into IC track (Principal/Staff PM focusing on product strategy) vs Management track (Group PM/Director focusing on people leadership). Both paths are valid and lead to senior roles. The specific titles and years vary by company, but this is the general progression framework.',
  },
  {
    id: 'mc2',
    question:
      'According to the content, what are the 8 core PM skills that need continuous development?',
    options: [
      'Coding, Design, Sales, Marketing, Finance, Legal, HR, Operations',
      'Product Strategy, User Research, Data Analysis, Communication, Technical Fluency, Design Thinking, Business Acumen, Leadership',
      'Python, SQL, JavaScript, React, AWS, Docker, Kubernetes, Git',
      'Writing, Speaking, Listening, Reading, Math, Science, Art, Music',
    ],
    correctAnswer: 1,
    explanation:
      'The 8 core PM skills are: (1) Product Strategy (vision, roadmapping, prioritization), (2) User Research (interviews, testing, insights), (3) Data Analysis (SQL, analytics tools, metrics), (4) Communication (writing, presenting, influencing), (5) Technical Fluency (understanding engineering), (6) Design Thinking (UX principles, collaboration), (7) Business Acumen (revenue models, unit economics), (8) Leadership (influence, decision-making, conflict resolution). PMs should develop all 8 skills continuously, though depth in each varies by role and seniority level. The content provides specific resources and practice methods for each skill.',
  },
  {
    id: 'mc3',
    question:
      'What is the key difference between the IC (Individual Contributor) track and Management track for PMs?',
    options: [
      'IC track pays more than Management track',
      'IC track focuses on product strategy/execution; Management track focuses on people development/org building',
      'IC track is for junior PMs; Management track is for senior PMs',
      'There is no difference - all senior PMs manage people',
    ],
    correctAnswer: 1,
    explanation:
      'IC Track (Individual Contributor): Focus on product strategy and execution, growth through larger scope and complex problems, peak at Principal/Staff PM. Management Track: Focus on people development and org building, growth through team size and org impact, peak at Director/VP Product/CPO. Choice should be based on: Do you love solving product problems (IC) or developing people (Manager)? Both tracks are equally valid and prestigious. You can switch tracks later - not a permanent decision. Some companies have Principal PM (IC) at same level/comp as Director (Manager).',
  },
  {
    id: 'mc4',
    question:
      'According to the 30-60-90 day plan, what should a new PM focus on in their first 30 days?',
    options: [
      'Shipping major features to prove themselves',
      'Listening and learning - understanding context, building relationships, absorbing information',
      'Proposing major changes to improve the product',
      'Managing the existing product team',
    ],
    correctAnswer: 1,
    explanation:
      'First 30 days: Listen and learn. Goal is to understand context, build relationships, and absorb information. Activities: Meet team (1-on-1s with everyone), use the product extensively, read documentation, understand users and market, study strategy and roadmap, ask lots of questions. Deliverable: "What I Learned" presentation. The principle is "Go slow to go fast" - build foundation before making changes. Common mistake: Proposing changes on Day 5 before understanding context. Days 31-60: Start contributing with small wins. Days 61-90: Take ownership and drive initiatives.',
  },
  {
    id: 'mc5',
    question:
      'What are effective ways to build your PM personal brand according to the content?',
    options: [
      'Only work at prestigious companies like Google/Meta',
      'Write publicly, speak at events, share work, be active online, build in public',
      'Get an MBA from a top business school',
      'Focus only on your day job and avoid any public presence',
    ],
    correctAnswer: 1,
    explanation:
      'Building PM brand: (1) Write publicly (Medium blog posts, LinkedIn posts, newsletters), (2) Speak at events (local PM meetups, company tech talks, conferences), (3) Share your work (open-source projects, case studies, product teardowns), (4) Be active online (Twitter product insights, LinkedIn industry comments, answer questions in communities), (5) Build in public (side projects, product experiments, share learning journey). Why brand matters: Career opportunities (recruiters find you), network effects (people want to work with you), thought leadership (industry influence), credibility (easier to land next role). Example content: lessons from launches, product frameworks, user research insights, PM career advice.',
  },
];
