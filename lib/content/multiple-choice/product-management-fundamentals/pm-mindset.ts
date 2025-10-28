/**
 * Multiple choice questions for The Product Manager Mindset
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmMindsetMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the key difference between "data-driven" and "data-informed" decision making?',
    options: [
      'Data-driven uses only quantitative data; data-informed uses only qualitative data',
      'Data-driven lets data make the decision; data-informed uses data to inform judgment and considers other factors',
      'Data-driven is for large companies; data-informed is for startups',
      'There is no difference; they mean the same thing',
    ],
    correctAnswer: 1,
    explanation:
      "Data-driven means letting data dictate decisions without human judgment, which can lead to local maxima and missing innovation opportunities. Data-informed means using data as a critical input but combining it with user empathy, strategic thinking, and intuition. Great PMs are data-informed: they use data to validate hypotheses, but don't outsource decision-making to metrics alone. This is especially important for 0-to-1 products where data is limited.",
  },
  {
    id: 'mc2',
    question:
      'According to the content, what did Facebook discover was the key metric that predicted user retention?',
    options: [
      'Posting 10 updates in the first week',
      'Adding 7 friends in 10 days',
      'Spending 30 minutes on the platform',
      'Uploading 5 photos',
    ],
    correctAnswer: 1,
    explanation:
      'Facebook\'s growth team discovered that users who added "7 friends in 10 days" were highly likely to become retained users. This data-driven insight led them to optimize the entire onboarding experience around achieving this metric, which contributed significantly to scaling from 100M to 1B users. This is a classic example of finding a North Star metric that predicts long-term success and focusing product efforts on optimizing for it.',
  },
  {
    id: 'mc3',
    question:
      "What was Dropbox's MVP strategy before building the full product?",
    options: [
      'Built a basic file storage app with limited features',
      'Created a 3-minute demo video and posted it on Hacker News to validate demand',
      'Offered $10 to early users to test a beta version',
      'Conducted 100 user interviews before writing any code',
    ],
    correctAnswer: 1,
    explanation:
      'Drew Houston created a 3-minute demo video showing how Dropbox would work and posted it on Hacker News. The waitlist exploded from 5,000 to 75,000 overnight, validating massive demand before building the full product. This demonstrates "bias for action" mindset—testing hypotheses quickly and cheaply before major investment. It\'s a brilliant example of validating demand without building the product first.',
  },
  {
    id: 'mc4',
    question: 'What was Instagram originally called, and why did it pivot?',
    options: [
      'Burbn; users loved the photo filter feature more than check-ins',
      'PhotoShare; users wanted video features instead',
      'SocialSnap; users wanted more privacy controls',
      'PicPost; users wanted integration with Facebook',
    ],
    correctAnswer: 0,
    explanation:
      'Instagram was originally called Burbn, a location check-in app like Foursquare. The founders noticed users didn\'t care about check-ins but loved the photo filters feature. They stripped everything except photos and filters, relaunched as Instagram, and achieved massive success. This demonstrates "comfort with being wrong"—the founders were willing to abandon their original idea when user behavior showed a better opportunity. Many failed startups die because founders can\'t pivot away from their original vision.',
  },
  {
    id: 'mc5',
    question:
      'According to the "11-star experience" framework from Airbnb, what does thinking about extreme experiences help PMs do?',
    options: [
      'Set unrealistic expectations that can never be achieved',
      'Push thinking beyond obvious improvements and imagine breakthrough experiences',
      'Focus only on luxury features for high-end customers',
      'Ignore practical constraints and focus on fantasy scenarios',
    ],
    correctAnswer: 1,
    explanation:
      'Brian Chesky\'s "11-star experience" exercise (where a 5-star is great and 11-star is Elon Musk picking you up in a rocket) helps PMs push beyond incremental thinking. The point isn\'t to literally build an 11-star experience, but to break mental barriers about what\'s possible. Then work backwards to what\'s feasible but still remarkable. This exercise demonstrates "customer obsession" by imagining truly transformative experiences, which often reveals opportunities for 6-7 star experiences that are achievable and differentiated.',
  },
];
