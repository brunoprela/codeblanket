/**
 * Competitive Programming - C++ & Problem Solving Module
 * Aggregates sections, discussions, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { buildingAlgorithmicIntuitionSection } from '../sections/cp-module-1-section-1';
import { whyCppForCpSection } from '../sections/cp-module-1-section-2';
import { environmentSetupCompilationSection } from '../sections/cp-module-1-section-3';
import { modernCpToolEcosystemSection } from '../sections/cp-module-1-section-4';
import { fastInputOutputTechniquesSection } from '../sections/cp-module-1-section-5';
import { cppBasicsReviewSection } from '../sections/cp-module-1-section-6';
import { cppModernFeaturesSection } from '../sections/cp-module-1-section-7';
import { macrosPreprocessorTricksSection } from '../sections/cp-module-1-section-8';
import { bitsBytesOperationsSection } from '../sections/cp-module-1-section-9';
import { memoryManagementCpSection } from '../sections/cp-module-1-section-10';
import { templateMetaprogrammingBasicsSection } from '../sections/cp-module-1-section-11';
import { commonCompilationErrorsSection } from '../sections/cp-module-1-section-12';
import { debuggingCompetitiveEnvironmentSection } from '../sections/cp-module-1-section-13';
import { readingOthersCppCodeSection } from '../sections/cp-module-1-section-14';
import { contestDayCppTipsSection } from '../sections/cp-module-1-section-15';
import { buildingRobustCpTemplateSection } from '../sections/cp-module-1-section-16';

// Import discussions
import cpM1S1Discussion from '../discussions/competitive-programming/cp-m1-s1-discussion';
import cpM1S2Discussion from '../discussions/competitive-programming/cp-m1-s2-discussion';
import cpM1S3Discussion from '../discussions/competitive-programming/cp-m1-s3-discussion';
import cpM1S4Discussion from '../discussions/competitive-programming/cp-m1-s4-discussion';
import cpM1S5Discussion from '../discussions/competitive-programming/cp-m1-s5-discussion';
import cpM1S6Discussion from '../discussions/competitive-programming/cp-m1-s6-discussion';
import cpM1S7Discussion from '../discussions/competitive-programming/cp-m1-s7-discussion';
import cpM1S8Discussion from '../discussions/competitive-programming/cp-m1-s8-discussion';
import cpM1S9Discussion from '../discussions/competitive-programming/cp-m1-s9-discussion';
import cpM1S10Discussion from '../discussions/competitive-programming/cp-m1-s10-discussion';
import cpM1S11Discussion from '../discussions/competitive-programming/cp-m1-s11-discussion';
import cpM1S12Discussion from '../discussions/competitive-programming/cp-m1-s12-discussion';
import cpM1S13Discussion from '../discussions/competitive-programming/cp-m1-s13-discussion';
import cpM1S14Discussion from '../discussions/competitive-programming/cp-m1-s14-discussion';
import cpM1S15Discussion from '../discussions/competitive-programming/cp-m1-s15-discussion';
import cpM1S16Discussion from '../discussions/competitive-programming/cp-m1-s16-discussion';

// Import multiple choice questions
import cpM1S1Quiz from '../multiple-choice/competitive-programming/cp-m1-s1-quiz';
import cpM1S2Quiz from '../multiple-choice/competitive-programming/cp-m1-s2-quiz';
import cpM1S3Quiz from '../multiple-choice/competitive-programming/cp-m1-s3-quiz';
import cpM1S4Quiz from '../multiple-choice/competitive-programming/cp-m1-s4-quiz';
import cpM1S5Quiz from '../multiple-choice/competitive-programming/cp-m1-s5-quiz';
import cpM1S6Quiz from '../multiple-choice/competitive-programming/cp-m1-s6-quiz';
import cpM1S7Quiz from '../multiple-choice/competitive-programming/cp-m1-s7-quiz';
import cpM1S8Quiz from '../multiple-choice/competitive-programming/cp-m1-s8-quiz';
import cpM1S9Quiz from '../multiple-choice/competitive-programming/cp-m1-s9-quiz';
import cpM1S10Quiz from '../multiple-choice/competitive-programming/cp-m1-s10-quiz';
import cpM1S11Quiz from '../multiple-choice/competitive-programming/cp-m1-s11-quiz';
import cpM1S12Quiz from '../multiple-choice/competitive-programming/cp-m1-s12-quiz';
import cpM1S13Quiz from '../multiple-choice/competitive-programming/cp-m1-s13-quiz';
import cpM1S14Quiz from '../multiple-choice/competitive-programming/cp-m1-s14-quiz';
import cpM1S15Quiz from '../multiple-choice/competitive-programming/cp-m1-s15-quiz';
import cpM1S16Quiz from '../multiple-choice/competitive-programming/cp-m1-s16-quiz';

export const competitiveProgrammingModule: Module = {
  id: 'competitive-programming',
  title: 'Competitive Programming: C++ & Problem Solving',
  description:
    'Master competitive programming with C++ fundamentals, optimization techniques, and contest strategies. Learn to write fast, efficient code under time pressure and build a robust CP toolkit.',
  category: 'Competitive Programming',
  difficulty: 'Beginner to Intermediate',
  estimatedTime: '30-40 hours',
  prerequisites: ['Basic programming knowledge'],
  icon: '⚡',
  keyTakeaways: [
    'Develop algorithmic intuition and problem-solving mindset',
    'Master C++ for competitive programming with modern features',
    'Set up optimal development environment and workflow',
    'Write blazing-fast I/O and optimized code',
    'Use advanced C++ features: templates, lambdas, structured bindings',
    'Master bit manipulation and bitwise operations',
    'Debug efficiently under contest time pressure',
    'Build and maintain a robust starter template',
    'Understand compilation errors and fix them quickly',
    'Develop contest-day strategies and mental resilience',
  ],
  learningObjectives: [
    'Build algorithmic intuition for problem pattern recognition',
    'Understand why C++ dominates competitive programming',
    'Configure compiler with optimal flags and settings',
    'Use modern CP tools: cf-tool, Competitive Companion',
    'Implement fast I/O with sync_with_stdio and cin.tie',
    'Master C++ STL: vectors, maps, sets, algorithms',
    'Leverage C++11/14/17/20 features effectively',
    'Write efficient macros and preprocessor tricks',
    'Perform bitwise operations and bit manipulation',
    'Manage memory efficiently with static vs dynamic allocation',
    'Use template metaprogramming for generic code',
    'Decode and fix cryptic compilation errors quickly',
    'Debug with print statements and stress testing',
    "Read and learn from others' CP code",
    'Execute effective contest-day workflow and time management',
    'Build and evolve a personal CP starter template',
  ],
  sections: [
    {
      ...buildingAlgorithmicIntuitionSection,
      quiz: cpM1S1Discussion,
      multipleChoice: cpM1S1Quiz,
    },
    {
      ...whyCppForCpSection,
      quiz: cpM1S2Discussion,
      multipleChoice: cpM1S2Quiz,
    },
    {
      ...environmentSetupCompilationSection,
      quiz: cpM1S3Discussion,
      multipleChoice: cpM1S3Quiz,
    },
    {
      ...modernCpToolEcosystemSection,
      quiz: cpM1S4Discussion,
      multipleChoice: cpM1S4Quiz,
    },
    {
      ...fastInputOutputTechniquesSection,
      quiz: cpM1S5Discussion,
      multipleChoice: cpM1S5Quiz,
    },
    {
      ...cppBasicsReviewSection,
      quiz: cpM1S6Discussion,
      multipleChoice: cpM1S6Quiz,
    },
    {
      ...cppModernFeaturesSection,
      quiz: cpM1S7Discussion,
      multipleChoice: cpM1S7Quiz,
    },
    {
      ...macrosPreprocessorTricksSection,
      quiz: cpM1S8Discussion,
      multipleChoice: cpM1S8Quiz,
    },
    {
      ...bitsBytesOperationsSection,
      quiz: cpM1S9Discussion,
      multipleChoice: cpM1S9Quiz,
    },
    {
      ...memoryManagementCpSection,
      quiz: cpM1S10Discussion,
      multipleChoice: cpM1S10Quiz,
    },
    {
      ...templateMetaprogrammingBasicsSection,
      quiz: cpM1S11Discussion,
      multipleChoice: cpM1S11Quiz,
    },
    {
      ...commonCompilationErrorsSection,
      quiz: cpM1S12Discussion,
      multipleChoice: cpM1S12Quiz,
    },
    {
      ...debuggingCompetitiveEnvironmentSection,
      quiz: cpM1S13Discussion,
      multipleChoice: cpM1S13Quiz,
    },
    {
      ...readingOthersCppCodeSection,
      quiz: cpM1S14Discussion,
      multipleChoice: cpM1S14Quiz,
    },
    {
      ...contestDayCppTipsSection,
      quiz: cpM1S15Discussion,
      multipleChoice: cpM1S15Quiz,
    },
    {
      ...buildingRobustCpTemplateSection,
      quiz: cpM1S16Discussion,
      multipleChoice: cpM1S16Quiz,
    },
  ],
};
