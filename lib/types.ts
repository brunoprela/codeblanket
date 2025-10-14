/**
 * Type definitions for the CodeBlanket application
 */

export interface TestCase {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  expected: any;
  explanation?: string;
  functionName?: string; // Optional function name to call for testing (useful for class-based problems)
}

export interface CustomTestCase {
  id: string;
  code: string; // Python code for the test case
  name?: string; // Optional name for the test case
}

export interface Example {
  input: string;
  output: string;
  explanation?: string;
}

export type Difficulty = 'Easy' | 'Medium' | 'Hard';

export interface Problem {
  id: string;
  title: string;
  difficulty: Difficulty;
  description: string;
  examples?: Example[];
  constraints?: string[];
  hints?: string[];
  starterCode?: string;
  testCases: TestCase[];
  solution?: string;
  timeComplexity?: string;
  spaceComplexity?: string;
  order?: number; // For sorting - Made optional
  topic: string; // e.g., "Binary Search", "Two Pointers"
  leetcodeUrl?: string; // Link to LeetCode problem
  youtubeUrl?: string; // Link to YouTube explanation
  approach?: string; // For design problems - detailed approach explanation
}

export interface TestResult {
  passed: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  expected: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  actual: any;
  error?: string;
  executionTime?: number;
}

/**
 * Module system types
 */

export interface QuizQuestion {
  id: string;
  question: string;
  hint?: string;
  sampleAnswer: string;
  keyPoints: string[];
}

export interface MultipleChoiceQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number; // Index of correct option (0-based)
  explanation: string;
}

export interface ModuleSection {
  id: string;
  title: string;
  content: string;
  codeExample?: string;
  quiz: QuizQuestion[];
  multipleChoice?: MultipleChoiceQuestion[];
}

export interface Module {
  id: string;
  title: string;
  description: string;
  icon: string;
  sections: ModuleSection[];
  keyTakeaways: string[];
  timeComplexity?: string;
  spaceComplexity?: string;
  relatedProblems: string[]; // Array of problem IDs
}

export interface ModuleCategory {
  id: string;
  title: string;
  description: string;
  icon: string;
  module: Module;
  problemCount: number;
}

export interface TopicSection {
  id: string;
  title: string;
  description: string;
  icon: string;
  modules: ModuleCategory[];
}
