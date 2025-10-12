export interface TestCase {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  expected: any;
  explanation?: string;
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
  order: number;
  description: string;
  examples: Example[];
  constraints?: string[];
  hints?: string[];
  starterCode: string;
  testCases: TestCase[];
  solution?: string;
  timeComplexity?: string;
  spaceComplexity?: string;
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
