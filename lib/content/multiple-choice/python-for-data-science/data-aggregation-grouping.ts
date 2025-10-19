import { MultipleChoiceQuestion } from '../../../types';

export const dataaggregationgroupingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-aggregation-grouping-mc-1',
    question:
      'What is the key difference between .agg() and .transform() in GroupBy operations?',
    options: [
      '.agg() is faster than .transform()',
      '.agg() returns aggregated results (fewer rows), .transform() returns same shape as input',
      'They do the same thing, just different names',
      '.transform() only works with numeric data',
    ],
    correctAnswer: 1,
    explanation:
      '.agg() performs aggregation and returns one value per group (reducing the DataFrame), while .transform() applies a function and returns a value for each row (maintaining the original shape). Use .agg() for summaries, .transform() to add group statistics to each row.',
  },
  {
    id: 'data-aggregation-grouping-mc-2',
    question: 'What does groupby().filter(lambda x: len(x) > 5) do?',
    options: [
      'Filters rows within each group where condition is met',
      'Removes groups that have 5 or fewer members',
      'Keeps only the first 5 rows of each group',
      'Filters out groups with more than 5 members',
    ],
    correctAnswer: 1,
    explanation:
      ".filter() with GroupBy removes entire groups that don't meet the condition. len(x) > 5 keeps only groups with more than 5 members, removing all groups with 5 or fewer members entirely from the result.",
  },
  {
    id: 'data-aggregation-grouping-mc-3',
    question: 'In a pivot table, what is the purpose of margins=True?',
    options: [
      'Adds padding around the table',
      'Adds row and column totals (subtotals)',
      'Removes missing values',
      'Sorts the results',
    ],
    correctAnswer: 1,
    explanation:
      'margins=True adds an "All" row and column to the pivot table showing the totals (or other aggregate) across all groups. This provides subtotals and a grand total, similar to Excel pivot tables.',
  },
  {
    id: 'data-aggregation-grouping-mc-4',
    question:
      'What does df.groupby("Category")["Value"].transform("mean") return?',
    options: [
      'A Series with one mean value per category',
      'A Series with the same length as the original DataFrame, with each row showing its group mean',
      'A DataFrame with mean values',
      'An error because transform requires a lambda function',
    ],
    correctAnswer: 1,
    explanation:
      '.transform("mean") returns a Series with the same length as the original DataFrame. Each row contains the mean of its group, allowing you to compare individual values to their group average.',
  },
  {
    id: 'data-aggregation-grouping-mc-5',
    question:
      'Why is df.groupby("Category")["Value"].mean() faster than df.groupby("Category")["Value"].agg(lambda x: x.mean())?',
    options: [
      'They are equally fast',
      'Built-in functions like mean() use optimized C code, while lambda functions use slower Python',
      'mean() uses less memory',
      'lambda functions are always faster',
    ],
    correctAnswer: 1,
    explanation:
      'Built-in aggregation functions (mean, sum, etc.) are implemented in optimized Cython/C code and are much faster than custom lambda functions which execute in Python. Always prefer built-in functions when possible.',
  },
];
