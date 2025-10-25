import { MultipleChoiceQuestion } from '../../../types';

export const pandasseriesdataframesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pandas-series-dataframes-mc-1',
    question:
      'What is the key difference between a Pandas Series and a NumPy array?',
    options: [
      'Series can only store numbers while arrays can store any type',
      'Series has an index (labels) while arrays only have positional indexing',
      'Series is faster than arrays for numerical operations',
      'Series cannot perform mathematical operations',
    ],
    correctAnswer: 1,
    explanation:
      'The key difference is that a Pandas Series has an index (labels) that can be used to access elements, while NumPy arrays only support positional integer indexing. This makes Series more flexible for labeled data.',
  },
  {
    id: 'pandas-series-dataframes-mc-2',
    question: "Given df = pd.DataFrame (data), what does df['column',] return?",
    options: [
      'A DataFrame with one column',
      'A Series',
      'A NumPy array',
      'A Python list',
    ],
    correctAnswer: 1,
    explanation:
      "Selecting a single column with df['column',] returns a Pandas Series. To get a DataFrame with one column, use df[['column',]] (double brackets).",
  },
  {
    id: 'pandas-series-dataframes-mc-3',
    question: 'What is the difference between .loc and .iloc in Pandas?',
    options: [
      '.loc is faster than .iloc',
      '.loc uses label-based indexing, .iloc uses position-based indexing',
      '.iloc only works with integers',
      'They are identical, just different syntax',
    ],
    correctAnswer: 1,
    explanation:
      '.loc uses label-based indexing (index names), while .iloc uses integer position-based indexing (0, 1, 2, ...). Both are useful in different scenarios.',
  },
  {
    id: 'pandas-series-dataframes-mc-4',
    question: "Why would you use the 'category' dtype for a DataFrame column?",
    options: [
      'It makes sorting faster',
      'It saves memory for columns with repeated values and enables ordering',
      'It is required for string columns',
      'It allows mathematical operations on strings',
    ],
    correctAnswer: 1,
    explanation:
      'The category dtype saves memory by storing repeated values as integers with a mapping, and enables ordering for ordered categorical data. This is especially useful for columns like "low", "medium", "high" or departments with limited unique values.',
  },
  {
    id: 'pandas-series-dataframes-mc-5',
    question:
      'What is the correct way to filter a DataFrame for rows where Age > 30 AND Department is "IT"?',
    options: [
      "df[df['Age',] > 30 and df['Department',] == 'IT',]",
      "df[(df['Age',] > 30) & (df['Department',] == 'IT')]",
      "df[df['Age',] > 30].filter (df['Department',] == 'IT')",
      "df.filter(('Age' > 30) and ('Department' == 'IT'))",
    ],
    correctAnswer: 1,
    explanation:
      "Boolean indexing requires bitwise operators (&, |, ~) not logical operators (and, or, not), and parentheses around each condition: (df['Age',] > 30) & (df['Department',] == 'IT').",
  },
];
