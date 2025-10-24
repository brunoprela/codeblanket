import { MultipleChoiceQuestion } from '../../../types';

export const datamanipulationpandasMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-manipulation-pandas-mc-1',
    question:
      'What is the key difference between .apply() and .map() in Pandas?',
    options: [
      '.apply() works on DataFrames, .map() only works on Series',
      '.map() is faster than .apply()',
      '.apply() can work row-wise or column-wise, .map() is element-wise on Series only',
      'They are identical, just different names',
    ],
    correctAnswer: 2,
    explanation:
      '.apply() can work on DataFrames (along rows or columns) or Series, while .map() is specifically for element-wise transformation of Series. .map() also uses a dictionary/Series for substitution, while .apply() typically uses functions.',
  },
  {
    id: 'data-manipulation-pandas-mc-2',
    question:
      'Which is the fastest way to create a new column based on a condition in Pandas?',
    options: [
      'df["new",] = df["col",].apply(lambda x: val1 if x > 5 else val2)',
      'df["new",] = np.where(df["col",] > 5, val1, val2)',
      'for i in range(len(df)): df.loc[i, "new",] = val1 if df.loc[i, "col",] > 5 else val2',
      'df["new",] = df["col",].map(lambda x: val1 if x > 5 else val2)',
    ],
    correctAnswer: 1,
    explanation:
      "np.where() is the fastest for conditional column creation because it's a vectorized NumPy operation. apply() and map() with lambdas are slower due to Python function call overhead, and loops are the slowest option.",
  },
  {
    id: 'data-manipulation-pandas-mc-3',
    question: 'What does the .str accessor do in Pandas?',
    options: [
      'Converts all values to strings',
      'Provides vectorized string methods for Series containing strings',
      'Creates a string representation of the DataFrame',
      'Filters for string columns only',
    ],
    correctAnswer: 1,
    explanation:
      'The .str accessor provides vectorized string methods (like .upper(), .contains(), .split()) that work element-wise on Series containing strings. It applies string operations efficiently without explicit loops.',
  },
  {
    id: 'data-manipulation-pandas-mc-4',
    question:
      'Given df["date",] as a datetime column, how do you extract the month name?',
    options: [
      'df["date",].month_name()',
      'df["date",].str.month_name()',
      'df["date",].dt.month_name()',
      'df["date",].datetime.month_name()',
    ],
    correctAnswer: 2,
    explanation:
      'The .dt accessor provides datetime-specific operations. Use df["date",].dt.month_name() to extract month names. The .dt accessor is specifically for datetime operations, similar to how .str is for strings.',
  },
  {
    id: 'data-manipulation-pandas-mc-5',
    question:
      'What is the difference between .replace() and .map() when substituting values?',
    options: [
      'No difference, they do the same thing',
      '.replace() keeps unmatched values unchanged, .map() returns NaN for unmatched',
      '.map() is faster than .replace()',
      '.replace() only works with strings',
    ],
    correctAnswer: 1,
    explanation:
      "Key difference: .replace() keeps values that don't match any replacement unchanged, while .map() returns NaN for values not in the mapping dictionary. This makes .replace() safer for partial substitutions.",
  },
];
