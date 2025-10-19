import { MultipleChoiceQuestion } from '../../../types';

export const mergingjoiningdataMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'merging-joining-data-mc-1',
    question:
      'What is the difference between pd.merge() with how="left" and how="inner"?',
    options: [
      'No difference, they produce the same result',
      'left keeps all rows from the left DataFrame, inner keeps only matching rows from both',
      'inner is faster than left',
      'left requires sorted data, inner does not',
    ],
    correctAnswer: 1,
    explanation:
      'A left join (how="left") keeps all rows from the left DataFrame and matches from the right (filling with NaN if no match). An inner join (how="inner") keeps only rows that have matching keys in both DataFrames.',
  },
  {
    id: 'merging-joining-data-mc-2',
    question:
      "When merging two DataFrames with overlapping column names (other than the merge key), what happens if you don't specify suffixes?",
    options: [
      'Pandas raises an error',
      'Pandas automatically adds "_x" and "_y" suffixes',
      'The columns are merged into one',
      'The right DataFrame columns are dropped',
    ],
    correctAnswer: 1,
    explanation:
      'By default, Pandas adds "_x" suffix to columns from the left DataFrame and "_y" suffix to columns from the right DataFrame when column names conflict. You can customize these with the suffixes parameter.',
  },
  {
    id: 'merging-joining-data-mc-3',
    question: 'What does the indicator=True parameter do in pd.merge()?',
    options: [
      'Shows progress during merge',
      'Adds a "_merge" column showing which DataFrame each row came from',
      'Validates the merge keys',
      'Sorts the result',
    ],
    correctAnswer: 1,
    explanation:
      'indicator=True adds a special "_merge" column with values "left_only", "right_only", or "both" to show whether each row came from only the left DataFrame, only the right DataFrame, or matched in both. This is useful for diagnosing merge issues.',
  },
  {
    id: 'merging-joining-data-mc-4',
    question: 'When would you use pd.merge_asof() instead of pd.merge()?',
    options: [
      'When you need an outer join',
      'When merging time series data with inexact timestamp matches',
      'When the DataFrames are very large',
      'When merging on multiple keys',
    ],
    correctAnswer: 1,
    explanation:
      "pd.merge_asof() is designed for time series data where you need to merge on the nearest timestamp within a tolerance, rather than requiring exact matches. It's useful for matching events to prices, aligning data streams, etc.",
  },
  {
    id: 'merging-joining-data-mc-5',
    question:
      'What is a potential problem when performing a many-to-many merge?',
    options: [
      'It always fails with an error',
      'It creates all combinations of matching rows, potentially exploding the dataset size',
      'It only keeps one row from each side',
      'It is slower than other join types',
    ],
    correctAnswer: 1,
    explanation:
      'In a many-to-many merge, if multiple rows in the left DataFrame match multiple rows in the right DataFrame, all combinations are created. This can dramatically increase the number of rows and is often unintended, indicating duplicate keys that should be handled.',
  },
];
