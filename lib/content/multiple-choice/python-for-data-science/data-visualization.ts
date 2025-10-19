import { MultipleChoiceQuestion } from '../../../types';

export const datavisualizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-visualization-mc-1',
    question: 'What is the main difference between matplotlib and seaborn?',
    options: [
      'Seaborn is faster than matplotlib',
      'Matplotlib provides low-level control, seaborn provides statistical plots with better defaults',
      'Matplotlib only works with DataFrames',
      'They are identical',
    ],
    correctAnswer: 1,
    explanation:
      'Matplotlib is a low-level plotting library giving full control over every element, while seaborn is built on matplotlib and provides high-level statistical visualizations with beautiful defaults. Seaborn is better for quick statistical plots, matplotlib for custom fine-tuning.',
  },
  {
    id: 'data-visualization-mc-2',
    question: 'What does plt.subplot(2, 3, 4) mean?',
    options: [
      'Create a 2x3 grid and activate the 4th subplot',
      'Create 2 rows, 3 columns, with 4 plots',
      'Create a 4x4 grid with 2-3 spacing',
      'It causes an error',
    ],
    correctAnswer: 0,
    explanation:
      'plt.subplot(nrows, ncols, index) creates a grid with nrows rows and ncols columns, and activates the subplot at position index (1-indexed). So subplot(2, 3, 4) creates a 2x3 grid and activates the 4th position (row 2, column 1).',
  },
  {
    id: 'data-visualization-mc-3',
    question: 'When should you use a box plot instead of a histogram?',
    options: [
      'Always use histograms, they show more information',
      'Box plots are better for comparing distributions across multiple groups',
      'Box plots are only for small datasets',
      'They show exactly the same information',
    ],
    correctAnswer: 1,
    explanation:
      'Box plots are excellent for comparing distributions across multiple categories side-by-side, showing median, quartiles, and outliers. Histograms show the full distribution shape but are harder to compare when you have many groups.',
  },
  {
    id: 'data-visualization-mc-4',
    question: 'What does the hue parameter do in seaborn plots?',
    options: [
      'Changes the plot color',
      'Groups data by a variable and colors them differently',
      'Adjusts plot brightness',
      'Creates 3D plots',
    ],
    correctAnswer: 1,
    explanation:
      'The hue parameter in seaborn adds an additional dimension to plots by coloring data points according to a categorical variable. For example, sns.scatterplot(x="age", y="income", hue="gender") would color points by gender.',
  },
  {
    id: 'data-visualization-mc-5',
    question: 'Why use plt.tight_layout()?',
    options: [
      'Makes plots load faster',
      'Automatically adjusts subplot spacing to prevent overlapping labels',
      'Reduces file size',
      'Creates tighter data clustering',
    ],
    correctAnswer: 1,
    explanation:
      "plt.tight_layout() automatically adjusts the spacing between subplots and around the figure to prevent axis labels, titles, and tick labels from overlapping. It's especially useful when creating complex multi-subplot figures.",
  },
];
