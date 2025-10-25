import { Quiz } from '@/lib/types';

export const jupyterLabQuantMultipleChoice: Quiz = {
  title: 'Jupyter Lab for Quantitative Research Quiz',
  description:
    'Test your knowledge of Jupyter Lab features and best practices for quantitative finance research.',
  questions: [
    {
      id: 'jupyter-1',
      question:
        'What is the primary advantage of using `%load_ext autoreload` with `%autoreload 2` at the beginning of a Jupyter notebook when developing trading strategies?',
      options: [
        'It automatically saves the notebook every 2 minutes',
        "It reloads external Python modules automatically when they're modified, without restarting the kernel",
        'It enables parallel processing across 2 CPU cores',
        'It creates 2 backup copies of the notebook',
      ],
      correctAnswer: 1,
      explanation:
        "`%autoreload 2` automatically reloads all modules before executing code, which is essential when developing trading strategies split between notebooks and .py files. Without this, you'd need to manually restart the kernel every time you modify an external module. This dramatically speeds up the development workflow: modify your strategy code in a .py file, and immediately test it in your notebook without kernel restart. The '2' specifies to reload all modules (vs '1' which only reloads modules imported with %aimport).",
    },
    {
      id: 'jupyter-2',
      question:
        'When organizing a quantitative research project in Jupyter Lab, why is it recommended to use sequential notebook numbering (e.g., 01_data_collection.ipynb, 02_eda.ipynb, 03_strategy_dev.ipynb)?',
      options: [
        'It ensures notebooks execute in alphabetical order automatically',
        'It provides clear indication of the intended workflow order and makes the project easier to navigate for collaborators',
        'Jupyter Lab requires numeric prefixes for proper rendering',
        'It enables automatic versioning of notebooks',
      ],
      correctAnswer: 1,
      explanation:
        "Sequential numbering (00-99) provides immediate visual indication of the intended workflow and analysis progression. This is crucial for: (1) Onboarding new team members who can follow the analysis from start to finish, (2) Returning to a project after months and understanding the flow, (3) Preventing circular dependencies between notebooks. The numbering doesn't automate execution - that's what tools like papermill are for - but it creates a clear narrative structure. Best practice: 00-09 for data collection, 10-19 for cleaning, 20-29 for feature engineering, 30-39 for modeling, etc.",
    },
    {
      id: 'jupyter-3',
      question:
        'What is the purpose of using Papermill to parameterize Jupyter notebooks in quantitative finance?',
      options: [
        'To convert notebooks to PDF format for client presentations',
        'To execute the same analysis notebook with different parameters (e.g., different tickers, date ranges) programmatically',
        'To compress notebooks for easier sharing via email',
        'To automatically detect and fix errors in notebook code',
      ],
      correctAnswer: 1,
      explanation:
        "Papermill allows you to execute notebooks programmatically with different parameters, which is invaluable for quantitative research. Example: You develop a backtest notebook for one stock, then use Papermill to run it across 500 S&P 500 stocks overnight. You tag one cell as 'parameters' containing `TICKER='AAPL'`, `START_DATE='2020-01-01'`, etc. Then run `papermill template.ipynb output_MSFT.ipynb -p TICKER MSFT` to execute with different parameters. This enables: batch processing, consistent analysis across instruments, automated reporting, and reproducible research at scale.",
    },
    {
      id: 'jupyter-4',
      question:
        'Why is it considered best practice to use Parquet or Feather format instead of CSV for loading large financial datasets in Jupyter notebooks?',
      options: [
        'Parquet and Feather are open-source while CSV requires a license',
        'Parquet and Feather support more data types than CSV',
        'Parquet and Feather load 5-10x faster and use less disk space due to columnar storage and compression',
        'CSV files cannot be read by pandas in Jupyter notebooks',
      ],
      correctAnswer: 2,
      explanation:
        "Parquet and Feather use columnar storage and compression, resulting in dramatic performance improvements for financial data. Typical improvements: 5-10x faster loading, 3-5x smaller file size, better preservation of data types (especially dates and categorical data). For a 5GB CSV file: CSV load might take 60 seconds, Parquet might take 8 seconds. When analyzing large datasets repeatedly, this adds up. Parquet also allows selecting specific columns without loading the entire file. Trade-off: Parquet/Feather aren't human-readable like CSV, but for programmatic analysis, the performance gains are worth it. Use CSV for sharing with Excel users, Parquet for analysis.",
    },
    {
      id: 'jupyter-5',
      question:
        'What is the purpose of using `nbstripout` in a quantitative research project managed with Git?',
      options: [
        'To automatically remove comments from code cells before committing',
        'To remove cell outputs and metadata from notebooks before committing to Git, reducing repository size and preventing merge conflicts',
        'To convert notebooks to Python scripts automatically',
        'To strip out sensitive API keys from notebook code',
      ],
      correctAnswer: 1,
      explanation:
        "`nbstripout` is essential for collaborative quantitative research projects. It automatically strips outputs, execution counts, and metadata from notebooks before Git commits. Benefits: (1) Repository stays small - outputs (especially charts/dataframes) can make notebooks 100MB+, (2) Meaningful diffs - without output removal, every execution changes the notebook even if code didn't change, (3) Prevents merge conflicts - execution metadata differs between users causing false conflicts, (4) Security - accidentally committed outputs might contain sensitive data. Install with `pip install nbstripout` and `nbstripout --install` to automatically clean notebooks on every commit. Outputs are preserved locally, just not committed.",
    },
  ],
};
