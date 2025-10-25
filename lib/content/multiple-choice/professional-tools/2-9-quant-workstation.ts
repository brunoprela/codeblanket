import { Quiz } from '@/lib/types';

export const quantWorkstationMultipleChoice: Quiz = {
  title: 'Building Your Quant Workstation Quiz',
  description:
    'Test your knowledge of hardware and software setup for quantitative trading.',
  questions: [
    {
      id: 'workstation-1',
      question:
        'For a quantitative researcher who regularly backtests strategies across 500 stocks, which hardware component provides the most significant performance improvement?',
      options: [
        'High-end GPU (RTX 4090)',
        'Ultra-fast NVMe SSD (7000 MB/s)',
        'High core-count CPU (16+ cores)',
        'Large 4K monitor for viewing data',
      ],
      correctAnswer: 2,
      explanation:
        'High core-count CPU provides the most impact for parallel backtesting. With 16 cores, you can backtest 16 stocks simultaneously instead of sequentially, providing 16x speedup. Backtesting 500 stocks: 16 cores = 15 minutes vs 1 core = 4 hours. GPU helps only for deep learning (not traditional backtesting), SSD speeds data loading but not computation, and monitor is for productivity not performance. For parallel workloads (backtesting, optimization, Monte Carlo), cores directly multiply throughput.',
    },
    {
      id: 'workstation-2',
      question:
        'Why is 32GB RAM recommended as a minimum for quantitative research rather than 16GB?',
      options: [
        'To run more applications simultaneously',
        'To load large financial datasets entirely in memory without swapping to disk, which is 100x slower',
        'To make the computer boot faster',
        '16GB is actually sufficient for all quantitative work',
      ],
      correctAnswer: 1,
      explanation:
        '32GB allows loading large datasets (S&P 500 minute bars for 1 year = ~8GB) entirely in RAM without swapping to disk. When RAM fills, OS swaps to SSD/HDD which is 100-1000x slower. With 16GB: load 8GB dataset, system swaps during analysis (painful). With 32GB: dataset stays in memory, analysis is fast. For multiple datasets or Jupyter notebooks, 16GB becomes unusable. 32GB provides comfortable working space for typical quant workloads. 64GB+ is luxury for larger universes or tick data.',
    },
    {
      id: 'workstation-3',
      question:
        'When setting up a Python environment for trading, what is the main advantage of using Conda over pip?',
      options: [
        'Conda is faster at installing packages than pip',
        'Conda handles both Python packages and system-level dependencies (C libraries, compilers), and creates isolated environments',
        'Conda packages are always more up-to-date than pip',
        'Conda is required by financial libraries',
      ],
      correctAnswer: 1,
      explanation:
        "Conda manages entire environments including system dependencies. Financial libraries like TA-Lib require C compilers and libraries - Conda handles this automatically, pip requires manual installation. Conda environments are isolated: 'trading' environment with Python 3.10, 'ml' environment with Python 3.11, no conflicts. Can specify exact versions for reproducibility. Conda channels (conda-forge) provide pre-compiled binaries for complex packages. Pip is great for Python-only packages, but Conda is superior for scientific/financial computing requiring system dependencies.",
    },
    {
      id: 'workstation-4',
      question:
        'What is the primary benefit of using VS Code Remote-SSH for quantitative research?',
      options: [
        'It makes your local computer faster',
        'It allows you to edit and run code on a powerful remote server while using your local VS Code interface',
        'It automatically backs up your code to the cloud',
        'It provides free access to market data',
      ],
      correctAnswer: 1,
      explanation:
        'VS Code Remote-SSH lets you work on a powerful cloud VPS as if it were local. Workflow: Open VS Code locally → Connect to remote server → Edit code, run notebooks, debug - all on server. Benefits: (1) Use 32-core server from laptop, (2) Long backtests run on server while you close laptop, (3) Access from any device, (4) No manual file syncing needed, (5) One source of truth for code. Perfect for running overnight backtests or using more powerful hardware than you own. Local VS Code interface with remote compute power.',
    },
    {
      id: 'workstation-5',
      question:
        'Why would a quant use Docker containers for their trading environment instead of installing everything directly on their system?',
      options: [
        'Docker makes code run faster than native installation',
        'Docker provides reproducibility, isolation, and easy deployment - ensuring code works identically across development, testing, and production',
        'Docker is required for all Python applications',
        'Docker reduces electricity consumption',
      ],
      correctAnswer: 1,
      explanation:
        'Docker ensures "works on my machine" → "works everywhere". Benefits: (1) Reproducibility - exact same environment on dev laptop, test server, production VPS, (2) Isolation - trading strategy in one container, ML research in another, no dependency conflicts, (3) Easy onboarding - new team member runs `docker-compose up`, everything works instantly, (4) Version control - Dockerfile documents exact environment, (5) Deployment - ship container to production, guaranteed to work. Example: Dockerfile specifies Python 3.10, pandas 2.0, Ubuntu 22.04 - every environment identical. Critical for production trading systems.',
    },
  ],
};
