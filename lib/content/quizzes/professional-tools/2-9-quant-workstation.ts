import { Discussion } from '@/lib/types';

export const quantWorkstationQuiz: Discussion = {
  title: 'Building Your Quant Workstation Discussion Questions',
  description:
    'Deep dive into hardware choices, software configuration, and workflow optimization for quantitative research.',
  questions: [
    {
      id: 'workstation-disc-1',
      question:
        'Design the optimal workstation setup for three different quant researcher profiles: (1) Beginner learning algorithmic trading, (2) Professional researcher at a hedge fund, (3) High-frequency trading developer. Include complete hardware specs, software stack, budget considerations, and justification for each choice.',
      sampleAnswer: `[Comprehensive comparison covering: Beginner ($1500 budget: Ryzen 5, 32GB RAM, 1TB SSD, no GPU, single monitor, Windows + WSL2, free tools), Professional ($4000: Ryzen 9/i9, 64GB RAM, 2TB NVMe, RTX 4070 Ti, dual monitors, Linux, Bloomberg Terminal), HFT Developer ($8000+: Threadripper, 128GB ECC RAM, NVMe RAID, 10GbE networking, specialized hardware for latency optimization). Justify each component choice based on workload characteristics, explain software stack differences, discuss tradeoffs between cost and performance.]`,
      keyPoints: [],
    },
    {
      id: 'workstation-disc-2',
      question:
        'Explain a complete development workflow using local workstation for development and cloud VPS for heavy compute. Cover: code editing, version control, data synchronization, remote execution, and result retrieval. Include specific tools and commands.',
      sampleAnswer: `[Detailed workflow: Local development (VS Code with Remote-SSH editing code on VPS), version control (Git with GitHub as source of truth), data sync (rsync for large datasets, rclone for cloud storage backups), remote execution (tmux for long-running jobs that persist after disconnect), result retrieval (automated scripts to sync results back to local). Include example commands for each step, discuss handling secrets and credentials, explain backup strategy, cover cost optimization by starting/stopping VPS as needed.]`,
      keyPoints: [],
    },
    {
      id: 'workstation-disc-3',
      question:
        'Describe a complete Docker-based development environment for a trading team. Include: services needed, networking configuration, data persistence, development workflow, and deployment strategy. How does this improve collaboration and reproducibility?',
      sampleAnswer: `[Complete Docker Compose architecture: PostgreSQL/TimescaleDB for price data, Redis for caching, Jupyter Lab for research, Flask/FastAPI for backtesting API, Grafana for monitoring. Network configuration for inter-service communication, volumes for data persistence, environment variables for configuration. Development workflow: developers work in containers matching production, code changes hot-reload, database seeds for testing. Deployment: same containers to staging then production. Benefits: eliminates "works on my machine" problems, new team members productive immediately, exact environment reproduction, easy rollback if issues.]`,
      keyPoints: [],
    },
  ],
};
