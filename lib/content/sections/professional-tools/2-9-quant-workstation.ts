import { Content } from '@/lib/types';

export const quantWorkstation: Content = {
  title: 'Building Your Quant Workstation',
  subtitle: 'Hardware and software setup for professional quantitative trading',
  description:
    'Design and build a high-performance workstation for quantitative research and trading. Learn optimal hardware configurations, software stack setup, and productivity optimization for professional quant work.',
  sections: [
    {
      title: 'Hardware Requirements',
      content: `
# Workstation Hardware for Quantitative Trading

## CPU: The Brain of Your Operation

### For Most Quants: High-Core-Count Intel/AMD
\`\`\`plaintext
Recommended CPUs (2024):

Budget ($200-400):
├── AMD Ryzen 5 7600X (6 cores, 12 threads)
├── Intel Core i5-13600K (14 cores, 20 threads)
└── Good for: Single-strategy backtesting, learning

Mid-Range ($400-700):
├── AMD Ryzen 7 7700X (8 cores, 16 threads)
├── Intel Core i7-13700K (16 cores, 24 threads)
└── Good for: Multi-strategy research, parallel backtests

High-End ($700-1500):
├── AMD Ryzen 9 7950X (16 cores, 32 threads)
├── Intel Core i9-13900K (24 cores, 32 threads)
└── Good for: Large-scale backtests, ML training

Extreme ($2000+):
├── AMD Threadripper PRO 5975WX (32 cores, 64 threads)
├── Intel Xeon W-3375 (38 cores, 76 threads)
└── Good for: Institution-grade computing
\`\`\`

### Why Core Count Matters
\`\`\`python
# Backtesting 500 stocks sequentially
for ticker in tickers:  # 500 iterations
    backtest (ticker)  # 30 seconds each
# Total time: 15,000 seconds (4.2 hours)

# Parallel backtesting with 16 cores
from joblib import Parallel, delayed
results = Parallel (n_jobs=16)(
    delayed (backtest)(ticker) for ticker in tickers
)
# Total time: 938 seconds (15.6 minutes) - 16x faster!
\`\`\`

## RAM: More is Better for Financial Data

### Minimum Requirements
\`\`\`plaintext
RAM Guidelines:

16GB: Minimum for basic work
├── Single-stock analysis
├── Small universes (<50 stocks)
└── Will page to disk with large datasets

32GB: Sweet spot for most quants
├── Multi-stock backtests
├── 500 stock universe
├── Comfortable for daily work
└── Recommended minimum

64GB: Professional grade
├── Large-scale backtests
├── Multiple Jupyter notebooks open
├── Database in memory
└── ML model training

128GB+: Institution/serious research
├── Tick data analysis
├── Massive parallel operations
├── Multiple strategies simultaneously
└── Future-proof
\`\`\`

### Why RAM Matters
\`\`\`python
# Loading S&P 500 minute bars (1 year)
import pandas as pd

# Low RAM (16GB): Painful
df = pd.read_csv('sp500_minute.csv')  # 8GB file
# System swaps to disk, everything slows down

# Adequate RAM (32GB): Smooth
df = pd.read_csv('sp500_minute.csv')  # Loads instantly
df['returns'] = df.groupby('ticker')['close'].pct_change()
# Fast, no swapping

# High RAM (64GB): Luxury
df_minute = pd.read_csv('sp500_minute.csv')  # 8GB
df_tick = pd.read_csv('sp500_tick.csv')      # 20GB
# Both in memory, instant switching between datasets
\`\`\`

## Storage: Speed Matters

### NVMe SSD is Essential
\`\`\`plaintext
Storage Configuration:

Primary Drive (OS + Applications):
├── 500GB-1TB NVMe SSD (PCIe 4.0)
├── Samsung 980 PRO, WD Black SN850X
├── ~7000 MB/s read speed
└── For: OS, Python, databases, active projects

Secondary Drive (Data):
├── 1-2TB NVMe SSD
├── For: Historical data, backtest results
└── Fast access to large datasets

Tertiary Drive (Archive):
├── 4-8TB HDD (optional)
├── For: Long-term storage, backups
└── Cheap bulk storage

Speed Comparison:
├── HDD: 150 MB/s (slow, mechanical)
├── SATA SSD: 550 MB/s (fast)
└── NVMe SSD: 7000 MB/s (very fast)
\`\`\`

### Example: Loading 10GB Dataset
\`\`\`plaintext
HDD (150 MB/s):      67 seconds
SATA SSD (550 MB/s): 18 seconds
NVMe SSD (7000 MB/s): 1.4 seconds

For quant work where you load data hundreds of times per day,
NVMe saves hours of waiting time.
\`\`\`

## GPU: Optional but Powerful

### When You Need a GPU
\`\`\`plaintext
GPU Use Cases:

Essential:
├── Deep learning for trading signals
├── Neural network training
├── Large-scale ML models
└── Computer vision (chart pattern recognition)

Helpful:
├── GPU-accelerated pandas (cuDF)
├── Large matrix operations
└── Monte Carlo simulations

Not Needed:
├── Traditional backtesting
├── Basic technical indicators
└── Statistical analysis
\`\`\`

### GPU Recommendations
\`\`\`plaintext
Budget ($300-500):
├── NVIDIA RTX 4060 Ti (16GB)
└── Good for: Learning ML, small models

Mid-Range ($800-1200):
├── NVIDIA RTX 4070 Ti (12GB)
└── Good for: Serious ML research

High-End ($1500-2500):
├── NVIDIA RTX 4090 (24GB)
└── Good for: Large models, fast iteration

Professional ($5000+):
├── NVIDIA A100 (40GB/80GB)
└── For: Production ML systems
\`\`\`

## Complete Build Examples

### Budget Quant Workstation ($1500)
\`\`\`plaintext
CPU: AMD Ryzen 5 7600X ($280)
Motherboard: B650 ($150)
RAM: 32GB DDR5 ($120)
Storage: 1TB NVMe SSD ($80)
GPU: Integrated graphics (save for later GPU)
PSU: 650W ($80)
Case: Mid-tower ($70)
Cooling: Air cooler ($40)
Monitor: 27" 1440p ($250)
Keyboard/Mouse: ($80)
Total: ~$1,500

Good for: Learning, single-strategy development
\`\`\`

### Professional Workstation ($3500)
\`\`\`plaintext
CPU: AMD Ryzen 9 7950X ($700)
Motherboard: X670E ($300)
RAM: 64GB DDR5 (32GB×2) ($240)
Storage: 2TB NVMe SSD + 1TB NVMe ($280)
GPU: NVIDIA RTX 4070 Ti ($900)
PSU: 850W Gold ($150)
Case: Full tower ($150)
Cooling: AIO liquid cooler ($180)
Monitors: 2× 27" 1440p ($500)
Keyboard/Mouse: Mechanical ($200)
UPS: 1500VA ($200)
Total: ~$3,500

Good for: Professional quant research, ML
\`\`\`

### Extreme Workstation ($8000+)
\`\`\`plaintext
CPU: AMD Threadripper PRO 5975WX ($3000)
Motherboard: WRX80 ($800)
RAM: 256GB ECC DDR4 (8×32GB) ($1200)
Storage: 4TB NVMe RAID + 2TB NVMe ($800)
GPU: NVIDIA RTX 4090 ($2000)
PSU: 1200W Platinum ($300)
Case: Server/workstation ($300)
Cooling: Custom water cooling ($500)
Monitors: 3× 32" 4K ($1500)
Network: 10GbE NIC ($200)
UPS: 3000VA ($600)
Total: ~$11,000

Good for: Institutional-grade research
\`\`\`

## Peripherals That Matter

### Multi-Monitor Setup
\`\`\`plaintext
Recommended Configuration:

Single Monitor (Minimum):
└── 27" 1440p (2560×1440)
    └── Enough for code + data

Dual Monitor (Recommended):
├── Main: 27" 1440p (code/charts)
└── Secondary: 27" 1440p (docs/monitoring)
    └── Massive productivity boost

Triple Monitor (Professional):
├── Center: 32" 4K (main work)
├── Left: 27" 1440p vertical (code)
└── Right: 27" 1440p (monitoring)
    └── Ultimate productivity

Ultra-Wide Alternative:
└── 34-49" ultra-wide (3440×1440 or wider)
    └── Single cable, less desk clutter
\`\`\`

### Mechanical Keyboard
Worth the investment for typing code all day:
- Cherry MX Brown switches (quiet, tactile)
- Keychron, Leopold, Varmilo brands
- $80-200 range

### Ergonomic Setup
\`\`\`plaintext
Prevent RSI and back pain:
├── Ergonomic chair ($300-800)
├── Sit-stand desk ($400-1000)
├── Monitor arms ($50-150)
├── Wrist rest ($20)
└── Footrest ($30)

Your health is worth the investment!
\`\`\`
      `,
    },
    {
      title: 'Software Stack Setup',
      content: `
# Complete Software Environment

## Operating System Choice

### Linux (Recommended for Serious Quants)
\`\`\`plaintext
Advantages:
├── Native Python/C++ development
├── Superior command-line tools
├── Docker and containers work better
├── Free and open-source
├── Better for server deployment
└── No licensing costs

Recommended Distributions:
├── Ubuntu 22.04 LTS (most popular, best support)
├── Fedora (cutting-edge packages)
└── Arch Linux (advanced users)

Disadvantages:
├── Steeper learning curve
├── Some proprietary software unavailable
└── Gaming support limited
\`\`\`

### Windows (Easiest Start)
\`\`\`plaintext
Advantages:
├── Familiar interface
├── All software available (Excel, Bloomberg)
├── Gaming if you need breaks
└── WSL2 gives Linux environment

Disadvantages:
├── License cost ($200)
├── Updates can be intrusive
├── Not ideal for production servers
└── Some performance overhead

Recommended Setup:
├── Windows 11 Pro
└── WSL2 with Ubuntu
    └── Best of both worlds!
\`\`\`

### macOS (Premium Option)
\`\`\`plaintext
Advantages:
├── Unix-based (similar to Linux)
├── Excellent hardware integration
├── Great for mobile work (MacBook)
└── Premium build quality

Disadvantages:
├── Expensive hardware
├── Limited GPU options
├── Can't build custom
└── Thermal throttling on laptops
\`\`\`

## Python Environment Setup

### Using Conda (Recommended)
\`\`\`bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create trading environment
conda create -n trading python=3.10
conda activate trading

# Install essential packages
conda install -c conda-forge \\
    numpy pandas scipy matplotlib seaborn \\
    jupyter jupyterlab ipython \\
    scikit-learn statsmodels \\
    sqlalchemy psycopg2 pymongo \\
    requests beautifulsoup4 lxml

# Financial libraries
pip install yfinance pandas-datareader \\
    ta-lib quantlib zipline-reloaded \\
    backtrader pyfolio empyrical

# Machine learning
conda install -c conda-forge \\
    tensorflow pytorch torchvision \\
    xgboost lightgbm catboost

# Visualization
pip install plotly dash altair

# Development tools
pip install black flake8 mypy pytest \\
    ipdb jupyter-lsp-python

# Save environment
conda env export > environment.yml
\`\`\`

### Environment Management
\`\`\`bash
# Create environment from file
conda env create -f environment.yml

# Update environment
conda env update -f environment.yml

# Multiple environments for different projects
conda create -n backtest python=3.10
conda create -n ml python=3.10
conda create -n production python=3.10

# Quick switch
conda activate backtest  # For backtesting work
conda activate ml        # For ML research
conda activate production  # For live trading
\`\`\`

## IDE Setup

### VS Code (Most Popular)
\`\`\`plaintext
Essential Extensions:
├── Python (Microsoft)
├── Jupyter
├── Pylance (IntelliSense)
├── GitLens
├── Docker
├── Remote - SSH
├── Black Formatter
└── autoDocstring

Settings (settings.json):
{
    "python.defaultInterpreterPath": "~/miniconda3/envs/trading/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "files.trimTrailingWhitespace": true
}
\`\`\`

### PyCharm (Professional Alternative)
\`\`\`plaintext
Advantages over VS Code:
├── Superior Python debugging
├── Built-in database tools
├── Better refactoring
└── Scientific mode for data analysis

Free for students, $200/year otherwise
\`\`\`

### Vim/Neovim (For Experts)
\`\`\`plaintext
Ultra-fast editing but steep learning curve
Popular with senior quants
Not recommended for beginners
\`\`\`

## Database Setup

### PostgreSQL with TimescaleDB
\`\`\`bash
# Docker setup (easiest)
docker run -d \\
    --name timescaledb \\
    -p 5432:5432 \\
    -e POSTGRES_PASSWORD=password \\
    -v pgdata:/var/lib/postgresql/data \\
    timescale/timescaledb:latest-pg15

# Connect
psql -h localhost -U postgres

# Create database
CREATE DATABASE marketdata;
\\c marketdata
CREATE EXTENSION timescaledb;
\`\`\`

### MongoDB (for Alternative Data)
\`\`\`bash
# Docker setup
docker run -d \\
    --name mongodb \\
    -p 27017:27017 \\
    -v mongodata:/data/db \\
    -e MONGO_INITDB_ROOT_USERNAME=admin \\
    -e MONGO_INITDB_ROOT_PASSWORD=password \\
    mongo:latest
\`\`\`

### Redis (for Caching)
\`\`\`bash
# Docker setup
docker run -d \\
    --name redis \\
    -p 6379:6379 \\
    -v redisdata:/data \\
    redis:latest

# Test connection
redis-cli ping
# Should return: PONG
\`\`\`

## Version Control

### Git Configuration
\`\`\`bash
# Global config
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Useful aliases
git config --global alias.st "status -sb"
git config --global alias.lg "log --graph --oneline"
git config --global alias.last "log -1 HEAD"

# Default branch name
git config --global init.defaultBranch main

# Editor
git config --global core.editor "code --wait"

# Line endings
git config --global core.autocrlf input  # Linux/Mac
git config --global core.autocrlf true   # Windows
\`\`\`

### GitHub CLI
\`\`\`bash
# Install gh CLI
# Linux
sudo apt install gh

# Mac
brew install gh

# Windows
winget install GitHub.cli

# Authenticate
gh auth login

# Create repo
gh repo create my-trading-strategy --private

# Clone with ease
gh repo clone username/repo
\`\`\`

## Productivity Tools

### Terminal: Alacritty or iTerm2
\`\`\`bash
# Fast, GPU-accelerated terminal
# Better than default terminals

# Install Alacritty (Linux)
sudo apt install alacritty

# Install iTerm2 (Mac)
brew install --cask iterm2
\`\`\`

### Shell: Zsh with Oh My Zsh
\`\`\`bash
# Install Zsh
sudo apt install zsh

# Set as default
chsh -s $(which zsh)

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Recommended plugins (~/.zshrc)
plugins=(
    git
    python
    docker
    kubectl
    zsh-autosuggestions
    zsh-syntax-highlighting
)

# Theme
ZSH_THEME="powerlevel10k/powerlevel10k"
\`\`\`

### tmux: Terminal Multiplexer
\`\`\`bash
# Install
sudo apt install tmux

# Basic usage
tmux new -s trading  # New session
tmux attach -t trading  # Reattach
Ctrl+b d  # Detach (session keeps running)

# Split panes
Ctrl+b %  # Vertical split
Ctrl+b "  # Horizontal split

# Great for running long backtests
# Detach and come back hours later!
\`\`\`

### Data Transfer: rsync
\`\`\`bash
# Sync data between workstation and server
rsync -avz --progress \\
    user@server:/data/markets/ \\
    ~/data/markets/

# Backup strategy code
rsync -avz --progress \\
    ~/projects/trading-strategies/ \\
    user@backup-server:~/backups/
\`\`\`
      `,
    },
    {
      title: 'Performance Optimization',
      content: `
# Optimizing Your Workstation

## Python Performance Tuning

### Use Numba for Speed
\`\`\`python
from numba import jit
import numpy as np

# Slow Python
def slow_backtest (prices):
    returns = []
    for i in range(1, len (prices)):
        returns.append((prices[i] - prices[i-1]) / prices[i-1])
    return returns

# Fast with Numba
@jit (nopython=True)
def fast_backtest (prices):
    returns = np.empty (len (prices) - 1)
    for i in range(1, len (prices)):
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns

# Speed test
import time
prices = np.random.randn(1000000).cumsum() + 100

start = time.time()
slow_backtest (prices)
print(f"Slow: {time.time() - start:.3f}s")  # 2.5s

start = time.time()
fast_backtest (prices)
print(f"Fast: {time.time() - start:.3f}s")  # 0.015s

# 166x faster!
\`\`\`

### Parallel Processing
\`\`\`python
from joblib import Parallel, delayed
import multiprocessing

# Number of CPU cores
n_cores = multiprocessing.cpu_count()
print(f"Available cores: {n_cores}")

# Parallel backtest
def backtest_ticker (ticker):
    # Heavy computation here
    return result

# Sequential
results = [backtest_ticker (t) for t in tickers]  # Slow

# Parallel
results = Parallel (n_jobs=n_cores)(
    delayed (backtest_ticker)(t) for t in tickers
)  # n_cores times faster!
\`\`\`

## Database Optimization

### PostgreSQL Tuning
\`\`\`sql
-- postgresql.conf optimizations for trading workstation

-- Memory settings (for 32GB RAM system)
shared_buffers = 8GB              -- 25% of RAM
effective_cache_size = 24GB       -- 75% of RAM
work_mem = 256MB                  -- Per-query memory
maintenance_work_mem = 2GB        -- For maintenance operations

-- Parallelism (for 16-core CPU)
max_parallel_workers_per_gather = 4
max_parallel_workers = 16
max_worker_processes = 16

-- Write performance
wal_buffers = 16MB
checkpoint_completion_target = 0.9
min_wal_size = 1GB
max_wal_size = 4GB

-- Query planner
random_page_cost = 1.1            -- For SSD (default 4.0 for HDD)
effective_io_concurrency = 200    -- For NVMe SSD

-- Connection pooling
max_connections = 100
\`\`\`

### Index Monitoring
\`\`\`sql
-- Find missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND n_distinct > 100
  AND correlation < 0.1
ORDER BY n_distinct DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- Unused indexes
ORDER BY pg_relation_size (indexrelid) DESC;

-- Remove unused indexes to improve write performance
\`\`\`

## SSD Optimization

### Linux: Enable TRIM
\`\`\`bash
# Check if TRIM is supported
sudo fstrim -v /

# Enable automatic TRIM
sudo systemctl enable fstrim.timer
sudo systemctl start fstrim.timer

# Verify
sudo systemctl status fstrim.timer
\`\`\`

### Disk I/O Monitoring
\`\`\`bash
# Install iotop
sudo apt install iotop

# Monitor disk I/O in real-time
sudo iotop -o  # Only show processes doing I/O

# Check disk statistics
iostat -x 1  # Update every second
\`\`\`

## Network Optimization

### For Market Data Streaming
\`\`\`bash
# Increase TCP buffer sizes (Linux)
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Make permanent
echo "net.core.rmem_max=134217728" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=134217728" | sudo tee -a /etc/sysctl.conf
\`\`\`

## Monitoring System Performance

### htop: Process Monitoring
\`\`\`bash
# Install
sudo apt install htop

# Run
htop

# Key metrics to watch:
# - CPU usage per core (should be balanced)
# - Memory usage (shouldn't swap)
# - Load average (should be < number of cores)
\`\`\`

### nvtop: GPU Monitoring
\`\`\`bash
# Install
sudo apt install nvtop

# Monitor GPU usage
nvtop

# Check if ML code is using GPU effectively
\`\`\`

### Custom Monitoring Script
\`\`\`python
#!/usr/bin/env python3
"""
System monitoring for trading workstation
Run in background to log performance
"""
import psutil
import time
import csv
from datetime import datetime

def log_system_stats (output_file='system_stats.csv'):
    with open (output_file, 'a', newline='') as f:
        writer = csv.writer (f)
        
        # Header
        if f.tell() == 0:
            writer.writerow([
                'timestamp', 'cpu_percent', 'memory_percent',
                'disk_read_mb', 'disk_write_mb',
                'net_sent_mb', 'net_recv_mb'
            ])
        
        while True:
            # Collect metrics
            cpu_percent = psutil.cpu_percent (interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            # Write row
            writer.writerow([
                datetime.now().isoformat(),
                cpu_percent,
                memory.percent,
                disk_io.read_bytes / 1024 / 1024,
                disk_io.write_bytes / 1024 / 1024,
                net_io.bytes_sent / 1024 / 1024,
                net_io.bytes_recv / 1024 / 1024
            ])
            
            f.flush()
            time.sleep(60)  # Log every minute

if __name__ == '__main__':
    log_system_stats()
\`\`\`

## Backup Strategy

### Automated Backups
\`\`\`bash
#!/bin/bash
# backup.sh - Run daily with cron

# Backup code
rsync -avz ~/projects/ /backup/projects/

# Backup databases
pg_dump marketdata | gzip > /backup/db/marketdata_$(date +%Y%m%d).sql.gz

# Backup research notebooks
rsync -avz ~/notebooks/ /backup/notebooks/

# Keep only last 30 days of backups
find /backup/db/ -name "*.sql.gz" -mtime +30 -delete

# Sync to cloud
rclone sync /backup/ remote:backup/

echo "Backup completed at $(date)"
\`\`\`

### Crontab Setup
\`\`\`bash
# Edit crontab
crontab -e

# Add backup job (run at 2 AM daily)
0 2 * * * /home/user/scripts/backup.sh >> /var/log/backup.log 2>&1
\`\`\`
      `,
    },
    {
      title: 'Remote Access and Cloud',
      content: `
# Working Remotely and Cloud Integration

## SSH Setup

### Secure Remote Access
\`\`\`bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy to remote server
ssh-copy-id user@remote-server.com

# Now can login without password
ssh user@remote-server.com

# SSH config (~/.ssh/config)
Host quant-server
    HostName 192.168.1.100
    User quant
    Port 22
    IdentityFile ~/.ssh/id_ed25519

# Now can use: ssh quant-server
\`\`\`

### VS Code Remote Development
\`\`\`plaintext
1. Install "Remote - SSH" extension in VS Code
2. Cmd+Shift+P → "Remote-SSH: Connect to Host"
3. Enter: user@remote-server.com
4. VS Code opens, running on remote server
5. Edit code, run notebooks, debug - all remote!

Benefits:
├── Use powerful remote server
├── Access from any device
├── Keep work in one place
└── No need to sync files
\`\`\`

## Cloud VPS Options

### For Additional Compute
\`\`\`plaintext
DigitalOcean Droplet:
├── CPU Optimized: $42/mo (4 vCPU, 8GB RAM)
├── General Purpose: $96/mo (8 vCPU, 16GB RAM)
└── Use for: Running long backtests while you sleep

AWS EC2:
├── c6i.2xlarge: $0.34/hr (8 vCPU, 16GB RAM)
├── c6i.8xlarge: $1.36/hr (32 vCPU, 64GB RAM)
└── Use for: Burst compute, then shut down

Linode:
├── Dedicated 16GB: $72/mo (8 vCPU, 16GB RAM)
└── Use for: 24/7 data collection

Hetzner (Europe):
├── CAX21: €8.46/mo (4 vCPU, 8GB RAM)
└── Use for: Budget-friendly compute
\`\`\`

### Quick Cloud Setup
\`\`\`bash
# After creating VPS, setup essentials
ssh root@your-vps-ip

# Update system
apt update && apt upgrade -y

# Install essentials
apt install -y build-essential python3-pip git htop tmux

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create trading environment
conda create -n trading python=3.10 -y
conda activate trading
pip install pandas numpy yfinance

# Clone your code
git clone https://github.com/yourusername/trading-strategies.git

# Run long backtest in tmux
tmux new -s backtest
python scripts/backtest_all_stocks.py
# Ctrl+b d to detach, close SSH, come back tomorrow!
\`\`\`

## Data Synchronization

### Using rsync
\`\`\`bash
# Download data from server
rsync -avz --progress \\
    user@server:/data/marketdata/ \\
    ~/data/marketdata/

# Upload results to server
rsync -avz --progress \\
    ~/results/ \\
    user@server:/backups/results/

# Two-way sync (use with caution!)
rsync -avuz --delete \\
    ~/projects/ \\
    user@server:~/projects/
\`\`\`

### Using rclone (for Cloud Storage)
\`\`\`bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure (interactive)
rclone config
# Add Google Drive, Dropbox, S3, etc.

# Sync to cloud
rclone sync ~/data/marketdata/ gdrive:marketdata/

# Sync from cloud
rclone sync gdrive:marketdata/ ~/data/marketdata/

# Schedule daily sync
crontab -e
# Add: 0 3 * * * rclone sync ~/data/marketdata/ gdrive:marketdata/
\`\`\`

## Docker for Reproducibility

### Containerize Your Environment
\`\`\`dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Default command
CMD ["python", "scripts/backtest.py"]
\`\`\`

### Docker Compose for Full Stack
\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: marketdata
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
  
  jupyter:
    build: .
    command: jupyter lab --ip=0.0.0.0 --no-browser --allow-root
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    depends_on:
      - postgres
      - redis

volumes:
  pgdata:
\`\`\`

### Usage
\`\`\`bash
# Start everything
docker-compose up -d

# Access Jupyter
# Check logs for token:
docker-compose logs jupyter

# Stop everything
docker-compose down

# Ship entire environment to colleague
# They run: docker-compose up
# Everything works identically!
\`\`\`

## Security Best Practices

### Firewall Setup
\`\`\`bash
# Ubuntu UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8888/tcp  # Jupyter (from specific IP better)
sudo ufw enable

# Allow SSH only from your IP
sudo ufw delete allow ssh
sudo ufw allow from YOUR_IP_HERE to any port 22
\`\`\`

### Fail2Ban (Prevent Brute Force)
\`\`\`bash
# Install
sudo apt install fail2ban

# Configure
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Check status
sudo fail2ban-client status sshd
\`\`\`

### VPN for Secure Access
\`\`\`plaintext
Options:
├── WireGuard (modern, fast, easy)
├── OpenVPN (traditional, well-supported)
└── Tailscale (zero-config mesh VPN)

Benefit: Encrypt all traffic between home and VPS
\`\`\`
      `,
    },
  ],
  exercises: [
    {
      title: 'Build Shopping List',
      description:
        'Create a complete hardware shopping list for a quantitative trading workstation within a $2500 budget, including all components and peripherals.',
      difficulty: 'beginner',
      hints: [
        'Balance CPU cores vs single-thread performance',
        'Prioritize 32GB RAM minimum',
        'NVMe SSD is essential, HDD optional',
        'Consider future GPU upgrade',
        'Include monitor, keyboard, mouse in budget',
      ],
    },
    {
      title: 'Development Environment Setup',
      description:
        'Set up a complete Python development environment with conda, install all necessary financial libraries, configure VS Code, and create a test backtest script.',
      difficulty: 'intermediate',
      hints: [
        'Create conda environment with Python 3.10',
        'Install pandas, numpy, yfinance, ta-lib',
        'Configure Black formatter and flake8 linter',
        'Set up Git with proper .gitignore',
        'Write simple backtest to verify setup',
      ],
    },
    {
      title: 'Docker Development Environment',
      description:
        'Create a Docker Compose setup that includes PostgreSQL/TimescaleDB, Redis, Jupyter Lab, and your trading code, all networked together.',
      difficulty: 'advanced',
      hints: [
        'Create Dockerfile for Python environment',
        'Use docker-compose.yml for orchestration',
        'Set up volumes for data persistence',
        'Configure networking between containers',
        'Test database connectivity from Jupyter',
      ],
    },
    {
      title: 'Remote Research Station',
      description:
        'Set up a cloud VPS as a remote research station, configure SSH access, install software stack, and sync code/data between local and remote.',
      difficulty: 'advanced',
      hints: [
        'Choose cloud provider and create VPS',
        'Configure SSH key-based authentication',
        'Install Miniconda and create environment',
        'Set up VS Code Remote-SSH',
        'Configure rsync or rclone for syncing',
        'Run test backtest in tmux session',
      ],
    },
  ],
};
