export const processResourceManagementDiscussion = [
  {
    id: 1,
    question:
      "Production application is hitting 'Too many open files' error during peak traffic. Design a comprehensive solution considering immediate fix, root cause, and prevention.",
    answer:
      '**Immediate fix:** `sudo systemctl edit myapp.service` → Set `LimitNOFILE=65536`, restart service. **Root cause analysis:** 1) Check current usage: `lsof -p $(pgrep myapp) | wc -l`. 2) Find file descriptor leaks: connections not closed, file handles leaked. 3) Profile application code for unclosed resources. **Prevention:** 1) Set `LimitNOFILE=65536` in systemd service. 2) Add monitoring alert when FD usage > 80%. 3) Implement connection pooling, proper resource cleanup. 4) Load test to verify limits. 5) Document limits in runbook. **Long-term:** Fix application code to reuse connections, add FD metrics to dashboards.',
  },
  {
    id: 2,
    question:
      'Database service is being OOM killed during backups, but memory usage shows only 70%. Why is it being killed and how do you fix it?',
    answer:
      '**Cause:** Linux OOM killer considers multiple factors: 1) Memory + swap usage. 2) Sudden allocation spikes. 3) OOM score (default calculation favors killing large memory users). 4) Kernel memory fragmentation. **Why 70% killed:** Backup process causes spike in memory requests, even if not allocated yet. Kernel sees potential OOM condition and kills preemptively. **Solution:** 1) Set `OOMScoreAdjust=-500` in systemd service to protect database. 2) Increase `MemoryMax=4G` with buffer above normal usage. 3) Schedule backups during low-traffic periods. 4) Use streaming backups (lower memory footprint). 5) Add swap (but tune `vm.swappiness=10`). 6) Monitor with `dmesg | grep oom` and CloudWatch. **Prevention:** Memory headroom = peak_usage × 1.5 + buffer.',
  },
  {
    id: 3,
    question:
      'Design a resource allocation strategy for a mixed workload EC2 instance running: critical API (needs 60% CPU), background jobs (30% CPU), and monitoring agents (10% CPU). Ensure API always has resources.',
    answer:
      '**Cgroups strategy:** 1) **API service:** `CPUQuota=200%` (2 cores), `CPUWeight=1000` (highest priority), `MemoryMax=4G`, `OOMScoreAdjust=-500`, `Nice=-5`. 2) **Background jobs:** `CPUQuota=100%` (1 core max), `CPUWeight=100` (normal), `Nice=10` (lower priority), can be throttled. 3) **Monitoring:** `CPUQuota=50%`, `CPUWeight=50`, `Nice=15`. **Implementation:** Create systemd slices for each workload. Set `CPUAccounting=true`, `MemoryAccounting=true` for monitoring. Use `systemd-cgtop` to verify allocation. **Result:** API gets priority during contention, background jobs throttled first, monitoring protected from starvation.',
  },
];
