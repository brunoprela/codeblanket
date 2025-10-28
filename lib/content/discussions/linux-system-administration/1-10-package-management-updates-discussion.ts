export const packageManagementUpdatesDiscussion = [
  {
    id: 1,
    question:
      'Design an automated patch management strategy for 500+ production EC2 instances running critical applications that must maintain 99.9% uptime.',
    answer:
      '**Multi-tier strategy:** 1) **Golden AMIs:** Bake monthly patches into new AMIs, test thoroughly. 2) **ASG rolling updates:** Use instance refresh with `MinHealthyPercentage=90` for zero-downtime. 3) **Staging first:** Test updates in staging 1 week before prod. 4) **Canary deployment:** Update 5% of fleet first, monitor for 24h. 5) **Security patches:** Apply critical CVEs within 48h using dnf-automatic security-only. 6) **Rollback plan:** Keep previous AMI active, ready to rollback via ASG. 7) **Maintenance windows:** Schedule non-critical updates during low-traffic periods. **Result:** 99.95%+ uptime maintained.',
  },
  {
    id: 2,
    question:
      'Application depends on Python 3.9, but security team mandates updating to Python 3.11 for CVE fixes. How do you manage this transition safely?',
    answer:
      '**Phased approach:** 1) **Version pinning:** Pin app to Python 3.9 while testing 3.11 compatibility (`dnf versionlock add python39`). 2) **Parallel installation:** Install Python 3.11 alongside 3.9 (use alternatives). 3) **Testing:** Clone prod environment, switch to 3.11, run full test suite. 4) **Gradual migration:** Update dev → staging → 10% prod canary → full prod. 5) **Virtual environments:** Use venv to isolate dependencies. 6) **Rollback:** Keep 3.9 packages available, document downgrade procedure. 7) **Monitoring:** Track errors, performance after migration. **Timeline:** 2-4 weeks for safe migration.',
  },
  {
    id: 3,
    question:
      'How would you prevent a scenario where a kernel update breaks a critical driver, causing downtime?',
    answer:
      '**Prevention strategy:** 1) **Test in staging:** Never update kernel directly in prod. 2) **AMI-based updates:** Bake kernel updates into AMI, test instance launch. 3) **Version locking:** Use `dnf versionlock` to prevent automatic kernel updates. 4) **Module testing:** Check critical modules (`lsmod`, `modinfo`) after update. 5) **Rollback plan:** Keep old kernel as bootloader option (GRUB). 6) **Blue-green:** Launch new instances with updated kernel, drain old instances. 7) **Canary testing:** Update 1 instance, monitor for 48h. 8) **Vendor advisories:** Check AWS/RHEL release notes for known issues. **Recovery:** Boot previous kernel via GRUB rescue mode.',
  },
];
