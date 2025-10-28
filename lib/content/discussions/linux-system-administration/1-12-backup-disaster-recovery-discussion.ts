export const backupDisasterRecoveryDiscussion = [
  {
    id: 1,
    question:
      'E-commerce site processes $1M/hour in transactions. Design a backup and DR strategy considering RTO=15min, RPO=1min, and multi-region compliance requirements.',
    answer:
      '**RPO=1min strategy:** 1) **Database:** RDS Multi-AZ with read replica in DR region (cross-region replication lag ~5sec). Enable automated backups (7d retention), manual snapshots before deployments. 2) **Application:** Blue-green deployment with instant rollback. Code in Git, deployments via CI/CD. 3) **Assets:** S3 with cross-region replication (CRR) enabled, versioning on. 4) **RTO=15min:** Active-passive DR: Standby RDS replica in us-west-2, Route53 health checks with automatic failover. Keep warm standby (scaled-down instances) or use ASG to launch on demand. **Testing:** Monthly DR drills, automated runbooks. **Cost:** ~$5K/mo for DR infrastructure + $2K/mo for cross-region data transfer. **Compliance:** Data residency handled via separate regional databases, GDPR-compliant S3 lifecycle policies.',
  },
  {
    id: 2,
    question:
      'Database backup took 6 hours last night, causing application slowdown. Design a zero-impact backup strategy.',
    answer:
      '**Solutions:** 1) **Read replica for backups:** Create RDS read replica, take pg_dump/mysqldump from replica (zero impact on primary). Cost: +$500/mo. 2) **EBS snapshots:** For EC2-hosted databases, use EBS snapshots (incremental, crash-consistent). Run during low-traffic window. 3) **Continuous archiving:** PostgreSQL WAL archiving or MySQL binlog streaming to S3. Real-time backup, point-in-time recovery. 4) **Parallel dumps:** Use `pg_dump -j 8` (8 parallel workers) to reduce time from 6h to 1h. 5) **Optimize dump:** Use `-Fc` custom format, exclude unnecessary data (logs, temp tables). 6) **Schedule wisely:** 3 AM backups during traffic minimum. 7) **AWS Backup:** Managed service with application-consistent backups. **Result:** Backup window reduced to 30min with zero user impact.',
  },
  {
    id: 3,
    question:
      'How would you verify that your backups are actually restorable? Design a comprehensive backup validation strategy.',
    answer:
      "**Multi-layer validation:** 1) **Automated integrity checks:** Run `tar -tzf` or `gunzip -t` on every backup, store checksums (MD5/SHA256), verify checksums match. 2) **Test restores:** Weekly automated restore to isolated test environment. Script: Restore backup → Start services → Run smoke tests → Compare data checksums. 3) **Quarterly DR drills:** Full failover exercise: Shutdown prod → Restore from backup in DR region → Validate functionality → Switch DNS → Monitor. Document time-to-restore (verify RTO). 4) **Continuous validation:** Use AWS Backup's automated restore testing feature. 5) **Data consistency:** For databases, run `ANALYZE` and consistency checks post-restore. 6) **Application testing:** Run full integration test suite on restored environment. 7) **Monitoring:** Alert if backup validation fails. **Documentation:** Runbook with restore procedures, test results logged for compliance. **Schedule:** Integrity checks (daily), test restores (weekly), DR drills (quarterly).",
  },
];
