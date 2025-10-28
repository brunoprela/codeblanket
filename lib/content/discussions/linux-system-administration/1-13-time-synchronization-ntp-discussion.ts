export const timeSynchronizationNtpDiscussion = [
  {
    id: 1,
    question:
      "Kubernetes cluster authentication suddenly failing across all nodes with 'certificate has expired or is not yet valid' errors. System shows correct time with 'date' command. Diagnose and fix.",
    answer:
      "**Root cause:** System time is correct, but **hardware clock (RTC) is wrong**. Kubernetes uses system time for certificates, but on reboot, system time initializes from RTC. **Diagnosis:** 1) `sudo hwclock --show` → Shows time 2 days behind. 2) `timedatectl` → Check if RTC in local TZ. 3) `chronyc sources` → NTP syncing but only updates system time. **Fix:** 1) Sync RTC to system time: `sudo hwclock --systohc`. 2) Verify: `sudo hwclock --show`. 3) Ensure chrony configured: `rtcsync` in `/etc/chrony.conf`. 4) Restart nodes to test. **Prevention:** Monitor RTC drift, alert if RTC offset > 1 hour from NTP, automate `hwclock --systohc` in cron. **Why date showed correct time:** NTP synced system clock but didn't sync hardware clock.",
  },
  {
    id: 2,
    question:
      'Distributed application shows inconsistent behavior across 3 AWS regions (us-east-1, eu-west-1, ap-southeast-1). Time-sensitive operations (cache expiration, JWT validation) failing randomly. Design solution.',
    answer:
      "**Problem:** Clock skew between regions causes cache invalidation race conditions, JWT 'nbf' (not before) failures. **Solution:** 1) **All servers use AWS Time Sync:** Configure `server 169.254.169.123 prefer iburst` in chrony. 2) **Time tolerance in code:** JWT validation with ±30s clock skew tolerance. Cache TTL with 5s grace period. 3) **Monitoring:** CloudWatch custom metric tracking time offset every minute. Alert if offset > 50ms. 4) **Time-based operations use UTC:** All timestamps stored as UTC milliseconds since epoch. 5) **Distributed tracing:** Add timestamp headers to all API calls for debugging. **Implementation:** Use NTP health check in load balancer, remove instances with time drift > 100ms. **Result:** Clock skew < 10ms across all regions, zero time-related failures.",
  },
  {
    id: 3,
    question:
      "After deploying 500 new EC2 instances, chrony logs show 'No suitable source found' and instances can't sync time. Other instances in VPC sync fine. What's wrong and how to fix?",
    answer:
      '**Root cause:** Security group or NACL blocking **UDP port 123** (NTP) for new instances. **Diagnosis:** 1) `nc -vuz 169.254.169.123 123` → Times out. 2) Check security group: No outbound rule for UDP 123. 3) Check NACL: Possibly stateless rule missing return traffic. **Fix:** 1) **Security group:** Ensure outbound rules allow all traffic (0.0.0.0/0) or specifically UDP 123 to 169.254.169.123. 2) **NACL:** If using custom NACL, add outbound rule 100: UDP 123, 169.254.169.123/32, allow. Add inbound rule 100: UDP 1024-65535, 0.0.0.0/0, allow (ephemeral ports for NTP response). 3) Test: `chronyc sources -v` → Should show 169.254.169.123 reachable. **Prevention:** Terraform default security group with NTP rule, automated testing of time sync in instance bootstrap script. **Why other instances work:** They were launched with different security group or before NACL change.',
  },
];
