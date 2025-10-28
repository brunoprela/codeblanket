export const securityHardeningDiscussion = [
  {
    id: 1,
    question:
      'Your production EC2 instances are experiencing brute force SSH attacks from multiple IPs. Design a comprehensive defense strategy.',
    answer:
      '**Multi-layer defense:** 1) **SSH hardening:** Disable password auth, change port (security by obscurity), key-based only, PermitRootLogin no. 2) **Fail2ban:** Ban IPs after 3 failed attempts for 24h. 3) **Security groups:** Whitelist office IPs only. 4) **Rate limiting:** MaxStartups in sshd_config. 5) **Alternative:** Switch to AWS Session Manager (no SSH port exposed). 6) **Monitoring:** CloudWatch alerts on failed SSH attempts. 7) **WAF:** If using bastion with ALB. **Result:** Reduce attacks by 99%.',
  },
  {
    id: 2,
    question: 'Compare SELinux and AppArmor. When would you use each?',
    answer:
      '**SELinux (RHEL/Amazon Linux):** Mandatory Access Control (MAC), label-based, complex but powerful, enterprise standard. Policies are comprehensive. **AppArmor (Ubuntu/Debian):** Path-based, easier to configure, simpler learning curve, good for web servers. **Choose SELinux** for high-security environments (government, finance), existing RHEL infrastructure. **Choose AppArmor** for Ubuntu-based systems, simpler security needs, easier maintenance. Both: Use enforcing mode in production, permissive for troubleshooting.',
  },
  {
    id: 3,
    question:
      'Design an automated security patching strategy that minimizes downtime while maintaining security.',
    answer:
      '**Strategy:** 1) **Dev/Staging:** Auto-patch nightly, test applications. 2) **Production:** Use ASG with rolling updates. 3) **Schedule:** Patch Tuesday+7 days (allow vendor testing). 4) **Process:** Create new AMI with patches → Launch new instances → Drain connections → Terminate old. 5) **Critical CVEs:** Emergency patch within 24h. 6) **Tools:** AWS Systems Manager Patch Manager, or dnf-automatic for security-only updates. 7) **Rollback:** Keep previous AMI for 30 days. 8) **Zero-downtime:** Blue-green deployment or rolling updates via ASG.',
  },
];
