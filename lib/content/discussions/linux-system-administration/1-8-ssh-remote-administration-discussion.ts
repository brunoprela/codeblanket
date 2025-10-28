export const sshRemoteAdministrationDiscussion = [
  {
    id: 1,
    question:
      'Design a secure SSH access architecture for a production environment with 100+ EC2 instances across public and private subnets, supporting developers, SREs, and contractors with different access levels.',
    answer:
      '**Architecture:** 1) **Bastion (jump host)** in public subnet with MFA. 2) **Private instances** in private subnets (no public IPs). 3) **Access tiers:** Devs SSH to bastion → dev servers; SREs to bastion → all servers; contractors to bastion → specific servers. 4) **Security:** SSH keys rotated every 90 days, AllowUsers per instance, fail2ban on bastion, CloudWatch logging, Session Manager as backup. 5) **Alternative:** AWS SSM Session Manager (no bastion needed, IAM-based access, full audit logs).',
  },
  {
    id: 2,
    question:
      'Compare SSH bastion host vs AWS Systems Manager Session Manager. When would you choose each?',
    answer:
      '**Bastion:** Pros: Traditional, works everywhere, supports tunneling, familiar. Cons: Single point of failure, requires public IP, SSH key management, security group maintenance. **Session Manager:** Pros: No public IP/bastion needed, IAM-based access (no keys), full audit logs to CloudTrail/S3, port forwarding support, no security group ingress. Cons: Requires SSM agent, internet/VPC endpoint for agent communication. **Choose SSM** for new AWS infrastructure (better security, audit). **Choose bastion** for hybrid/multi-cloud or complex tunneling needs.',
  },
  {
    id: 3,
    question:
      'How would you implement temporary SSH access for a contractor who needs database access for 8 hours?',
    answer:
      "**Approach 1 (Keys):** Generate time-limited SSH key with certificate authority. Upload public key to authorized_keys with `command=` restriction. Remove after 8 hours via cron. **Approach 2 (Session Manager):** Grant IAM role with time-based condition: `aws:CurrentTime` between start/end. Attach to contractor's IAM user. Auto-expires. **Approach 3 (Best):** SSH tunnel through bastion with database access. `ssh -L 5432:rds.internal:5432 contractor@bastion`. Revoke bastion access after 8 hours. Log all queries via RDS audit logs.",
  },
];
