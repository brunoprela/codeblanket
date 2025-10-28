/**
 * Linux System Administration & DevOps Foundations Module
 * Aggregates sections, discussions, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { linuxFundamentalsProductionSection } from '../sections/linux-system-administration/linux-fundamentals-production';
import { shellScriptingAutomationSection } from '../sections/linux-system-administration/shell-scripting-automation';
import { systemMonitoringPerformanceSection } from '../sections/linux-system-administration/system-monitoring-performance';
import { storageFileSystemsSection } from '../sections/linux-system-administration/storage-file-systems';
import { networkingBasicsSection } from '../sections/linux-system-administration/networking-basics';
import { systemdServiceManagementSection } from '../sections/linux-system-administration/systemd-service-management';
import { logManagementSection } from '../sections/linux-system-administration/log-management';
import { sshRemoteAdministrationSection } from '../sections/linux-system-administration/ssh-remote-administration';
import { securityHardeningSection } from '../sections/linux-system-administration/security-hardening';
import { packageManagementUpdatesSection } from '../sections/linux-system-administration/package-management-updates';
import { processResourceManagementSection } from '../sections/linux-system-administration/process-resource-management';
import { backupDisasterRecoverySection } from '../sections/linux-system-administration/backup-disaster-recovery';
import { timeSynchronizationNtpSection } from '../sections/linux-system-administration/time-synchronization-ntp';
import { debuggingProductionIssuesSection } from '../sections/linux-system-administration/debugging-production-issues';

// Import discussions
import { linuxFundamentalsProductionDiscussion } from '../discussions/linux-system-administration/1-1-linux-fundamentals-production-discussion';
import { shellScriptingAutomationDiscussion } from '../discussions/linux-system-administration/1-2-shell-scripting-automation-discussion';
import { systemMonitoringPerformanceDiscussion } from '../discussions/linux-system-administration/1-3-system-monitoring-performance-discussion';
import { storageFileSystemsDiscussion } from '../discussions/linux-system-administration/1-4-storage-file-systems-discussion';
import { networkingBasicsDiscussion } from '../discussions/linux-system-administration/1-5-networking-basics-discussion';
import { systemdServiceManagementDiscussion } from '../discussions/linux-system-administration/1-6-systemd-service-management-discussion';
import { logManagementDiscussion } from '../discussions/linux-system-administration/1-7-log-management-discussion';
import { sshRemoteAdministrationDiscussion } from '../discussions/linux-system-administration/1-8-ssh-remote-administration-discussion';
import { securityHardeningDiscussion } from '../discussions/linux-system-administration/1-9-security-hardening-discussion';
import { packageManagementUpdatesDiscussion } from '../discussions/linux-system-administration/1-10-package-management-updates-discussion';
import { processResourceManagementDiscussion } from '../discussions/linux-system-administration/1-11-process-resource-management-discussion';
import { backupDisasterRecoveryDiscussion } from '../discussions/linux-system-administration/1-12-backup-disaster-recovery-discussion';
import { timeSynchronizationNtpDiscussion } from '../discussions/linux-system-administration/1-13-time-synchronization-ntp-discussion';
import { debuggingProductionIssuesDiscussion } from '../discussions/linux-system-administration/1-14-debugging-production-issues-discussion';

// Import multiple choice questions
import { linuxFundamentalsProductionMultipleChoice } from '../multiple-choice/linux-system-administration/1-1-linux-fundamentals-production';
import { shellScriptingAutomationMultipleChoice } from '../multiple-choice/linux-system-administration/1-2-shell-scripting-automation';
import { systemMonitoringPerformanceMultipleChoice } from '../multiple-choice/linux-system-administration/1-3-system-monitoring-performance';
import { storageFileSystemsMultipleChoice } from '../multiple-choice/linux-system-administration/1-4-storage-file-systems';
import { networkingBasicsMultipleChoice } from '../multiple-choice/linux-system-administration/1-5-networking-basics';
import { systemdServiceManagementMultipleChoice } from '../multiple-choice/linux-system-administration/1-6-systemd-service-management';
import { logManagementMultipleChoice } from '../multiple-choice/linux-system-administration/1-7-log-management';
import { sshRemoteAdministrationMultipleChoice } from '../multiple-choice/linux-system-administration/1-8-ssh-remote-administration';
import { securityHardeningMultipleChoice } from '../multiple-choice/linux-system-administration/1-9-security-hardening';
import { packageManagementUpdatesMultipleChoice } from '../multiple-choice/linux-system-administration/1-10-package-management-updates';
import { processResourceManagementMultipleChoice } from '../multiple-choice/linux-system-administration/1-11-process-resource-management';
import { backupDisasterRecoveryMultipleChoice } from '../multiple-choice/linux-system-administration/1-12-backup-disaster-recovery';
import { timeSynchronizationNtpMultipleChoice } from '../multiple-choice/linux-system-administration/1-13-time-synchronization-ntp';
import { debuggingProductionIssuesMultipleChoice } from '../multiple-choice/linux-system-administration/1-14-debugging-production-issues';

export const linuxSystemAdministrationModule: Module = {
  id: 'linux-system-administration',
  title: 'Linux System Administration & DevOps Foundations',
  description:
    'Master production-level Linux system administration for AWS cloud environments. Learn everything from file systems and process management to security hardening, monitoring, and debugging production issues.',
  category: 'DevOps & AWS',
  difficulty: 'Beginner',
  estimatedTime: '40-50 hours',
  prerequisites: [],
  icon: '🐧',
  keyTakeaways: [
    'Administer production Linux servers on AWS EC2',
    'Write production-ready shell scripts for automation',
    'Monitor and optimize system performance',
    'Manage storage with EBS, EFS, and FSx',
    'Configure and harden SSH access',
    'Implement systemd services with resource limits',
    'Set up centralized logging with CloudWatch',
    'Harden security and pass compliance audits',
    'Design backup and disaster recovery strategies',
    'Debug production issues systematically',
  ],
  learningObjectives: [
    'Understand Linux file systems (ext4, XFS) and inodes',
    'Master process management with systemd',
    'Write robust shell scripts with error handling',
    'Monitor systems with top, htop, iostat, and CloudWatch',
    'Configure storage (EBS, EFS, FSx) for production',
    'Design VPC networking and troubleshoot connectivity',
    'Create production-ready systemd services',
    'Implement log rotation and centralized logging',
    'Secure SSH with key-based authentication and bastions',
    'Harden systems with SELinux, fail2ban, and CIS benchmarks',
    'Manage packages and automate security updates',
    'Implement resource limits with cgroups and ulimits',
    'Design backup strategies with RTO/RPO requirements',
    'Configure NTP and AWS Time Sync Service',
    'Debug issues with strace, tcpdump, and perf',
  ],
  sections: [
    {
      ...linuxFundamentalsProductionSection,
      quiz: linuxFundamentalsProductionDiscussion,
      multipleChoice: linuxFundamentalsProductionMultipleChoice,
    },
    {
      ...shellScriptingAutomationSection,
      quiz: shellScriptingAutomationDiscussion,
      multipleChoice: shellScriptingAutomationMultipleChoice,
    },
    {
      ...systemMonitoringPerformanceSection,
      quiz: systemMonitoringPerformanceDiscussion,
      multipleChoice: systemMonitoringPerformanceMultipleChoice,
    },
    {
      ...storageFileSystemsSection,
      quiz: storageFileSystemsDiscussion,
      multipleChoice: storageFileSystemsMultipleChoice,
    },
    {
      ...networkingBasicsSection,
      quiz: networkingBasicsDiscussion,
      multipleChoice: networkingBasicsMultipleChoice,
    },
    {
      ...systemdServiceManagementSection,
      quiz: systemdServiceManagementDiscussion,
      multipleChoice: systemdServiceManagementMultipleChoice,
    },
    {
      ...logManagementSection,
      quiz: logManagementDiscussion,
      multipleChoice: logManagementMultipleChoice,
    },
    {
      ...sshRemoteAdministrationSection,
      quiz: sshRemoteAdministrationDiscussion,
      multipleChoice: sshRemoteAdministrationMultipleChoice,
    },
    {
      ...securityHardeningSection,
      quiz: securityHardeningDiscussion,
      multipleChoice: securityHardeningMultipleChoice,
    },
    {
      ...packageManagementUpdatesSection,
      quiz: packageManagementUpdatesDiscussion,
      multipleChoice: packageManagementUpdatesMultipleChoice,
    },
    {
      ...processResourceManagementSection,
      quiz: processResourceManagementDiscussion,
      multipleChoice: processResourceManagementMultipleChoice,
    },
    {
      ...backupDisasterRecoverySection,
      quiz: backupDisasterRecoveryDiscussion,
      multipleChoice: backupDisasterRecoveryMultipleChoice,
    },
    {
      ...timeSynchronizationNtpSection,
      quiz: timeSynchronizationNtpDiscussion,
      multipleChoice: timeSynchronizationNtpMultipleChoice,
    },
    {
      ...debuggingProductionIssuesSection,
      quiz: debuggingProductionIssuesDiscussion,
      multipleChoice: debuggingProductionIssuesMultipleChoice,
    },
  ],
};
