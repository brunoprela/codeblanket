/**
 * Multiple choice questions for PII Detection & Removal section
 */

export const piidetectionremovalMultipleChoice = [
  {
    id: 'pii-det-mc-1',
    question:
      'Your PII detector uses regex to detect emails. It catches "john@example.com" but misses "john[at]example.com". What is the issue?',
    options: [
      'The regex pattern is too strict',
      'Users are intentionally obfuscating emails to bypass detection',
      'The detector needs NLP to understand context',
      'Email validation is not being applied',
    ],
    correctAnswer: 1,
    explanation:
      'Users often obfuscate emails with [at], (at), or " at " to bypass filters. While regex can be updated to catch these, it\'s an ongoing cat-and-mouse game. Context-aware detection (C) helps, but B is the root cause. Option A is backwards—regex is too loose, not too strict.',
  },
  {
    id: 'pii-det-mc-2',
    question:
      'A user requests deletion under GDPR Article 17. You delete their data from the database but backups still contain it. Is this compliant?',
    options: [
      'Yes—GDPR allows data in backups',
      'No—must delete from backups or make inaccessible',
      'Yes—backups are exempt from GDPR',
      'No—must immediately destroy all backups',
    ],
    correctAnswer: 1,
    explanation:
      'GDPR requires deletion from backups OR making the data inaccessible (e.g., encrypt with keys you then delete). Option D (destroy all backups) is extreme and unnecessary. Option A is incorrect—backups are not automatically exempt. You must either delete from backups or ensure data cannot be restored.',
  },
  {
    id: 'pii-det-mc-3',
    question:
      'Your PII detector flags 20% of requests. Manual review shows 10% have real PII, 10% are false positives. What is the precision?',
    options: [
      '90% (10% false positives / 100% total)',
      '50% (10% real PII / 20% flagged)',
      '80% (100% - 20% flagged)',
      '10% (real PII rate)',
    ],
    correctAnswer: 1,
    explanation:
      'Precision = True Positives / (True Positives + False Positives) = 10% / (10% + 10%) = 10% / 20% = 50%. Half of flagged items are false positives. Option A calculates false positive rate incorrectly. Options C and D are not precision calculations.',
  },
  {
    id: 'pii-det-mc-4',
    question:
      'You use Luhn algorithm to validate credit card numbers. What is its PRIMARY purpose?',
    options: [
      'Encrypt credit card numbers for secure storage',
      'Check if a number is a mathematically valid credit card number',
      'Detect stolen credit cards',
      'Generate new credit card numbers',
    ],
    correctAnswer: 1,
    explanation:
      'Luhn algorithm checks if a number passes the checksum validation that all valid credit card numbers must pass. This reduces false positives (random 16-digit numbers flagged as cards). It does NOT encrypt (A), detect fraud (C), or generate cards (D).',
  },
  {
    id: 'pii-det-mc-5',
    question:
      'Your system pseudonymizes user IDs: user_123 becomes anonymous_abc. Is this reversible?',
    options: [
      'Yes—you can reverse the hash to get original ID',
      'No—if using a one-way hash (SHA-256), it cannot be reversed',
      'Yes—all hashes can be reversed with enough computing power',
      'No—anonymization always destroys the original data',
    ],
    correctAnswer: 1,
    explanation:
      'One-way cryptographic hashes (SHA-256, BLAKE2) cannot be reversed. However, you can look up the original if you store the mapping. True anonymization destroys the link. Pseudonymization retains a way to link back. Option C is wrong—secure hashes resist brute force. Option D confuses anonymization with pseudonymization.',
  },
];
