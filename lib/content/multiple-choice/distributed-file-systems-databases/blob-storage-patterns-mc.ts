/**
 * Multiple choice questions for Blob Storage Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const blobStorageMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When should you use multipart upload in S3?',
    options: [
      'All files regardless of size',
      'Files smaller than 1 MB',
      'Files larger than 100 MB (required for > 5 GB)',
      'Only for images',
    ],
    correctAnswer: 2,
    explanation:
      'Multipart upload is required for files larger than 5 GB (single PUT limit). It is recommended for files larger than 100 MB for better performance, resilience, and ability to resume failed uploads. Multipart enables parallel uploads and retry of individual parts.',
  },
  {
    id: 'mc2',
    question: 'What is the primary benefit of the presigned URL pattern?',
    options: [
      'Encrypt data automatically',
      'Compress files before upload',
      'Allow clients to upload directly to S3, offloading backend',
      'Scan for viruses',
    ],
    correctAnswer: 2,
    explanation:
      "Presigned URLs allow clients to upload directly to S3 without going through your backend servers. This offloads bandwidth and processing from your backend, enables better scalability, reduces costs, and leverages S3's global infrastructure. Backend only generates the presigned URL (lightweight operation).",
  },
  {
    id: 'mc3',
    question: 'How does content deduplication save storage in blob storage?',
    options: [
      'By compressing all files',
      'By deleting old files automatically',
      'By storing unique files once and using references for duplicates',
      'By reducing file resolution',
    ],
    correctAnswer: 2,
    explanation:
      'Content deduplication calculates hash (SHA-256) of files and stores each unique file only once. Multiple users uploading the same file share a single blob copy. The system maintains references (user_id → blob_key mapping). This can save 50-75% storage for file sharing platforms like Dropbox.',
  },
  {
    id: 'mc4',
    question: 'What is the purpose of S3 lifecycle policies?',
    options: [
      'Monitor website traffic',
      'Automatically transition objects to cheaper storage classes or delete them',
      'Replicate data to other regions',
      'Compress objects',
    ],
    correctAnswer: 1,
    explanation:
      'S3 lifecycle policies automate cost optimization by transitioning objects to cheaper storage classes (e.g., Standard → Standard-IA → Glacier after X days) or deleting objects after expiration. This eliminates manual management and significantly reduces storage costs for infrequently accessed data.',
  },
  {
    id: 'mc5',
    question: 'Why integrate blob storage with a CDN for static content?',
    options: [
      'To increase storage capacity',
      'To cache content at edge locations for faster global delivery',
      'To backup data automatically',
      'To encrypt data',
    ],
    correctAnswer: 1,
    explanation:
      'Integrating blob storage (S3) with CDN (CloudFront) caches static content at edge locations worldwide. Users download from nearest edge location (low latency) instead of origin S3 bucket (high latency). Cache hit = fast delivery without S3 request. Reduces latency, bandwidth costs, and load on origin.',
  },
];
