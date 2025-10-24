/**
 * Multiple choice questions for Dropbox Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dropboxarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What is the average chunk size Dropbox uses for file deduplication?',
        options: [
            '512 KB with fixed-size chunking',
            '1 MB with boundary detection',
            '4 MB with variable-size chunking (1-16 MB range)',
            '16 MB with content-defined boundaries',
        ],
        correctAnswer: 2,
        explanation: 'Dropbox uses variable-size chunking with an average of 4 MB and a range of 1-16 MB. Chunk boundaries are determined using Rabin fingerprinting (content-defined chunking), which ensures that small edits don\'t shift all subsequent chunk boundaries. Each chunk is hashed with SHA-256 for global deduplication. This enables 95%+ bandwidth savings for common files and efficient incremental uploads.',
    },
    {
        id: 'mc2',
        question: 'What was Dropbox\'s primary motivation for migrating from AWS to Magic Pocket (own datacenters)?',
        options: [
            'AWS had reliability issues affecting Dropbox availability',
            'Cost savings (30-40%) and operational control',
            'AWS lacked required compliance certifications',
            'Needed custom hardware for AI/ML workloads',
        ],
        correctAnswer: 1,
        explanation: 'Dropbox migrated from AWS to Magic Pocket (own datacenters) primarily for cost savings (30-40%) and operational control. Storing exabytes on AWS S3 cost $75M+/year. Building custom datacenters with high-density storage nodes and erasure coding allowed Dropbox to amortize costs over years and optimize for their specific workload. The migration (2014-2016) also improved performance by co-locating compute and storage.',
    },
    {
        id: 'mc3',
        question: 'How does Dropbox handle conflicting edits to the same file from multiple devices?',
        options: [
            'Operational transform like Google Docs to auto-merge changes',
            'Lock-based concurrency preventing simultaneous edits',
            'Last-write-wins with conflict marker files for the losing version',
            'Version control system with manual merge UI',
        ],
        correctAnswer: 2,
        explanation: 'Dropbox uses last-write-wins with conflict markers. When concurrent edits occur, the first upload to sync becomes the canonical version. The conflicting version is renamed to "file (conflicted copy YYYY-MM-DD).txt" and synced to all devices. Users manually resolve conflicts. Unlike Google Docs, Dropbox syncs arbitrary file types (can\'t auto-merge binaries), so operational transform isn\'t feasible. Version history (30 days) provides a fallback for recovery.',
    },
    {
        id: 'mc4',
        question: 'What hashing algorithm does Dropbox use for chunk deduplication?',
        options: [
            'MD5 for speed despite collision risks',
            'SHA-1 with collision detection',
            'SHA-256 for cryptographic security',
            'xxHash for performance',
        ],
        correctAnswer: 2,
        explanation: 'Dropbox uses SHA-256 for chunk hashing and deduplication. SHA-256 provides cryptographic security against collision attacks, ensuring that two different chunks won\'t have the same hash. This is critical for deduplication correctness—serving the wrong chunk due to a collision would corrupt files. While SHA-256 is slower than MD5 or xxHash, the security benefits outweigh the performance cost for Dropbox\'s use case.',
    },
    {
        id: 'mc5',
        question: 'How long does Dropbox retain file version history for free users?',
        options: [
            '7 days of version history',
            '30 days of version history',
            '90 days of version history',
            'Unlimited version history',
        ],
        correctAnswer: 1,
        explanation: 'Dropbox retains 30 days of version history for free users. Users can restore previous versions of files within this window. This provides a safety net for accidental deletions or unwanted edits. Paid plans offer extended version history (180 days or unlimited). Version history is stored in the same deduplicated chunk format, minimizing storage overhead since most chunks are shared across versions.',
    },
];

