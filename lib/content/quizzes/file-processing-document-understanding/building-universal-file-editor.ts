/**
 * Quiz questions for Building a Universal File Editor section
 */

export const buildinguniversalfileeditorQuiz = [
    {
        id: 'fpdu-editor-q-1',
        question: 'Design the complete architecture for a Cursor-like universal file editor. Include file detection, processing, editing, validation, and LLM integration.',
        hint: 'Consider modularity, safety, extensibility, and production requirements.',
        sampleAnswer: 'Universal editor architecture: (1) File Type Detection: magic number + extension based routing. (2) Processor Registry: modular processors for each file type (text, Python, Excel, PDF). (3) Read Layer: format-specific reading with structure extraction. (4) Edit Layer: unified edit interface (FileEdit objects), validation before application, atomic writes with temp files. (5) Safety System: automatic backups, rollback on failure, change validation. (6) LLM Integration: file summarization for context, edit generation from natural language, diff preview before applying. (7) Monitoring: logging all operations, error tracking, performance metrics. (8) Extension System: plugin architecture for new file types. Production requirements: concurrent access handling, large file support, memory efficiency.',
        keyPoints: ['Modular processors per file type', 'Unified edit interface', 'Atomic writes with backups', 'LLM integration for AI editing', 'Validation and rollback', 'Plugin architecture', 'Production-grade error handling'],
    },
    {
        id: 'fpdu-editor-q-2',
        question: 'How would you ensure data safety when implementing file modifications? Design a complete safety system with backup, validation, and rollback.',
        hint: 'Think about backup strategies, validation points, atomic operations, and recovery.',
        sampleAnswer: 'File safety system: (1) Pre-modification: validate file exists, check permissions, verify file is not corrupted. (2) Backup: create timestamped backup before any change, store in separate .backups directory, keep N most recent backups. (3) Atomic writes: write to temp file first, validate temp file, atomic rename to target. (4) Validation: check file opens correctly, validate structure for typed files (Python AST, Excel schema), compare checksums. (5) Rollback: on any error, restore from backup immediately, log failure reason. (6) Monitoring: track all modifications, alert on repeated failures, regular backup cleanup. (7) User confirmation: show diff preview, require explicit approval for large changes. Similar to how Cursor handles code modifications safely.',
        keyPoints: ['Create backup before modification', 'Atomic writes via temp files', 'Validate before and after', 'Immediate rollback on failure', 'Show diff preview to user', 'Track all modifications', 'Test recovery procedures'],
    },
    {
        id: 'fpdu-editor-q-3',
        question: 'Explain how you would integrate LLM capabilities into a universal file editor. What natural language commands would you support and how would you implement them?',
        hint: 'Consider command parsing, context generation, edit generation, and validation.',
        sampleAnswer: 'LLM integration: (1) Commands: "Add function to calculate X", "Refactor class Y", "Fix formatting", "Add column Z to Excel". (2) Context Generation: provide file type, structure summary, relevant code, nearby context. (3) Command Parsing: use LLM to extract intent, target location, desired changes. (4) Edit Generation: LLM generates diff or new content, validate syntax, check for breaking changes. (5) Preview: show proposed changes with diff highlighting, user approves before applying. (6) Application: use file editor to apply validated edits atomically. (7) Feedback Loop: learn from user corrections. Implementation: maintain conversation context, cache file embeddings, incremental reanalysis on changes. Similar to Cursor\'s Cmd+K functionality.',
        keyPoints: ['Parse natural language commands', 'Generate context from file structure', 'LLM generates diffs', 'Preview changes before applying', 'Validate generated edits', 'Learn from user feedback', 'Maintain conversation context'],
    },
];

