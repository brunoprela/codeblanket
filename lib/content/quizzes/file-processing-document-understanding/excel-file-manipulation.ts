/**
 * Quiz questions for Excel File Manipulation section
 */

export const excelfilemanipulationQuiz = [
    {
        id: 'fpdu-excel-manip-q-1',
        question:
            'You are building a Cursor-like tool for Excel that allows users to say "add a column calculating profit margin." Explain the complete architecture and implementation approach, including how you would parse the intent, modify the Excel file, and present the result.',
        hint: 'Consider LLM parsing, Excel manipulation libraries, formula generation, and user confirmation.',
        sampleAnswer:
            'Complete architecture: (1) Parse user intent with LLM - extract action (add column), column name (profit margin), and formula logic (Revenue - Cost / Revenue). (2) Load Excel file using openpyxl to preserve formatting, or pandas for data operations. (3) Validate: check if required columns (Revenue, Cost) exist in the file. (4) Generate formula: either pandas calculation (df["Profit Margin"] = (df["Revenue"] - df["Cost"]) / df["Revenue"]) or Excel formula (=((Revenue_col - Cost_col) / Revenue_col) * 100). (5) Add column to DataFrame/workbook with proper formatting. (6) Show preview to user before saving. (7) Save atomically with backup. (8) Return summary of changes. Key considerations: handle missing values, divide-by-zero errors, preserve existing formatting, support undo operation.',
        keyPoints: [
            'Use LLM to parse natural language into structured operations',
            'Validate that required columns exist before modification',
            'Choose openpyxl for formatting preservation, pandas for data ops',
            'Generate either pandas calculations or Excel formulas',
            'Show preview before committing changes',
            'Create backup before modifying original file',
            'Handle edge cases: nulls, divide-by-zero, data type mismatches',
        ],
    },
    {
        id: 'fpdu-excel-manip-q-2',
        question:
            'Compare openpyxl, pandas, and xlwings for Excel manipulation. When would you choose each library, and how would you combine them in a production system?',
        hint: 'Consider formatting preservation, performance, cross-platform support, and use cases.',
        sampleAnswer:
            'Choose based on needs: (1) openpyxl: Use when you need to preserve formatting, charts, formulas. Best for modifying existing Excel files without changing their appearance. Pure Python, cross-platform. Slower for large datasets. (2) pandas: Use for data analysis, transformation, bulk operations. Excellent performance with large data. Limited formatting control. Best for ETL pipelines. (3) xlwings: Use for complex Excel automation, running macros, real-time Excel control. Requires Excel installed (not cross-platform). Best for Windows-based automation. Production strategy: Use pandas for data processing (fast), then openpyxl to write with formatting. Read with pandas for analysis, use openpyxl to apply business-specific formatting. For web applications, avoid xlwings (requires Excel). Combine: read with pandas for speed, manipulate data, write with openpyxl for formatting.',
        keyPoints: [
            'openpyxl: Formatting preservation, cross-platform, slower',
            'pandas: Fast data operations, limited formatting, best for analysis',
            'xlwings: Full Excel control, requires Excel, Windows-focused',
            'Combine: pandas for data, openpyxl for formatting',
            'Web apps: use openpyxl/pandas only (cross-platform)',
            'Test file size limits - pandas better for large files',
        ],
    },
    {
        id: 'fpdu-excel-manip-q-3',
        question:
            'Design a safe file modification system for Excel files that supports atomic operations, backup/restore, and rollback. How would you implement this for a production application?',
        hint: 'Think about transaction-like behavior, temporary files, validation, and error recovery.',
        sampleAnswer:
            'Safe modification system design: (1) Pre-modification validation: check file exists, validate format, check file size, ensure no corruption. (2) Create timestamped backup: copy original to .bak or backup folder with timestamp. (3) Load and validate: read file, check structure matches expectations. (4) Apply modifications in memory: make all changes to DataFrame/workbook object, validate results. (5) Write to temporary file: save modifications to .tmp file in same directory. (6) Atomic rename: move .tmp to original (atomic operation on most filesystems). (7) On success: optionally delete backup or keep for history. (8) On failure: restore from backup, log error, return detailed failure message. Additional features: Change tracking - log what was modified for undo, checksum validation to detect corruption, file locking to prevent concurrent modifications, version history (keep multiple backups). Similar to how Cursor saves code files safely.',
        keyPoints: [
            'Always create timestamped backup before modification',
            'Write to temporary file first, then atomic rename',
            'Validate data before and after modifications',
            'Keep backups for rollback capability',
            'Log all changes for audit trail and undo',
            'Handle concurrent access with file locking',
            'Test error recovery paths thoroughly',
        ],
    },
];

