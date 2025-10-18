# Content Structure Refactoring

## Overview

We've refactored the content structure to use a compositional model that makes it easier to work with and maintain the dataset.

## New Structure

```
lib/content/
├── topics/           # Top-level topic definitions
│   ├── python.ts
│   └── index.ts
├── modules/          # Module definitions that compose sections
│   ├── python-fundamentals.ts
│   └── index.ts
├── sections/         # Individual section content
│   └── python-fundamentals/
│       ├── variables-types.ts
│       ├── control-flow.ts
│       ├── ...
│       └── index.ts
├── quizzes/          # Quiz questions organized by module
│   └── python-fundamentals/
│       ├── variables-types.ts
│       ├── control-flow.ts
│       ├── ...
│       └── index.ts
├── multiple-choice/  # Multiple choice questions organized by module
│   └── python-fundamentals/
│       ├── variables-types.ts
│       ├── control-flow.ts
│       ├── ...
│       └── index.ts
└── problems/         # Problems organized by module (to be migrated)
```

## Benefits

1. **Separation of Concerns**: Each type of content (sections, quizzes, multiple-choice) is in its own folder
2. **Easy to Navigate**: Clear hierarchy makes it easy to find specific content
3. **Reusable Components**: Sections, quizzes, and multiple-choice can be reused across modules
4. **Better Collaboration**: Multiple people can work on different sections without merge conflicts
5. **Easier Testing**: Individual components can be tested in isolation
6. **Scalability**: Easy to add new content types or modules

## Migration Status

### Completed

- ✅ Python Fundamentals (prototype/example)
  - 10 sections extracted
  - 10 quiz sets extracted
  - 10 multiple-choice sets extracted
  - Module composition file created
  - Index files created
  - Main modules index updated

### Pending

- Python Intermediate
- Python OOP
- Python Advanced
- All DSA modules (arrays-hashing, binary-search, etc.)
- System Design modules
- ML/AI modules

## How to Use

### Importing a Module

```typescript
import { pythonFundamentalsModule } from '@/lib/content/modules/python-fundamentals';
```

### Importing Individual Sections

```typescript
import { variablestypesSection } from '@/lib/content/sections/python-fundamentals/variables-types';
```

### Importing All Sections for a Module

```typescript
import * as pythonFundamentalsSections from '@/lib/content/sections/python-fundamentals';
```

## Extraction Tool

Use `extract-sections.js` to extract content from existing monolithic module files:

```bash
node extract-sections.js
```

The script will:

1. Parse the module file
2. Extract each section with its content
3. Extract quiz questions
4. Extract multiple-choice questions
5. Create individual files for each component
6. Organize them in the new structure

## Next Steps

1. Run the extraction script for remaining modules
2. Update imports in the main modules index
3. Verify all modules work correctly
4. Remove old monolithic module files
5. Update documentation and examples
