/**
 * Quiz questions for Multi-File Code Generation section
 */

export const multifilecodegenerationQuiz = [
  {
    id: 'bcgs-multifile-q-1',
    question:
      'Design a strategy for determining which files need to be modified when implementing a new feature that spans multiple files. How would you minimize unnecessary changes while ensuring all required updates are made?',
    hint: 'Think about dependency analysis, impact analysis, and change propagation.',
    sampleAnswer:
      '**Multi-File Change Strategy:** **1) Dependency Analysis** - Build import graph showing which files depend on which. Use AST parsing to find: Direct imports (from X import Y), Indirect dependencies (X imports Y, Y imports Z), Type dependencies (functions expecting specific types). **2) Impact Analysis** - Starting from changed file, trace: Files that import it (need updates if interface changes), Files it imports (might need usage updates), Shared base classes/interfaces. **3) Change Classification** - **Required Changes:** Interface changes (function signature modified → all callers need updates), New dependencies (new import → add to dependent files), Type changes (User → DetailedUser → all type hints update). **Optional Changes:** Style consistency updates, Documentation updates, Test updates. **4) Minimization Rules** - Only modify files with breaking changes, Use backward-compatible changes when possible (add parameter with default vs changing signature), Update tests only for actually changed behavior. **5) Verification** - After identifying files: Check each has necessary changes only, Verify no files missed (run tests), Look for over-eager changes (modified file unnecessarily). **Example:** Adding "email_verified" field to User model → **Must change:** user model, database migration, user creation, serialization **Should change:** tests **No change needed:** files that just pass User around.',
    keyPoints: [
      'Build dependency graph to trace impact of changes',
      'Required: interface changes, new dependencies, type changes',
      'Minimize by using backward-compatible changes',
      'Verify no files missed but avoid over-eager modifications',
    ],
  },
  {
    id: 'bcgs-multifile-q-2',
    question:
      "Explain transactional application of multi-file changes. What makes this challenging, how would you implement it, and what's your rollback strategy if changes partially fail?",
    hint: 'Consider file system atomicity, validation timing, and state consistency.',
    sampleAnswer:
      "**Transactional Multi-File Changes:** **Challenges:** 1) File system isn't transactional by default, 2) Can't rollback after partial writes without backups, 3) Files depend on each other - inconsistent state if one fails, 4) Validation might catch errors only after applying changes. **Implementation:** **Phase 1: Preparation** - Create backups of all files to be modified, Validate each change individually (syntax, imports), Check cross-file consistency (new imports resolve, types match). **Phase 2: Atomic Application** ```python\nclass FileTransaction:\n    def __init__(self):\n        self.backups = {}\n        self.modified_files = []\n    \n    def modify_file(self, path, new_content):\n        # Backup original\n        self.backups[path] = read_file(path)\n        # Write new content\n        write_file(path, new_content)\n        self.modified_files.append(path)\n    \n    def commit(self):\n        # Validate all files together\n        if self.validate_all():\n            self.backups.clear()\n            return True\n        else:\n            self.rollback()\n            return False\n    \n    def rollback(self):\n        for path, content in self.backups.items():\n            write_file(path, content)\n``` **Phase 3: Validation** - Import all modified modules (catch import errors), Run tests (catch logic errors), Type check (catch type mismatches). **Rollback Strategy:** Keep complete backups until validation passes, On any failure: Restore all files from backups, Report which file/check failed, Preserve error info for retry. **Safety:** Write to temp files first, Validate, then atomic move to actual location.",
    keyPoints: [
      'Create backups before any modifications',
      'Apply all changes, then validate as a unit',
      'Complete rollback on any validation failure',
      'Use temp files + atomic moves for extra safety',
    ],
  },
  {
    id: 'bcgs-multifile-q-3',
    question:
      'When generating changes across multiple files, how would you ensure import statements are updated correctly? Design a system that handles both adding new imports and removing unused ones.',
    hint: 'Consider detecting required imports, avoiding duplicates, and organizing properly.',
    sampleAnswer:
      '**Import Management System:** **1) Detect Required Imports** - Parse generated code to find: Undefined names (NameError candidates), Type hints needing imports (List, Dict, Optional, custom types), Decorators needing imports. **2) Resolve Import Paths** - For each undefined name: Check if defined in current file (no import needed), Search project for definition: ```python\\ndef find_definition(name, project_root):\\n    for file in project_files:\\n        if name in get_exported_names(file):\\n            return module_path_from_file(file)``` Map to import statement: "User" in "app/models/user.py" → "from app.models.user import User" **3) Insert Imports Correctly** - Parse existing imports, Determine insertion point (after docstring, in appropriate group), Group by type: stdlib, third-party, local, Add alphabetically within group. **4) Remove Unused Imports** - After modification, scan file: Build set of all names used, Compare to imported names, Remove imports not in used set, Be careful with wildcard imports. **5) Avoid Duplicates** - Before adding import, check if already imported, Check if imported from different module (conflict!), Handle aliases (import X as Y). **Algorithm:** ```python\\ndef update_imports(file_path, new_code):\\n    existing = parse_imports(file_path)\\n    required = detect_required_imports(new_code)\\n    \\n    to_add = required - existing\\n    \\n    for imp in to_add:\\n        if not conflicts_with_existing(imp, existing):\\n            insert_import(file_path, imp)\\n    \\n    remove_unused_imports(file_path)\\n``` **Example:** Generated code uses "User" and "datetime". Check: User defined in app/models/user.py → add "from app.models.user import User", datetime is stdlib → add "import datetime".',
    keyPoints: [
      'Parse generated code to detect undefined names',
      'Resolve names to import paths by searching project',
      'Insert imports in correct groups (stdlib/third-party/local)',
      'Remove unused imports after modifications',
    ],
  },
];
