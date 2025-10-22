/**
 * Quiz questions for Symbol Resolution & References section
 */

export const symbolresolutionQuiz = [
    {
        id: 'cuam-symbolresolution-q-1',
        question:
            'How does symbol resolution enable the "Rename Symbol" refactoring in IDEs? What edge cases must be handled to avoid breaking code?',
        hint: 'Consider scoped names, shadowing, imported symbols, and attribute access.',
        sampleAnswer:
            "Symbol resolution enables rename by: 1) **Finding the definition** - resolve the symbol to its declaration point, 2) **Finding all references** - use cross-reference table to find every usage, 3) **Respecting scope** - only rename within the symbol's scope, not shadowed names in inner scopes, 4) **Updating systematically** - change definition + all references atomically. **Edge cases**: 1) **Shadowing** - don't rename `x` in global scope if inner function has its own `x`, 2) **Import aliases** - renaming `import pandas as pd` must update all `pd.` references, 3) **String references** - dynamic access like `getattr(obj, 'method_name')` requires string updates too, 4) **Cross-file** - must track imports/exports across files, 5) **Attributes** - renaming `self.name` requires understanding it's an attribute, not a variable. Without proper symbol resolution, rename could: skip references in other scopes, incorrectly rename shadowed variables, or miss dynamically constructed names.",
        keyPoints: [
            'Resolve symbol to definition, find all references',
            'Respect scope boundaries and shadowing',
            'Handle imports, attributes, and cross-file usage',
            'Edge cases: dynamic access, string references',
        ],
    },
    {
        id: 'cuam-symbolresolution-q-2',
        question:
            'Explain how import resolution allows Cursor to provide accurate auto-complete for external libraries. What happens when you type "np." after "import numpy as np"?',
        hint: 'Think about alias tracking, module introspection, and symbol tables.',
        sampleAnswer:
            "Import resolution tracks: 1) **Alias mapping**: `import numpy as np` → {alias: 'np', module: 'numpy'}, 2) When you type 'np.', Cursor: resolves 'np' to 'numpy' module via import table, 3) **Introspects module** (or uses cached symbol table) to find available attributes, 4) **Filters by type** - shows functions, classes, constants from numpy, 5) **Provides documentation** from docstrings. Without resolution, typing 'np.' means nothing - just text. With resolution: knows np=numpy, loads numpy's public API (~600 functions), suggests np.array, np.sum, etc. with type hints and descriptions. For local imports (`from .utils import helper`), resolution maps relative imports to actual files, extracting symbols. This is how Cursor knows what's available without re-analyzing libraries constantly - maintains symbol tables per import, resolves aliases, provides context-aware suggestions. Import resolution is the bridge between local code and external modules.",
        keyPoints: [
            'Maps import aliases to actual modules',
            'Introspects or caches module symbol tables',
            'Resolves relative imports to file paths',
            'Enables accurate cross-module completions',
        ],
    },
    {
        id: 'cuam-symbolresolution-q-3',
        question:
            'Why is attribute resolution more complex than simple name resolution? How do you resolve "user.profile.avatar.url" in Python?',
        hint: 'Consider object types, class hierarchies, and dynamic attributes.',
        sampleAnswer:
            "Attribute resolution is complex because it requires **knowing types**, not just names. For `user.profile.avatar.url`: 1) Resolve 'user' to its definition, infer or look up its type (e.g., User class), 2) Find 'profile' attribute in User class (check __init__, annotations, class attributes), determine profile's type (Profile class), 3) Find 'avatar' attribute in Profile class, determine type (Avatar class), 4) Find 'url' attribute in Avatar class, determine type (probably str). **Challenges**: 1) **Type inference** - if types aren't annotated, must infer from assignments/returns, 2) **Inheritance** - attribute might be in parent class, requires walking MRO, 3) **Dynamic attributes** - `__getattr__` can create attributes on-the-fly, impossible to resolve statically, 4) **Conditional attributes** - set in some code paths but not others. Partial solution: use type hints where available, infer from common patterns, accept that Python's dynamism means some attributes can't be resolved. Cursor prioritizes common cases (90% coverage) and marks uncertain cases.",
        keyPoints: [
            'Requires type information at each step',
            'Must walk class hierarchies for inherited attributes',
            'Dynamic attributes challenge static analysis',
            'Type hints essential for accurate resolution',
        ],
    },
];
