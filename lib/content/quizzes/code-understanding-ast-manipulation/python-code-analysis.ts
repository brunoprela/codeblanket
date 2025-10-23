/**
 * Quiz questions for Python Code Analysis with AST section
 */

export const pythoncodeanalysisQuiz = [
  {
    id: 'cuam-pythoncodeanalysis-q-1',
    question:
      'How would you design a system to track all class instance attributes (self.x = y) across a Python codebase? What AST nodes and traversal patterns would you use?',
    hint: 'Think about ast.Attribute nodes, context, and the __init__ method.',
    sampleAnswer:
      "To track instance attributes: 1) Visit all **ClassDef** nodes to identify classes, 2) Within each class, find the **__init__ method** (FunctionDef with name='__init__'), 3) Walk the __init__ body looking for **ast.Assign and ast.AnnAssign** nodes, 4) Check if the target is an **ast.Attribute** node where value is ast.Name with id='self', 5) Extract the attribute name from node.attr, 6) Store mapping of {class_name: [attribute_names]}. Also check other methods for attribute assignments. The key is recognizing the pattern: node.target is Attribute, node.target.value is Name('self'), node.target.attr is the attribute name. This allows Cursor to suggest self.attributes when typing 'self.' in a method. Extra: track type annotations from ast.AnnAssign for better type inference.",
    keyPoints: [
      'Visit ClassDef nodes, find __init__ methods',
      'Look for Assign/AnnAssign with Attribute targets',
      'Check if target.value is Name(id="self")',
      'Extract attribute name from target.attr',
    ],
  },
  {
    id: 'cuam-pythoncodeanalysis-q-2',
    question:
      'Why is scope analysis essential for accurate symbol resolution? Give an example where incorrect scope handling leads to wrong results.',
    hint: 'Consider variable shadowing and the LEGB rule (Local, Enclosing, Global, Built-in).',
    sampleAnswer:
      "Scope analysis is essential because **the same name can refer to different things** in different scopes. Without proper scope tracking, you can't determine which 'x' a reference points to. Example of failure: ```python\nx = 1  # Global\ndef outer():\n    x = 2  # Outer scope\n    def inner():\n        x = 3  # Inner scope\n        print(x)  # Which x?\n```\nWithout scope analysis, you might always resolve 'x' to the global (wrong!). Correct resolution requires following Python's LEGB rule: check Local scope first (inner x=3), then Enclosing (outer x=2), then Global (module x=1), then Built-ins. Each function creates a new scope, and you must maintain a scope stack/chain to resolve names correctly. This is why Cursor knows what 'user' refers to in nested functions - it walks the scope chain from innermost to outermost until it finds the definition.",
    keyPoints: [
      'Same name can exist in multiple scopes',
      'Must follow LEGB resolution order',
      'Track scope stack during traversal',
      'Critical for accurate go-to-definition',
    ],
  },
  {
    id: 'cuam-pythoncodeanalysis-q-3',
    question:
      'How would you build a call graph analyzer that handles both regular function calls and method calls (obj.method())? What challenges arise with dynamic Python code?',
    hint: "Consider ast.Call, ast.Name vs ast.Attribute, and Python's dynamic nature.",
    sampleAnswer:
      "Build a call graph by: 1) Track all function/method definitions in symbol table, 2) Visit ast.Call nodes to find calls, 3) For **ast.Name** func (direct calls like foo()), extract func.id as the callee, 4) For **ast.Attribute** func (method calls like obj.method()), extract func.attr as method name, ideally resolve obj's type to find which class the method belongs to. Store edges: (caller_function, callee_function). **Challenges with dynamic Python**: 1) Can't resolve dynamic calls like globals()[name]() - requires runtime info, 2) Method calls depend on object type which may be unknown statically, 3) Higher-order functions (passing functions as args) create indirect calls, 4) eval(), exec(), __getattr__ make static analysis incomplete. Solution: use type inference where possible, mark unresolved calls, and accept that Python's dynamism means some calls can't be determined statically. Focus on common patterns (90% of real code) and flag dynamic cases for human review.",
    keyPoints: [
      'Handle ast.Name (direct) and ast.Attribute (method) calls',
      'Resolve types for accurate method resolution',
      'Dynamic calls (eval, getattr) need special handling',
      'Accept limitations of static analysis on dynamic code',
    ],
  },
];
