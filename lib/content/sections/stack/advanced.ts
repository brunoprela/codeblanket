/**
 * Advanced Techniques Section
 */

export const advancedSection = {
  id: 'advanced',
  title: 'Advanced Techniques',
  content: `**Technique 1: Two-Stack Expression Evaluation**

Use separate stacks for operands and operators:
\`\`\`python
def evaluate_expression (expression: str) -> int:
    def precedence (op):
        if op in '+-': return 1
        if op in '*/': return 2
        return 0
    
    def apply_op (a, b, op):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a // b
    
    operands = []
    operators = []
    
    i = 0
    while i < len (expression):
        if expression[i].isdigit():
            num = 0
            while i < len (expression) and expression[i].isdigit():
                num = num * 10 + int (expression[i])
                i += 1
            operands.append (num)
            continue
        
        if expression[i] == '(':
            operators.append (expression[i])
        elif expression[i] == ')':
            while operators[-1] != '(':
                b = operands.pop()
                a = operands.pop()
                op = operators.pop()
                operands.append (apply_op (a, b, op))
            operators.pop()  # Remove '('
        elif expression[i] in '+-*/':
            while (operators and operators[-1] != '(' and
                   precedence (operators[-1]) >= precedence (expression[i])):
                b = operands.pop()
                a = operands.pop()
                op = operators.pop()
                operands.append (apply_op (a, b, op))
            operators.append (expression[i])
        
        i += 1
    
    while operators:
        b = operands.pop()
        a = operands.pop()
        op = operators.pop()
        operands.append (apply_op (a, b, op))
    
    return operands[0]
\`\`\`

**Technique 2: Stack for DFS (Iterative)**

Replace recursion with an explicit stack:
\`\`\`python
def dfs_iterative (graph, start):
    """
    Depth-first search using stack instead of recursion.
    """
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        
        visited.add (node)
        print(node)  # Process node
        
        # Add neighbors to stack (reverse order for same traversal as recursive)
        for neighbor in reversed (graph[node]):
            if neighbor not in visited:
                stack.append (neighbor)
\`\`\`

**Technique 3: Stack for Backtracking**

Use stack to track decision points:
\`\`\`python
def generate_parentheses (n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    """
    result = []
    stack = [(', 0, 0)]  # (current_string, open_count, close_count)
    
    while stack:
        s, open_count, close_count = stack.pop()
        
        if len (s) == 2 * n:
            result.append (s)
            continue
        
        if open_count < n:
            stack.append((s + '(', open_count + 1, close_count))
        if close_count < open_count:
            stack.append((s + ')', open_count, close_count + 1))
    
    return result
\`\`\``,
};
