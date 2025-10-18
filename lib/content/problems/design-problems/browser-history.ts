/**
 * Design Browser History
 * Problem ID: browser-history
 * Order: 6
 */

import { Problem } from '../../../types';

export const browser_historyProblem: Problem = {
  id: 'browser-history',
  title: 'Design Browser History',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `You have a **browser** of one tab where you start on the \`homepage\` and you can visit another \`url\`, get back in the history number of \`steps\` or move forward in the history number of \`steps\`.

Implement the \`BrowserHistory\` class:

- \`BrowserHistory(string homepage)\` Initializes the object with the \`homepage\` of the browser.
- \`void visit(string url)\` Visits \`url\` from the current page. It clears up all the forward history.
- \`string back(int steps)\` Move \`steps\` back in history. If you can only return \`x\` steps in the history and \`steps > x\`, you will return only \`x\` steps. Return the current \`url\` after moving back in history **at most** \`steps\`.
- \`string forward(int steps)\` Move \`steps\` forward in history. If you can only forward \`x\` steps in the history and \`steps > x\`, you will forward only \`x\` steps. Return the current \`url\` after forwarding in history **at most** \`steps\`.`,
  hints: [
    'Think of browser history as linear timeline with current position',
    'Two stacks: back_stack for history, forward_stack for forward pages',
    'Alternative: array with current_index pointer',
    'visit() must clear forward history (creates new branch)',
  ],
  approach: `## Intuition

Browser maintains **linear history** with current position. We need:
- Go back: Move position backward
- Go forward: Move position forward  
- Visit: Add to history, clear forward

---

## Approach 1: Two Stacks

Use two stacks to track position:
- **back_stack**: Pages behind current
- **forward_stack**: Pages ahead of current
- **current**: Current page

### Example:

\`\`\`
BrowserHistory("home"):
  back=[], current="home", forward=[]

visit("page1"):
  back=["home"], current="page1", forward=[]

visit("page2"):
  back=["home", "page1"], current="page2", forward=[]

back(1):
  Move current to forward, pop from back
  back=["home"], current="page1", forward=["page2"]

back(1):
  back=[], current="home", forward=["page2", "page1"]

forward(1):
  back=["home"], current="page1", forward=["page2"]

visit("page3"):
  CLEARS forward! (new branch)
  back=["home", "page1"], current="page3", forward=[]
\`\`\`

---

## Approach 2: Array with Pointer

Store all history in array, track current index:

\`\`\`python
history = ["home"]
current_idx = 0

visit("page1"):
  # Remove everything after current
  history = history[:current_idx + 1]
  history.append("page1")
  current_idx += 1
  # history = ["home", "page1"], idx = 1

back(1):
  current_idx = max(0, current_idx - 1)
  return history[current_idx]
\`\`\`

**Simpler!** No need to manage two stacks.

---

## Why Clear Forward on Visit?

When you visit new page from middle of history, forward pages become alternate timeline:

\`\`\`
Timeline: A -> B -> C (at B, visit D)
Result:   A -> B -> D (C is lost, new branch)
\`\`\`

Matches real browser behavior.

---

## Time Complexity: O(1) per step for back/forward
## Space Complexity: O(N) where N = total pages visited`,
  testCases: [
    {
      input: [
        ['BrowserHistory', 'home'],
        ['visit', 'page1'],
        ['visit', 'page2'],
        ['back', 1],
        ['back', 1],
        ['forward', 1],
        ['visit', 'page3'],
        ['forward', 2],
        ['back', 2],
        ['back', 7],
      ],
      expected: [
        null,
        null,
        null,
        'page1',
        'home',
        'page1',
        null,
        'page1',
        'home',
        'home',
      ],
    },
  ],
  solution: `# Approach 1: Two Stacks
class BrowserHistory:
    def __init__(self, homepage: str):
        self.back_stack = []
        self.forward_stack = []
        self.current = homepage
    
    def visit(self, url: str) -> None:
        """Visit new URL, clear forward history - O(1)"""
        self.back_stack.append(self.current)
        self.current = url
        self.forward_stack = []  # Clear forward!
    
    def back(self, steps: int) -> str:
        """Go back steps - O(steps)"""
        while steps > 0 and self.back_stack:
            self.forward_stack.append(self.current)
            self.current = self.back_stack.pop()
            steps -= 1
        return self.current
    
    def forward(self, steps: int) -> str:
        """Go forward steps - O(steps)"""
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.current)
            self.current = self.forward_stack.pop()
            steps -= 1
        return self.current


# Approach 2: Array with Pointer (Recommended - Simpler!)
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current_idx = 0
    
    def visit(self, url: str) -> None:
        """Visit new URL - O(1)"""
        # Remove everything after current (clear forward)
        self.history = self.history[:self.current_idx + 1]
        self.history.append(url)
        self.current_idx += 1
    
    def back(self, steps: int) -> str:
        """Go back steps - O(1)"""
        self.current_idx = max(0, self.current_idx - steps)
        return self.history[self.current_idx]
    
    def forward(self, steps: int) -> str:
        """Go forward steps - O(1)"""
        self.current_idx = min(len(self.history) - 1, 
                               self.current_idx + steps)
        return self.history[self.current_idx]

# Example usage:
# browser = BrowserHistory("home")
# browser.visit("page1")  # history=["home", "page1"], idx=1
# browser.visit("page2")  # history=["home", "page1", "page2"], idx=2
# browser.back(1)         # idx=1, returns "page1"
# browser.forward(1)      # idx=2, returns "page2"
# browser.visit("page3")  # history=["home", "page1", "page2", "page3"], idx=3`,
  timeComplexity: 'visit: O(1), back/forward: O(1) per step (or O(steps))',
  spaceComplexity: 'O(N) where N is number of unique pages visited',
  patterns: ['Stack', 'Design', 'Array'],
  companies: ['Google', 'Amazon', 'Facebook'],
};
