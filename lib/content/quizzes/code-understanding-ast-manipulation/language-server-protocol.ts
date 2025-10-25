/**
 * Quiz questions for Language Server Protocol section
 */

export const languageserverprotocolQuiz = [
  {
    id: 'cuam-languageserverprotocol-q-1',
    question:
      "Explain the client-server architecture of LSP and why it's better than having each editor implement language support directly.",
    hint: 'Think about N editors × M languages and the benefits of separation.',
    sampleAnswer:
      "LSP uses **client-server architecture**: Editor (client) communicates with Language Server (server) via JSON-RPC. **Without LSP**: Each editor implements support for each language → N editors × M languages = N×M implementations. VS Code for Python, Vim for Python, Emacs for Python - all separate! **With LSP**: Each language needs ONE server, each editor needs ONE client → N + M implementations. **Benefits**: 1) **Language authors** - write server once, works in all editors, 2) **Editor authors** - implement LSP client once, get all languages, 3) **Feature parity** - all editors get same features for each language, 4) **Maintenance** - fix bug in server, all editors benefit. Example: Python\'s pylsp server works in VS Code, Vim, Emacs, Sublime. This is why Cursor has excellent multi-language support - leverages existing LSP servers (pylsp, typescript-language-server, rust-analyzer) instead of reimplementing everything. LSP is the standard that made modern IDE features universal.",
    keyPoints: [
      'Reduces N×M to N+M implementations',
      'Language servers work across all editors',
      'Consistent features everywhere',
      'Easier maintenance and improvement',
    ],
  },
  {
    id: 'cuam-languageserverprotocol-q-2',
    question:
      'Why is incremental document synchronization critical for LSP performance? What would happen if the entire document was re-sent on every keystroke?',
    hint: 'Consider file sizes, parsing time, and network overhead.',
    sampleAnswer:
      "Incremental sync sends only **changes**, not entire document. **Full sync problems**: 1) **Network overhead** - sending 10,000 line file on every keystroke is slow, especially remote servers, 2) **Parsing cost** - re-parsing entire file on each change wastes CPU, 3) **Memory churn** - constantly allocating/deallocating large documents, 4) **Latency** - delays before features update. **Incremental sync**: Send only what changed - 'at line 42, insert \"x\"'. Server applies change, re-parses affected region only. Benefits: 1) **Fast updates** - typically <5ms, 2) **Low bandwidth** - changes are tiny (bytes vs kilobytes), 3) **Efficient parsing** - tree-sitter's incremental parsing only re-parses changed nodes, 4) **Responsive IDE** - features update in real-time. For Cursor: editing 10,000-line file feels instant because only changed lines are sent/re-analyzed. This is essential for modern IDE UX - sub-100ms response to any edit. Without incremental sync, typing would be laggy on large files - unacceptable for professional tools.",
    keyPoints: [
      'Sends only changes, not entire document',
      'Reduces network overhead dramatically',
      'Enables incremental parsing (faster)',
      'Critical for real-time IDE responsiveness',
    ],
  },
  {
    id: 'cuam-languageserverprotocol-q-3',
    question:
      'How do LSP diagnostics differ from code actions? Give examples of when you would use each.',
    hint: 'Think about reporting problems vs offering solutions.',
    sampleAnswer:
      "**Diagnostics** = **reporting problems** (errors, warnings). Sent from server to client automatically, shown as squiggles/indicators. Examples: 1) Syntax error: 'Expected \")\" on line 42' (Error), 2) Unused variable: 'x is never used' (Warning), 3) Deprecated API: 'Method foo() is deprecated' (Info). **Code Actions** = **offering solutions** to problems. Triggered by user (right-click, lightbulb), provide quick fixes or refactorings. Examples: 1) Quick fix for diagnostic: 'unused variable' → action: 'Remove variable', 2) Refactoring: 'Extract method', 'Rename symbol', 3) Code improvements: 'Convert to list comprehension'. **Key difference**: Diagnostics are passive (showing issues), Code Actions are active (doing something). Workflow: Server sends diagnostic about problem → User requests code actions at that location → Server returns possible fixes → User picks one → Edit applied. For Cursor: diagnostics show red squiggles, code actions show 'Fix' suggestions. Diagnostics identify issues, Code Actions resolve them. Both essential for complete IDE experience.",
    keyPoints: [
      'Diagnostics: report problems (passive)',
      'Code Actions: offer solutions (active)',
      'Diagnostics trigger automatically',
      'Code Actions triggered by user',
    ],
  },
];
