const languageServerProtocol = {
  id: 'language-server-protocol',
  title: 'Language Server Protocol (LSP)',
  content: `
# Language Server Protocol (LSP)

## Introduction

The Language Server Protocol (LSP) is the standard that powers IDE features like auto-complete, go-to-definition, hover information, and error checking. Understanding LSP is crucial for building AI coding tools like Cursor that integrate with editors and provide intelligent code assistance.

**Why LSP Matters:**

LSP enables:
- Universal IDE feature support
- Real-time code intelligence
- Cross-editor compatibility
- Scalable language support
- Efficient code analysis
- Standardized tool integration

This section teaches you how LSP works and how to implement it.

## Deep Technical Explanation

### LSP Architecture

**Client-Server Model:**
\`\`\`
┌─────────────┐         ┌──────────────────┐
│   Editor    │◄───────►│  Language Server │
│  (Client)   │   LSP   │   (Server)       │
└─────────────┘         └──────────────────┘
     │                          │
     │                          ├─ Parse code
     │                          ├─ Analyze symbols
     │                          ├─ Check types
     │                          └─ Provide completions
\`\`\`

**Communication:**
- JSON-RPC 2.0 protocol
- Bi-directional messages
- Notifications and requests
- Standard message format

### Core LSP Features

**1. Text Document Synchronization:**
- didOpen: Document opened
- didChange: Document changed
- didClose: Document closed
- didSave: Document saved

**2. Language Features:**
- Completion: Auto-complete suggestions
- Hover: Show information on hover
- Definition: Go-to-definition
- References: Find all references
- Diagnostics: Errors and warnings
- Code Actions: Quick fixes and refactorings
- Formatting: Code formatting

**3. Workspace Features:**
- Symbol search: Find symbols in workspace
- File watching: Track file changes
- Configuration: Server settings

## Code Implementation

### Basic LSP Server

\`\`\`python
from pygls.server import LanguageServer
from pygls.lsp.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    COMPLETION,
    HOVER,
    DEFINITION,
    REFERENCES,
    CompletionItem,
    CompletionList,
    CompletionParams,
    Hover,
    Location,
    Position,
    Range,
    TextDocumentIdentifier,
    DidOpenTextDocumentParams,
    DidChangeTextDocumentParams,
)
import ast

class PythonLanguageServer(LanguageServer):
    """
    Basic Python language server implementing LSP.
    Provides auto-complete, hover, and go-to-definition.
    """
    
    def __init__(self):
        super().__init__()
        self.documents = {}  # Store document content
        self.symbol_tables = {}  # Store symbol information
    
    def parse_document (self, uri: str, text: str):
        """Parse document and build symbol table."""
        try:
            tree = ast.parse (text)
            # Build symbol table (simplified)
            self.symbol_tables[uri] = self._build_symbol_table (tree)
        except SyntaxError:
            # Document has syntax errors
            pass
    
    def _build_symbol_table (self, tree: ast.Module) -> dict:
        """Build symbol table from AST."""
        symbols = {}
        
        for node in ast.walk (tree):
            if isinstance (node, ast.FunctionDef):
                symbols[node.name] = {
                    'type': 'function',
                    'line': node.lineno,
                    'params': [arg.arg for arg in node.args.args]
                }
            elif isinstance (node, ast.ClassDef):
                symbols[node.name] = {
                    'type': 'class',
                    'line': node.lineno
                }
        
        return symbols

# Create server instance
server = PythonLanguageServer()

@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open (ls: PythonLanguageServer, params: DidOpenTextDocumentParams):
    """Handle document open event."""
    uri = params.text_document.uri
    text = params.text_document.text
    
    # Store document
    ls.documents[uri] = text
    
    # Parse and analyze
    ls.parse_document (uri, text)
    
    # Could send diagnostics here
    # ls.publish_diagnostics (uri, diagnostics)

@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change (ls: PythonLanguageServer, params: DidChangeTextDocumentParams):
    """Handle document change event."""
    uri = params.text_document.uri
    
    # Get updated text
    for change in params.content_changes:
        text = change.text
        ls.documents[uri] = text
        ls.parse_document (uri, text)

@server.feature(COMPLETION)
async def completions (ls: PythonLanguageServer, params: CompletionParams):
    """Provide completion suggestions."""
    uri = params.text_document.uri
    position = params.position
    
    # Get symbol table
    symbols = ls.symbol_tables.get (uri, {})
    
    # Create completion items
    items = []
    for name, info in symbols.items():
        item = CompletionItem(
            label=name,
            kind=2 if info['type'] == 'function' else 7,  # 2=Method, 7=Class
            detail=f"{info['type']} {name}"
        )
        items.append (item)
    
    return CompletionList (is_incomplete=False, items=items)

@server.feature(HOVER)
async def hover (ls: PythonLanguageServer, params):
    """Provide hover information."""
    uri = params.text_document.uri
    position = params.position
    
    # Find symbol at position
    # Simplified: just show position
    return Hover(
        contents=f"Line {position.line}, Character {position.character}"
    )

@server.feature(DEFINITION)
async def definition (ls: PythonLanguageServer, params):
    """Provide go-to-definition."""
    uri = params.text_document.uri
    position = params.position
    
    # Find symbol definition
    # Simplified: return empty
    return []

@server.feature(REFERENCES)
async def references (ls: PythonLanguageServer, params):
    """Find all references to symbol."""
    uri = params.text_document.uri
    position = params.position
    
    # Find all references
    # Simplified: return empty
    return []

# To start server:
# server.start_io()
\`\`\`

### LSP Client Integration

\`\`\`python
import json
from typing import Dict, Any

class LSPClient:
    """
    Simple LSP client for testing.
    In practice, editors implement this.
    """
    
    def __init__(self, server_command: list):
        self.server_command = server_command
        self.message_id = 0
        self.process = None
    
    def start (self):
        """Start language server process."""
        import subprocess
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def send_request (self, method: str, params: Dict[str, Any]) -> Dict:
        """Send request to server."""
        self.message_id += 1
        
        message = {
            'jsonrpc': '2.0',
            'id': self.message_id,
            'method': method,
            'params': params
        }
        
        # Send message
        content = json.dumps (message)
        header = f"Content-Length: {len (content)}\\r\\n\\r\\n"
        
        self.process.stdin.write((header + content).encode())
        self.process.stdin.flush()
        
        # Read response (simplified)
        return {}
    
    def send_notification (self, method: str, params: Dict[str, Any]):
        """Send notification (no response expected)."""
        message = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params
        }
        
        content = json.dumps (message)
        header = f"Content-Length: {len (content)}\\r\\n\\r\\n"
        
        self.process.stdin.write((header + content).encode())
        self.process.stdin.flush()
    
    def initialize (self, root_uri: str):
        """Initialize server."""
        return self.send_request('initialize', {
            'processId': None,
            'rootUri': root_uri,
            'capabilities': {}
        })
    
    def did_open (self, uri: str, text: str, language: str = 'python'):
        """Notify server of opened document."""
        self.send_notification('textDocument/didOpen', {
            'textDocument': {
                'uri': uri,
                'languageId': language,
                'version': 1,
                'text': text
            }
        })
    
    def completion (self, uri: str, line: int, character: int):
        """Request completions."""
        return self.send_request('textDocument/completion', {
            'textDocument': {'uri': uri},
            'position': {'line': line, 'character': character}
        })
    
    def hover (self, uri: str, line: int, character: int):
        """Request hover information."""
        return self.send_request('textDocument/hover', {
            'textDocument': {'uri': uri},
            'position': {'line': line, 'character': character}
        })
    
    def shutdown (self):
        """Shutdown server."""
        self.send_request('shutdown', {})
        self.send_notification('exit', {})

# Example usage:
# client = LSPClient(['python', '-m', 'my_language_server'])
# client.start()
# client.initialize('file:///workspace')
# client.did_open('file:///workspace/test.py', 'def hello(): pass')
# completions = client.completion('file:///workspace/test.py', 0, 5)
\`\`\`

### Implementing Diagnostics

\`\`\`python
from pygls.lsp.types import Diagnostic, DiagnosticSeverity, Range, Position

class DiagnosticProvider:
    """
    Provide diagnostics (errors/warnings) for code.
    """
    
    def __init__(self):
        pass
    
    def analyze (self, uri: str, text: str) -> list:
        """
        Analyze code and return diagnostics.
        
        Returns:
            List of Diagnostic objects
        """
        diagnostics = []
        
        try:
            tree = ast.parse (text)
        except SyntaxError as e:
            # Syntax error diagnostic
            diagnostic = Diagnostic(
                range=Range(
                    start=Position (line=e.lineno - 1, character=e.offset or 0),
                    end=Position (line=e.lineno - 1, character=(e.offset or 0) + 1)
                ),
                message=e.msg,
                severity=DiagnosticSeverity.Error,
                source='python-lsp'
            )
            diagnostics.append (diagnostic)
            return diagnostics
        
        # Find other issues
        for node in ast.walk (tree):
            # Check for bare except
            if isinstance (node, ast.ExceptHandler):
                if node.type is None:
                    diagnostic = Diagnostic(
                        range=Range(
                            start=Position (line=node.lineno - 1, character=0),
                            end=Position (line=node.lineno - 1, character=100)
                        ),
                        message="Bare 'except:' clause catches all exceptions",
                        severity=DiagnosticSeverity.Warning,
                        source='python-lsp'
                    )
                    diagnostics.append (diagnostic)
        
        return diagnostics

# Integrate with server:
diagnostic_provider = DiagnosticProvider()

@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open_with_diagnostics (ls: PythonLanguageServer, params):
    uri = params.text_document.uri
    text = params.text_document.text
    
    # Store and parse
    ls.documents[uri] = text
    ls.parse_document (uri, text)
    
    # Get diagnostics
    diagnostics = diagnostic_provider.analyze (uri, text)
    
    # Publish to client
    ls.publish_diagnostics (uri, diagnostics)
\`\`\`

### Implementing Code Actions

\`\`\`python
from pygls.lsp.types import (
    CODE_ACTION,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    TextEdit,
    WorkspaceEdit
)

@server.feature(CODE_ACTION)
async def code_actions (ls: PythonLanguageServer, params: CodeActionParams):
    """
    Provide code actions (quick fixes, refactorings).
    """
    uri = params.text_document.uri
    range_param = params.range
    context = params.context
    
    actions = []
    
    # Example: Add docstring action
    actions.append(CodeAction(
        title="Add docstring",
        kind=CodeActionKind.Refactor,
        edit=WorkspaceEdit(
            changes={
                uri: [
                    TextEdit(
                        range=Range(
                            start=Position (line=1, character=0),
                            end=Position (line=1, character=0)
                        ),
                        new_text='    """TODO: Add docstring."""\\n'
                    )
                ]
            }
        )
    ))
    
    # Example: Fix issues from diagnostics
    for diagnostic in context.diagnostics:
        if 'Bare except' in diagnostic.message:
            actions.append(CodeAction(
                title="Replace with 'except Exception:'",
                kind=CodeActionKind.QuickFix,
                edit=WorkspaceEdit(
                    changes={
                        uri: [
                            TextEdit(
                                range=diagnostic.range,
                                new_text='except Exception:'
                            )
                        ]
                    }
                ),
                diagnostics=[diagnostic]
            ))
    
    return actions
\`\`\`

## Real-World Case Study: How Cursor Uses LSP

Cursor integrates with LSP servers for intelligent features:

**1. Multi-Language Support:**
\`\`\`
Python: Uses python-lsp-server (pylsp)
TypeScript: Uses typescript-language-server
Go: Uses gopls
Rust: Uses rust-analyzer

All through standard LSP interface!
\`\`\`

**2. Real-Time Analysis:**
\`\`\`python
# As you type, LSP server:
def calculate (x, y):
    result = x + y  # ← Hover shows types
    return result   # ← Diagnostics check
# ← Auto-complete for 'result.'
\`\`\`

**3. Intelligent Refactoring:**
\`\`\`python
def process_data (data):  # ← Right-click
    # Shows code actions:
    # - Extract method
    # - Rename symbol
    # - Add type hints
    # All via LSP!
\`\`\`

**4. Workspace Intelligence:**
\`\`\`python
# Cursor can:
# - Find all usages (via LSP references)
# - Rename across files (via LSP rename)
# - Search symbols (via LSP workspace symbols)
# - Track dependencies (via LSP)
\`\`\`

## Common Pitfalls

### 1. Not Handling Partial Updates

\`\`\`python
# ❌ Wrong: Re-parse entire file on every keystroke
@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change (params):
    text = get_entire_document()  # Expensive!
    parse_everything (text)

# ✅ Correct: Incremental updates
@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change (params):
    for change in params.content_changes:
        apply_incremental_change (change)
    reparse_only_affected_regions()
\`\`\`

### 2. Blocking Operations

\`\`\`python
# ❌ Wrong: Slow synchronous operations
@server.feature(COMPLETION)
def completions (params):
    results = expensive_operation()  # Blocks!
    return results

# ✅ Correct: Fast, async operations
@server.feature(COMPLETION)
async def completions (params):
    results = await fast_async_operation()
    return results
\`\`\`

### 3. Poor Error Handling

\`\`\`python
# ❌ Wrong: Crash on invalid input
@server.feature(HOVER)
def hover (params):
    return get_hover_info (params)  # May crash

# ✅ Correct: Graceful error handling
@server.feature(HOVER)
def hover (params):
    try:
        return get_hover_info (params)
    except Exception as e:
        logger.error (f"Hover failed: {e}")
        return None  # Return null, don't crash
\`\`\`

## Production Checklist

### Server Implementation
- [ ] Handle all document lifecycle events
- [ ] Implement core features (completion, hover, definition)
- [ ] Provide diagnostics in real-time
- [ ] Support code actions
- [ ] Handle errors gracefully
- [ ] Use async operations

### Performance
- [ ] Incremental parsing
- [ ] Cache analysis results
- [ ] Fast response times (<100ms)
- [ ] Handle large files efficiently
- [ ] Support cancellation
- [ ] Profile bottlenecks

### Features
- [ ] Auto-completion
- [ ] Hover information
- [ ] Go-to-definition
- [ ] Find references
- [ ] Rename symbol
- [ ] Code formatting
- [ ] Diagnostics (errors/warnings)
- [ ] Code actions (quick fixes)

### Robustness
- [ ] Handle syntax errors
- [ ] Recover from crashes
- [ ] Log errors appropriately
- [ ] Validate messages
- [ ] Handle edge cases
- [ ] Test with real editors

### Integration
- [ ] Follow LSP specification
- [ ] Support multiple editors
- [ ] Configuration options
- [ ] Documentation
- [ ] Version compatibility

## Summary

Language Server Protocol enables universal IDE features:

- **Standard Protocol**: JSON-RPC based communication
- **Core Features**: Completion, hover, definition, references
- **Real-Time Analysis**: Fast, incremental updates
- **Editor Independent**: Works with VS Code, Vim, Emacs, etc.
- **Production Ready**: Used by major tools

Understanding LSP is essential for building AI coding tools that integrate seamlessly with editors. It provides the infrastructure for real-time code intelligence that makes tools like Cursor possible—enabling features like auto-complete, go-to-definition, and refactoring across any programming language.

In the final section, we'll bring everything together to build a complete code understanding engine that combines all the techniques we've learned.
`,
};

export default languageServerProtocol;
