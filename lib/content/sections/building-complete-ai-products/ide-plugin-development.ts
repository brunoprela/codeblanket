export const idePluginDevelopment = {
  title: 'IDE Plugin Development & Extensions',
  id: 'ide-plugin-development',
  content: `
# IDE Plugin Development & Extensions

## Introduction

Building standalone AI applications is valuable, but **integrating AI directly into developers' IDEs** is transformative. This section covers how to build IDE plugins and extensions that bring AI assistance into VSCode, JetBrains IDEs, and other development environments.

We'll build a complete Cursor-like VSCode extension that provides:
- Real-time code suggestions
- Natural language code editing
- Chat interface within the IDE
- File context awareness
- Custom commands and keybindings

### Why IDE Plugins?

**User Adoption**: Developers spend 6-8 hours/day in their IDE. Integrating AI where they already work removes friction.

**Seamless Experience**: No context switching between IDE and separate AI tool.

**Deep Integration**: Access to IDE features like debugging, version control, terminal.

**Distribution**: VSCode Marketplace has 50M+ developers, instant distribution channel.

### VSCode Extension Architecture

\`\`\`
┌──────────────────────────────────────────────────────┐
│             VSCode Extension                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐        ┌──────────────┐           │
│  │ Extension   │        │  Language    │           │
│  │ Host        │◄──────►│  Server      │           │
│  │ (Node.js)   │        │  Protocol    │           │
│  └─────┬───────┘        └──────────────┘           │
│        │                                            │
│        │                                            │
│  ┌─────▼──────────────────────────────────┐        │
│  │         VSCode Extension API            │        │
│  ├─────────────────────────────────────────┤        │
│  │  • TextEditor                           │        │
│  │  • Commands                             │        │
│  │  • TreeView                             │        │
│  │  • WebView                              │        │
│  │  • Language Features                    │        │
│  │  • Workspace                            │        │
│  └─────┬───────────────────────────────────┘        │
│        │                                            │
│        ▼                                            │
│  ┌─────────────────┐                               │
│  │  Backend API    │                               │
│  │  (Python/Node)  │                               │
│  └─────────────────┘                               │
└──────────────────────────────────────────────────────┘
\`\`\`

---

## VSCode Extension Basics

### Project Structure

\`\`\`
cursor-ai-extension/
├── package.json          # Extension manifest
├── tsconfig.json         # TypeScript config
├── .vscodeignore        # Files to exclude from package
├── src/
│   ├── extension.ts     # Main entry point
│   ├── commands/
│   │   ├── editCode.ts
│   │   ├── chatAI.ts
│   │   └── explainCode.ts
│   ├── providers/
│   │   ├── completionProvider.ts
│   │   ├── hoverProvider.ts
│   │   └── codeActionProvider.ts
│   ├── views/
│   │   ├── chatView.ts
│   │   └── historyView.ts
│   ├── services/
│   │   ├── aiService.ts
│   │   └── contextManager.ts
│   └── webviews/
│       └── chatPanel.html
└── README.md
\`\`\`

### package.json - Extension Manifest

\`\`\`json
{
  "name": "cursor-ai-clone",
  "displayName": "Cursor AI Clone",
  "description": "AI-powered code editor with natural language understanding",
  "version": "0.0.1",
  "publisher": "your-name",
  "engines": {
    "vscode": "^1.80.0"
  },
  "categories": [
    "Programming Languages",
    "Machine Learning",
    "Other"
  ],
  "activationEvents": [
    "onStartupFinished"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "cursorAI.editCode",
        "title": "Cursor AI: Edit Code with AI"
      },
      {
        "command": "cursorAI.chatAI",
        "title": "Cursor AI: Chat with AI"
      },
      {
        "command": "cursorAI.explainCode",
        "title": "Cursor AI: Explain Selected Code"
      },
      {
        "command": "cursorAI.generateTests",
        "title": "Cursor AI: Generate Tests"
      }
    ],
    "keybindings": [
      {
        "command": "cursorAI.editCode",
        "key": "ctrl+k",
        "mac": "cmd+k",
        "when": "editorTextFocus"
      },
      {
        "command": "cursorAI.chatAI",
        "key": "ctrl+l",
        "mac": "cmd+l"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "cursorAI.editCode",
          "group": "cursorAI@1",
          "when": "editorHasSelection"
        },
        {
          "command": "cursorAI.explainCode",
          "group": "cursorAI@2",
          "when": "editorHasSelection"
        }
      ]
    },
    "viewsContainers": {
      "activitybar": [
        {
          "id": "cursorAI",
          "title": "Cursor AI",
          "icon": "resources/icon.svg"
        }
      ]
    },
    "views": {
      "cursorAI": [
        {
          "id": "cursorAI.chatView",
          "name": "AI Chat"
        },
        {
          "id": "cursorAI.historyView",
          "name": "Edit History"
        }
      ]
    },
    "configuration": {
      "title": "Cursor AI",
      "properties": {
        "cursorAI.apiKey": {
          "type": "string",
          "default": "",
          "description": "OpenAI API Key"
        },
        "cursorAI.model": {
          "type": "string",
          "enum": ["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus"],
          "default": "gpt-4-turbo",
          "description": "LLM Model to use"
        },
        "cursorAI.maxTokens": {
          "type": "number",
          "default": 8000,
          "description": "Maximum tokens for context"
        }
      }
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "package": "vsce package"
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0",
    "@types/node": "^16.x",
    "typescript": "^5.0.0",
    "@vscode/vsce": "^2.19.0"
  },
  "dependencies": {
    "openai": "^4.20.0",
    "axios": "^1.6.0"
  }
}
\`\`\`

---

## Extension Entry Point

\`\`\`typescript
/**
 * Main extension entry point
 * src/extension.ts
 */

import * as vscode from 'vscode';
import { EditCodeCommand } from './commands/editCode';
import { ChatAICommand } from './commands/chatAI';
import { ExplainCodeCommand } from './commands/explainCode';
import { CompletionProvider } from './providers/completionProvider';
import { AIService } from './services/aiService';
import { ContextManager } from './services/contextManager';
import { ChatViewProvider } from './views/chatView';

// Extension state
let aiService: AIService;
let contextManager: ContextManager;

export function activate(context: vscode.ExtensionContext) {
    console.log('Cursor AI extension is now active');

    // Initialize services
    const config = vscode.workspace.getConfiguration('cursorAI');
    const apiKey = config.get<string>('apiKey');
    
    if (!apiKey) {
        vscode.window.showWarningMessage(
            'Cursor AI: Please set your API key in settings'
        );
    }

    aiService = new AIService(apiKey || ', config);
    contextManager = new ContextManager();

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'cursorAI.editCode',
            () => new EditCodeCommand(aiService, contextManager).execute()
        )
    );

    context.subscriptions.push(
        vscode.commands.registerCommand(
            'cursorAI.chatAI',
            () => new ChatAICommand(aiService, contextManager).execute()
        )
    );

    context.subscriptions.push(
        vscode.commands.registerCommand(
            'cursorAI.explainCode',
            () => new ExplainCodeCommand(aiService, contextManager).execute()
        )
    );

    // Register completion provider
    const completionProvider = new CompletionProvider(aiService, contextManager);
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            { scheme: 'file' },
            completionProvider,
            '.' // Trigger on dot
        )
    );

    // Register tree view for chat
    const chatViewProvider = new ChatViewProvider(
        context.extensionUri,
        aiService,
        contextManager
    );
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'cursorAI.chatView',
            chatViewProvider
        )
    );

    // Listen for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('cursorAI')) {
                const newConfig = vscode.workspace.getConfiguration('cursorAI');
                aiService.updateConfig(newConfig);
            }
        })
    );

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = '$(robot) Cursor AI';
    statusBarItem.command = 'cursorAI.chatAI';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
}

export function deactivate() {
    console.log('Cursor AI extension is now deactivated');
}
\`\`\`

---

## Commands Implementation

### Edit Code Command

\`\`\`typescript
/**
 * Edit Code with AI command
 * src/commands/editCode.ts
 */

import * as vscode from 'vscode';
import { AIService } from '../services/aiService';
import { ContextManager } from '../services/contextManager';

export class EditCodeCommand {
    constructor(
        private aiService: AIService,
        private contextManager: ContextManager
    ) {}

    async execute() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        // Get selection or entire document
        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select code to edit');
            return;
        }

        // Get instruction from user
        const instruction = await vscode.window.showInputBox({
            prompt: 'What would you like to do with this code?',
            placeHolder: 'e.g., Add error handling, Add type hints, Refactor to use async',
            ignoreFocusOut: true
        });

        if (!instruction) {
            return;
        }

        // Show progress
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Cursor AI: Editing code...',
            cancellable: true
        }, async (progress, token) => {
            try {
                // Get context
                const context = await this.contextManager.getContext(
                    editor.document,
                    selection.start
                );

                // Call AI service
                const result = await this.aiService.editCode({
                    code: selectedText,
                    instruction,
                    context,
                    filePath: editor.document.fileName,
                    language: editor.document.languageId
                });

                if (token.isCancellationRequested) {
                    return;
                }

                // Show diff
                const accepted = await this.showDiff(
                    editor.document,
                    selection,
                    result.newCode
                );

                if (accepted) {
                    // Apply changes
                    await editor.edit(editBuilder => {
                        editBuilder.replace(selection, result.newCode);
                    });

                    vscode.window.showInformationMessage('✓ Code updated');
                }

            } catch (error: any) {
                vscode.window.showErrorMessage(
                    \`Cursor AI: \${error.message}\`
                );
            }
        });
    }

    private async showDiff(
        document: vscode.TextDocument,
        selection: vscode.Selection,
        newCode: string
    ): Promise<boolean> {
        // Create temporary document with new code
        const tempDoc = await vscode.workspace.openTextDocument({
            content: newCode,
            language: document.languageId
        });

        // Show diff
        await vscode.commands.executeCommand(
            'vscode.diff',
            document.uri,
            tempDoc.uri,
            'Current ↔ AI Suggestion'
        );

        // Ask user to accept
        const choice = await vscode.window.showInformationMessage(
            'Accept AI changes?',
            'Accept',
            'Reject'
        );

        return choice === 'Accept';
    }
}
\`\`\`

### Chat Command

\`\`\`typescript
/**
 * Chat with AI command
 * src/commands/chatAI.ts
 */

import * as vscode from 'vscode';
import { AIService } from '../services/aiService';
import { ContextManager } from '../services/contextManager';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export class ChatAICommand {
    private conversationHistory: ChatMessage[] = [];

    constructor(
        private aiService: AIService,
        private contextManager: ContextManager
    ) {}

    async execute() {
        // Create webview panel for chat
        const panel = vscode.window.createWebviewPanel(
            'cursorAIChat',
            'Cursor AI Chat',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        // Set HTML content
        panel.webview.html = this.getWebviewContent();

        // Handle messages from webview
        panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'sendMessage':
                        await this.handleChatMessage(
                            panel.webview,
                            message.text
                        );
                        break;
                    
                    case 'clearHistory':
                        this.conversationHistory = [];
                        panel.webview.postMessage({
                            command: 'clearChat'
                        });
                        break;
                }
            }
        );
    }

    private async handleChatMessage(
        webview: vscode.Webview,
        userMessage: string
    ) {
        // Add user message to history
        this.conversationHistory.push({
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        });

        // Show user message in chat
        webview.postMessage({
            command: 'addMessage',
            role: 'user',
            content: userMessage
        });

        // Get current editor context
        const editor = vscode.window.activeTextEditor;
        let context: string = ';
        
        if (editor) {
            context = await this.contextManager.getContext(
                editor.document,
                editor.selection.start
            );
        }

        // Stream AI response
        try {
            let fullResponse = ';

            await this.aiService.chat({
                message: userMessage,
                context,
                history: this.conversationHistory.map(m => ({
                    role: m.role,
                    content: m.content
                })),
                onStream: (chunk: string) => {
                    fullResponse += chunk;
                    webview.postMessage({
                        command: 'streamMessage',
                        content: chunk
                    });
                }
            });

            // Add assistant response to history
            this.conversationHistory.push({
                role: 'assistant',
                content: fullResponse,
                timestamp: new Date()
            });

            // Finalize message
            webview.postMessage({
                command: 'finalizeMessage'
            });

        } catch (error: any) {
            webview.postMessage({
                command: 'error',
                message: error.message
            });
        }
    }

    private getWebviewContent(): string {
        return \`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cursor AI Chat</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 10px;
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        #chat-container {
            max-width: 800px;
            margin: 0 auto;
        }
        #messages {
            height: calc(100vh - 150px);
            overflow-y: auto;
            padding: 10px;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: var(--vscode-input-background);
            text-align: right;
        }
        .assistant-message {
            background-color: var(--vscode-editor-inactiveSelectionBackground);
        }
        .timestamp {
            font-size: 0.8em;
            color: var(--vscode-descriptionForeground);
            margin-top: 5px;
        }
        #input-container {
            display: flex;
            gap: 10px;
        }
        #message-input {
            flex: 1;
            padding: 10px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 3px;
        }
        button {
            padding: 10px 20px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }
        code {
            font-family: var(--vscode-editor-font-family);
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-container">
            <textarea 
                id="message-input" 
                placeholder="Ask about your code..."
                rows="3"
            ></textarea>
            <button id="send-button">Send</button>
        </div>
        <button id="clear-button" style="margin-top: 10px;">Clear History</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const inputField = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-button');

        let currentMessage = null;

        sendButton.addEventListener('click', sendMessage);
        clearButton.addEventListener('click', () => {
            vscode.postMessage({ command: 'clearHistory' });
        });

        inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                sendMessage();
            }
        });

        function sendMessage() {
            const text = inputField.value.trim();
            if (!text) return;

            vscode.postMessage({
                command: 'sendMessage',
                text: text
            });

            inputField.value = ';
        }

        window.addEventListener('message', event => {
            const message = event.data;

            switch (message.command) {
                case 'addMessage':
                    addMessage(message.role, message.content);
                    break;
                
                case 'streamMessage':
                    streamMessage(message.content);
                    break;
                
                case 'finalizeMessage':
                    currentMessage = null;
                    break;
                
                case 'clearChat':
                    messagesDiv.innerHTML = ';
                    break;
                
                case 'error':
                    addMessage('assistant', \`Error: \${message.message}\`);
                    break;
            }
        });

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${role}-message\`;
            
            const contentDiv = document.createElement('div');
            contentDiv.innerHTML = formatContent(content);
            
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'timestamp';
            timestampDiv.textContent = new Date().toLocaleTimeString();
            
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(timestampDiv);
            messagesDiv.appendChild(messageDiv);
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            if (role === 'assistant') {
                currentMessage = contentDiv;
            }
        }

        function streamMessage(chunk) {
            if (!currentMessage) {
                addMessage('assistant', ');
            }
            
            currentMessage.innerHTML += formatContent(chunk);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function formatContent(content) {
            // Simple markdown-style formatting
            content = content.replace(/\`\`\`(\\w+)?\\n([\\s\\S]*?)\\n\`\`\`/g, 
                '<pre><code>$2</code></pre>');
            content = content.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
            content = content.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
            return content;
        }
    </script>
</body>
</html>\`;
    }
}
\`\`\`

---

## Language Server Protocol

For advanced features like hover information and diagnostics:

\`\`\`typescript
/**
 * Language Server Protocol integration
 * src/providers/hoverProvider.ts
 */

import * as vscode from 'vscode';
import { AIService } from '../services/aiService';

export class AIHoverProvider implements vscode.HoverProvider {
    constructor(private aiService: AIService) {}

    async provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.Hover | undefined> {
        // Get word at position
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) {
            return undefined;
        }

        const word = document.getText(wordRange);
        
        // Get surrounding code for context
        const lineStart = Math.max(0, position.line - 5);
        const lineEnd = Math.min(document.lineCount, position.line + 5);
        const contextRange = new vscode.Range(lineStart, 0, lineEnd, 0);
        const context = document.getText(contextRange);

        try {
            // Ask AI to explain the symbol
            const explanation = await this.aiService.explainSymbol({
                symbol: word,
                context,
                language: document.languageId
            });

            const markdown = new vscode.MarkdownString();
            markdown.appendMarkdown(\`### \${word}\\n\\n\`);
            markdown.appendMarkdown(explanation);
            markdown.isTrusted = true;

            return new vscode.Hover(markdown, wordRange);

        } catch (error) {
            return undefined;
        }
    }
}
\`\`\`

---

## Publishing to Marketplace

### Package Extension

\`\`\`bash
# Install vsce (VS Code Extension Manager)
npm install -g @vscode/vsce

# Package extension
vsce package

# This creates: cursor-ai-clone-0.0.1.vsix
\`\`\`

### Publish to Marketplace

\`\`\`bash
# Create publisher account at:
# https://marketplace.visualstudio.com/manage

# Login
vsce login <publisher-name>

# Publish
vsce publish
\`\`\`

---

## JetBrains Plugin Development

For IntelliJ, PyCharm, WebStorm, etc:

\`\`\`kotlin
/**
 * JetBrains Plugin (Kotlin/Java)
 * src/main/kotlin/com/cursor/plugin/CursorAIPlugin.kt
 */

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.ui.Messages

class EditCodeAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val project = e.project ?: return
        
        // Get selected text
        val selectionModel = editor.selectionModel
        val selectedText = selectionModel.selectedText ?: return
        
        // Get instruction from user
        val instruction = Messages.showInputDialog(
            project,
            "What would you like to do with this code?",
            "Cursor AI",
            null
        ) ?: return
        
        // Call AI service (async)
        AIService.editCode(selectedText, instruction) { newCode ->
            // Apply changes
            WriteCommandAction.runWriteCommandAction(project) {
                val document = editor.document
                val start = selectionModel.selectionStart
                val end = selectionModel.selectionEnd
                document.replaceString(start, end, newCode)
            }
        }
    }
}
\`\`\`

---

## Conclusion

Key takeaways for IDE plugin development:

1. **Understand Extension API**: VSCode/JetBrains APIs are well-documented
2. **User Experience**: Seamless integration is critical
3. **Performance**: Keep UI responsive, run AI calls async
4. **Configuration**: Let users customize model, API keys
5. **Distribution**: Marketplace makes reach easy

Building IDE plugins transforms your AI application from a separate tool into an integrated development experience.
`,
};
