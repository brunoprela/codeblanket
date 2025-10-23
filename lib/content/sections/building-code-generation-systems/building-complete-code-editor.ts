/**
 * Building a Complete Code Editor Section
 * Module 5: Building Code Generation Systems
 */

export const buildingcompletecodeeditorSection = {
  id: 'building-complete-code-editor',
  title: 'Building a Complete Code Editor',
  content: `# Building a Complete Code Editor

Bring it all together - build a production-ready code editor like Cursor that users will love.

## Overview: Complete System Architecture

A production code editor needs:
- **File Management**: Read, write, watch files
- **Code Analysis**: Parse and understand code
- **Generation Engine**: Generate and edit code
- **Validation**: Test and validate changes
- **User Interface**: Interactive editing experience
- **Version Control**: Track changes, undo/redo
- **Multi-File Support**: Edit across files
- **Safety**: Sandboxing and validation

### Project Structure

Let's build a complete system step by step.

## Core Architecture

### Main Application

\`\`\`python
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

@dataclass
class EditorConfig:
    """Configuration for the code editor."""
    project_root: str
    max_file_size: int = 1_000_000  # 1MB
    supported_languages: List[str] = None
    auto_save: bool = True
    enable_sandbox: bool = True
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["python", "javascript", "typescript"]

class CodeEditorApp:
    """Complete code editor application."""
    
    def __init__(self, config: EditorConfig):
        self.config = config
        self.project_root = Path(config.project_root)
        
        # Initialize components
        self.analyzer = ProjectAnalyzer(config.project_root)
        self.editor = InteractiveCodeEditor()
        self.validator = CodeValidator()
        self.file_watcher = FileWatcher(config.project_root)
        self.conversation_manager = ConversationManager()
        
        # Active sessions
        self.sessions: Dict[str, EditSession] = {}
    
    def start(self):
        """Start the code editor."""
        print(f"Code Editor started for: {self.project_root}")
        print(f"Supported languages: {', '.join(self.config.supported_languages)}")
        
        # Start file watcher
        self.file_watcher.start()
        
        # Run main loop
        self.run()
    
    def run(self):
        """Main editor loop."""
        while True:
            try:
                command = input("\\n> ").strip()
                
                if not command:
                    continue
                
                if command == "exit":
                    break
                elif command.startswith("edit "):
                    file_path = command[5:]
                    self.edit_file(file_path)
                elif command.startswith("new "):
                    file_path = command[4:]
                    self.create_file(file_path)
                elif command == "list":
                    self.list_files()
                elif command == "help":
                    self.show_help()
                else:
                    print(f"Unknown command: {command}")
            
            except KeyboardInterrupt:
                print("\\n\\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def edit_file(self, file_path: str):
        """Start editing a file interactively."""
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            print(f"File not found: {file_path}")
            return
        
        # Read file
        try:
            content = full_path.read_text()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        # Start editing session
        session = self.editor.start_editing(file_path, content)
        self.sessions[file_path] = session
        
        print(f"\\nEditing: {file_path}")
        print("Type your requests (or 'done' to finish)\\n")
        
        # Interactive editing loop
        while True:
            user_request = input(f"{file_path}> ").strip()
            
            if user_request == "done":
                break
            elif user_request == "show":
                print(session.current_content)
            elif user_request == "diff":
                if session.pending_edits:
                    diff = self.editor.show_diff(session)
                    print(diff)
                else:
                    print("No pending changes")
            elif user_request == "accept":
                if session.pending_edits:
                    success = self.editor.accept_edits(session)
                    if success:
                        print("✓ Changes applied")
                        # Save file
                        self.save_file(file_path, session.current_content)
                    else:
                        print("✗ Failed to apply changes")
                else:
                    print("No pending changes")
            elif user_request == "reject":
                self.editor.reject_edits(session)
                print("✗ Changes rejected")
            elif user_request == "undo":
                success = self.editor.undo(session)
                if success:
                    print("✓ Undone")
                else:
                    print("Nothing to undo")
            else:
                # Process as editing request
                response = self.editor.process_request(session, user_request)
                print(f"\\nAI: {response}\\n")
    
    def create_file(self, file_path: str):
        """Create a new file with AI assistance."""
        full_path = self.project_root / file_path
        
        if full_path.exists():
            print(f"File already exists: {file_path}")
            return
        
        description = input("Describe the file to create: ")
        
        # Detect language
        language = self._detect_language_from_path(file_path)
        
        # Generate file
        generator = MultiLanguageGenerator()
        content = generator.generate(language, description, "file")
        
        # Show preview
        print("\\nGenerated content:")
        print(content)
        print()
        
        # Confirm
        confirm = input("Create this file? (yes/no): ")
        if confirm.lower() == "yes":
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            print(f"✓ Created: {file_path}")
    
    def save_file(self, file_path: str, content: str):
        """Save file with backup."""
        full_path = self.project_root / file_path
        
        # Create backup
        if full_path.exists():
            backup_path = full_path.with_suffix(full_path.suffix + ".bak")
            backup_path.write_text(full_path.read_text())
        
        # Save
        full_path.write_text(content)
        print(f"Saved: {file_path}")
    
    def list_files(self):
        """List all files in project."""
        print("\\nProject files:")
        for py_file in self.project_root.rglob("*.py"):
            relative = py_file.relative_to(self.project_root)
            print(f"  {relative}")
    
    def show_help(self):
        """Show help message."""
        print("""
Commands:
  edit <file>  - Edit a file interactively
  new <file>   - Create a new file
  list         - List all files
  help         - Show this help
  exit         - Exit the editor

During editing:
  <request>    - Make an editing request
  show         - Show current file content
  diff         - Show pending changes
  accept       - Accept and apply changes
  reject       - Reject changes
  undo         - Undo last change
  done         - Finish editing
""")
    
    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix
        
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".rs": "rust",
            ".cpp": "cpp",
            ".go": "go"
        }
        
        return mapping.get(ext, "python")

# Usage
config = EditorConfig(
    project_root="/path/to/project",
    auto_save=True,
    enable_sandbox=True
)

app = CodeEditorApp(config)
app.start()

# Example interaction:
# > edit app/routes.py
# Editing: app/routes.py
# Type your requests (or 'done' to finish)
#
# app/routes.py> add error handling to the login function
# AI: I'll add comprehensive error handling...
#
# app/routes.py> diff
# [shows diff]
#
# app/routes.py> accept
# ✓ Changes applied
# Saved: app/routes.py
#
# app/routes.py> done
\`\`\`

## File Watching

### Real-Time File Monitoring

\`\`\`python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class CodeFileHandler(FileSystemEventHandler):
    """Handle file system events."""
    
    def __init__(self, callback):
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory:
            self.callback("modified", event.src_path)
    
    def on_created(self, event):
        if not event.is_directory:
            self.callback("created", event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.callback("deleted", event.src_path)

class FileWatcher:
    """Watch files for changes."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.observer = None
    
    def start(self):
        """Start watching files."""
        def handle_event(event_type, path):
            print(f"\\n[File {event_type}: {path}]")
        
        event_handler = CodeFileHandler(handle_event)
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            str(self.root_path),
            recursive=True
        )
        self.observer.start()
    
    def stop(self):
        """Stop watching files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

# Usage with editor
# The file watcher runs in background and notifies when files change
\`\`\`

## Web Interface (Optional)

### Flask-Based Web UI

\`\`\`python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize editor
config = EditorConfig(project_root="/path/to/project")
editor_app = CodeEditorApp(config)

@app.route("/")
def index():
    """Serve the editor UI."""
    return render_template("editor.html")

@app.route("/api/files", methods=["GET"])
def list_files():
    """List all files in project."""
    files = []
    for py_file in Path(config.project_root).rglob("*.py"):
        relative = str(py_file.relative_to(config.project_root))
        files.append(relative)
    return jsonify(files)

@app.route("/api/file/<path:file_path>", methods=["GET"])
def get_file(file_path):
    """Get file content."""
    full_path = Path(config.project_root) / file_path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    content = full_path.read_text()
    return jsonify({"content": content})

@app.route("/api/edit", methods=["POST"])
def edit_file():
    """Process an edit request."""
    data = request.json
    file_path = data["file_path"]
    user_request = data["request"]
    
    # Get or create session
    if file_path not in editor_app.sessions:
        full_path = Path(config.project_root) / file_path
        content = full_path.read_text()
        session = editor_app.editor.start_editing(file_path, content)
        editor_app.sessions[file_path] = session
    else:
        session = editor_app.sessions[file_path]
    
    # Process request
    response = editor_app.editor.process_request(session, user_request)
    
    # Return response and pending edits
    return jsonify({
        "response": response,
        "has_edits": len(session.pending_edits) > 0,
        "current_content": session.current_content
    })

@app.route("/api/accept", methods=["POST"])
def accept_changes():
    """Accept pending changes."""
    data = request.json
    file_path = data["file_path"]
    
    if file_path not in editor_app.sessions:
        return jsonify({"error": "No active session"}), 400
    
    session = editor_app.sessions[file_path]
    success = editor_app.editor.accept_edits(session)
    
    if success:
        # Save file
        editor_app.save_file(file_path, session.current_content)
        return jsonify({"success": True})
    
    return jsonify({"success": False}), 400

@app.route("/api/reject", methods=["POST"])
def reject_changes():
    """Reject pending changes."""
    data = request.json
    file_path = data["file_path"]
    
    if file_path in editor_app.sessions:
        editor_app.editor.reject_edits(editor_app.sessions[file_path])
    
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# Frontend would be a React/Vue app that connects to this API
\`\`\`

## CLI Tool

### Command-Line Interface

\`\`\`python
import click

@click.group()
def cli():
    """AI Code Editor - Edit code with AI assistance."""
    pass

@cli.command()
@click.argument("file_path")
@click.option("--request", "-r", help="Edit request")
@click.option("--project", "-p", default=".", help="Project root")
def edit(file_path, request, project):
    """Edit a file with AI assistance."""
    config = EditorConfig(project_root=project)
    editor_app = CodeEditorApp(config)
    
    # Read file
    full_path = Path(project) / file_path
    if not full_path.exists():
        click.echo(f"File not found: {file_path}")
        return
    
    content = full_path.read_text()
    
    # Start session
    session = editor_app.editor.start_editing(file_path, content)
    
    # Process request
    if request:
        response = editor_app.editor.process_request(session, request)
        click.echo(f"\\nAI: {response}\\n")
        
        if session.pending_edits:
            # Show diff
            diff = editor_app.editor.show_diff(session)
            click.echo("Proposed changes:")
            click.echo(diff)
            
            # Confirm
            if click.confirm("\\nApply these changes?"):
                success = editor_app.editor.accept_edits(session)
                if success:
                    full_path.write_text(session.current_content)
                    click.echo("✓ Changes applied and saved")
                else:
                    click.echo("✗ Failed to apply changes")
    else:
        click.echo("No request provided. Use --request or -r")

@cli.command()
@click.argument("file_path")
@click.option("--description", "-d", required=True, help="File description")
@click.option("--project", "-p", default=".", help="Project root")
def create(file_path, description, project):
    """Create a new file with AI."""
    full_path = Path(project) / file_path
    
    if full_path.exists():
        click.echo(f"File already exists: {file_path}")
        return
    
    # Detect language
    ext = full_path.suffix
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript"
    }
    language = language_map.get(ext, "python")
    
    # Generate
    generator = MultiLanguageGenerator()
    content = generator.generate(language, description, "file")
    
    # Show preview
    click.echo("\\nGenerated content:")
    click.echo(content)
    
    # Confirm
    if click.confirm("\\nCreate this file?"):
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        click.echo(f"✓ Created: {file_path}")

@cli.command()
@click.argument("file_path")
@click.option("--project", "-p", default=".", help="Project root")
def review(file_path, project):
    """Review code for issues."""
    full_path = Path(project) / file_path
    
    if not full_path.exists():
        click.echo(f"File not found: {file_path}")
        return
    
    content = full_path.read_text()
    
    # Run review
    reviewer = CodeReviewer()
    review = reviewer.review_code(content)
    
    # Generate report
    report = reviewer.generate_review_report(review)
    
    click.echo(report)

if __name__ == "__main__":
    cli()

# Usage:
# $ ai-editor edit app/routes.py --request "Add error handling"
# $ ai-editor create app/new_service.py --description "User authentication service"
# $ ai-editor review app/routes.py
\`\`\`

## Production Deployment

### Docker Container

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
\`\`\`

### Docker Compose

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  editor:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./project:/workspace
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
    restart: unless-stopped
  
  sandbox:
    image: python:3.11-slim
    command: sleep infinity
    mem_limit: 512m
    cpus: 0.5
    network_mode: none
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Use modular architecture** - separate components
2. **Implement file watching** for real-time updates
3. **Add interactive editing** with conversation
4. **Provide multiple interfaces** (CLI, Web, API)
5. **Use sandboxing** for code execution
6. **Implement undo/redo** functionality
7. **Add comprehensive validation**
8. **Save backups** before editing
9. **Support multiple languages**
10. **Provide good UX** with clear feedback

### ❌ DON'T:
1. **Skip error handling**
2. **Forget to backup files**
3. **Execute code without sandboxing**
4. **Ignore user confirmation**
5. **Skip validation**
6. **Forget multi-language support**
7. **Neglect documentation**
8. **Skip testing**

## Capstone Project

Build your own Cursor-like editor! Include:
- ✅ File management
- ✅ Interactive editing
- ✅ Multi-file support
- ✅ Code validation
- ✅ Test execution
- ✅ Version control
- ✅ Web or CLI interface
- ✅ Language support

## Congratulations!

You've completed Module 5: Building Code Generation Systems! 

You now know how to:
- Generate code from natural language
- Edit code with precision
- Work across multiple files
- Refactor code automatically
- Generate tests and documentation
- Review code for bugs
- Build interactive editors
- Validate and execute code
- Support multiple languages
- Build complete production systems

**You can now build your own Cursor! 🎉**

Next steps:
- Build your capstone project
- Deploy to production
- Share with the community
- Keep improving your system

Remember: **Great Tools Are Built Iteratively**
`,
};
