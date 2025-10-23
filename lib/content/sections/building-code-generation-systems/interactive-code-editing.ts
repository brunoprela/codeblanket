/**
 * Interactive Code Editing Section
 * Module 5: Building Code Generation Systems
 */

export const interactivecodeeditingSection = {
  id: 'interactive-code-editing',
  title: 'Interactive Code Editing',
  content: `# Interactive Code Editing

Master building interactive code editing experiences with conversational AI, like Cursor's chat interface.

## Overview: Conversational Code Editing

Interactive editing allows users to:
- Have conversations about code
- Iterate on changes with follow-ups
- Ask clarifying questions
- Provide feedback and refine
- Undo/redo changes
- Review before applying

### How Cursor Does Interactive Editing

Cursor's strength is its conversational interface:
1. User makes a request
2. AI asks clarifying questions if needed
3. AI generates changes
4. User reviews and can:
   - Accept
   - Reject
   - Ask for modifications
   - Provide more context
5. Changes are applied only when user accepts

## Conversation State Management

### Maintain Editing Context

\`\`\`python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EditSession:
    """An interactive editing session."""
    file_path: str
    original_content: str
    current_content: str
    messages: List[Message] = field(default_factory=list)
    pending_edits: List[SearchReplace] = field(default_factory=list)
    applied_edits: List[SearchReplace] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add message to conversation."""
        self.messages.append(Message(role, content))
    
    def apply_pending_edits(self):
        """Apply pending edits to current content."""
        applicator = SafeEditApplicator()
        success, new_content, errors = applicator.apply_multiple_edits(
            self.current_content,
            self.pending_edits
        )
        
        if success:
            self.applied_edits.extend(self.pending_edits)
            self.current_content = new_content
            self.pending_edits.clear()
            return True
        
        return False
    
    def undo_last_edit(self):
        """Undo the most recent edit."""
        if not self.applied_edits:
            return False
        
        # Revert to original and reapply all but last edit
        self.current_content = self.original_content
        last_edit = self.applied_edits.pop()
        
        # Reapply remaining edits
        if self.applied_edits:
            applicator = SafeEditApplicator()
            success, new_content, _ = applicator.apply_multiple_edits(
                self.original_content,
                self.applied_edits
            )
            if success:
                self.current_content = new_content
        
        return True

class ConversationManager:
    """Manage interactive editing conversations."""
    
    def __init__(self):
        self.sessions: Dict[str, EditSession] = {}
    
    def create_session(
        self,
        file_path: str,
        content: str
    ) -> EditSession:
        """Create a new editing session."""
        session = EditSession(
            file_path=file_path,
            original_content=content,
            current_content=content
        )
        self.sessions[file_path] = session
        return session
    
    def get_session(self, file_path: str) -> Optional[EditSession]:
        """Get existing session."""
        return self.sessions.get(file_path)
    
    def end_session(self, file_path: str):
        """End an editing session."""
        if file_path in self.sessions:
            del self.sessions[file_path]

# Usage
manager = ConversationManager()

# Start editing session
session = manager.create_session(
    "app/routes.py",
    original_code
)

# Add user request
session.add_message("user", "Add error handling to the login function")

# ... AI generates edits ...
# session.pending_edits = generated_edits

# User reviews and asks for modification
session.add_message("user", "Make the error messages more specific")

# ... AI refines edits ...

# User accepts
session.apply_pending_edits()
\`\`\`

## Interactive Code Editor

### Full Interactive Editor

\`\`\`python
from openai import OpenAI

class InteractiveCodeEditor:
    """Interactive code editor with conversation."""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_manager = ConversationManager()
        self.editor = CursorStyleEditor()
    
    def start_editing(
        self,
        file_path: str,
        content: str
    ) -> EditSession:
        """Start interactive editing session."""
        session = self.conversation_manager.create_session(file_path, content)
        
        # Add system message
        session.add_message(
            "system",
            f"You are editing {file_path}. "
            "Ask clarifying questions if needed. "
            "Generate precise edits when you understand the request."
        )
        
        return session
    
    def process_request(
        self,
        session: EditSession,
        user_request: str
    ) -> str:
        """Process user request and generate response."""
        
        # Add user message
        session.add_message("user", user_request)
        
        # Build conversation history for LLM
        messages = self._build_messages(session)
        
        # Get AI response
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.2
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to session
        session.add_message("assistant", ai_response)
        
        # Check if response contains edits
        if "<<<<<<< SEARCH" in ai_response:
            # Parse edits
            parser = SearchReplaceParser()
            edits = parser.parse(ai_response)
            session.pending_edits = edits
        
        return ai_response
    
    def _build_messages(self, session: EditSession) -> List[dict]:
        """Build message array for LLM."""
        messages = []
        
        # Add file context as first user message
        file_context = f"""Current file: {session.file_path}

Content:
{session.current_content}
"""
        messages.append({
            "role": "user",
            "content": file_context
        })
        
        # Add conversation history
        for msg in session.messages:
            if msg.role != "system":  # Skip system messages
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return messages
    
    def accept_edits(self, session: EditSession) -> bool:
        """Accept pending edits."""
        return session.apply_pending_edits()
    
    def reject_edits(self, session: EditSession):
        """Reject pending edits."""
        session.pending_edits.clear()
    
    def undo(self, session: EditSession) -> bool:
        """Undo last edit."""
        return session.undo_last_edit()
    
    def show_diff(self, session: EditSession) -> str:
        """Show diff of pending changes."""
        import difflib
        
        original_lines = session.current_content.split("\\n")
        
        # Apply pending edits to get preview
        applicator = SafeEditApplicator()
        _, new_content, _ = applicator.apply_multiple_edits(
            session.current_content,
            session.pending_edits
        )
        
        new_lines = new_content.split("\\n")
        
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            lineterm=""
        )
        
        return "\\n".join(diff)

# Usage
editor = InteractiveCodeEditor()

# Start session
session = editor.start_editing("app/routes.py", file_content)

# User request
response = editor.process_request(
    session,
    "Add error handling to the login function"
)
print(f"AI: {response}")

# If AI asks clarifying question
if "?" in response:
    # User provides more context
    response = editor.process_request(
        session,
        "Handle both authentication errors and database errors separately"
    )
    print(f"AI: {response}")

# Show diff before accepting
if session.pending_edits:
    diff = editor.show_diff(session)
    print("\\nProposed changes:")
    print(diff)
    
    # User reviews and decides
    user_decision = input("\\nAccept changes? (yes/no/modify): ")
    
    if user_decision == "yes":
        editor.accept_edits(session)
        print("✓ Changes applied")
    
    elif user_decision == "no":
        editor.reject_edits(session)
        print("✗ Changes rejected")
    
    elif user_decision == "modify":
        # User can provide feedback
        feedback = input("What would you like to change? ")
        response = editor.process_request(session, feedback)
        print(f"AI: {response}")
\`\`\`

## Clarifying Questions

### Smart Question Generation

\`\`\`python
class QuestionGenerator:
    """Generate clarifying questions when needed."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def should_ask_question(
        self,
        user_request: str,
        file_content: str
    ) -> tuple[bool, Optional[str]]:
        """Determine if clarification is needed."""
        
        prompt = f"""Analyze this user request for code editing:

Request: {user_request}

File content:
{file_content}

Is the request:
1. Clear and specific enough to implement?
2. Ambiguous and needs clarification?

If clarification is needed, what question should be asked?

Output as JSON:
{{
    "needs_clarification": true/false,
    "question": "clarifying question or null"
}}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at understanding code editing requests."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        needs_clarification = result.get("needs_clarification", False)
        question = result.get("question")
        
        return needs_clarification, question

# Usage
question_gen = QuestionGenerator()

# Vague request
needs_clarification, question = question_gen.should_ask_question(
    "Make it better",
    file_content
)

if needs_clarification:
    print(f"AI: {question}")
    # e.g., "What aspect would you like to improve? 
    #        Performance, readability, error handling, or something else?"

# Specific request
needs_clarification, question = question_gen.should_ask_question(
    "Add type hints to all function parameters and return types",
    file_content
)

if not needs_clarification:
    print("AI: I understand. I'll add comprehensive type hints.")
\`\`\`

## Multi-Turn Refinement

### Iterative Improvement

\`\`\`python
class IterativeRefiner:
    """Support multi-turn refinement of edits."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def refine_edits(
        self,
        original_edits: List[SearchReplace],
        current_content: str,
        user_feedback: str
    ) -> List[SearchReplace]:
        """Refine edits based on user feedback."""
        
        # Apply current edits to see what they produce
        applicator = SafeEditApplicator()
        _, preview_content, _ = applicator.apply_multiple_edits(
            current_content,
            original_edits
        )
        
        # Show original edits
        edits_str = "\\n\\n".join(str(edit) for edit in original_edits)
        
        prompt = f"""Refine these code edits based on user feedback:

Original content:
{current_content}

Current edits:
{edits_str}

Preview after edits:
{preview_content}

User feedback: {user_feedback}

Generate refined edits that incorporate the feedback.
Use SEARCH/REPLACE format.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refining code edits based on feedback."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Parse refined edits
        parser = SearchReplaceParser()
        refined_edits = parser.parse(response.choices[0].message.content)
        
        return refined_edits

# Usage
refiner = IterativeRefiner()

# Original edits
original_edits = [
    SearchReplace(
        "def login(username, password):",
        "def login(username: str, password: str) -> bool:"
    )
]

# User provides feedback
user_feedback = "Also add a docstring explaining what the function does"

# Refine
refined_edits = refiner.refine_edits(
    original_edits,
    file_content,
    user_feedback
)

print("Refined edits:")
for edit in refined_edits:
    print(edit)
\`\`\`

## Edit History & Undo

### Version Control for Edits

\`\`\`python
@dataclass
class EditVersion:
    """A version in edit history."""
    content: str
    edits: List[SearchReplace]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

class EditHistory:
    """Track edit history with undo/redo."""
    
    def __init__(self, initial_content: str):
        self.versions: List[EditVersion] = [
            EditVersion(
                content=initial_content,
                edits=[],
                message="Initial version"
            )
        ]
        self.current_index = 0
    
    def add_version(
        self,
        content: str,
        edits: List[SearchReplace],
        message: str
    ):
        """Add a new version."""
        # Remove any versions after current (if we're not at the end)
        self.versions = self.versions[:self.current_index + 1]
        
        # Add new version
        self.versions.append(EditVersion(
            content=content,
            edits=edits,
            message=message
        ))
        
        self.current_index += 1
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.versions) - 1
    
    def undo(self) -> Optional[EditVersion]:
        """Undo to previous version."""
        if not self.can_undo():
            return None
        
        self.current_index -= 1
        return self.versions[self.current_index]
    
    def redo(self) -> Optional[EditVersion]:
        """Redo to next version."""
        if not self.can_redo():
            return None
        
        self.current_index += 1
        return self.versions[self.current_index]
    
    def get_current(self) -> EditVersion:
        """Get current version."""
        return self.versions[self.current_index]
    
    def get_history(self) -> List[str]:
        """Get history as list of messages."""
        return [v.message for v in self.versions]

# Usage
history = EditHistory(original_content)

# Apply edit
history.add_version(
    new_content,
    applied_edits,
    "Added error handling"
)

# User wants to undo
if history.can_undo():
    previous = history.undo()
    print(f"Undone to: {previous.message}")
    current_content = previous.content

# User wants to redo
if history.can_redo():
    next_version = history.redo()
    print(f"Redone to: {next_version.message}")
    current_content = next_version.content

# Show history
print("\\nEdit history:")
for i, msg in enumerate(history.get_history()):
    marker = "→" if i == history.current_index else " "
    print(f"{marker} {i}: {msg}")
\`\`\`

## User Confirmation Flow

### Safe Confirmation System

\`\`\`python
class ConfirmationManager:
    """Manage user confirmation for edits."""
    
    def __init__(self):
        self.pending_confirmations: Dict[str, Any] = {}
    
    def request_confirmation(
        self,
        session_id: str,
        edits: List[SearchReplace],
        preview: str,
        diff: str
    ) -> Dict:
        """Request user confirmation."""
        
        confirmation = {
            "session_id": session_id,
            "edits": edits,
            "preview": preview,
            "diff": diff,
            "timestamp": datetime.now(),
            "status": "pending"
        }
        
        self.pending_confirmations[session_id] = confirmation
        
        return confirmation
    
    def show_confirmation_prompt(
        self,
        confirmation: Dict
    ) -> str:
        """Generate confirmation prompt for user."""
        
        prompt = f"""
{'='*60}
PROPOSED CHANGES
{'='*60}

{confirmation['diff']}

{'='*60}
PREVIEW
{'='*60}

{confirmation['preview'][:500]}...

{'='*60}

Options:
1. Accept - Apply changes
2. Reject - Discard changes
3. Modify - Request modifications
4. Diff - Show full diff

What would you like to do?
"""
        
        return prompt
    
    def handle_confirmation(
        self,
        session_id: str,
        action: str,
        feedback: Optional[str] = None
    ) -> Dict:
        """Handle user's confirmation decision."""
        
        if session_id not in self.pending_confirmations:
            return {"error": "No pending confirmation"}
        
        confirmation = self.pending_confirmations[session_id]
        
        if action == "accept":
            confirmation["status"] = "accepted"
            return {"success": True, "action": "apply"}
        
        elif action == "reject":
            confirmation["status"] = "rejected"
            return {"success": True, "action": "discard"}
        
        elif action == "modify":
            confirmation["status"] = "modifying"
            return {
                "success": True,
                "action": "refine",
                "feedback": feedback
            }
        
        else:
            return {"error": "Invalid action"}

# Usage
conf_manager = ConfirmationManager()

# Request confirmation
confirmation = conf_manager.request_confirmation(
    session_id="session_123",
    edits=pending_edits,
    preview=preview_content,
    diff=diff_content
)

# Show to user
prompt = conf_manager.show_confirmation_prompt(confirmation)
print(prompt)

# Get user input
user_choice = input("Your choice: ")

if user_choice == "3":  # Modify
    feedback = input("What would you like to change? ")
    result = conf_manager.handle_confirmation(
        "session_123",
        "modify",
        feedback
    )
    
    if result["action"] == "refine":
        # Refine edits with feedback
        refined = refiner.refine_edits(
            pending_edits,
            current_content,
            result["feedback"]
        )
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Maintain conversation state** across turns
2. **Ask clarifying questions** when needed
3. **Show diffs** before applying
4. **Support undo/redo** functionality
5. **Allow iterative refinement**
6. **Request confirmation** for changes
7. **Track edit history**
8. **Provide clear feedback** to user

### ❌ DON'T:
1. **Apply changes without confirmation**
2. **Lose conversation context**
3. **Skip clarifying questions** for vague requests
4. **Forget edit history**
5. **Make undo impossible**
6. **Hide what changes will be made**
7. **Ignore user feedback**
8. **Rush to apply edits**

## Next Steps

You've mastered interactive editing! Next:
- Code execution and validation
- Language-specific generation
- Building complete code editors

Remember: **User Control + Transparency = Trust**
`,
};
