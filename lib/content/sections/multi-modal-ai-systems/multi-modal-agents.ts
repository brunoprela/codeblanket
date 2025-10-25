export const multiModalAgents = {
  title: 'Multi-Modal Agents',
  id: 'multi-modal-agents',
  description:
    'Master building intelligent agents that can perceive, reason, and act across multiple modalities - vision, audio, text, and actions.',
  content: `
# Multi-Modal Agents

## Introduction

Multi-modal agents represent the convergence of perception, reasoning, and action across multiple modalities. These agents can see (vision), hear (audio), read and write (text), and take actions in their environment, making them powerful for complex real-world tasks.

In this section, we'll explore how to build agents that leverage multiple modalities to understand their environment and accomplish goals.

## Agent Architecture

### Core Components

\`\`\`
[Perception Layer]
  ├─ Vision (cameras, images)
  ├─ Audio (microphones, sound)
  └─ Text (documents, interfaces)
          ↓
[Multi-Modal Understanding]
  ├─ Fusion of modalities
  ├─ Context building
  └─ State representation
          ↓
[Reasoning & Planning]
  ├─ Goal setting
  ├─ Action planning
  └─ Decision making
          ↓
[Action Layer]
  ├─ Tool use
  ├─ Communication
  └─ Environment interaction
\`\`\`

## Building Multi-Modal Agents

### Basic Multi-Modal Agent

\`\`\`python
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import base64
from openai import OpenAI

client = OpenAI()

class ModalityType(Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

@dataclass
class Perception:
    """Agent perception from a modality."""
    modality: ModalityType
    content: Any
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class Action:
    """Agent action."""
    action_type: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None

class MultiModalAgent:
    """Basic multi-modal agent."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.perceptions: List[Perception] = []
        self.actions: List[Action] = []
        self.memory: List[Dict[str, Any]] = []
    
    def perceive_text (self, text: str):
        """Add text perception."""
        perception = Perception(
            modality=ModalityType.TEXT,
            content=text,
            timestamp=time.time()
        )
        self.perceptions.append (perception)
    
    def perceive_image (self, image_path: str):
        """Add image perception."""
        perception = Perception(
            modality=ModalityType.IMAGE,
            content=image_path,
            timestamp=time.time()
        )
        self.perceptions.append (perception)
    
    def perceive_audio (self, audio_path: str):
        """Add audio perception (transcribe first)."""
        # Transcribe audio
        with open (audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        perception = Perception(
            modality=ModalityType.AUDIO,
            content=transcription.text,
            timestamp=time.time(),
            metadata={"original_audio": audio_path}
        )
        self.perceptions.append (perception)
    
    def reason (self, goal: str) -> str:
        """Reason about perceptions to achieve goal."""
        # Build context from all perceptions
        context_parts = []
        
        for perception in self.perceptions:
            if perception.modality == ModalityType.TEXT:
                context_parts.append (f"Text: {perception.content}")
            elif perception.modality == ModalityType.AUDIO:
                context_parts.append (f"Audio (transcribed): {perception.content}")
        
        context = "\\n".join (context_parts)
        
        # Build message with images
        message_content = [{
            "type": "text",
            "text": f"""Based on these perceptions, determine how to achieve this goal: {goal}

Perceptions:
{context}

Think step-by-step and describe your reasoning."""
        }]
        
        # Add images
        for perception in self.perceptions:
            if perception.modality == ModalityType.IMAGE:
                with open (perception.content, "rb") as f:
                    img_data = base64.b64encode (f.read()).decode()
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data}",
                        "detail": "low"
                    }
                })
        
        # Reason with multi-modal context
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": message_content}],
            max_tokens=500
        )
        
        reasoning = response.choices[0].message.content
        
        # Store in memory
        self.memory.append({
            "goal": goal,
            "perceptions": len (self.perceptions),
            "reasoning": reasoning,
            "timestamp": time.time()
        })
        
        return reasoning
    
    def act (self, action_type: str, **parameters) -> Action:
        """Perform an action."""
        action = Action(
            action_type=action_type,
            parameters=parameters
        )
        
        # Execute action (simplified)
        if action_type == "speak":
            # Generate speech
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=parameters.get("text", "")
            )
            response.stream_to_file("agent_speech.mp3")
            action.result = "agent_speech.mp3"
        
        elif action_type == "generate_image":
            # Generate image
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=parameters.get("prompt", "")
            )
            action.result = response.data[0].url
        
        elif action_type == "write":
            # Write text
            text = parameters.get("text", "")
            filename = parameters.get("filename", "output.txt")
            with open (filename, "w") as f:
                f.write (text)
            action.result = filename
        
        self.actions.append (action)
        return action
    
    def clear_perceptions (self):
        """Clear current perceptions."""
        self.perceptions = []

# Example usage
agent = MultiModalAgent (openai_api_key=os.getenv("OPENAI_API_KEY"))

# Agent perceives environment
agent.perceive_text("The user asked about the weather")
agent.perceive_image("weather_map.png")
agent.perceive_audio("user_question.mp3")

# Agent reasons about goal
reasoning = agent.reason("Provide weather information to the user")
print(f"Agent reasoning: {reasoning}")

# Agent acts
action = agent.act("speak", text="Based on the weather map, it will rain today.")
print(f"Agent action: {action.action_type} -> {action.result}")
\`\`\`

### Vision-Enabled Agent

\`\`\`python
class VisionAgent:
    """Agent with vision capabilities."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.visual_memory: List[Dict[str, Any]] = []
    
    def observe (self, image_path: str, context: str = "") -> Dict[str, Any]:
        """Observe environment through vision."""
        with open (image_path, "rb") as f:
            img_data = base64.b64encode (f.read()).decode()
        
        prompt = f"""Observe this image and describe:

1. What objects or elements are present
2. Their spatial relationships
3. Any text visible
4. The overall scene or context
5. Any potential actions or tasks suggested

{f'Context: {context}' if context else '}

Return as JSON:
{{
  "objects": ["object1", "object2", ...],
  "scene": "description",
  "text_visible": "any text",
  "suggested_actions": ["action1", "action2"]
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    }
                ]
            }],
            max_tokens=500
        )
        
        import json
        observation = json.loads (response.choices[0].message.content)
        
        # Store in visual memory
        self.visual_memory.append({
            "image": image_path,
            "observation": observation,
            "timestamp": time.time()
        })
        
        return observation
    
    def spatial_reasoning(
        self,
        image_path: str,
        question: str
    ) -> str:
        """Answer spatial reasoning questions about image."""
        with open (image_path, "rb") as f:
            img_data = base64.b64encode (f.read()).decode()
        
        prompt = f"""Spatial reasoning question: {question}

Analyze the image carefully and provide a clear answer based on spatial relationships."""

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    }
                ]
            }],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def navigate_visual_scene(
        self,
        image_path: str,
        goal: str
    ) -> List[str]:
        """Determine navigation steps in visual scene."""
        observation = self.observe (image_path, context=f"Goal: {goal}")
        
        # Plan navigation
        planning_prompt = f"""Given this observation of the scene:

Objects: {', '.join (observation['objects'])}
Scene: {observation['scene']}
Goal: {goal}

Provide step-by-step navigation instructions to achieve the goal.

Return as JSON array of strings: ["step 1", "step 2", ...]"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": planning_prompt}]
        )
        
        import json
        steps = json.loads (response.choices[0].message.content)
        
        return steps

# Usage
vision_agent = VisionAgent (openai_api_key=os.getenv("OPENAI_API_KEY"))

# Observe scene
observation = vision_agent.observe("room.jpg", context="Finding the remote control")
print(f"Objects: {observation['objects']}")
print(f"Suggested actions: {observation['suggested_actions']}")

# Spatial reasoning
answer = vision_agent.spatial_reasoning(
    "room.jpg",
    "What is to the left of the couch?"
)
print(f"Answer: {answer}")

# Navigation
steps = vision_agent.navigate_visual_scene(
    "room.jpg",
    "Get to the window"
)
print("Navigation steps:")
for step in steps:
    print(f"  - {step}")
\`\`\`

### Autonomous Task Agent

\`\`\`python
class AutonomousTaskAgent:
    """Agent that can autonomously complete multi-modal tasks."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.task_log: List[Dict[str, Any]] = []
    
    def complete_task(
        self,
        task_description: str,
        available_inputs: Dict[str, str]  # {modality: path/content}
    ) -> Dict[str, Any]:
        """
        Autonomously complete a task using available inputs.
        
        Args:
            task_description: Description of task to complete
            available_inputs: Available multi-modal inputs
        
        Returns:
            Task result
        """
        # Step 1: Analyze task and inputs
        task_plan = self._plan_task (task_description, available_inputs)
        
        # Step 2: Execute plan
        result = self._execute_plan (task_plan, available_inputs)
        
        # Step 3: Verify result
        verification = self._verify_result (task_description, result)
        
        task_log = {
            "task": task_description,
            "plan": task_plan,
            "result": result,
            "verification": verification,
            "timestamp": time.time()
        }
        
        self.task_log.append (task_log)
        
        return task_log
    
    def _plan_task(
        self,
        task: str,
        inputs: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Plan how to complete task."""
        inputs_desc = "\\n".join([f"- {k}: {v}" for k, v in inputs.items()])
        
        planning_prompt = f"""Task: {task}

Available inputs:
{inputs_desc}

Create a step-by-step plan to complete this task. Each step should specify:
- What to do
- Which inputs to use
- Expected output

Return as JSON array:
[
  {{
    "step": 1,
    "action": "action description",
    "inputs": ["input1", "input2"],
    "expected_output": "what this step produces"
  }}
]"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.3
        )
        
        import json
        plan = json.loads (response.choices[0].message.content)
        
        return plan
    
    def _execute_plan(
        self,
        plan: List[Dict[str, Any]],
        inputs: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute the plan."""
        results = {}
        
        for step in plan:
            step_num = step['step']
            action = step['action']
            
            # Execute based on action type
            if "analyze image" in action.lower():
                # Image analysis
                image_input = inputs.get('image')
                if image_input:
                    with open (image_input, "rb") as f:
                        img_data = base64.b64encode (f.read()).decode()
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": action},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_data}"
                                    }
                                }
                            ]
                        }]
                    )
                    
                    results[f"step_{step_num}"] = response.choices[0].message.content
            
            elif "transcribe" in action.lower():
                # Audio transcription
                audio_input = inputs.get('audio')
                if audio_input:
                    with open (audio_input, "rb") as f:
                        transcription = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f
                        )
                    
                    results[f"step_{step_num}"] = transcription.text
            
            elif "generate" in action.lower() and "image" in action.lower():
                # Image generation
                prompt = results.get('step_1', ')  # Use previous step result
                
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt
                )
                
                results[f"step_{step_num}"] = response.data[0].url
            
            else:
                # Text processing
                context = "\\n".join([f"{k}: {v}" for k, v in results.items()])
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "user",
                        "content": f"{action}\\n\\nContext:\\n{context}"
                    }]
                )
                
                results[f"step_{step_num}"] = response.choices[0].message.content
        
        return results
    
    def _verify_result(
        self,
        task: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify that task was completed successfully."""
        results_summary = "\\n".join([f"{k}: {v}" for k, v in result.items()])
        
        verification_prompt = f"""Task: {task}

Results:
{results_summary}

Verify if the task was completed successfully. Return JSON:
{{
  "success": true/false,
  "completeness": 0.0-1.0,
  "issues": ["issue1", "issue2", ...],
  "summary": "verification summary"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.3
        )
        
        import json
        verification = json.loads (response.choices[0].message.content)
        
        return verification

# Usage
task_agent = AutonomousTaskAgent (openai_api_key=os.getenv("OPENAI_API_KEY"))

result = task_agent.complete_task(
    task_description="Create a presentation slide about the product in this image",
    available_inputs={
        "image": "product_photo.jpg",
        "text": "Product is a smart watch with fitness tracking"
    }
)

print(f"Task completed: {result['verification']['success']}")
print(f"Completeness: {result['verification']['completeness']:.0%}")
print(f"\\nPlan:")
for step in result['plan']:
    print(f"  Step {step['step']}: {step['action']}")
\`\`\`

## Real-World Applications

### 1. Customer Service Agent

\`\`\`python
class CustomerServiceAgent(MultiModalAgent):
    """Customer service agent with multi-modal capabilities."""
    
    def handle_customer_query(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle customer query from any modality."""
        # Perceive all provided inputs
        if query_text:
            self.perceive_text (query_text)
        
        if query_image:
            self.perceive_image (query_image)
        
        if query_audio:
            self.perceive_audio (query_audio)
        
        # Reason about how to help
        response_text = self.reason("Help the customer with their query")
        
        # Generate response (both text and speech)
        text_response = self.act("write", text=response_text, filename="response.txt")
        audio_response = self.act("speak", text=response_text)
        
        self.clear_perceptions()
        
        return {
            "text_response": response_text,
            "audio_response": audio_response.result,
            "handled": True
        }

# Usage
cs_agent = CustomerServiceAgent (openai_api_key=os.getenv("OPENAI_API_KEY"))

response = cs_agent.handle_customer_query(
    query_text="My product looks different from the website",
    query_image="product_photo.jpg"
)

print(f"Response: {response['text_response']}")
\`\`\`

### 2. Home Automation Agent

\`\`\`python
class HomeAutomationAgent(VisionAgent):
    """Agent for smart home automation with vision."""
    
    def monitor_home(
        self,
        camera_images: List[str]
    ) -> Dict[str, Any]:
        """Monitor home using camera images."""
        observations = []
        alerts = []
        
        for img_path in camera_images:
            obs = self.observe (img_path, context="Home security monitoring")
            observations.append (obs)
            
            # Check for alerts
            if any (word in str (obs).lower() for word in ["person", "motion", "open", "unusual"]):
                alerts.append({
                    "image": img_path,
                    "reason": obs['scene'],
                    "suggested_actions": obs.get('suggested_actions', [])
                })
        
        return {
            "observations": observations,
            "alerts": alerts,
            "status": "normal" if not alerts else "alert"
        }
\`\`\`

## Best Practices

### 1. Modality Fusion

- Combine modalities for richer context
- Prioritize based on reliability
- Handle missing modalities gracefully
- Validate cross-modal consistency

### 2. Agent Safety

- Validate actions before execution
- Implement kill switches
- Log all decisions and actions
- Human oversight for critical tasks
- Test extensively before deployment

### 3. Performance

- Cache perception results
- Batch process when possible
- Optimize for latency-critical tasks
- Use appropriate detail levels
- Implement timeouts

## Summary

Multi-modal agents combine perception, reasoning, and action across modalities:

**Key Capabilities:**
- Perceive through vision, audio, and text
- Reason with multi-modal context
- Act in environment through various means
- Maintain memory across interactions
- Autonomous task completion

**Production Patterns:**
- Separate perception, reasoning, and action layers
- Maintain multi-modal memory
- Implement verification loops
- Log all agent decisions
- Human oversight for critical actions

**Applications:**
- Customer service agents
- Home automation
- Robotic control
- Personal assistants
- Educational tutors
- Healthcare assistants

Next, we'll explore accessibility applications of multi-modal AI.
`,
  codeExamples: [
    {
      title: 'Multi-Modal Agent',
      description:
        'Complete agent with vision, audio, and text perception plus reasoning and action',
      language: 'python',
      code: `# See MultiModalAgent class in content above`,
    },
  ],
  practicalTips: [
    'Separate perception, reasoning, and action layers for cleaner architecture',
    'Maintain multi-modal memory to provide context across interactions',
    'Use vision for spatial understanding and navigation tasks',
    'Combine modalities for richer context - vision + audio + text together',
    'Implement verification loops to validate agent actions before execution',
    'Log all agent decisions and actions for debugging and auditing',
    'Start with simple agents and gradually add complexity',
    'Test agents extensively in controlled environments before production',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/multi-modal-agents',
};
