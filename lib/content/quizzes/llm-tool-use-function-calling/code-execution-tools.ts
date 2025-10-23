export const codeExecutionToolsQuiz = [
  {
    id: 'q1',
    question:
      'Design a secure code execution environment similar to ChatGPT Code Interpreter. What sandboxing strategies would you use, and how would you balance security, performance, and user experience?',
    sampleAnswer: `Secure code execution requires multiple layers of isolation:

**Sandboxing Approaches:**

1. **Docker Containers (Recommended):**
- Full OS-level isolation
- Resource limits (CPU, memory, disk)
- Network isolation
- Ephemeral environments
- Best security-to-performance ratio

2. **Virtual Machines:**
- Maximum isolation
- Higher resource overhead
- Slower startup times
- Good for untrusted code

3. **RestrictedPython:**
- Lightweight
- Python-specific
- Good for simple cases
- Limited protection

**Implementation:**
\`\`\`python
class SecureCodeExecutor:
    def execute(self, code: str) -> dict:
        container = docker.run(
            image="python-sandbox",
            command=["python", "/code/script.py"],
            mem_limit="256m",
            cpu_quota=50000,
            network_disabled=True,
            timeout=30,
            volumes={"/tmp/code": {"bind": "/code", "mode": "ro"}}
        )
\`\`\`

**Security Measures:**
- No network access
- Limited file system access
- CPU/memory quotas
- Execution timeouts
- Input validation
- Output sanitization
- Audit logging

**Performance Optimization:**
- Container pooling
- Warm containers
- Caching common libraries
- Parallel execution

**User Experience:**
- Progress indicators
- Partial results on timeout
- Clear error messages
- File upload/download
- Visualization support`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how you would handle file generation in code execution tools (plots, CSVs, etc.). How would you manage storage, enforce limits, and make files accessible to users?',
    sampleAnswer: `File handling in code execution requires careful resource management:

**Storage Strategy:**
\`\`\`python
class FileManager:
    def __init__(self):
        self.storage = S3Storage()
        self.limits = {
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "max_files": 10,
            "ttl": 3600  # 1 hour
        }
    
    def store_output(self, user_id: str, execution_id: str, files: List[File]):
        # Check limits
        if len(files) > self.limits["max_files"]:
            raise TooManyFilesError()
        
        for file in files:
            if file.size > self.limits["max_file_size"]:
                raise FileTooLargeError()
            
            # Upload to S3 with expiration
            key = f"{user_id}/{execution_id}/{file.name}"
            self.storage.upload(key, file.content, ttl=self.limits["ttl"])
            
            # Generate signed URL
            url = self.storage.generate_url(key, expires_in=3600)
            file.url = url
\`\`\`

**File Detection:**
\`\`\`python
def detect_generated_files(output_dir: str) -> List[File]:
    files = []
    for filepath in os.listdir(output_dir):
        if filepath != "script.py":
            with open(filepath, "rb") as f:
                files.append(File(
                    name=filepath,
                    content=f.read(),
                    mime_type=guess_mime_type(filepath),
                    size=os.path.getsize(filepath)
                ))
    return files
\`\`\`

**Best Practices:**
- Temporary storage with TTL
- Size limits per file
- Total storage quotas per user
- Virus scanning for uploads
- Content-Type validation
- Signed URLs for downloads
- Auto-cleanup of old files`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'q3',
    question:
      'Discuss the security risks of allowing LLMs to generate and execute code. What validation and monitoring strategies would you implement to prevent malicious use?',
    sampleAnswer: `LLM-generated code execution poses significant security risks:

**Risk Categories:**

1. **Resource Exhaustion:**
- Infinite loops
- Memory bombs
- Fork bombs
- CPU-intensive operations

2. **Data Exfiltration:**
- Reading sensitive files
- Network requests to external servers
- Encoding data in outputs

3. **System Compromise:**
- Privilege escalation
- Breaking out of sandbox
- Modifying system files

**Prevention Strategies:**

**1. Code Validation:**
\`\`\`python
def validate_code(code: str) -> Tuple[bool, str]:
    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Check for dangerous patterns
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if node.names[0].name in BLOCKED_MODULES:
                return False, f"Import of {node.names[0].name} not allowed"
        
        if isinstance(node, (ast.Exec, ast.Eval)):
            return False, "eval/exec not allowed"
    
    return True, ""
\`\`\`

**2. Runtime Monitoring:**
\`\`\`python
class ExecutionMonitor:
    def monitor(self, container):
        # CPU usage
        if container.stats()["cpu_percent"] > 90:
            container.kill()
            raise ResourceExhaustedError("CPU limit exceeded")
        
        # Memory usage
        if container.stats()["memory_usage"] > MAX_MEMORY:
            container.kill()
            raise ResourceExhaustedError("Memory limit exceeded")
        
        # Execution time
        if elapsed > TIMEOUT:
            container.kill()
            raise TimeoutError()
\`\`\`

**3. Audit Logging:**
\`\`\`python
def log_execution(user_id, code, result):
    logger.info({
        "event": "code_execution",
        "user_id": user_id,
        "code_hash": hash(code),
        "success": result.success,
        "duration_ms": result.duration,
        "resources_used": result.resources
    })
\`\`\`

**Best Practices:**
- Multiple layers of defense
- Assume all code is malicious
- Monitor everything
- Rate limit executions
- Require user authentication
- Alert on suspicious patterns
- Regular security audits
- Incident response plan`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
