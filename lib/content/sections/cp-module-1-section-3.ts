export const environmentSetupCompilationSection = {
  id: 'cp-m1-s3',
  title: 'Environment Setup & Compilation',
  content: `

# Environment Setup & Compilation

## Introduction

Before writing your first line of competitive programming code, you need a **properly configured development environment**. A good setup can save you hours of frustration and help you focus on solving problems, not fighting with tools.

In this section, we'll cover everything you need: installing compilers, choosing IDEs, understanding compiler flags, and setting up an efficient workflow that top competitive programmers use.

**Goal**: By the end of this section, you'll have a professional CP environment ready for contests.

---

## Operating System Considerations

### Linux (Ubuntu/Debian) ✅ **RECOMMENDED**

**Advantages:**
- GCC comes pre-installed or easy to install
- Same environment as online judges
- Fast terminal workflows
- Lightweight and efficient

**Best for**: Serious competitive programmers

### macOS ✅ **GOOD**

**Advantages:**
- Unix-based like Linux
- Clang/GCC available via Xcode or Homebrew
- Good terminal support

**Caveats:**
- Some GCC-specific features might differ
- Need to install command-line tools

**Best for**: Mac users who want convenience

### Windows ⚠️ **WORKABLE**

**Advantages:**
- Most familiar to beginners
- Good IDE support (VS Code, Visual Studio)

**Challenges:**
- Need to install MinGW or use WSL
- Path setup can be tricky
- Some differences from Linux environment

**Best for**: Windows users (but consider WSL2)

### Windows Subsystem for Linux (WSL2) ✅ **GREAT ALTERNATIVE**

**Advantages:**
- Linux environment on Windows
- Best of both worlds
- Same as online judges

**Best for**: Windows users serious about CP

---

## Installing GCC/G++ Compiler

The **GNU Compiler Collection (GCC)** is the standard compiler for competitive programming. Specifically, we use **g++** for C++ code.

### Linux Installation

**Ubuntu/Debian:**
\`\`\`bash
# Update package list
sudo apt update

# Install build-essential (includes g++)
sudo apt install build-essential

# Verify installation
g++ --version
# Should show: g++ (Ubuntu 11.x or higher)
\`\`\`

**Expected output:**
\`\`\`
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
\`\`\`

**Fedora/RedHat:**
\`\`\`bash
sudo dnf install gcc-c++
g++ --version
\`\`\`

### macOS Installation

**Option 1: Xcode Command Line Tools (Clang)**
\`\`\`bash
# Install command-line tools
xcode-select --install

# This installs clang++, not g++, but works similarly
clang++ --version
\`\`\`

**Option 2: Install GCC via Homebrew**
\`\`\`bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GCC
brew install gcc

# Use g++-13 (or whatever version installed)
g++-13 --version
\`\`\`

### Windows Installation

**Option 1: MinGW-w64 (Native Windows)**

1. Download MinGW-w64 from: https://www.mingw-w64.org/
2. Install to \`C: \\mingw64\`
3. Add to PATH: \`C: \\mingw64\\bin\`
4. Verify:
\`\`\`bash
g++ --version
\`\`\`

**Option 2: WSL2 (Recommended)**

1. Enable WSL2:
\`\`\`powershell
# Run in PowerShell as Administrator
wsl --install
\`\`\`

2. Install Ubuntu from Microsoft Store
3. Follow Linux installation steps above

**Option 3: MSYS2**

1. Download from: https://www.msys2.org/
2. Install and update:
\`\`\`bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc
\`\`\`

---

## Understanding Compiler Flags

Compiler flags control how your code is compiled. For competitive programming, certain flags are crucial.

### Essential Flags for CP

\`\`\`bash
g++ -o solution solution.cpp -std=c++17 -O2 -Wall
\`\`\`

Let's break down each flag:

#### \`- std=c++17\` or \` - std=c++20\`
- **What**: Specifies C++ standard version
- **Why**: Use modern C++ features
- **Options**: \`- std=c++11\`, \` - std=c++14\`, \` - std=c++17\`, \` - std=c++20\`
- **Recommended**: \`- std=c++17\` (supported by most judges)

\`\`\`cpp
// C++17 features you'll use:
auto [x, y] = make_pair(1, 2);  // Structured bindings
if (auto it = s.find(x); it != s.end()) { }  // Init in if
\`\`\`

#### \`- O2\` (Optimization Level 2)
- **What**: Enables compiler optimizations
- **Why**: Makes code run 2-5x faster!
- **Options**: \`- O0\` (no optimization), \` - O1\`, \` - O2\`, \` - O3\`
- **Recommended**: \`- O2\` (same as online judges)

\`\`\`bash
# Without optimization
g++ solution.cpp -o slow
# Time: 2.5 seconds

# With -O2
g++ solution.cpp -o fast -O2
# Time: 0.5 seconds
\`\`\`

#### \`- Wall\` (Warnings All)
- **What**: Enable all common warnings
- **Why**: Catch potential bugs early
- **Catches**: Uninitialized variables, unused variables, etc.

\`\`\`cpp
int main() {
    int x;
    cout << x;  // Warning: 'x' may be used uninitialized
}
\`\`\`

### Additional Useful Flags

#### \`- Wextra\` (Extra Warnings)
\`\`\`bash
g++ solution.cpp -Wall -Wextra
\`\`\`
- More strict warnings
- Catches edge cases

#### \`- fsanitize=address\` (Address Sanitizer)
\`\`\`bash
g++ solution.cpp -fsanitize=address -g
\`\`\`
- Detects memory errors
- Buffer overflows, use-after-free
- **Use during development, not in contests!**

#### \`- fsanitize=undefined\` (Undefined Behavior Sanitizer)
\`\`\`bash
g++ solution.cpp -fsanitize=undefined -g
\`\`\`
- Detects undefined behavior
- Integer overflow, null pointer dereference
- **Use during development!**

#### \`- g\` (Debug Information)
\`\`\`bash
g++ solution.cpp -g
\`\`\`
- Adds debugging symbols
- Needed for GDB debugger
- **Don't use in contests** (makes binary larger)

### Contest-Standard Compilation

**For practice/contests:**
\`\`\`bash
g++ -std=c++17 -O2 -Wall solution.cpp -o solution
\`\`\`

**For debugging:**
\`\`\`bash
g++ -std=c++17 -O2 -Wall -g -fsanitize=address,undefined solution.cpp -o solution
\`\`\`

---

## Creating an Alias/Script

Typing long compilation commands every time is tedious. Let's create shortcuts!

### Bash Alias (Linux/macOS)

Add to \`~/.bashrc\` or \`~/.zshrc\`:

\`\`\`bash
# Basic compilation
alias cprun='g++ -std=c++17 -O2 -Wall -Wextra'

# Compilation with debug info
alias cpdebug='g++ -std=c++17 -g -Wall -Wextra -fsanitize=address,undefined'

# Compile and run
cpcompile() {
    g++ -std=c++17 -O2 -Wall "$1.cpp" -o "$1" && ./"$1"
}
\`\`\`

**Usage:**
\`\`\`bash
cprun solution.cpp -o solution
cpdebug solution.cpp -o solution
cpcompile solution  # Compiles solution.cpp and runs ./solution
\`\`\`

### Windows Batch Script

Create \`compile.bat\`:
\`\`\`batch
@echo off
g++ -std=c++17 -O2 -Wall %1.cpp -o %1.exe
if %errorlevel%==0 (
    echo Compilation successful!
    %1.exe
) else (
    echo Compilation failed!
)
\`\`\`

**Usage:**
\`\`\`
compile solution
\`\`\`

### Makefile

Create \`Makefile\`:
\`\`\`makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

%: %.cpp
\t$(CXX) $(CXXFLAGS) $< -o $@

clean:
\trm -f solution

debug: CXXFLAGS += -g -fsanitize=address,undefined
debug: solution
\`\`\`

**Usage:**
\`\`\`bash
make solution     # Compile solution.cpp
make debug       # Compile with debug flags
./solution       # Run
make clean       # Remove binary
\`\`\`

---

## IDE Options

Choosing the right IDE can significantly improve your productivity.

### Visual Studio Code (VS Code) ✅ **HIGHLY RECOMMENDED**

**Why VS Code:**
- Lightweight and fast
- Excellent C++ support
- Integrated terminal
- Massive extension ecosystem
- Free and open-source

**Installation:**
1. Download from: https://code.visualstudio.com/
2. Install C/C++ extension (Microsoft)
3. Install Code Runner extension (optional)

**Essential Extensions:**
- **C/C++** (Microsoft): IntelliSense, debugging
- **C/C++ Extension Pack** (Microsoft): Comprehensive C++ support
- **Code Runner**: Run code with one click
- **Competitive Programming Helper** (DivyanshuAgrawal): CP tools integration
- **Better Comments**: Enhanced comment highlighting
- **Error Lens**: Inline error/warning display

**VS Code Setup for CP:**

1. Install extensions
2. Configure \`settings.json\`:

\`\`\`json
{
    "code-runner.executorMap": {
        "cpp": "cd $dir && g++ -std=c++17 -O2 -Wall $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt"
    },
    "code-runner.runInTerminal": true,
    "files.associations": {
        "*.cpp": "cpp",
        "iostream": "cpp",
        "vector": "cpp"
    },
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.compilerPath": "/usr/bin/g++"
}
\`\`\`

3. Create \`tasks.json\` for building:

\`\`\`json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "C++ Compile",
            "type": "shell",
            "command": "g++",
            "args": [
                "-std=c++17",
                "-O2",
                "-Wall",
                "\${file}",
                "-o",
                "\${fileDirname}/\${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
\`\`\`

**Usage:**
- \`Ctrl + Alt + N\` (Code Runner): Compile and run
- \`Ctrl + Shift + B\`: Build task
- \`F5\`: Debug

### CLion ✅ **EXCELLENT (Paid)**

**Why CLion:**
- Professional IDE by JetBrains
- Smart code completion
- Excellent refactoring tools
- Integrated debugger

**Drawbacks:**
- Paid (free for students)
- Heavier than VS Code
- Requires CMake setup

**Best for**: Serious developers with student license

### Sublime Text ⚠️ **FAST BUT LIMITED**

**Why Sublime:**
- Extremely fast
- Lightweight
- Popular among competitive programmers

**Setup:**
1. Download from: https://www.sublimetext.com/
2. Install Package Control
3. Install "C++ Snippets" and "SublimeREPL"

**Build System** (\`Tools > Build System > New Build System\`):
\`\`\`json
{
    "cmd": ["g++", "-std=c++17", "-O2", "-Wall", "\${file}", "-o", "\${file_path}/\${file_base_name}"],
    "file_regex": "^(..[^:]*):([0-9]+):?([0-9]+)?:? (.*)$",
    "working_dir": "\${file_path}",
    "selector": "source.c++",
    "variants":
    [
        {
            "name": "Run",
            "cmd": ["bash", "-c", "g++ -std=c++17 -O2 -Wall '\${file}' -o '\${file_path}/\${file_base_name}' && '\${file_path}/\${file_base_name}'"]
        }
    ]
}
\`\`\`

### Vim/Neovim ⚡ **FOR ADVANCED USERS**

**Why Vim:**
- Ultimate speed (no mouse needed)
- Available everywhere
- Highly customizable

**Setup:**
1. Install \`vim - plug\` plugin manager
2. Add plugins in \`.vimrc\`:

\`\`\`vim
call plug#begin()
Plug 'neoclide/coc.nvim', {'branch': 'release'}  " LSP
Plug 'vim-airline/vim-airline'                   " Status line
Plug 'preservim/nerdtree'                        " File explorer
Plug 'jiangmiao/auto-pairs'                      " Auto brackets
call plug#end()

" Compile and run
autocmd FileType cpp nnoremap <F5> :w <bar> !g++ -std=c++17 -O2 -Wall % -o %:r && ./%:r <CR>
\`\`\`

**Best for**: Vim enthusiasts

### CP Editor ✅ **CP-SPECIFIC TOOL**

**Why CP Editor:**
- Built for competitive programming
- Test case management
- Timer for practice
- Competitive Companion integration

**Installation:**
Download from: https://cpeditor.org/

**Features:**
- Multiple test cases side-by-side
- Diff viewer for output
- Code formatting
- Template management

**Best for**: Dedicated CP practitioners

---

## Online IDEs and Custom Invocation

Sometimes you need to test code quickly without local setup, or test with exact judge environment.

### Codeforces Custom Invocation

**Access**: Any Codeforces problem page → "Custom Invocation" tab

**Advantages:**
- Exact Codeforces environment
- Test with judge's compiler
- Check compilation errors
- Verify TLE/MLE

**Usage:**
1. Write/paste code
2. Provide input
3. Click "Run"
4. See output, time, memory

**Compiler versions:**
- GNU G++17 7.3.0
- GNU G++20 11.2.0 (64-bit)

**When to use:**
- Before final submission
- Test edge cases
- Verify time limits

### Other Online IDEs

**Compiler Explorer (godbolt.org)**
- See assembly code
- Compare optimizations
- Multiple compilers

**Example:**
\`\`\`
Visit: https://godbolt.org/
Paste code
See optimized assembly
Compare -O2 vs -O3
\`\`\`

**Ideone.com**
- Quick online IDE
- Share code snippets
- Multiple languages

**Repl.it**
- Full development environment
- Collaborative coding
- Auto-save

---

## Debugging Setup

Debugging is crucial for finding bugs quickly. Let's set up **GDB** (GNU Debugger).

### GDB Installation

**Linux:**
\`\`\`bash
sudo apt install gdb
gdb --version
\`\`\`

**macOS:**
\`\`\`bash
brew install gdb
# May need codesigning on macOS
\`\`\`

**Windows (MinGW):**
- Included with MinGW installation

### Basic GDB Usage

**Compile with debug info:**
\`\`\`bash
g++ -g -std=c++17 solution.cpp -o solution
\`\`\`

**Start GDB:**
\`\`\`bash
gdb ./solution
\`\`\`

**Common GDB commands:**
\`\`\`
(gdb) break main          # Set breakpoint at main
(gdb) run                 # Start execution
(gdb) next                # Next line (step over)
(gdb) step                # Step into function
(gdb) print variable      # Print variable value
(gdb) continue            # Continue execution
(gdb) quit                # Exit GDB
\`\`\`

**VS Code Debugging:**

\`launch.json\` configuration:
\`\`\`json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "C++ Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "\${fileDirname}/\${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "\${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "preLaunchTask": "C++ Compile"
        }
    ]
}
\`\`\`

**Usage in VS Code:**
1. Set breakpoints (click left of line numbers)
2. Press \`F5\`
3. Use debug console to inspect variables

---

## Build Automation

For larger projects or complex setups, automate your builds.

### Simple Build Script

**build.sh:**
\`\`\`bash
#!/bin/bash

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
NC='\\033[0m'  # No Color

# Compile
echo "Compiling $1.cpp..."
g++ -std=c++17 -O2 -Wall -Wextra "$1.cpp" -o "$1"

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "\${GREEN}✓ Compilation successful!\${NC}"
    
    # Run if input file exists
    if [ -f "input.txt" ]; then
        echo "Running with input.txt..."
        ./"$1" < input.txt
    else
        ./"$1"
    fi
else
    echo "\${RED}✗ Compilation failed!\${NC}"
    exit 1
fi
\`\`\`

**Make executable:**
\`\`\`bash
chmod +x build.sh
./build.sh solution
\`\`\`

### Advanced Makefile

\`\`\`makefile
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
DEBUGFLAGS := -g -fsanitize=address,undefined
TARGET := solution

.PHONY: all clean debug run test

all: $(TARGET)

$(TARGET): $(TARGET).cpp
\t$(CXX) $(CXXFLAGS) $< -o $@

debug: CXXFLAGS += $(DEBUGFLAGS)
debug: clean $(TARGET)

run: $(TARGET)
\t./$(TARGET)

test: $(TARGET)
\t@if [ -f input.txt ]; then \\
\t\t./$(TARGET) < input.txt; \\
\telse \\
\t\techo "No input.txt found"; \\
\tfi

clean:
\trm -f $(TARGET)
\`\`\`

---

## Windows vs Linux Differences

### Path Separators
\`\`\`cpp
// Windows
#include <windows.h>
string path = "C:\\\\Users\\\\name\\\\file.txt";

// Linux/macOS
#include <unistd.h>
string path = "/home/user/file.txt";
\`\`\`

### Line Endings
- **Windows**: \`\\r\\n\` (CRLF)
- **Linux/macOS**: \`\\n\` (LF)

**Fix in VS Code:**
- Bottom right corner: Select "LF"
- Or set in settings: \`"files.eol": "\\n"\`

### Case Sensitivity
- **Linux**: Case-sensitive (\`file.txt\` ≠ \`File.txt\`)
- **Windows**: Case-insensitive
- **macOS**: Usually case-insensitive

### Compiler Differences
- **Linux/Windows (GCC)**: True G++ compiler
- **macOS**: Clang masquerading as G++

**Check what you have:**
\`\`\`bash
g++ --version

# If output says "clang", you have Clang, not GCC
# For true GCC: brew install gcc && use g++-13
\`\`\`

---

## Testing Your Environment

Let's verify everything works!

### Test Program

Create \`test.cpp\`:
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    cout << "Environment test successful!" << endl;
    cout << "C++ version: " << __cplusplus << endl;
    
    // Test STL
    vector<int> v = {1, 2, 3, 4, 5};
    cout << "Vector size: " << v.size() << endl;
    
    // Test C++17 structured bindings
    auto [x, y] = make_pair(10, 20);
    cout << "Pair: (" << x << ", " << y << ")" << endl;
    
    return 0;
}
\`\`\`

### Compile and Run

\`\`\`bash
g++ -std=c++17 -O2 -Wall test.cpp -o test
./test
\`\`\`

**Expected output:**
\`\`\`
Environment test successful!
C++ version: 201703
Vector size: 5
Pair: (10, 20)
\`\`\`

If you see this, **congratulations!** Your environment is ready!

---

## Common Setup Issues

### Issue 1: "g++ not found"

**Solution:**
\`\`\`bash
# Linux
sudo apt install build-essential

# macOS
xcode-select --install

# Windows
# Add MinGW bin directory to PATH
\`\`\`

### Issue 2: "bits/stdc++.h: No such file"

**Solution (macOS):**
\`\`\`bash
# Create the file manually
sudo mkdir -p /usr/local/include/bits
sudo curl https://raw.githubusercontent.com/gcc-mirror/gcc/master/libstdc++-v3/include/precompiled/stdc++.h -o /usr/local/include/bits/stdc++.h
\`\`\`

### Issue 3: Compilation is slow

**Solution:**
\`\`\`bash
# Precompile bits/stdc++.h
sudo g++ -std=c++17 /usr/local/include/bits/stdc++.h
\`\`\`

### Issue 4: Permission denied

**Solution:**
\`\`\`bash
chmod +x solution
./solution
\`\`\`

---

## Summary

**Essential Setup Checklist:**

- ✅ Install G++ compiler (version 7+)
- ✅ Verify with \`g++ --version\`
- ✅ Create compilation alias/script
- ✅ Choose and configure IDE (VS Code recommended)
- ✅ Set up debugging (GDB + VS Code)
- ✅ Test with sample program
- ✅ Understand compiler flags (\`- std=c++17 - O2 - Wall\`)

**Your typical workflow:**
1. Write code in IDE
2. Compile: \`g++ - std=c++17 - O2 - Wall solution.cpp - o solution\`
3. Run: \`./ solution < input.txt\`
4. Debug if needed
5. Submit to online judge

**Pro tip:** Spend time on good setup now, save hours later!

---

## Next Steps

Now that your environment is ready, let's explore the **Modern CP Tool Ecosystem**—tools that top competitive programmers use to work even faster!

**Key Takeaway**: A well-configured environment is your foundation. Invest time here, and you'll code faster and debug quicker throughout your CP journey.
`,
  quizId: 'cp-m1-s3-quiz',
  discussionId: 'cp-m1-s3-discussion',
} as const;
