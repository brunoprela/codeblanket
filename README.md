# CodeBlanket - Comprehensive Learning Platform

An interactive, all-in-one learning platform covering Python, algorithms, system design, finance, AI/ML, and more through hands-on coding practice, structured modules, quizzes, and real-world problems.

## 🎯 Features

### Core Learning Tools

- **567+ Coding Problems** across multiple difficulty levels and topics
- **110+ Interactive Modules** with structured learning paths
- **1,216+ Multiple Choice Questions** for concept reinforcement
- **1,197+ Quiz Questions** with detailed explanations
- **Discussion Questions** with video recording capability
- **In-Browser Python Execution** using Pyodide (no backend needed!)
- **Interactive Code Editor** with Monaco Editor (VS Code powered)
- **Custom Test Cases** - create and save your own test cases for each problem
- **Instant Test Feedback** - run test cases immediately with execution time metrics
- **Progress Tracking** - track your learning journey across all content types
- **Export/Import Progress** - backup and restore your learning progress and video recordings
- **Beautiful, Modern UI** built with Next.js 15 and Tailwind CSS
- **100% Client-Side** - all code runs in your browser with IndexedDB for local storage

### Content Types

- **Hands-on Coding Problems** with predefined and custom test cases, solutions, and explanations
- **Comprehensive Module Sections** with theory, code examples, and interactive exercises
- **Interactive Quizzes** with sample answers for knowledge validation
- **Multiple Choice Assessments** with detailed explanations and key points
- **Discussion Prompts** with video recording for deeper understanding and practice
- **Progress Dashboards** showing completion across sections, problems, quizzes, and discussions

## 📚 Topics Covered

### Core Programming

- **🐍 Python** - Fundamentals, Intermediate, Advanced, OOP, Async, Django, FastAPI, Testing
- **⚡ Algorithms & Data Structures** - Arrays, Trees, Graphs, Dynamic Programming, Backtracking, and more
- **🏆 Competitive Programming** - Advanced problem-solving techniques

### Software Engineering

- **🏗️ System Design** - Fundamentals, Databases, Networking, Microservices, Distributed Systems, Real-world Architectures
- **⚛️ Frontend Development** - React & Next.js fundamentals
- **☁️ DevOps & AWS** - Linux System Administration, Cloud Infrastructure

### Data Science & AI

- **🤖 Applied AI / LLM Engineering** - Prompt Engineering, RAG, Multi-Agent Systems, Code Generation, Tool Use, Production Systems
- **🧠 Machine Learning** - Mathematical Foundations, Supervised/Unsupervised Learning, Deep Learning, NLP, Large Language Models
- **📊 Quantitative Programming** - Financial ML, Time Series Analysis, Statistical Methods

### Finance & Trading

- **💰 Finance Foundations** - Corporate Finance, Financial Statements, Valuation, Portfolio Theory
- **📈 Trading & Markets** - Algorithmic Trading, Risk Management, Options & Derivatives, Market Microstructure
- **🔢 Quantitative Finance** - Time Series, Financial ML, Backtesting, Trading Infrastructure

### Career Development

- **💼 Product Management** - Fundamentals and best practices
- **👔 Engineering Management** - Leadership and team management
- **💬 Behavioral Interview Prep** - Common interview questions and answers
- **📦 Big Data** - Distributed systems and data processing
- **🔐 Crypto & Blockchain** - Blockchain fundamentals

## 📊 Content Statistics

- **567 Coding Problems** - From easy warm-ups to hard challenges
- **110 Modules** - Comprehensive learning paths
- **1,241 Sections** - Detailed content sections with theory, examples, and exercises
- **1,216 Multiple Choice Questions** - Test your understanding with instant feedback
- **1,197 Quiz Questions** - Reinforce concepts with detailed sample answers
- **84 Discussion Questions** - Deep dive into topics with video recording support
- **14 Learning Topics** - Organized into clear learning paths

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ (you have v22.13.1 ✅)
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🎓 Learning Paths

The platform organizes content into **topic-based learning paths**, each containing multiple modules:

1. **Choose a Topic** - Select from 14 different topic areas
2. **Explore Modules** - Each topic contains 5-20+ modules with structured content
3. **Complete Sections** - Work through detailed sections with theory and examples
4. **Practice Problems** - Solve coding problems related to each module
5. **Take Quizzes** - Validate your understanding with multiple choice and discussion questions
6. **Track Progress** - Monitor your completion across sections, problems, quizzes, and discussions

### Example Learning Journey

**Python Fundamentals Path:**

1. Start with Python Fundamentals module
2. Read through sections on syntax, data types, and control flow
3. Complete multiple choice questions to reinforce concepts
4. Practice with coding problems from easy to hard
5. Record discussion question responses to solidify understanding
6. Move to Python Intermediate → Advanced → OOP → etc.

**System Design Path:**

1. Begin with System Design Fundamentals
2. Learn core building blocks (databases, caching, load balancing)
3. Study trade-offs and design patterns
4. Explore real-world architectures (Twitter, Uber, Netflix)
5. Practice with case studies and practical projects

## 🏗️ Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout with Pyodide script
│   ├── page.tsx                # Home page with learning paths
│   ├── modules/[slug]/         # Module viewing pages
│   ├── problems/[id]/          # Individual problem pages
│   └── topics/[slug]/          # Topic problem listing pages
├── components/
│   ├── PythonCodeRunner.tsx    # Code editor + test runner with custom test cases
│   ├── SimpleCodeEditor.tsx    # Alternative code editor for simpler problems
│   ├── MultipleChoiceQuiz.tsx  # Interactive quiz component
│   ├── VideoRecorder.tsx       # Discussion question video recorder
│   ├── ExportImportMenu.tsx    # Progress backup/restore functionality
│   ├── InteractiveCodeBlock.tsx # Syntax-highlighted code blocks
│   ├── MathText.tsx            # Math rendering with KaTeX
│   └── ...                     # Other interactive components
├── lib/
│   ├── content/
│   │   ├── modules/            # 110 module definitions
│   │   ├── sections/           # 1,241 content sections
│   │   ├── problems/           # 567 problem definitions
│   │   ├── multiple-choice/    # 1,216+ MCQ definitions
│   │   ├── quizzes/            # 1,197+ quiz definitions
│   │   ├── discussions/        # 84+ discussion questions
│   │   └── topics/             # Topic organization
│   ├── pyodide.ts              # Pyodide loader
│   ├── types.ts                # TypeScript interfaces
│   └── helpers/                # Storage and utilities
└── curricula/                  # 13 curriculum markdown files
```

## 💡 How It Works

1. **Pyodide**: Python (CPython 3.11) compiled to WebAssembly runs directly in the browser - no server needed!
2. **Monaco Editor**: Same editor that powers VS Code for a familiar coding experience with syntax highlighting
3. **Next.js 15**: React framework with App Router for optimal performance and SEO
4. **IndexedDB**: Client-side storage for progress tracking, code persistence, custom test cases, and video recordings
5. **LocalStorage Sync**: Hybrid storage approach with IndexedDB for large data (videos) and localStorage for fast access
6. **No Backend**: Everything runs client-side - zero infrastructure costs, infinite scalability!

## 🛠️ Tech Stack

- **Framework**: Next.js 15 (App Router) with React 19
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS 4 with custom Dracula theme
- **Code Editor**: Monaco Editor (@monaco-editor/react)
- **Python Runtime**: Pyodide (CPython 3.11 compiled to WebAssembly)
- **Storage**: IndexedDB + localStorage hybrid for optimal performance
- **Math Rendering**: KaTeX via react-katex
- **Syntax Highlighting**: react-syntax-highlighter and Prism.js
- **Deployment**: Vercel with optimized build configuration

## 📝 Adding Content

### Adding a New Problem

Edit the appropriate file in `lib/content/problems/[topic]/`:

```typescript
{
  id: 'problem-slug',
  title: 'Problem Title',
  difficulty: 'Easy', // or 'Medium', 'Hard'
  topic: 'Arrays & Hashing',
  description: '...',
  examples: [...],
  starterCode: `def solution(...):
    pass
`,
  testCases: [...],
  // ... other fields
}
```

### Adding a New Module

1. Create a module file in `lib/content/modules/`
2. Import sections, quizzes, and multiple choice questions
3. Define the module object with sections array
4. Add the module to the appropriate topic in `lib/content/topics/`

### Adding Content to a Module

- **Sections**: Add content files in `lib/content/sections/[module-name]/`
- **Quizzes**: Add quiz files in `lib/content/quizzes/[module-name]/`
- **Multiple Choice**: Add MCQ files in `lib/content/multiple-choice/[module-name]/`

## 🎨 Customization

- **Colors**: Edit Tailwind classes in components (uses Dracula theme by default)
- **Pyodide Version**: Update CDN URL in `app/layout.tsx`
- **Editor Theme**: Change `theme` prop in `PythonCodeRunner.tsx`
- **Progress Tracking**: Modify storage helpers in `lib/helpers/`

## 🚀 Deployment

### Deploy to Vercel

The easiest way to deploy this Next.js app is using [Vercel](https://vercel.com):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/brunoprela/codeblanket-frontend)

#### Manual Deployment Steps:

1. **Install Vercel CLI** (optional):

   ```bash
   npm i -g vercel
   ```

2. **Connect to Vercel**:

   ```bash
   vercel login
   ```

3. **Deploy**:

   ```bash
   vercel
   ```

4. **Production deployment**:
   ```bash
   vercel --prod
   ```

#### Automatic Deployment:

1. Push your code to GitHub
2. Import your repository in [Vercel Dashboard](https://vercel.com/dashboard)
3. Vercel will automatically detect Next.js and configure the build
4. Every push to `main` will trigger a new deployment

#### Environment Variables:

If you need environment variables:

1. Go to your project in Vercel Dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add your variables

### Build Configuration:

The project includes `vercel.json` with optimized settings:

- Build command: `pnpm run build`
- Framework detection: Next.js 15
- Security headers configured
- Pyodide CDN proxying for better performance

## 🐛 Troubleshooting

**Python not loading?**

- Check internet connection (Pyodide is ~10MB CDN download)
- Check browser console for errors
- Try refreshing the page

**Code not running?**

- Ensure function name matches the problem's expected function name
- Check for Python syntax errors
- Make sure you're returning the correct data type

**Progress not saving?**

- Check browser console for IndexedDB errors
- Ensure cookies/local storage is enabled
- Try using the Export/Import feature to backup and restore
- Use the "Force Sync to IndexedDB" button if sync seems stuck

**Videos not saving?**

- Videos are stored in IndexedDB (larger storage capacity)
- Check browser storage quota in DevTools → Application → Storage
- Ensure sufficient disk space available

**Vercel Build Failing?**

- Ensure `pnpm-lock.yaml` is committed
- Check build logs in Vercel Dashboard
- Run `pnpm run build` locally first
- Verify all dependencies are in `package.json`

## 📄 License

MIT - Feel free to use this for your own learning!

## 🙏 Built With

- [Next.js](https://nextjs.org/)
- [Pyodide](https://pyodide.org/)
- [Monaco Editor](https://microsoft.github.io/monaco-editor/)
- [Tailwind CSS](https://tailwindcss.com/)

---

**Happy Learning! 🚀** Master programming, algorithms, system design, finance, AI, and more, one module at a time.
