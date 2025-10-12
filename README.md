# CodeBlanket - Binary Search Learning Platform

An interactive learning platform focused on mastering binary search algorithms through hands-on Python coding practice.

## 🎯 Features

- **3 Curated Binary Search Problems** (Easy, Medium, Hard)
- **In-Browser Python Execution** using Pyodide (no backend needed!)
- **Interactive Code Editor** with Monaco Editor
- **Instant Test Feedback** - run test cases immediately
- **Beautiful, Modern UI** built with Next.js 14 and Tailwind CSS
- **100% Client-Side** - all code runs in your browser

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

## 📚 Problems Included

### Easy

1. **Binary Search** - Classic binary search implementation on a sorted array

### Medium

2. **Search in Rotated Sorted Array** - Find target in a rotated sorted array

### Hard

3. **Median of Two Sorted Arrays** - Find median with O(log(m+n)) complexity

## 🏗️ Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with Pyodide script
│   ├── page.tsx            # Home page with problem list
│   └── problems/[id]/
│       └── page.tsx        # Individual problem page
├── components/
│   └── PythonCodeRunner.tsx # Code editor + test runner
├── lib/
│   ├── pyodide.ts          # Pyodide loader
│   ├── types.ts            # TypeScript interfaces
│   └── problems/
│       └── binary-search.ts # Problem definitions
└── ...
```

## 💡 How It Works

1. **Pyodide**: Python (CPython 3.11) compiled to WebAssembly runs directly in the browser
2. **Monaco Editor**: Same editor that powers VS Code
3. **Next.js 14**: React framework with App Router for static generation
4. **No Backend**: Everything runs client-side - zero infrastructure costs!

## 🎓 Learning Path

Start with the **Easy** problem to understand the fundamental binary search pattern, then progress to **Medium** and **Hard** problems to learn advanced variations.

Each problem includes:

- Clear problem description
- Multiple examples with explanations
- Constraints
- Hints (expandable)
- Expected time/space complexity
- Pre-written test cases

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Code Editor**: Monaco Editor
- **Python Runtime**: Pyodide (WebAssembly)

## 📝 Adding New Problems

Edit `lib/problems/binary-search.ts` and add a new problem object:

```typescript
{
  id: 'problem-slug',
  title: 'Problem Title',
  difficulty: 'Easy', // or 'Medium', 'Hard'
  order: 4,
  description: '...',
  examples: [...],
  starterCode: `def solution(...):
    pass
`,
  testCases: [...],
  // ... other fields
}
```

## 🎨 Customization

- **Colors**: Edit Tailwind classes in components
- **Pyodide Version**: Update CDN URL in `app/layout.tsx`
- **Editor Theme**: Change `theme` prop in `PythonCodeRunner.tsx`

## 🐛 Troubleshooting

**Python not loading?**

- Check internet connection (Pyodide is ~10MB CDN download)
- Check browser console for errors
- Try refreshing the page

**Code not running?**

- Ensure function name matches the problem's expected function name
- Check for Python syntax errors
- Make sure you're returning the correct data type

## 📄 License

MIT - Feel free to use this for your own learning!

## 🙏 Built With

- [Next.js](https://nextjs.org/)
- [Pyodide](https://pyodide.org/)
- [Monaco Editor](https://microsoft.github.io/monaco-editor/)
- [Tailwind CSS](https://tailwindcss.com/)

---

**Happy Learning! 🚀** Master binary search one problem at a time.
