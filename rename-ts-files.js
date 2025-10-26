const fs = require('fs');
const path = require('path');

const base = '/Users/bruno/Developer/codeblanket/frontend/lib/content';

// Process quiz files
const quizDir = path.join(base, 'quizzes/time-series-analysis');
const quizFiles = fs.readdirSync(quizDir).filter(f => f.endsWith('-discussion.ts'));

quizFiles.forEach(oldName => {
  const oldPath = path.join(quizDir, oldName);
  const newName = oldName.replace('-discussion.ts', '.ts');
  const newPath = path.join(quizDir, newName);
  
  // Read and update content
  let content = fs.readFileSync(oldPath, 'utf8');
  content = content.replace(/DiscussionQuestions/g, 'Quiz');
  
  // Write to new path
  fs.writeFileSync(newPath, content);
  
  // Delete old if different
  if (oldPath !== newPath) {
    fs.unlinkSync(oldPath);
    console.log(`Renamed: ${oldName} -> ${newName}`);
  }
});

// Process multiple-choice files
const mcDir = path.join(base, 'multiple-choice/time-series-analysis');
const mcFiles = fs.readdirSync(mcDir).filter(f => f.endsWith('-quiz.ts'));

mcFiles.forEach(oldName => {
  const oldPath = path.join(mcDir, oldName);
  const newName = oldName.replace('-quiz.ts', '.ts');
  const newPath = path.join(mcDir, newName);
  
  // Read and update content
  let content = fs.readFileSync(oldPath, 'utf8');
  content = content.replace(/MultipleChoiceQuestions/g, 'MultipleChoice');
  
  // Write to new path
  fs.writeFileSync(newPath, content);
  
  // Delete old if different
  if (oldPath !== newPath) {
    fs.unlinkSync(oldPath);
    console.log(`Renamed: ${oldName} -> ${newName}`);
  }
});

console.log('Done!');

