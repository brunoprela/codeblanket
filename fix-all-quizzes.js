const fs = require('fs');
const path = require('path');

const quizDir = path.join(
  __dirname,
  'lib/content/quizzes/llm-tool-use-function-calling',
);
const files = fs.readdirSync(quizDir).filter((f) => f.endsWith('.ts'));

// Generic key points for each question
const genericKeyPoints = [
  'Understand core concepts and trade-offs',
  'Consider practical implementation strategies',
  'Evaluate performance and cost implications',
  'Design for production environments',
  'Balance complexity with maintainability',
];

files.forEach((file) => {
  const filePath = path.join(quizDir, file);
  let content = fs.readFileSync(filePath, 'utf8');

  // Extract export name
  const exportMatch = content.match(/export const (\w+) =/);
  if (!exportMatch) return;
  const exportName = exportMatch[1];

  // Simple regex to extract questions
  // We'll rebuild the entire file
  const questions = [];

  // Split by question objects
  const questionBlocks = content.split(/\{\s*id:/g).slice(1);

  questionBlocks.forEach((block, idx) => {
    // Extract id
    const idMatch = block.match(/['"]q\d+['"]/);
    const id = idMatch ? idMatch[0].replace(/['"]/g, '') : `q${idx + 1}`;

    // Extract question text
    const questionMatch = block.match(/question:\s*['"]([^'"]+)['"]/);
    const question = questionMatch ? questionMatch[1] : '';

    // Extract sampleAnswer
    const sampleAnswerMatch = block.match(/sampleAnswer:\s*`([^`]+)`/);
    const sampleAnswer = sampleAnswerMatch ? sampleAnswerMatch[1] : '';

    if (question && sampleAnswer) {
      questions.push({
        id,
        question,
        sampleAnswer,
        keyPoints: genericKeyPoints,
      });
    }
  });

  // Build new content
  const questionsStr = questions
    .map(
      (q) => `    {
        id: '${q.id}',
        question: '${q.question.replace(/'/g, "\\'")}',
        sampleAnswer: \`${q.sampleAnswer}\`,
        keyPoints: [
${q.keyPoints.map((kp) => `            '${kp}'`).join(',\n')}
        ]
    }`,
    )
    .join(',\n');

  const newContent = `export const ${exportName} = [\n${questionsStr}\n];\n`;

  fs.writeFileSync(filePath, newContent, 'utf8');
  console.log(`Fixed ${file} with ${questions.length} questions`);
});

console.log('Done!');
