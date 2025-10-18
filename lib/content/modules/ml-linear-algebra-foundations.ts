/**
 * Linear Algebra Foundations Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { vectorsfundamentalsSection } from '../sections/ml-linear-algebra-foundations/vectors-fundamentals';
import { vectoroperationsSection } from '../sections/ml-linear-algebra-foundations/vector-operations';
import { matricesfundamentalsSection } from '../sections/ml-linear-algebra-foundations/matrices-fundamentals';
import { matrixoperationsSection } from '../sections/ml-linear-algebra-foundations/matrix-operations';
import { specialmatricesSection } from '../sections/ml-linear-algebra-foundations/special-matrices';
import { matrixinversedeterminantsSection } from '../sections/ml-linear-algebra-foundations/matrix-inverse-determinants';
import { systemslinearequationsSection } from '../sections/ml-linear-algebra-foundations/systems-linear-equations';
import { vectorspacesSection } from '../sections/ml-linear-algebra-foundations/vector-spaces';
import { eigenvalueseigenvectorsSection } from '../sections/ml-linear-algebra-foundations/eigenvalues-eigenvectors';
import { matrixdecompositionsSection } from '../sections/ml-linear-algebra-foundations/matrix-decompositions';
import { principalcomponentanalysisSection } from '../sections/ml-linear-algebra-foundations/principal-component-analysis';
import { lineartransformationsSection } from '../sections/ml-linear-algebra-foundations/linear-transformations';
import { tensoroperationsSection } from '../sections/ml-linear-algebra-foundations/tensor-operations';
import { sparselinearalgebraSection } from '../sections/ml-linear-algebra-foundations/sparse-linear-algebra';

// Import quizzes
import { vectorsfundamentalsQuiz } from '../quizzes/ml-linear-algebra-foundations/vectors-fundamentals';
import { vectoroperationsQuiz } from '../quizzes/ml-linear-algebra-foundations/vector-operations';
import { matricesfundamentalsQuiz } from '../quizzes/ml-linear-algebra-foundations/matrices-fundamentals';
import { matrixoperationsQuiz } from '../quizzes/ml-linear-algebra-foundations/matrix-operations';
import { specialmatricesQuiz } from '../quizzes/ml-linear-algebra-foundations/special-matrices';
import { matrixinversedeterminantsQuiz } from '../quizzes/ml-linear-algebra-foundations/matrix-inverse-determinants';
import { systemslinearequationsQuiz } from '../quizzes/ml-linear-algebra-foundations/systems-linear-equations';
import { vectorspacesQuiz } from '../quizzes/ml-linear-algebra-foundations/vector-spaces';
import { eigenvalueseigenvectorsQuiz } from '../quizzes/ml-linear-algebra-foundations/eigenvalues-eigenvectors';
import { matrixdecompositionsQuiz } from '../quizzes/ml-linear-algebra-foundations/matrix-decompositions';
import { principalcomponentanalysisQuiz } from '../quizzes/ml-linear-algebra-foundations/principal-component-analysis';
import { lineartransformationsQuiz } from '../quizzes/ml-linear-algebra-foundations/linear-transformations';
import { tensoroperationsQuiz } from '../quizzes/ml-linear-algebra-foundations/tensor-operations';
import { sparselinearalgebraQuiz } from '../quizzes/ml-linear-algebra-foundations/sparse-linear-algebra';

// Import multiple choice
import { vectorsfundamentalsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/vectors-fundamentals';
import { vectoroperationsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/vector-operations';
import { matricesfundamentalsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/matrices-fundamentals';
import { matrixoperationsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/matrix-operations';
import { specialmatricesMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/special-matrices';
import { matrixinversedeterminantsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/matrix-inverse-determinants';
import { systemslinearequationsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/systems-linear-equations';
import { vectorspacesMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/vector-spaces';
import { eigenvalueseigenvectorsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/eigenvalues-eigenvectors';
import { matrixdecompositionsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/matrix-decompositions';
import { principalcomponentanalysisMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/principal-component-analysis';
import { lineartransformationsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/linear-transformations';
import { tensoroperationsMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/tensor-operations';
import { sparselinearalgebraMultipleChoice } from '../multiple-choice/ml-linear-algebra-foundations/sparse-linear-algebra';

export const mlLinearAlgebraFoundationsModule: Module = {
  id: 'ml-linear-algebra-foundations',
  title: 'Linear Algebra Foundations',
  description:
    'Master vectors, matrices, tensors, and linear transformations - the language of machine learning and deep learning',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [
    'Basic algebra and functions',
    'Python programming fundamentals',
    'NumPy basics',
  ],
  icon: '🔷',
  keyTakeaways: [
    'Vectors represent data points, features, and parameters in ML',
    'Dot product measures similarity and is fundamental to neural networks',
    'Matrices enable compact representation of datasets and transformations',
    'Matrix multiplication is the core operation in neural networks',
    'Different norms (L1, L2, L∞) serve different purposes in ML',
    'Eigenvalues and eigenvectors reveal important data structure',
    'SVD is a powerful decomposition used in dimensionality reduction',
    'PCA uses linear algebra to find principal components',
    'Sparse matrices enable efficient large-scale computation',
    'Understanding linear algebra is essential for deep learning',
  ],
  learningObjectives: [
    'Understand vectors and their geometric interpretation',
    'Perform vector operations: dot product, norms, distances',
    'Master matrix operations and multiplication',
    'Apply linear transformations using matrices',
    'Compute eigenvalues, eigenvectors, and matrix decompositions',
    'Implement PCA for dimensionality reduction',
    'Work with sparse matrices efficiently',
    'Use linear algebra in machine learning algorithms',
    'Implement neural network operations with matrices',
    'Debug common linear algebra errors in ML code',
  ],
  sections: [
    {
      ...vectorsfundamentalsSection,
      quiz: vectorsfundamentalsQuiz,
      multipleChoice: vectorsfundamentalsMultipleChoice,
    },
    {
      ...vectoroperationsSection,
      quiz: vectoroperationsQuiz,
      multipleChoice: vectoroperationsMultipleChoice,
    },
    {
      ...matricesfundamentalsSection,
      quiz: matricesfundamentalsQuiz,
      multipleChoice: matricesfundamentalsMultipleChoice,
    },
    {
      ...matrixoperationsSection,
      quiz: matrixoperationsQuiz,
      multipleChoice: matrixoperationsMultipleChoice,
    },
    {
      ...specialmatricesSection,
      quiz: specialmatricesQuiz,
      multipleChoice: specialmatricesMultipleChoice,
    },
    {
      ...matrixinversedeterminantsSection,
      quiz: matrixinversedeterminantsQuiz,
      multipleChoice: matrixinversedeterminantsMultipleChoice,
    },
    {
      ...systemslinearequationsSection,
      quiz: systemslinearequationsQuiz,
      multipleChoice: systemslinearequationsMultipleChoice,
    },
    {
      ...vectorspacesSection,
      quiz: vectorspacesQuiz,
      multipleChoice: vectorspacesMultipleChoice,
    },
    {
      ...eigenvalueseigenvectorsSection,
      quiz: eigenvalueseigenvectorsQuiz,
      multipleChoice: eigenvalueseigenvectorsMultipleChoice,
    },
    {
      ...matrixdecompositionsSection,
      quiz: matrixdecompositionsQuiz,
      multipleChoice: matrixdecompositionsMultipleChoice,
    },
    {
      ...principalcomponentanalysisSection,
      quiz: principalcomponentanalysisQuiz,
      multipleChoice: principalcomponentanalysisMultipleChoice,
    },
    {
      ...lineartransformationsSection,
      quiz: lineartransformationsQuiz,
      multipleChoice: lineartransformationsMultipleChoice,
    },
    {
      ...tensoroperationsSection,
      quiz: tensoroperationsQuiz,
      multipleChoice: tensoroperationsMultipleChoice,
    },
    {
      ...sparselinearalgebraSection,
      quiz: sparselinearalgebraQuiz,
      multipleChoice: sparselinearalgebraMultipleChoice,
    },
  ],
};
