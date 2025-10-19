/**
 * Module: Deep Learning Fundamentals
 * Topic: ML/AI
 *
 * Comprehensive coverage of neural networks, backpropagation, optimization,
 * regularization, PyTorch/TensorFlow, and efficient training techniques
 */

import { Module } from '../../types';

// Import sections
import { neuralNetworksIntroductionSection } from '../sections/ml-deep-learning-fundamentals/neural-networks-introduction';
import { activationFunctionsSection } from '../sections/ml-deep-learning-fundamentals/activation-functions';
import { forwardPropagationSection } from '../sections/ml-deep-learning-fundamentals/forward-propagation';
import { lossFunctionsSection } from '../sections/ml-deep-learning-fundamentals/loss-functions';
import { backpropagationAlgorithmSection } from '../sections/ml-deep-learning-fundamentals/backpropagation-algorithm';
import { optimizationAlgorithmsSection } from '../sections/ml-deep-learning-fundamentals/optimization-algorithms';
import { weightInitializationSection } from '../sections/ml-deep-learning-fundamentals/weight-initialization';
import { regularizationTechniquesSection } from '../sections/ml-deep-learning-fundamentals/regularization-techniques';
import { trainingNeuralNetworksSection } from '../sections/ml-deep-learning-fundamentals/training-neural-networks';
import { pytorchFundamentalsSection } from '../sections/ml-deep-learning-fundamentals/pytorch-fundamentals';
import { tensorflowKerasFundamentalsSection } from '../sections/ml-deep-learning-fundamentals/tensorflow-keras-fundamentals';
import { deepLearningBestPracticesSection } from '../sections/ml-deep-learning-fundamentals/deep-learning-best-practices';
import { efficientTrainingTechniquesSection } from '../sections/ml-deep-learning-fundamentals/efficient-training-techniques';

// Import quizzes (discussion questions)
import { neuralNetworksIntroductionQuiz } from '../quizzes/ml-deep-learning-fundamentals/neural-networks-introduction';
import { activationFunctionsQuiz } from '../quizzes/ml-deep-learning-fundamentals/activation-functions';
import { forwardPropagationQuiz } from '../quizzes/ml-deep-learning-fundamentals/forward-propagation';
import { lossFunctionsQuiz } from '../quizzes/ml-deep-learning-fundamentals/loss-functions';
import { backpropagationAlgorithmQuiz } from '../quizzes/ml-deep-learning-fundamentals/backpropagation-algorithm';
import { optimizationAlgorithmsQuiz } from '../quizzes/ml-deep-learning-fundamentals/optimization-algorithms';
import { weightInitializationQuiz } from '../quizzes/ml-deep-learning-fundamentals/weight-initialization';
import { regularizationTechniquesQuiz } from '../quizzes/ml-deep-learning-fundamentals/regularization-techniques';
import { trainingNeuralNetworksQuiz } from '../quizzes/ml-deep-learning-fundamentals/training-neural-networks';
import { pytorchFundamentalsQuiz } from '../quizzes/ml-deep-learning-fundamentals/pytorch-fundamentals';
import { tensorflowKerasFundamentalsQuiz } from '../quizzes/ml-deep-learning-fundamentals/tensorflow-keras-fundamentals';
import { deepLearningBestPracticesQuiz } from '../quizzes/ml-deep-learning-fundamentals/deep-learning-best-practices';
import { efficientTrainingTechniquesQuiz } from '../quizzes/ml-deep-learning-fundamentals/efficient-training-techniques';

// Import multiple choice questions
import { neuralNetworksIntroductionMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/neural-networks-introduction';
import { activationFunctionsMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/activation-functions';
import { forwardPropagationMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/forward-propagation';
import { lossFunctionsMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/loss-functions';
import { backpropagationAlgorithmMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/backpropagation-algorithm';
import { optimizationAlgorithmsMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/optimization-algorithms';
import { weightInitializationMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/weight-initialization';
import { regularizationTechniquesMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/regularization-techniques';
import { trainingNeuralNetworksMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/training-neural-networks';
import { pytorchFundamentalsMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/pytorch-fundamentals';
import { tensorflowKerasFundamentalsMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/tensorflow-keras-fundamentals';
import { deepLearningBestPracticesMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/deep-learning-best-practices';
import { efficientTrainingTechniquesMultipleChoice } from '../multiple-choice/ml-deep-learning-fundamentals/efficient-training-techniques';

export const mlDeepLearningFundamentals: Module = {
  id: 'ml-deep-learning-fundamentals',
  title: 'Deep Learning Fundamentals',
  description:
    'Master neural networks from first principles: forward/backpropagation, optimization, regularization, PyTorch/TensorFlow frameworks, and efficient training techniques for production-scale models',
  category: 'ml-ai',
  difficulty: 'advanced',
  estimatedTime: '20-25 hours',
  prerequisites: [
    'Module 6: Python for Data Science',
    'Module 5: Statistics Fundamentals',
    'Module 7: Classical Machine Learning - Supervised Learning',
    'Linear Algebra (vectors, matrices, derivatives)',
  ],
  icon: '🧠',
  keyTakeaways: [
    'Neural networks are universal function approximators',
    'Perceptron is the basic building block; MLP stacks multiple layers',
    'Activation functions introduce non-linearity; ReLU is modern default',
    'Forward propagation computes predictions through matrix multiplications',
    'Loss functions measure prediction error; cross-entropy for classification',
    'Backpropagation computes gradients via chain rule',
    'Gradient descent updates weights: w = w - lr * gradient',
    'SGD uses mini-batches; momentum accelerates convergence',
    'Adam combines momentum and adaptive learning rates',
    'He initialization for ReLU; Xavier for sigmoid/tanh',
    'Batch Normalization normalizes activations, stabilizes training',
    'Dropout randomly deactivates neurons, prevents overfitting',
    'L2 regularization penalizes large weights',
    'Learning rate scheduling essential for convergence',
    'Gradient clipping prevents exploding gradients',
    'PyTorch: dynamic graphs, explicit training loops',
    'TensorFlow/Keras: high-level API, production deployment',
    'Mixed precision training: 2x speed with FP16',
    'Gradient accumulation enables larger effective batch sizes',
    'Data parallelism for multi-GPU training',
  ],
  learningObjectives: [
    'Understand neural network architecture and universal approximation',
    'Implement forward propagation with various activation functions',
    'Derive and implement backpropagation algorithm',
    'Apply optimization algorithms: SGD, Momentum, Adam',
    'Initialize weights properly to prevent vanishing/exploding gradients',
    'Use regularization techniques: dropout, batch norm, L1/L2',
    'Train neural networks with proper batch size and learning rate scheduling',
    'Build models in PyTorch with nn.Module and autograd',
    'Build models in TensorFlow/Keras with Sequential and Functional APIs',
    'Debug training issues systematically',
    'Find optimal hyperparameters with learning rate finder',
    'Scale training with mixed precision, gradient accumulation, multi-GPU',
    'Deploy production-ready models with proper versioning and validation',
  ],
  sections: [
    {
      ...neuralNetworksIntroductionSection,
      quiz: neuralNetworksIntroductionQuiz,
      multipleChoice: neuralNetworksIntroductionMultipleChoice,
    },
    {
      ...activationFunctionsSection,
      quiz: activationFunctionsQuiz,
      multipleChoice: activationFunctionsMultipleChoice,
    },
    {
      ...forwardPropagationSection,
      quiz: forwardPropagationQuiz,
      multipleChoice: forwardPropagationMultipleChoice,
    },
    {
      ...lossFunctionsSection,
      quiz: lossFunctionsQuiz,
      multipleChoice: lossFunctionsMultipleChoice,
    },
    {
      ...backpropagationAlgorithmSection,
      quiz: backpropagationAlgorithmQuiz,
      multipleChoice: backpropagationAlgorithmMultipleChoice,
    },
    {
      ...optimizationAlgorithmsSection,
      quiz: optimizationAlgorithmsQuiz,
      multipleChoice: optimizationAlgorithmsMultipleChoice,
    },
    {
      ...weightInitializationSection,
      quiz: weightInitializationQuiz,
      multipleChoice: weightInitializationMultipleChoice,
    },
    {
      ...regularizationTechniquesSection,
      quiz: regularizationTechniquesQuiz,
      multipleChoice: regularizationTechniquesMultipleChoice,
    },
    {
      ...trainingNeuralNetworksSection,
      quiz: trainingNeuralNetworksQuiz,
      multipleChoice: trainingNeuralNetworksMultipleChoice,
    },
    {
      ...pytorchFundamentalsSection,
      quiz: pytorchFundamentalsQuiz,
      multipleChoice: pytorchFundamentalsMultipleChoice,
    },
    {
      ...tensorflowKerasFundamentalsSection,
      quiz: tensorflowKerasFundamentalsQuiz,
      multipleChoice: tensorflowKerasFundamentalsMultipleChoice,
    },
    {
      ...deepLearningBestPracticesSection,
      quiz: deepLearningBestPracticesQuiz,
      multipleChoice: deepLearningBestPracticesMultipleChoice,
    },
    {
      ...efficientTrainingTechniquesSection,
      quiz: efficientTrainingTechniquesQuiz,
      multipleChoice: efficientTrainingTechniquesMultipleChoice,
    },
  ],
};
