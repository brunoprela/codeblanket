/**
 * Advanced Deep Learning Architectures Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { convolutionalNeuralNetworksSection } from '../sections/ml-advanced-deep-learning/convolutional-neural-networks';
import { cnnArchitecturesSection } from '../sections/ml-advanced-deep-learning/cnn-architectures';
import { imageProcessingWithCnnsSection } from '../sections/ml-advanced-deep-learning/image-processing-with-cnns';
import { recurrentNeuralNetworksSection } from '../sections/ml-advanced-deep-learning/recurrent-neural-networks';
import { lstmGruSection } from '../sections/ml-advanced-deep-learning/lstm-gru';
import { sequenceToSequenceSection } from '../sections/ml-advanced-deep-learning/sequence-to-sequence';
import { attentionMechanismSection } from '../sections/ml-advanced-deep-learning/attention-mechanism';
import { transformerArchitectureSection } from '../sections/ml-advanced-deep-learning/transformer-architecture';
import { transferLearningSection } from '../sections/ml-advanced-deep-learning/transfer-learning';
import { autoencodersSection } from '../sections/ml-advanced-deep-learning/autoencoders';
import { generativeAdversarialNetworksSection } from '../sections/ml-advanced-deep-learning/generative-adversarial-networks';
import { graphNeuralNetworksSection } from '../sections/ml-advanced-deep-learning/graph-neural-networks';

// Import quizzes
import { convolutionalNeuralNetworksQuiz } from '../quizzes/ml-advanced-deep-learning/convolutional-neural-networks';
import { cnnArchitecturesQuiz } from '../quizzes/ml-advanced-deep-learning/cnn-architectures';
import { imageProcessingWithCnnsQuiz } from '../quizzes/ml-advanced-deep-learning/image-processing-with-cnns';
import { recurrentNeuralNetworksQuiz } from '../quizzes/ml-advanced-deep-learning/recurrent-neural-networks';
import { lstmGruQuiz } from '../quizzes/ml-advanced-deep-learning/lstm-gru';
import { sequenceToSequenceQuiz } from '../quizzes/ml-advanced-deep-learning/sequence-to-sequence';
import { attentionMechanismQuiz } from '../quizzes/ml-advanced-deep-learning/attention-mechanism';
import { transformerArchitectureQuiz } from '../quizzes/ml-advanced-deep-learning/transformer-architecture';
import { transferLearningQuiz } from '../quizzes/ml-advanced-deep-learning/transfer-learning';
import { autoencodersQuiz } from '../quizzes/ml-advanced-deep-learning/autoencoders';
import { generativeAdversarialNetworksQuiz } from '../quizzes/ml-advanced-deep-learning/generative-adversarial-networks';
import { graphNeuralNetworksQuiz } from '../quizzes/ml-advanced-deep-learning/graph-neural-networks';

// Import multiple choice
import { convolutionalNeuralNetworksMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/convolutional-neural-networks';
import { cnnArchitecturesMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/cnn-architectures';
import { imageProcessingWithCnnsMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/image-processing-with-cnns';
import { recurrentNeuralNetworksMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/recurrent-neural-networks';
import { lstmGruMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/lstm-gru';
import { sequenceToSequenceMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/sequence-to-sequence';
import { attentionmechanismMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/attention-mechanism';
import { transformerArchitectureMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/transformer-architecture';
import { transferLearningMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/transfer-learning';
import { autoencodersMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/autoencoders';
import { generativeAdversarialNetworksMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/generative-adversarial-networks';
import { graphNeuralNetworksMultipleChoice } from '../multiple-choice/ml-advanced-deep-learning/graph-neural-networks';

export const mlAdvancedDeepLearningModule: Module = {
  id: 'ml-advanced-deep-learning',
  title: 'Advanced Deep Learning Architectures',
  description:
    'Master cutting-edge deep learning architectures including CNNs, RNNs, Transformers, GANs, and Graph Neural Networks. Learn to build state-of-the-art models for computer vision, NLP, and structured data',
  category: 'machine-learning',
  difficulty: 'advanced',
  estimatedTime: '30 hours',
  prerequisites: [
    'Deep Learning Fundamentals',
    'Neural Networks',
    'Python for Deep Learning',
  ],
  icon: '🧠',
  keyTakeaways: [
    'CNNs use convolutional layers to extract spatial hierarchies of features from images',
    'Modern CNN architectures (ResNet, EfficientNet) use residual connections and efficient scaling',
    'RNNs process sequences by maintaining hidden state, but suffer from vanishing gradients',
    'LSTMs and GRUs solve vanishing gradients with gating mechanisms for long-term dependencies',
    'Seq2Seq models with attention enable variable-length input-output mapping for translation',
    'Attention mechanism allows models to focus on relevant parts of input dynamically',
    'Transformers eliminate recurrence, enabling parallelization and capturing long-range dependencies',
    'Transfer learning leverages pre-trained models to achieve high performance with limited data',
    'Autoencoders learn compressed representations for dimensionality reduction and generation',
    'GANs train generator and discriminator adversarially to produce realistic synthetic data',
    'VAEs learn structured probabilistic latent spaces for controlled generation',
    'Graph Neural Networks extend deep learning to graph-structured data through message passing',
  ],
  learningObjectives: [
    'Understand and implement Convolutional Neural Networks for computer vision tasks',
    'Apply modern CNN architectures (ResNet, VGG, EfficientNet) to image classification',
    'Implement image processing techniques: data augmentation, object detection, segmentation',
    'Build Recurrent Neural Networks for sequence modeling and time series prediction',
    'Implement LSTM and GRU architectures to capture long-term dependencies',
    'Create sequence-to-sequence models with encoder-decoder architecture for translation',
    'Implement attention mechanisms to focus on relevant input for each output',
    'Build Transformer models from scratch with multi-head self-attention',
    'Apply transfer learning with pre-trained models (BERT, GPT, ResNet) for custom tasks',
    'Implement autoencoders for dimensionality reduction, denoising, and anomaly detection',
    'Build Variational Autoencoders (VAEs) for generative modeling with structured latent space',
    'Implement Generative Adversarial Networks (GANs) to generate realistic synthetic data',
    'Apply Graph Neural Networks to graph-structured data for node and graph classification',
    'Fine-tune and adapt pre-trained models with discriminative learning rates',
    'Evaluate deep learning models with appropriate metrics and visualization techniques',
  ],
  sections: [
    {
      ...convolutionalNeuralNetworksSection,
      quiz: convolutionalNeuralNetworksQuiz,
      multipleChoice: convolutionalNeuralNetworksMultipleChoice,
    },
    {
      ...cnnArchitecturesSection,
      quiz: cnnArchitecturesQuiz,
      multipleChoice: cnnArchitecturesMultipleChoice,
    },
    {
      ...imageProcessingWithCnnsSection,
      quiz: imageProcessingWithCnnsQuiz,
      multipleChoice: imageProcessingWithCnnsMultipleChoice,
    },
    {
      ...recurrentNeuralNetworksSection,
      quiz: recurrentNeuralNetworksQuiz,
      multipleChoice: recurrentNeuralNetworksMultipleChoice,
    },
    {
      ...lstmGruSection,
      quiz: lstmGruQuiz,
      multipleChoice: lstmGruMultipleChoice,
    },
    {
      ...sequenceToSequenceSection,
      quiz: sequenceToSequenceQuiz,
      multipleChoice: sequenceToSequenceMultipleChoice,
    },
    {
      ...attentionMechanismSection,
      quiz: attentionMechanismQuiz,
      multipleChoice: attentionmechanismMultipleChoice,
    },
    {
      ...transformerArchitectureSection,
      quiz: transformerArchitectureQuiz,
      multipleChoice: transformerArchitectureMultipleChoice,
    },
    {
      ...transferLearningSection,
      quiz: transferLearningQuiz,
      multipleChoice: transferLearningMultipleChoice,
    },
    {
      ...autoencodersSection,
      quiz: autoencodersQuiz,
      multipleChoice: autoencodersMultipleChoice,
    },
    {
      ...generativeAdversarialNetworksSection,
      quiz: generativeAdversarialNetworksQuiz,
      multipleChoice: generativeAdversarialNetworksMultipleChoice,
    },
    {
      ...graphNeuralNetworksSection,
      quiz: graphNeuralNetworksQuiz,
      multipleChoice: graphNeuralNetworksMultipleChoice,
    },
  ],
};
