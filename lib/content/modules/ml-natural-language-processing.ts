/**
 * Module: Natural Language Processing
 *
 * Comprehensive module covering text processing, embeddings, transformers,
 * and advanced NLP tasks including financial applications
 */

import { Module } from '../../types';

// Section imports
import { textPreprocessingSection } from '../sections/ml-natural-language-processing/text-preprocessing';
import { textRepresentationSection } from '../sections/ml-natural-language-processing/text-representation';
import { wordEmbeddingsSection } from '../sections/ml-natural-language-processing/word-embeddings';
import { contextualizedEmbeddingsSection } from '../sections/ml-natural-language-processing/contextualized-embeddings';
import { sequenceModelingNlpSection } from '../sections/ml-natural-language-processing/sequence-modeling-nlp';
import { attentionNlpSection } from '../sections/ml-natural-language-processing/attention-nlp';
import { transformerModelsNlpSection } from '../sections/ml-natural-language-processing/transformer-models-nlp';
import { finetuningTransformersSection } from '../sections/ml-natural-language-processing/finetuning-transformers';
import { textClassificationSection } from '../sections/ml-natural-language-processing/text-classification';
import { namedEntityRecognitionSection } from '../sections/ml-natural-language-processing/named-entity-recognition';
import { questionAnsweringSection } from '../sections/ml-natural-language-processing/question-answering-information-retrieval';
import { advancedNlpTasksSection } from '../sections/ml-natural-language-processing/advanced-nlp-tasks';

// Quiz imports
import { textPreprocessingQuiz } from '../quizzes/ml-natural-language-processing/text-preprocessing';
import { textRepresentationQuiz } from '../quizzes/ml-natural-language-processing/text-representation';
import { wordEmbeddingsQuiz } from '../quizzes/ml-natural-language-processing/word-embeddings';
import { contextualizedEmbeddingsQuiz } from '../quizzes/ml-natural-language-processing/contextualized-embeddings';
import { sequenceModelingNlpQuiz } from '../quizzes/ml-natural-language-processing/sequence-modeling-nlp';
import { attentionNlpQuiz } from '../quizzes/ml-natural-language-processing/attention-nlp';
import { transformerModelsNlpQuiz } from '../quizzes/ml-natural-language-processing/transformer-models-nlp';
import { finetuningTransformersQuiz } from '../quizzes/ml-natural-language-processing/finetuning-transformers';
import { textClassificationQuiz } from '../quizzes/ml-natural-language-processing/text-classification';
import { namedEntityRecognitionQuiz } from '../quizzes/ml-natural-language-processing/named-entity-recognition';
import { questionAnsweringQuiz } from '../quizzes/ml-natural-language-processing/question-answering-information-retrieval';
import { advancedNlpTasksQuiz } from '../quizzes/ml-natural-language-processing/advanced-nlp-tasks';

// Multiple choice imports
import { textPreprocessingMultipleChoice } from '../multiple-choice/ml-natural-language-processing/text-preprocessing';
import { textRepresentationMultipleChoice } from '../multiple-choice/ml-natural-language-processing/text-representation';
import { wordEmbeddingsMultipleChoice } from '../multiple-choice/ml-natural-language-processing/word-embeddings';
import { contextualizedEmbeddingsMultipleChoice } from '../multiple-choice/ml-natural-language-processing/contextualized-embeddings';
import { sequenceModelingNlpMultipleChoice } from '../multiple-choice/ml-natural-language-processing/sequence-modeling-nlp';
import { attentionNlpMultipleChoice } from '../multiple-choice/ml-natural-language-processing/attention-nlp';
import { transformerModelsNlpMultipleChoice } from '../multiple-choice/ml-natural-language-processing/transformer-models-nlp';
import { finetuningTransformersMultipleChoice } from '../multiple-choice/ml-natural-language-processing/finetuning-transformers';
import { textClassificationMultipleChoice } from '../multiple-choice/ml-natural-language-processing/text-classification';
import { namedEntityRecognitionMultipleChoice } from '../multiple-choice/ml-natural-language-processing/named-entity-recognition';
import { questionAnsweringMultipleChoice } from '../multiple-choice/ml-natural-language-processing/question-answering-information-retrieval';
import { advancedNlpTasksMultipleChoice } from '../multiple-choice/ml-natural-language-processing/advanced-nlp-tasks';

export const naturalLanguageProcessingModule: Module = {
  id: 'ml-natural-language-processing',
  title: 'Natural Language Processing',
  description: 'Master text processing, embeddings, and NLP with transformers',
  icon: '📝',
  keyTakeaways: [
    'Text preprocessing and tokenization are fundamental to NLP pipelines',
    'Word embeddings capture semantic relationships in vector space',
    'Transformers revolutionized NLP through self-attention mechanisms',
    'BERT and GPT represent different approaches to language understanding',
    'Fine-tuning pretrained models achieves state-of-the-art results',
  ],
  sections: [
    {
      ...textPreprocessingSection,
      quiz: textPreprocessingQuiz,
      multipleChoice: textPreprocessingMultipleChoice,
    },
    {
      ...textRepresentationSection,
      quiz: textRepresentationQuiz,
      multipleChoice: textRepresentationMultipleChoice,
    },
    {
      ...wordEmbeddingsSection,
      quiz: wordEmbeddingsQuiz,
      multipleChoice: wordEmbeddingsMultipleChoice,
    },
    {
      ...contextualizedEmbeddingsSection,
      quiz: contextualizedEmbeddingsQuiz,
      multipleChoice: contextualizedEmbeddingsMultipleChoice,
    },
    {
      ...sequenceModelingNlpSection,
      quiz: sequenceModelingNlpQuiz,
      multipleChoice: sequenceModelingNlpMultipleChoice,
    },
    {
      ...attentionNlpSection,
      quiz: attentionNlpQuiz,
      multipleChoice: attentionNlpMultipleChoice,
    },
    {
      ...transformerModelsNlpSection,
      quiz: transformerModelsNlpQuiz,
      multipleChoice: transformerModelsNlpMultipleChoice,
    },
    {
      ...finetuningTransformersSection,
      quiz: finetuningTransformersQuiz,
      multipleChoice: finetuningTransformersMultipleChoice,
    },
    {
      ...textClassificationSection,
      quiz: textClassificationQuiz,
      multipleChoice: textClassificationMultipleChoice,
    },
    {
      ...namedEntityRecognitionSection,
      quiz: namedEntityRecognitionQuiz,
      multipleChoice: namedEntityRecognitionMultipleChoice,
    },
    {
      ...questionAnsweringSection,
      quiz: questionAnsweringQuiz,
      multipleChoice: questionAnsweringMultipleChoice,
    },
    {
      ...advancedNlpTasksSection,
      quiz: advancedNlpTasksQuiz,
      multipleChoice: advancedNlpTasksMultipleChoice,
    },
  ],
};
