/**
 * Module: Video & Audio Generation
 * Module 9 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { videoGenerationFundamentals } from '../sections/video-audio-generation/video-generation-fundamentals';
import { textToVideoModels } from '../sections/video-audio-generation/text-to-video-models';
import { imageToVideoAnimation } from '../sections/video-audio-generation/image-to-video-animation';
import { videoEditingWithAI } from '../sections/video-audio-generation/video-editing-with-ai';
import { speechToTextWhisper } from '../sections/video-audio-generation/speech-to-text-whisper';
import { textToSpeechElevenlabs } from '../sections/video-audio-generation/text-to-speech-elevenlabs';
import { musicAudioGeneration } from '../sections/video-audio-generation/music-audio-generation';
import { audioProcessingAnalysis } from '../sections/video-audio-generation/audio-processing-analysis';
import { lipSyncAvatarGeneration } from '../sections/video-audio-generation/lip-sync-avatar-generation';
import { buildingMediaGenerationStudio } from '../sections/video-audio-generation/building-media-generation-studio';

// Quiz imports
import { videoGenerationFundamentalsQuiz } from '../quizzes/video-audio-generation/video-generation-fundamentals';
import { textToVideoModelsQuiz } from '../quizzes/video-audio-generation/text-to-video-models';
import { imageToVideoAnimationQuiz } from '../quizzes/video-audio-generation/image-to-video-animation';
import { videoEditingWithAIQuiz } from '../quizzes/video-audio-generation/video-editing-with-ai';
import { speechToTextWhisperQuiz } from '../quizzes/video-audio-generation/speech-to-text-whisper';
import { textToSpeechElevenlabsQuiz } from '../quizzes/video-audio-generation/text-to-speech-elevenlabs';
import { musicAudioGenerationQuiz } from '../quizzes/video-audio-generation/music-audio-generation';
import { audioProcessingAnalysisQuiz } from '../quizzes/video-audio-generation/audio-processing-analysis';
import { lipSyncAvatarGenerationQuiz } from '../quizzes/video-audio-generation/lip-sync-avatar-generation';
import { buildingMediaGenerationStudioQuiz } from '../quizzes/video-audio-generation/building-media-generation-studio';

// Multiple choice imports
import { videogenerationfundamentalsMultipleChoice } from '../multiple-choice/video-audio-generation/video-generation-fundamentals';
import { texttovideommodelsMultipleChoice } from '../multiple-choice/video-audio-generation/text-to-video-models';
import { imagetovideanimationMultipleChoice } from '../multiple-choice/video-audio-generation/image-to-video-animation';
import { videoeditingwithaiMultipleChoice } from '../multiple-choice/video-audio-generation/video-editing-with-ai';
import { speechtotextwhisperMultipleChoice } from '../multiple-choice/video-audio-generation/speech-to-text-whisper';
import { texttospeechelevenlabsMultipleChoice } from '../multiple-choice/video-audio-generation/text-to-speech-elevenlabs';
import { musicaudiogenerationMultipleChoice } from '../multiple-choice/video-audio-generation/music-audio-generation';
import { audioprocessinganalysisMultipleChoice } from '../multiple-choice/video-audio-generation/audio-processing-analysis';
import { lipsyncavatargenerationMultipleChoice } from '../multiple-choice/video-audio-generation/lip-sync-avatar-generation';
import { buildingmediagenerationstudioMultipleChoice } from '../multiple-choice/video-audio-generation/building-media-generation-studio';

export const videoAudioGenerationModule: Module = {
  id: 'applied-ai-video-audio',
  title: 'Video & Audio Generation',
  description:
    'Master the cutting-edge field of generative media AI. Learn to generate and manipulate videos, create realistic speech, produce music, and build production-ready media generation platforms with comprehensive hands-on implementations.',
  icon: '🎬',
  keyTakeaways: [
    'Master text-to-video generation with Runway, Pika, and Stable Video Diffusion',
    'Transform images into animated videos with consistent motion',
    'Generate professional voiceovers with ElevenLabs and text-to-speech APIs',
    'Transcribe audio with Whisper for multi-language speech recognition',
    'Create music and audio effects with AI generation models',
    'Edit and manipulate videos programmatically with AI-powered tools',
    'Build lip-synced avatars for video content generation',
    'Design complete media generation studios with production pipelines',
    'Optimize costs and quality for generative media at scale',
    'Deploy media generation APIs and services in production',
  ],
  sections: [
    {
      ...videoGenerationFundamentals,
      quiz: videoGenerationFundamentalsQuiz,
      multipleChoice: videogenerationfundamentalsMultipleChoice,
    },
    {
      ...textToVideoModels,
      quiz: textToVideoModelsQuiz,
      multipleChoice: texttovideommodelsMultipleChoice,
    },
    {
      ...imageToVideoAnimation,
      quiz: imageToVideoAnimationQuiz,
      multipleChoice: imagetovideanimationMultipleChoice,
    },
    {
      ...videoEditingWithAI,
      quiz: videoEditingWithAIQuiz,
      multipleChoice: videoeditingwithaiMultipleChoice,
    },
    {
      ...speechToTextWhisper,
      quiz: speechToTextWhisperQuiz,
      multipleChoice: speechtotextwhisperMultipleChoice,
    },
    {
      ...textToSpeechElevenlabs,
      quiz: textToSpeechElevenlabsQuiz,
      multipleChoice: texttospeechelevenlabsMultipleChoice,
    },
    {
      ...musicAudioGeneration,
      quiz: musicAudioGenerationQuiz,
      multipleChoice: musicaudiogenerationMultipleChoice,
    },
    {
      ...audioProcessingAnalysis,
      quiz: audioProcessingAnalysisQuiz,
      multipleChoice: audioprocessinganalysisMultipleChoice,
    },
    {
      ...lipSyncAvatarGeneration,
      quiz: lipSyncAvatarGenerationQuiz,
      multipleChoice: lipsyncavatargenerationMultipleChoice,
    },
    {
      ...buildingMediaGenerationStudio,
      quiz: buildingMediaGenerationStudioQuiz,
      multipleChoice: buildingmediagenerationstudioMultipleChoice,
    },
  ],
};
