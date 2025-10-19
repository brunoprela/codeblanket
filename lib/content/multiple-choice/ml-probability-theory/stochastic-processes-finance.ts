/**
 * Multiple choice questions for Stochastic Processes for Finance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stochasticprocessesfinanceMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'In Geometric Brownian Motion dS = μS dt + σS dW, what does μ represent?',
      options: [
        'Volatility',
        'Expected return (drift)',
        'Initial price',
        'Random shock',
      ],
      correctAnswer: 1,
      explanation:
        'In GBM, μ is the drift parameter representing the expected return or growth rate of the asset. σ is the volatility, and dW is the random Brownian motion shock.',
    },
    {
      id: 'mc2',
      question:
        'Which stochastic process is most appropriate for modeling stock prices?',
      options: [
        'Simple Random Walk',
        'Geometric Brownian Motion',
        'Ornstein-Uhlenbeck',
        'Poisson Process',
      ],
      correctAnswer: 1,
      explanation:
        'Geometric Brownian Motion (GBM) is the standard model for stock prices because it ensures prices stay positive, models percentage changes rather than absolute changes, and has log-normal distribution of returns.',
    },
    {
      id: 'mc3',
      question:
        'In a mean-reverting process dX = θ(μ - X)dt + σdW, what happens when X > μ?',
      options: [
        'X tends to increase',
        'X tends to decrease toward μ',
        'X stays constant',
        'X follows random walk',
      ],
      correctAnswer: 1,
      explanation:
        'When X > μ (above the long-term mean), the drift term θ(μ - X) is negative, causing X to tend downward back toward the mean μ. This is the mean-reverting property.',
    },
    {
      id: 'mc4',
      question:
        'What distribution do inter-arrival times follow in a Poisson process with rate λ?',
      options: ['Normal', 'Uniform', 'Exponential(λ)', 'Poisson(λ)'],
      correctAnswer: 2,
      explanation:
        'In a Poisson process, the times between consecutive events follow an Exponential distribution with rate parameter λ. The count of events in a fixed interval follows a Poisson distribution.',
    },
    {
      id: 'mc5',
      question:
        'Which process is most appropriate for pairs trading (modeling the spread between two cointegrated stocks)?',
      options: [
        'Geometric Brownian Motion',
        'Simple Random Walk',
        'Ornstein-Uhlenbeck (mean-reverting)',
        'Poisson Process',
      ],
      correctAnswer: 2,
      explanation:
        'Pairs trading exploits mean reversion in the spread between two cointegrated stocks. The Ornstein-Uhlenbeck process is a mean-reverting process ideal for modeling this spread.',
    },
  ];
