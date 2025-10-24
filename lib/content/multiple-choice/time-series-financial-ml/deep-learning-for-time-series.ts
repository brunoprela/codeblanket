import { MultipleChoiceQuestion } from '@/lib/types';

export const deepLearningForTimeSeriesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dlts-mc-1',
      question:
        'What is the main advantage of LSTMs over vanilla RNNs for financial time series?',
      options: [
        'LSTMs are faster to train',
        'LSTMs solve the vanishing gradient problem, enabling long-term dependencies',
        'LSTMs require less data',
        'LSTMs have fewer parameters',
      ],
      correctAnswer: 1,
      explanation:
        'LSTMs solve the vanishing gradient problem through gating mechanisms (forget gate, input gate, output gate) and cell state. This allows gradient to flow unchanged through many time steps, enabling learning of long-term dependencies (e.g., pattern from 30 days ago). Vanilla RNNs lose gradient information after ~10 steps. LSTMs actually have MORE parameters and train SLOWER than RNNs, but capture dependencies better.',
    },
    {
      id: 'dlts-mc-2',
      question:
        'For stock price prediction with 2,000 samples, which model is most appropriate?',
      options: [
        'Transformer with 6 layers and d_model=512',
        'Simple LSTM with 2 layers and hidden_size=50',
        'Deep CNN with 10 convolutional layers',
        'GPT-style large language model',
      ],
      correctAnswer: 1,
      explanation:
        'With limited data (2K samples), use simple models to avoid overfitting. LSTM(layers=2, hidden=50) has ~50K parameters, appropriate for 2K samples. Transformer with d_model=512 has millions of parameters → severe overfitting. Rule: Parameters should be < 10% of samples. 2K samples → max 200K parameters → simple LSTM is optimal. Deep/complex models need 10K+ samples.',
    },
    {
      id: 'dlts-mc-3',
      question:
        'What does the attention mechanism in Transformers enable for time series?',
      options: [
        'Faster training only',
        'Reduced memory usage',
        'Direct connections between any two time steps, capturing long-range dependencies',
        'Automatic feature engineering',
      ],
      correctAnswer: 2,
      explanation:
        'Attention allows any time step t to directly attend to any past time step t-k without sequential processing. Example: Price at t=0 can directly attend to earnings announcement at t=-30. LSTM must sequentially process 30 steps. Attention matrix[TxT] shows which past steps influence each prediction. This is why Transformers excel at long sequences (100+ steps) where dependencies span distant time steps.',
    },
    {
      id: 'dlts-mc-4',
      question:
        'Why is walk-forward validation critical for deep learning time series models?',
      options: [
        'It makes training faster',
        'It respects temporal ordering and prevents lookahead bias',
        'It increases the amount of training data',
        'It reduces model complexity',
      ],
      correctAnswer: 1,
      explanation:
        'Walk-forward validation trains on past data and tests on future data, simulating real deployment where you only have historical information. Random split would train on scattered future data, creating lookahead bias (unrealistic). Example: Train on 2020-2022, test on 2023 (correct). Random split might train on 2023 days to predict 2020 (wrong, uses future). Critical for trading systems to evaluate realistic performance.',
    },
    {
      id: 'dlts-mc-5',
      question:
        'What is the typical R² (coefficient of determination) for daily stock return prediction with deep learning?',
      options: [
        'R² > 0.8 (highly predictable)',
        'R² = 0.4-0.6 (moderately predictable)',
        'R² = 0.05-0.15 (weakly predictable)',
        'R² < 0 (worse than baseline)',
      ],
      correctAnswer: 2,
      explanation:
        "Financial returns are nearly random walk (weak-form market efficiency). Even best models achieve R² = 0.05-0.15 for daily predictions. R² = 0.10 means model explains 10% of variance. This is realistic and actually useful for trading (slight edge compounds). R² > 0.5 would suggest market inefficiency or overfitting. Don't expect high R² - focus on directional accuracy (52-56%) and profitable signals instead.",
    },
  ];
