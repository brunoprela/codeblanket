import { MultipleChoiceQuestion } from '../../../types';

export const tensorflowKerasFundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tensorflow-keras-mc1',
      question:
        'Which Keras API would you need to use to build a model with skip connections (residual connections)?',
      options: [
        'Sequential API',
        'Functional API',
        'Subclassing API',
        'Either Sequential or Functional API',
      ],
      correctAnswer: 1,
      explanation:
        'The Functional API is required for skip connections because Sequential API only supports linear layer stacks. Skip connections require branching (splitting the data path) and merging (adding/concatenating paths), which the Functional API supports through its flexible input-output specification. Subclassing also works but Functional is more common for this.',
    },
    {
      id: 'tensorflow-keras-mc2',
      question: 'What is the purpose of the EarlyStopping callback in Keras?',
      options: [
        'To stop training after a fixed number of epochs',
        'To stop training when validation loss stops improving for a specified number of epochs',
        'To stop training when training loss reaches zero',
        'To stop training if the model starts overfitting',
      ],
      correctAnswer: 1,
      explanation:
        "EarlyStopping monitors a specified metric (typically validation loss) and stops training if it doesn't improve for a specified number of epochs (patience). It can also restore the weights from the best epoch. This prevents overfitting and saves training time by stopping automatically when further training won't help.",
    },
    {
      id: 'tensorflow-keras-mc3',
      question:
        'In TensorFlow/Keras, what is the difference between model.save() and model.save_weights()?',
      options: [
        'save() is faster but save_weights() is more reliable',
        'save() saves only weights, save_weights() saves the entire model',
        'save() saves architecture + weights + optimizer state, save_weights() saves only weights',
        'They are equivalent, just different naming conventions',
      ],
      correctAnswer: 2,
      explanation:
        'model.save() saves the complete model including architecture, weights, optimizer state, and compilation settings. You can load it and immediately train or predict. model.save_weights() saves only the weight values - you need to recreate the model architecture first, then load the weights. save() is more convenient, save_weights() is useful when you want to transfer weights to a different architecture.',
    },
    {
      id: 'tensorflow-keras-mc4',
      question:
        "How does TensorFlow's GradientTape differ from PyTorch's autograd in terms of usage?",
      options: [
        'GradientTape requires explicit context manager, autograd tracks automatically',
        'GradientTape is faster but less flexible',
        'GradientTape only works with Keras models',
        'There is no meaningful difference in usage',
      ],
      correctAnswer: 0,
      explanation:
        'TensorFlow\'s GradientTape requires you to explicitly wrap the forward pass in a "with tf.GradientTape() as tape:" context manager. PyTorch\'s autograd automatically tracks operations on tensors with requires_grad=True. This makes PyTorch slightly more convenient but GradientTape gives explicit control over what is recorded.',
    },
    {
      id: 'tensorflow-keras-mc5',
      question:
        'What format should you use to deploy a TensorFlow/Keras model for serving predictions in a production environment?',
      options: [
        'The .keras file format',
        'SavedModel format',
        'HDF5 (.h5) format',
        'Pickle the model object',
      ],
      correctAnswer: 1,
      explanation:
        "SavedModel is TensorFlow's universal format for production deployment. It's language-neutral and works with TensorFlow Serving (scalable serving), TensorFlow Lite (mobile/edge), and TensorFlow.js (browser). The .keras and .h5 formats are good for checkpointing during development but SavedModel is the production standard.",
    },
  ];
