import { MultipleChoiceQuestion } from '../../../types';

export const pytorchFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pytorch-mc1',
    question:
      'What is the purpose of the requires_grad=True parameter when creating a PyTorch tensor?',
    options: [
      'It enables GPU acceleration for that tensor',
      'It allows the tensor to be modified during training',
      'It tells PyTorch to track operations on this tensor for automatic differentiation',
      'It improves numerical stability during computations',
    ],
    correctAnswer: 2,
    explanation:
      'requires_grad=True tells PyTorch to track all operations on this tensor, building a computational graph for automatic differentiation. When you call backward(), PyTorch uses this graph to compute gradients. By default, leaf tensors have requires_grad=False for efficiency, but model parameters automatically have it set to True.',
  },
  {
    id: 'pytorch-mc2',
    question:
      'In a PyTorch training loop, why is it necessary to call optimizer.zero_grad() before each backward pass?',
    options: [
      'To reset the model weights to zero',
      'To clear the computational graph from the previous iteration',
      'To clear accumulated gradients, as PyTorch accumulates gradients by default',
      'To initialize the optimizer for the next training step',
    ],
    correctAnswer: 2,
    explanation:
      'PyTorch accumulates gradients by default, adding new gradients to existing .grad values. This is useful for gradient accumulation but problematic for normal training. Without zero_grad(), gradients from previous iterations would accumulate, leading to incorrect weight updates. You must explicitly zero gradients before each backward pass.',
  },
  {
    id: 'pytorch-mc3',
    question:
      'What is the difference between model.train() and model.eval() in PyTorch?',
    options: [
      'train() enables backpropagation while eval() disables it',
      'train() enables training-specific layers like Dropout and BatchNorm, eval() disables them',
      'train() uses the training dataset while eval() uses the validation dataset',
      'train() allows weight updates while eval() freezes weights',
    ],
    correctAnswer: 1,
    explanation:
      'model.train() sets the model to training mode, enabling training-specific behavior like Dropout (randomly dropping neurons) and BatchNorm (computing batch statistics). model.eval() sets evaluation mode, disabling Dropout and using running statistics for BatchNorm. Forgetting to call eval() during inference can lead to incorrect predictions.',
  },
  {
    id: 'pytorch-mc4',
    question: 'When should you use the torch.no_grad() context manager?',
    options: [
      'During training to speed up backpropagation',
      'During validation/inference to disable gradient tracking and save memory',
      'When initializing model weights',
      'When saving the model to disk',
    ],
    correctAnswer: 1,
    explanation: `torch.no_grad() disables gradient tracking during the enclosed code block. This is essential during validation/inference because you don't need gradients (no backpropagation), and tracking them wastes memory and computation. It can reduce memory usage by ~50% during inference. Never use it during training as you need gradients for weight updates.`,
  },
  {
    id: 'pytorch-mc5',
    question:
      'What is the recommended way to save and load a PyTorch model for later use?',
    options: [
      'Save the entire model object using torch.save(model, path)',
      'Save only the model.state_dict() which contains just the weights',
      'Convert the model to ONNX format',
      `Pickle the entire model using Python's pickle module`,
    ],
    correctAnswer: 1,
    explanation:
      'The recommended approach is to save model.state_dict() which contains only the learned parameters (weights and biases). To load, you first create the model architecture, then load the state dict with model.load_state_dict(). This is more portable and flexible than saving the entire model object, which ties you to specific Python/PyTorch versions and code structure.',
  },
];
