/**
 * Convolutional Neural Networks (CNNs) Multiple Choice Questions
 */
export const convolutionalNeuralNetworksMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question:
      'A convolutional layer with input (28, 28, 1), kernel size 3×3, stride 1, and NO padding will produce an output of which spatial dimensions?',
    options: [
      '(28, 28) - same as input',
      '(26, 26) - shrinks by 2 in each dimension',
      '(30, 30) - expands by 2 in each dimension',
      '(14, 14) - half the input size',
    ],
    correctAnswer: 1,
    explanation:
      "With kernel size k=3, stride s=1, and padding p=0, the output size is: (H - k + 2p)/s + 1 = (28 - 3 + 0)/1 + 1 = 26. Without padding ('valid' convolution), the output shrinks by (k-1) = 2 pixels in each dimension. This is because the 3×3 filter can't be centered on the border pixels.",
  },
  {
    id: 'cnn-mc-2',
    question:
      'What is the primary advantage of using convolutional layers over fully connected layers for image processing?',
    options: [
      'Convolutional layers are always faster to compute',
      'Convolutional layers can only process grayscale images',
      'Convolutional layers use parameter sharing, dramatically reducing the number of parameters',
      "Convolutional layers don't require activation functions",
    ],
    correctAnswer: 2,
    explanation:
      'The key advantage is **parameter sharing** - the same filter (with its weights) is applied across the entire image, reusing parameters. A 3×3 filter has only 9 weights that are used at every position, versus a fully connected layer that would need unique weights for every input-output connection. This reduces parameters by orders of magnitude, prevents overfitting, and exploits spatial structure. For example, a 3×3 conv filter on a 28×28 image needs 9 parameters, while a fully connected layer would need 784 parameters per output neuron.',
  },
  {
    id: 'cnn-mc-3',
    question:
      'In a CNN, what does increasing the stride from 1 to 2 accomplish?',
    options: [
      'Doubles the output spatial dimensions',
      'Reduces the output spatial dimensions by approximately half and decreases computation',
      'Increases the number of parameters in the layer',
      'Only affects the number of channels, not spatial dimensions',
    ],
    correctAnswer: 1,
    explanation:
      "Stride=2 means the filter moves 2 pixels at a time instead of 1, effectively skipping every other position. This reduces the output spatial dimensions by approximately half in each direction (height and width), resulting in ~4× less computation overall. It's often used as an alternative to pooling for downsampling. For example, a 32×32 input with stride=2 produces roughly a 16×16 output, while stride=1 would produce ~32×32 (with same padding).",
  },
  {
    id: 'cnn-mc-4',
    question:
      "What is the 'receptive field' of a neuron in a CNN, and how does it change with network depth?",
    options: [
      'The receptive field is the filter size, which stays constant through all layers',
      "The receptive field is the region of the input that affects that neuron's output; it increases with depth",
      'The receptive field only applies to the output layer',
      'The receptive field is the same as the number of parameters in the layer',
    ],
    correctAnswer: 1,
    explanation:
      "The receptive field is the region of the original input image that influences a particular neuron's activation. Early layers have small receptive fields (e.g., 3×3), seeing only local features. As we go deeper, each neuron's receptive field grows because it aggregates information from neurons in the previous layer, which themselves see larger regions. For example, after 3 conv layers with 3×3 filters, a neuron might see a 7×7 region of the original image. Pooling layers further increase receptive field size. This allows deep layers to capture global context while early layers capture local details.",
  },
  {
    id: 'cnn-mc-5',
    question:
      "You're building a CNN for 224×224 RGB images with 1000 classes. After several conv/pool layers, your feature maps are 7×7×512. Which approach for the final classification layers would be most parameter-efficient?",
    options: [
      'Flatten to 25,088 neurons, then FC layer to 1000 classes (25M+ parameters)',
      'Global average pooling to 512 features, then FC to 1000 classes (512K parameters)',
      'Add more convolutional layers until you reach 1×1×1000',
      'Use max pooling repeatedly until you get 1×1 spatial dimensions',
    ],
    correctAnswer: 1,
    explanation:
      "Global average pooling (GAP) is the most parameter-efficient approach. It averages each 7×7 feature map to a single value, reducing 7×7×512 to 1×512. Then one FC layer: 512 × 1000 + 1000 = ~512K parameters. Option A (flatten) would need 25,088 × 1000 = 25M+ parameters. GAP dramatically reduces parameters, prevents overfitting, and is translation invariant. This technique is used in modern architectures like ResNet and GoogLeNet. It essentially says 'detect if this feature appears anywhere in the image' rather than caring about exact location.",
  },
];
