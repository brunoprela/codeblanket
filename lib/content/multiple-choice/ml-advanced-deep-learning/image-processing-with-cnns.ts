/**
 * Image Processing with CNNs Multiple Choice Questions
 */
export const imageProcessingWithCnnsMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question:
      "When using transfer learning with a pretrained ResNet on a small dataset (5,000 images), what's the recommended approach?",
    options: [
      'Fine-tune the entire network with high learning rate (0.01)',
      'Freeze all convolutional layers, train only the final classifier with moderate learning rate',
      'Add more layers to the network to increase capacity',
      'Remove batch normalization layers and train from scratch',
    ],
    correctAnswer: 1,
    explanation:
      "With small datasets (<10K images), the best approach is to **freeze all convolutional layers and train only the final classifier**. This prevents overfitting because: (1) Pretrained features (edges, textures, objects) from ImageNet already generalize well, (2) Only ~10K parameters in classifier vs millions in conv layers, (3) Reduces training time dramatically. Use moderate learning rate (0.001) since you're not risking damaging pretrained weights. Fine-tuning all layers with small data leads to overfitting. The pretrained network already learned 1000 ImageNet classes, so it has rich features for most visual tasks.",
  },
  {
    id: 'cnn-mc-2',
    question:
      'Which data augmentation technique would be INAPPROPRIATE for training a model to recognize handwritten digits (0-9)?',
    options: [
      'Random rotation (±15 degrees)',
      'Random translation (shift by a few pixels)',
      'Vertical flip (flip upside down)',
      'Small elastic deformations',
    ],
    correctAnswer: 2,
    explanation:
      "**Vertical flip is inappropriate** for digit recognition because it changes the semantic meaning: flipping '6' vertically creates '9' - a different class! Valid augmentations preserve the label. Appropriate augmentations for digits: (1) Small rotations (people write at slight angles), (2) Translation (digit position varies), (3) Elastic deformations (natural handwriting variation), (4) Scaling (different sizes). Bad augmentations: Vertical/horizontal flip (changes digit), extreme rotation (±180°), color changes (digits are grayscale). Rule: Augmentations must simulate realistic variations that don't change the correct label.",
  },
  {
    id: 'cnn-mc-3',
    question: 'In YOLO object detection, what does each grid cell predict?',
    options: [
      'A single bounding box for the largest object in that cell',
      'Multiple bounding boxes (x, y, w, h, confidence) plus class probabilities',
      'Only class probabilities, with boxes predicted separately',
      'A segmentation mask for objects in that region',
    ],
    correctAnswer: 1,
    explanation:
      'Each YOLO grid cell predicts **multiple bounding boxes plus class probabilities**. Specifically: (1) B bounding boxes (typically 2-3), each with (x, y, width, height, confidence), (2) C class probabilities shared across boxes. Total output per cell: B×5 + C values. Example: 7×7 grid, 2 boxes, 20 classes → 7×7×30 output. Why multiple boxes? To handle multiple objects or different aspect ratios in same grid region. The confidence score indicates how certain the model is that a box contains an object. This single-shot approach makes YOLO very fast (~30-60 FPS) compared to region proposal methods.',
  },
  {
    id: 'cnn-mc-4',
    question:
      "What is the primary advantage of U-Net\'s skip connections for semantic segmentation?",
    options: [
      'They reduce the number of parameters needed',
      'They preserve spatial information from encoder to help decoder reconstruct precise boundaries',
      'They enable training much deeper networks (100+ layers)',
      'They eliminate the need for pooling layers',
    ],
    correctAnswer: 1,
    explanation:
      "Skip connections **preserve spatial information** lost during downsampling. During encoding, pooling reduces 256×256 → 8×8, destroying precise locations. Skip connections copy encoder feature maps directly to decoder, bypassing the bottleneck. This provides: (1) Fine spatial details (edges, textures) from shallow layers, (2) Semantic understanding from deep layers, (3) Decoder combines both for precise, meaningful segmentation. Without skip connections, decoder must guess spatial structure from tiny bottleneck → blurry boundaries. With skips: Sharp boundaries, +20-30% accuracy. Note: This is different from ResNet\'s skip connections (enable depth); U-Net's skips preserve spatial resolution.",
  },
  {
    id: 'cnn-mc-5',
    question:
      "You're deploying a CNN for medical image segmentation where precision is critical. Which metric would be most appropriate for evaluation?",
    options: [
      'Accuracy (% of correctly classified pixels)',
      'Dice coefficient (2×|A∩B| / (|A|+|B|))',
      'Mean Squared Error between predicted and true masks',
      'Top-5 accuracy',
    ],
    correctAnswer: 1,
    explanation:
      "**Dice coefficient** is the gold standard for segmentation evaluation, especially in medical imaging. Here's why: (1) **Handles class imbalance**: Medical images often have small regions of interest (tumor = 1% of pixels). Accuracy would be 99% by predicting all background! (2) **Measures overlap**: Dice = 2×(intersection)/(sum of sets) directly measures how well predicted mask overlaps ground truth, (3) **Clinically meaningful**: Dice=0.8 means 80% overlap, easy to interpret, (4) **Symmetric**: Treats false positives and false negatives equally. Range [0,1], higher is better. Medical imaging typically requires Dice > 0.9 for deployment. Also called F1 score in binary segmentation. IoU (Intersection over Union) is similar and also widely used.",
  },
];
