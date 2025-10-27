import { QuizQuestion } from '../../../types';

export const tensorflowKerasFundamentalsQuiz: QuizQuestion[] = [
  {
    id: 'tensorflow-keras-q1',
    question:
      'Compare Keras Sequential API with the Functional API. What are the key differences, advantages, and use cases for each? When would you be forced to use the Functional API instead of Sequential?',
    sampleAnswer: `Keras provides two main APIs for building models, each suited to different architectures:

**Sequential API - Linear Architectures:**

\`\`\`python
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
\`\`\`

**Advantages:**
- Simple, intuitive syntax
- Minimal boilerplate
- Perfect for beginners
- Good for rapid prototyping

**Limitations:**
- Only linear layer stacks
- Single input, single output
- No branching or merging
- No skip connections
- Cannot share layers

**Functional API - Complex Architectures:**

\`\`\`python
inputs = keras.Input (shape=(784,))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model (inputs=inputs, outputs=outputs)
\`\`\`

**Advantages:**
- Supports any topology
- Multiple inputs/outputs
- Skip connections (ResNets)
- Layer sharing
- Access intermediate layers

**When Functional API is Required:**1. **Skip Connections (ResNets):**
\`\`\`python
x = layers.Dense(64)(inputs)
shortcut = x
x = layers.Dense(64)(x)
x = layers.Add()([x, shortcut])  # Cannot do in Sequential!
\`\`\`

2. **Multiple Inputs:**
\`\`\`python
text_input = keras.Input (shape=(100,))
num_input = keras.Input (shape=(10,))
combined = layers.concatenate([text_features, num_features])
\`\`\`

3. **Multiple Outputs:**
\`\`\`python
main_output = layers.Dense(1, name='main')(x)
aux_output = layers.Dense(1, name='auxiliary')(x)
\`\`\`

4. **Non-Linear Topology:**
- Inception modules
- U-Net (encoder-decoder with skip connections)
- Siamese networks

**When to Use Each:**

Sequential: Simple feedforward, learning, prototyping
Functional: ResNets, multi-task, attention, modern architectures`,
    keyPoints: [
      'Sequential: linear stacks only, simple syntax',
      'Functional: any topology, multiple inputs/outputs',
      'Sequential limitations: no branching, skip connections, layer sharing',
      'Functional required for: ResNets, multi-input/output, complex topologies',
      'Both compile and train identically',
      'Functional is superset of Sequential',
      'Modern complex architectures require Functional API',
    ],
  },
  {
    id: 'tensorflow-keras-q2',
    question:
      'Explain how Keras callbacks work and why they are useful. Describe at least three built-in callbacks and their purposes. How would you implement a custom callback to perform actions during training?',
    sampleAnswer: `Callbacks provide hooks to customize training behavior at various points during the training process:

**How Callbacks Work:**

Callbacks are objects with methods called at specific points:
- on_epoch_begin/end
- on_batch_begin/end
- on_train_begin/end

**Built-in Callbacks:**

**1. EarlyStopping:**
\`\`\`python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
\`\`\`
- Stops training when metric stops improving
- Restores weights from best epoch
- Prevents overfitting automatically

**2. ModelCheckpoint:**
\`\`\`python
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)
\`\`\`
- Saves model periodically
- Can save only best model
- Essential for long training runs

**3. ReduceLROnPlateau:**
\`\`\`python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
\`\`\`
- Reduces learning rate when stuck
- Adaptive to training progress
- Helps fine-tune convergence

**Custom Callback:**

\`\`\`python
class CustomCallback (keras.callbacks.Callback):
    def on_epoch_end (self, epoch, logs=None):
        print(f"Epoch {epoch}: loss={logs['loss',]:.4f}")
        
        if logs['val_accuracy',] > 0.95:
            print("Reached target accuracy!")
            self.model.stop_training = True
\`\`\`

**Usage:**
\`\`\`python
model.fit(
    X_train, y_train,
    epochs=100,
    callbacks=[early_stop, checkpoint, custom_callback]
)
\`\`\`

**Why Callbacks Are Powerful:**
- No need to modify training loop
- Reusable across models
- Standard training interface
- Easy to combine multiple callbacks`,
    keyPoints: [
      'Callbacks: hooks at specific training points',
      'EarlyStopping: stops when metric stops improving',
      'ModelCheckpoint: saves best model',
      'ReduceLROnPlateau: adapts learning rate',
      'Custom callbacks: subclass Callback, override methods',
      'Access model, logs through callback methods',
      'Can stop training: self.model.stop_training = True',
      'Combine multiple callbacks in fit()',
    ],
  },
  {
    id: 'tensorflow-keras-q3',
    question:
      'Compare PyTorch and TensorFlow/Keras from a practical perspective. What are the strengths and weaknesses of each framework? In what scenarios would you choose one over the other?',
    sampleAnswer: `Both frameworks are excellent, but they have different design philosophies and practical trade-offs:

**PyTorch:**

**Strengths:**
- Pythonic, intuitive API
- Dynamic computational graphs (flexible)
- Easy debugging (standard Python debugger)
- Explicit control over training loop
- Research-friendly
- Growing production ecosystem

**Weaknesses:**
- More verbose training code
- Manual training loop
- Less mature deployment tools (improving)
- Smaller production footprint historically

**TensorFlow/Keras:**

**Strengths:**
- High-level Keras API (easy to learn)
- Automated training (fit() method)
- Mature production tools (TF Serving, TF Lite)
- Excellent deployment ecosystem
- Strong mobile/edge support
- TensorBoard visualization

**Weaknesses:**
- Less intuitive than PyTorch
- Historically complex (much improved)
- Eager execution added later
- More abstraction layers

**When to Choose Each:**

**Choose PyTorch for:**
- Research projects
- Custom architectures
- Need debugging flexibility
- Academic environment
- Prototyping new ideas
- When you want explicit control

**Choose TensorFlow/Keras for:**
- Production deployment
- Mobile/edge devices
- Large-scale serving
- Enterprise environments
- Quick prototyping with Keras
- When mature deployment tools needed

**Modern Reality:**
- Both are excellent choices
- Converging in capabilities
- PyTorch dominant in research (most papers)
- TensorFlow strong in production (Google scale)
- Personal/team preference matters

**Practical Advice:**
- Learn both (transferable skills)
- Use PyTorch for research/learning
- Use TensorFlow for production at scale
- Or use PyTorch for both (improving production support)`,
    keyPoints: [
      'PyTorch: pythonic, dynamic graphs, easy debugging',
      'TensorFlow/Keras: high-level API, mature deployment',
      'PyTorch dominant in research (papers, academia)',
      'TensorFlow strong in production (serving, mobile)',
      'Both excellent, choice often based on ecosystem',
      'PyTorch: explicit control, research-friendly',
      'TensorFlow: automated training, deployment tools',
      'Both converging, learn transferable concepts',
    ],
  },
];
