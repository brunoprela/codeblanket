/**
 * Section: TensorFlow/Keras Fundamentals
 * Module: Deep Learning Fundamentals
 *
 * Covers TensorFlow basics, Keras Sequential and Functional APIs, callbacks,
 * model compilation, training, and production deployment
 */

export const tensorflowKerasFundamentalsSection = {
  id: 'tensorflow-keras-fundamentals',
  title: 'TensorFlow/Keras Fundamentals',
  content: `
# TensorFlow/Keras Fundamentals

## Introduction

**TensorFlow** is Google's deep learning framework, powering production systems at massive scale. **Keras** is its high-level API, making deep learning accessible with minimal code.

**Why TensorFlow/Keras?**
- Production-ready (TensorFlow Serving, TensorFlow Lite)
- Excellent deployment tools
- Strong ecosystem (TensorBoard, TensorFlow Extended)
- Keras: user-friendly, rapid prototyping
- Industry standard for production ML

**What You'll Learn:**
- TensorFlow basics and eager execution
- Keras Sequential API (simple)
- Keras Functional API (complex architectures)
- Custom layers and models
- Callbacks for training control
- Saving and deployment

## Installing TensorFlow

\`\`\`bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow[and-cuda]

# Check installation
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
\`\`\`

## TensorFlow Basics

### Tensors and Operations

\`\`\`python
import tensorflow as tf
import numpy as np

# Create tensors
x = tf.constant([1, 2, 3, 4, 5])
print(x)  # tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)

# From NumPy
arr = np.array([1.0, 2.0, 3.0])
x = tf.constant(arr)

# Common creation functions
zeros = tf.zeros((3, 4))
ones = tf.ones((2, 3))
rand = tf.random.uniform((2, 3))  # Uniform [0, 1)
randn = tf.random.normal((2, 3))  # Normal N(0, 1)

# Tensor operations
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])

print(x + y)        # Element-wise addition
print(x * y)        # Element-wise multiplication
print(tf.tensordot(x, y, axes=1))  # Dot product

# Matrix operations
A = tf.random.normal((3, 4))
B = tf.random.normal((4, 5))
C = tf.matmul(A, B)  # Matrix multiplication
# Or use @ operator
C = A @ B
\`\`\`

### Automatic Differentiation

\`\`\`python
# Compute gradients with GradientTape
x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx)  # 4.0 (since dy/dx = 2x = 2*2)

# Multi-variable
x = tf.Variable(2.0)
w = tf.Variable(3.0)
b = tf.Variable(1.0)

with tf.GradientTape() as tape:
    y = w * x + b
    loss = (y - 10.0) ** 2

# Compute gradients
gradients = tape.gradient(loss, [x, w, b])
print(f"dL/dx = {gradients[0]}")
print(f"dL/dw = {gradients[1]}")
print(f"dL/db = {gradients[2]}")
\`\`\`

## Keras Sequential API

### Building Simple Models

\`\`\`python
from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# Print model summary
model.summary()

'''
Output:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
batch_normalization (Batch)  (None, 512)               2048      
dropout (Dropout)            (None, 512)               0         
dense_1 (Dense)              (None, 256)               131328    
batch_normalization_1 (Batch)(None, 256)               1024      
dropout_1 (Dropout)          (None, 256)               0         
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 538,890
Trainable params: 537,354
Non-trainable params: 1,536
'''
\`\`\`

### Compiling and Training

\`\`\`python
# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Or with more control
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Prepare data
import numpy as np
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)
X_val = np.random.randn(200, 784)
y_val = np.random.randint(0, 10, 200)

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test accuracy: {test_acc:.4f}")

# Predict
predictions = model.predict(X_val)
print(predictions.shape)  # (200, 10) - probabilities for each class
\`\`\`

## Keras Functional API

### For Complex Architectures

\`\`\`python
# Input layer
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

# Output layer
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name='functional_model')

model.summary()
\`\`\`

### Multi-Input/Multi-Output Models

\`\`\`python
# Example: Combine text and numerical features

# Text input
text_input = keras.Input(shape=(100,), name='text')
text_features = layers.Embedding(10000, 128)(text_input)
text_features = layers.LSTM(64)(text_features)

# Numerical input
num_input = keras.Input(shape=(10,), name='numerical')
num_features = layers.Dense(64, activation='relu')(num_input)

# Combine
combined = layers.concatenate([text_features, num_features])
x = layers.Dense(64, activation='relu')(combined)

# Multiple outputs
priority_output = layers.Dense(1, activation='sigmoid', name='priority')(x)
category_output = layers.Dense(5, activation='softmax', name='category')(x)

# Create model
model = keras.Model(
    inputs=[text_input, num_input],
    outputs=[priority_output, category_output]
)

# Compile with multiple losses
model.compile(
    optimizer='adam',
    loss={
        'priority': 'binary_crossentropy',
        'category': 'sparse_categorical_crossentropy'
    },
    loss_weights={'priority': 1.0, 'category': 0.5},
    metrics={
        'priority': 'accuracy',
        'category': 'accuracy'
    }
)

# Train with multiple inputs/outputs
model.fit(
    {'text': text_data, 'numerical': num_data},
    {'priority': priority_labels, 'category': category_labels},
    epochs=10
)
\`\`\`

### Residual Connections (ResNet-style)

\`\`\`python
def residual_block(x, filters):
    """Residual block with skip connection"""
    shortcut = x
    
    # Main path
    x = layers.Dense(filters, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Build model with residual blocks
inputs = keras.Input(shape=(512,))
x = layers.Dense(512, activation='relu')(inputs)

x = residual_block(x, 512)
x = residual_block(x, 512)
x = residual_block(x, 512)

outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
\`\`\`

## Custom Layers and Models

### Custom Layer

\`\`\`python
class CustomDense(layers.Layer):
    def __init__(self, units, activation=None):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        # Initialize weights
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        # Forward pass
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            output = self.activation(output)
        return output

# Use custom layer
model = keras.Sequential([
    CustomDense(512, activation='relu'),
    layers.Dropout(0.5),
    CustomDense(10, activation='softmax')
])
\`\`\`

### Custom Model

\`\`\`python
class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        
        self.dense2 = layers.Dense(256, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        return self.output_layer(x)

# Create and train
model = CustomModel(num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
\`\`\`

## Callbacks

### Built-in Callbacks

\`\`\`python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model checkpoint (save best model)
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard logging
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)

# Learning rate schedule
def lr_schedule(epoch, lr):
    if epoch > 10:
        lr = lr * 0.9
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# Train with callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, reduce_lr, tensorboard, lr_scheduler]
)
\`\`\`

### Custom Callback

\`\`\`python
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Called at end of each epoch
        print(f"\\nEpoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")
        
        # Custom logic (e.g., save samples, logging)
        if logs['val_accuracy'] > 0.95:
            print("Reached 95% validation accuracy!")
            self.model.stop_training = True

# Use custom callback
model.fit(X_train, y_train, epochs=100, callbacks=[CustomCallback()])
\`\`\`

## Saving and Loading Models

### Save/Load Complete Model

\`\`\`python
# Save entire model (architecture + weights + optimizer state)
model.save('my_model.keras')

# Load model
loaded_model = keras.models.load_model('my_model.keras')

# Continue training or predict
loaded_model.fit(X_train, y_train, epochs=5)
predictions = loaded_model.predict(X_test)
\`\`\`

### Save/Load Weights Only

\`\`\`python
# Save weights
model.save_weights('model_weights.h5')

# Load weights (must create model first)
new_model = create_model()  # Same architecture
new_model.load_weights('model_weights.h5')
\`\`\`

### Export for Production

\`\`\`python
# SavedModel format (for TensorFlow Serving)
model.export('saved_model/')

# Load SavedModel
loaded = tf.saved_model.load('saved_model/')

# Convert to TensorFlow Lite (mobile/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
\`\`\`

## Key Differences: PyTorch vs TensorFlow/Keras

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|------------------|
| **API Style** | Pythonic, explicit | High-level, declarative |
| **Computation** | Dynamic graphs | Eager by default (static optional) |
| **Model Definition** | nn.Module, forward() | Sequential/Functional/Subclassing |
| **Training Loop** | Manual (full control) | fit() method (automated) |
| **Debugging** | Easy (Python debugger) | Harder (but improving) |
| **Production** | TorchServe, ONNX | TF Serving, TF Lite (mature) |
| **Research** | Dominant | Growing |
| **Industry** | Growing | Dominant |

## Key Takeaways

1. **Keras Sequential** - simple linear models
2. **Keras Functional** - complex architectures with multiple inputs/outputs
3. **Custom layers/models** - full flexibility
4. **Callbacks** - control training (early stopping, checkpoints, LR scheduling)
5. **Saving** - .keras for complete model, SavedModel for production
6. **TensorBoard** - excellent visualization tool
7. **Production** - TensorFlow has mature deployment ecosystem

## What's Next

You've learned two major frameworks. Next: **Deep Learning Best Practices** - how to debug, find good hyperparameters, and build production-ready models!
`,
};
