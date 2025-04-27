import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU Detected:", gpus)
else:
    print("❌ No GPU detected! Training will use CPU.")

# Create a dummy dataset
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)

# Simple model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model and measure time
start = time.time()
history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
end = time.time()

print(f"✅ Training Completed in {end - start:.2f} seconds")
