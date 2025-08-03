import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️ No GPU found. Running on CPU.")

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0  # shape: (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Define the CNN model using Functional API
inputs = keras.Input(shape=(28, 28, 1))
x = data_augmentation(inputs)
x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

# Train the model
model.fit(
    x_train, y_train,
    validation_split=0.1,
    batch_size=64,
    epochs=20,
    callbacks=[tensorboard_cb, early_stop_cb, checkpoint_cb],
    verbose=2
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# Load and reuse the best model
best_model = keras.models.load_model("best_model.keras")
predictions = best_model.predict(x_test[:5])
print("\n🔍 Sample predictions (first 5):")
print(tf.argmax(predictions, axis=1).numpy())
