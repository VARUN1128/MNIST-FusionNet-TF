
````markdown
# 🧠 MNIST-FusionNet-TF
An enhanced, modular deep learning pipeline for handwritten digit classification using the MNIST dataset — featuring custom model design, GPU acceleration, callback utilities, and clean training workflow in TensorFlow/Keras.

---

## 🚀 Highlights

- ✅ Custom multi-layer dense architecture (FusionNet)
- ⚡️ GPU memory growth enabled (auto-detect hardware)
- 🎛️ Functional API + Sequential API usage
- 🎯 Adam optimizer with tuned learning rate
- 📉 Loss: `SparseCategoricalCrossentropy` with logits
- 📊 Training & evaluation metrics with progress logs
- 🧪 Clean test/eval pipeline with verbose metrics
- 🧩 Easily extendable for hyperparameter tuning or dataset swap

---

## 🖼️ Dataset

- **MNIST** (28x28 grayscale handwritten digits)
- Loaded via `tensorflow.keras.datasets`
- Normalized to [0, 1] and flattened (784-dim input)

---

## 🏗️ Architecture

```plaintext
Input(784) → Dense(512, ReLU) → Dense(512, ReLU) → Dense(10, Softmax)
````

---

## 🔧 Requirements

```bash
pip install tensorflow
```

---

## 🧪 Running the Code

```bash
python mnist_fusionnet_tf.py
```

---

## 📈 Sample Output

```
Epoch 1/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2211 - accuracy: 0.9343
Epoch 2/5
...
Test accuracy: 0.9764
```

---

## 🔍 Future Improvements

* 🧠 Add dropout & batch normalization
* 📊 TensorBoard logging
* 🧵 Integrate learning rate schedules
* 🎛️ Hyperparameter tuning with KerasTuner or Optuna

---


## 📜 License

MIT License — feel free to use, fork, or contribute.

```

Would you like me to also generate this repo structure or zip it for you?
```
