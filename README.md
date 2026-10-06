# MiniNN

**A minimal neural network framework built from scratch with NumPy.**

MiniNN is an educational project focused on understanding how neural networks work under the hood. Layers, gradients, losses, and optimization are implemented directly with NumPy, without automatic differentiation or a high-level deep learning framework.

The goal is **learning**, not building a production-ready library.

## Features

- **Layers:** Fully connected (`Linear`) layers and a `Sequential` container
- **Activations:** ReLU, Sigmoid, Tanh, and Softmax
- **Loss functions:** MSE, Binary Cross Entropy, BCE with Logits, and multiclass Cross Entropy
- **Optimization:** Gradient descent using `SGD`
- **Initialization:** He Normal and He Uniform
- **Backpropagation:** Explicit gradients for every layer and loss

## Installation

```bash
git clone https://github.com/JohnsoN98X/MiniNN.git
cd MiniNN
pip install -e .
```

## Quick start

The following example trains a small network on a synthetic regression problem:

```python
import numpy as np

from mini_nn.sequential import Sequential
from mini_nn.layers.linear import Linear
from mini_nn.layers.relu import ReLu
from mini_nn.losses import MSELoss
from mini_nn.optim.sgd import SGD

# Generate data
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 2))
y = (3 * X[:, 0] - 2 * X[:, 1] + 0.5).reshape(-1, 1)

# Build model
model = Sequential([
    Linear(2, 8, initialize="he_normal"),
    ReLu(),
    Linear(8, 1),
])

loss_fn = MSELoss()
optimizer = SGD(layers=model.layers, learning_rate=0.01)

# Train
for epoch in range(1000):
    predictions = model(X)
    loss = loss_fn(y, predictions)

    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE = {loss:.6f}")
```

The `Sequential` container executes forward propagation in layer order and backward propagation in reverse order. The optimizer updates the trainable parameters using gradients computed during backpropagation.

## Experiments

The `notebooks/` directory contains practical demonstrations and small component-level playgrounds, including:

- **Regression:** Training a network to approximate a nonlinear function
- **Binary classification:** Learning a nonlinear boundary with BCE with Logits
- **Multiclass classification:** Training with Softmax and Cross Entropy
- **Component experiments:** Activations, losses, weight initialization, and optimization

## Tests

A small `pytest` test suite checks core computations such as layer forward/backward propagation and selected loss functions.

Run the tests from the project root:

```bash
python -m pytest tests/
```

## What this project explores

MiniNN was built to practice the mathematical and software foundations of neural networks:

1. Matrix operations in fully connected layers
2. The chain rule and explicit backpropagation
3. Loss functions for regression and classification
4. Weight initialization and gradient-based optimization
5. Composing layers into reusable models

## Scope

MiniNN intentionally omits automatic differentiation, GPU acceleration, and production-oriented training infrastructure. It is a compact learning exercise rather than a general-purpose machine learning framework.