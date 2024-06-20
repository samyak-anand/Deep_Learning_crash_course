# Deep Learning

## Common Architectural Principles of Deep Networks
#### Core components of Deep Networks are
    - Parameters
    - Layers
    - Activation Functions
    - Loss Functions
    - Optmization Methods
    - Hyperparameters

#### Blocks of deep network 
    - RBM's
    - Autoencoders

#### Deep network architecture

    - UPN's
    - CNN's

## Core Components of Deep Networks

### 1. Parameters
**Definition:**
Parameters are the elements within the network that are learned from the training data. In neural networks, these primarily include weights and biases.



**Details:**

- **Weights:**
  - Each connection between neurons has an associated weight. These weights determine the strength and direction of the influence one neuron has on another.
  - Initial Values: Weights are often initialized randomly or using specific strategies like He initialization or Xavier initialization to start the training process.
  - Updates: During training, weights are adjusted through backpropagation to minimize the loss function.
  - Formula for a simple neuron:
    <pre>
    y = σ(Σ w_i x_i + b)
    </pre>

- **Biases:**
  - Each neuron has a bias that is added to the weighted sum of inputs. This bias helps the network to better fit the data by allowing the activation function to shift.
  - Role: Biases ensure that neurons can activate even when all input values are zero, providing more flexibility in learning.

### 2. Layers
**Definition:**
Layers are the building blocks of neural networks. Each layer consists of a set of neurons (nodes) that perform specific transformations on the input data.

**Types of Layers:**

- **Input Layer:**
  - Function: Receives the raw input data and passes it to the next layer.
  - Characteristics: Does not perform any computations, simply forwards the data.

- **Hidden Layers:**
  - Function: Perform computations and extract features from the input data.
  - Characteristics: Can have multiple hidden layers, making the network "deep."
  - Types:
    - Dense (Fully Connected) Layers:
      <pre>
      z = W • x + b
      </pre>
    - Convolutional Layers:
      <pre>
      z = (x * w) + b
      </pre>
    - Recurrent Layers:
      <pre>
      h_t = σ(W_h h_{t-1} + W_x x_t + b)
      </pre>

Hidden layers are concerned with extracting progressively higher order features form the raw data.

- **Output Layer:**
  - Function: Produces the final output of the network.
  - Characteristics: The number of neurons in the output layer typically matches the number of target classes in classification tasks or the dimensionality of the target variable in regression tasks.

### 3. Activation Functions
**Definition:**
Activation functions introduce non-linearity into the network, enabling it to learn and model complex patterns in the data.

**Common Activation Functions:**

- **Sigmoid:**
  <pre>
  σ(x) = 1 / (1 + e^(-x))
  </pre>
  - Range: (0, 1)
  - Use Cases: Binary classification, output layer for probability predictions.
  - Characteristics: Can cause vanishing gradient problems, leading to slow convergence in deep networks.

- **Tanh:**
  <pre>
  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  </pre>
  - Range: (-1, 1)
  - Use Cases: Hidden layers to achieve zero-centered outputs.
  - Characteristics: Similar vanishing gradient issues as sigmoid, but zero-centered.

- **ReLU (Rectified Linear Unit):**
  <pre>
  ReLU(x) = max(0, x)
  </pre>
  - Range: [0, ∞)
  - Use Cases: Most commonly used in hidden layers due to its simplicity and effectiveness.
  - Characteristics: Can suffer from dying ReLU problem, where neurons get stuck in zero state.

- **Leaky ReLU:**
  <pre>
  Leaky ReLU(x) = max(0.01x, x)
  </pre>
  - Range: (-∞, ∞)
  - Use Cases: Overcomes the dying ReLU problem by allowing a small, non-zero gradient for negative inputs.

### 4. Loss Functions
**Definition:**
Loss functions measure the discrepancy between the predicted values and the actual target values. They provide a signal for updating the model’s parameters.

**Common Loss Functions:**

- **Mean Squared Error (MSE):**
  <pre>
  MSE = (1/n) Σ (y_i - ŷ_i)^2
  </pre>
  - Use Cases: Regression tasks.
  - Characteristics: Sensitive to outliers due to squaring of errors.

- **Cross-Entropy Loss:**
  - Equation (Binary):
    <pre>
    Binary Cross-Entropy = - (1/n) Σ [y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i)]
    </pre>
  - Equation (Categorical):
    <pre>
    Categorical Cross-Entropy = - (1/n) Σ Σ y_ic log(ŷ_ic)
    </pre>
  - Use Cases: Classification tasks.
  - Characteristics: Measures the performance of a classification model whose output is a probability value between 0 and 1.

### 5. Optimization Methods
**Definition:**
Optimization methods are algorithms used to adjust the model’s parameters to minimize the loss function. They play a crucial role in training the network effectively.

**Common Optimization Algorithms:**

- **Gradient Descent:**
  <pre>
  θ = θ - η ∇J(θ)
  </pre>
  - Characteristics: Iteratively updates parameters by computing the gradient of the loss function with respect to the parameters.

- **Stochastic Gradient Descent (SGD):**
  <pre>
  θ = θ - η ∇J(θ; x^(i), y^(i))
  </pre>
  - Characteristics: Updates parameters using one random data poinhttps://www.youtube.com/watch?v=442n7jTW5fAt at a time, leading to more noisy updates but often faster convergence.

- **Adam (Adaptive Moment Estimation):**
  <pre>https://www.youtube.com/watch?v=442n7jTW5fA
  m_t = β_1 m_{t-1} + (1 - β_1) ∇J(θ)
  v_t = β_2 v_{t-1} + (1 - β_2) (∇J(θ))^2
  m̂_t = m_t / (1 - β_1^t)
  v̂_t = v_t / (1 - β_2^t)
  θ = θ - η (m̂_t / (√v̂_t + ε))
  </pre>
  - Characteristics: Combines the benefits of AdaGrad and RMSProp, maintaining a per-parameter learning rate that adapts over time.

### 6. Hyperparameters
**Definition:**
Hyperparameters are external configurations of the model set before training that govern the training process and structure of the model. They are not learned from the data.

**Examples of Hyperparameters:**

- **Learning Rate:**
  - Definition: Controls the size of the steps taken during parameter updates.
  - Characteristics: A small learning rate might lead to slow convergence, while a large learning rate might cause the model to overshoot the optimal solution.

- **Number of Layers and Neurons:**
  - Definition: Determines the depth (number of layers) and width (number of neurons per layer) of the network.
  - Characteristics: More layers and neurons can capture more complex patterns but also increase the risk of overfitting.

- **Batch Size:**
  - Definition: The number of training examples used in one iteration to update the model’s parameters.
  - Characteristics: Larger batch sizes provide more stable estimates of the gradient but require more memory and can lead to slower updates.

- **Epochs:**
  - Definition: The number of times the entire training dataset passes through the network.
  - Characteristics: More epochs can lead to better training, but excessive epochs may cause overfitting.

**Hyperparameter Tuning:**

- Grid Search: Systematically testing a range of hyperparameter values.
- Random Search: Testing random combinations of hyperparameter values.
- Bayesian Optimization: Using probabilistic models to find the optimal set of hyperparameters.
