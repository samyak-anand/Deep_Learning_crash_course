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


#### Parameters
Definition:
Parameters are the elements within the network that are learned from the training data. In neural networks, these primarily include weights and biases.

Details:

- Weights: Each connection between neurons has an associated weight. These weights determine the strength and direction of the influence one neuron has on another.

- Initial Values: Weights are often initialized randomly or using specific strategies like He initialization or Xavier initialization to start the training process.

- Updates: During training, weights are adjusted through backpropagation to minimize the loss function.

- Biases: Each neuron has a bias that is added to the weighted sum of inputs. This bias helps the network to better fit the data by allowing the activation function to shift.

- Role: Biases ensure that neurons can activate even when all input values are zero, providing more flexibility in learning.


#### Layers 
Definition:
Layers are the building blocks of neural networks. Each layer consists of a set of neurons (nodes) that perform specific transformations on the input data.

Types of Layers:

    Input Layer:

    - Function: Receives the raw input data and passes it to the next layer.
    - Characteristics: Does not perform any computations, simply forwards the data.

    Hidden Layers:

    - Function: Perform computations and extract features from the input data.
    - Characteristics: Can have multiple hidden layers, making the network "deep."

Types:
- Dense (Fully Connected) Layers: Each neuron is connected to every neuron in the previous and next layers.
- Convolutional Layers: Apply convolution operations to capture spatial hierarchies, used primarily in image processing.
- Recurrent Layers: Have connections that loop back to themselves, used for sequential data to capture temporal dependencies.

#### Activation Functions

Definition:
Activation functions introduce non-linearity into the network, enabling it to learn and model complex patterns in the data.
Common Activation Functions:

Sigmoid:

Equation: 
𝜎
(
𝑥
)
=
1
1
+
𝑒
−
𝑥
σ(x)= 
1+e 
−x
 
1
​
 
Range: (0, 1)
Use Cases: Binary classification, output layer for probability predictions.
Characteristics: Can cause vanishing gradient problems, leading to slow convergence in deep networks.
'''blash
σ(x)= 
1+e 
−x
 
1
​
 
'''