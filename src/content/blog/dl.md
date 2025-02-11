---
title: 'Deep Learning'
pubDate: 2010-01-01
description: 'Deep Learning for beginners'
heroImage: 'https://i.wolves.top/picgo/202501071054048.png'
---

<p style="color: aquamarine;text-align: center">POST   ON 2024-03-05 BY WOLVES</p>
<p style="color: lightblue;text-align: center">UPDATE ON 2025-01-06 BY WOLVES</p>

<link rel="stylesheet" href="/katex/katex.min.css">
<script defer src="/katex/katex.min.js"></script>
<script defer src="/katex/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ]
        });
    });
</script>


## Deep Learning

You can see the code of this blog on <a href="https://github.com/lWolvesl/AI-learning.git" target="_blank">github</a>

### 1.Neurons and Artificial Neural Networks

| Biological Neurons | Artificial Neural Networks |
|--------------------|----------------------------|
| Composed of cell body, dendrites, and axon | Composed of nodes (artificial neurons) |
| Dendrites receive signals, axon transmits signals | Nodes receive input signals and perform weighted summation |
| Signals are transmitted through synapses | Signals are transmitted through connections (weights) |
| Use chemical or electrical signals | Use mathematical functions and algorithms |
| Complex biological structure | Computational model, divided into input layer, hidden layers, and output layer |

This table shows the comparison between biological neurons and artificial neural networks, helping to understand their similarities and differences.

![](https://i.wolves.top/picgo/202501201624508.png)

### 2. Forward Propagation

In a neural network, forward propagation refers to the process of signal transmission from the input layer to the output layer. Each node (neuron) receives input signals, performs a weighted summation, and generates output signals through an activation function. The basic steps of forward propagation are as follows:

1. **Input Layer**: Receives input data $\vec{x}$.
2. **Hidden Layer**: Each node calculates the weighted sum $z = \sum (w_i \cdot x_i) + b$, where $w_i$ is the weight and $b$ is the bias.
3. **Activation Function**: The weighted sum $z$ is passed through an activation function $a = f(z)$ to generate the output.
4. **Output Layer**: Outputs the final result.

The purpose of forward propagation is to compute the output of the neural network for prediction or classification.

### 3. Backward Propagation

Backpropagation is a key algorithm used in training neural networks. It involves propagating the error from the output layer back through the network to update the weights and biases, minimizing the error in predictions. The basic steps of backpropagation are as follows:

1. **Calculate Error**: Determine the error at the output layer by comparing the predicted output with the actual target values.
2. **Output Layer**: Compute the gradient of the loss function with respect to the output of the network.
3. **Hidden Layers**: Propagate the error back through the network, calculating the gradient of the loss function with respect to each layer's weights and biases.
4. **Update Weights and Biases**: Adjust the weights and biases using the calculated gradients and a learning rate to minimize the error.

The purpose of backpropagation is to optimize the neural network's parameters, improving its accuracy in making predictions or classifications.

### 4. Activation Function

Activation functions are used to introduce non-linearity into the neural network, allowing it to handle complex patterns in data. Common activation functions include:

- Sigmoid function: $a = \frac{1}{1 + e^{-z}}$
- ReLU (Rectified Linear Unit): $a = \max(0, z)$
- Tanh (Hyperbolic Tangent): $a = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

These functions help the neural network learn and generalize better.

#### 4.1 How to choose activation function

- Select the activation function based on the value of y
- The most common activation function is ReLU

- For output layer (In common classification problems)
    - If the value of y is negative or positive, use sigmoid function
    - If the value of y is always equal or greater than 0, use Linear function
    - Relu is not recommended for output layer

- For hidden layer
    - Relu is the most common activation function

### 5. Multiclass Classification

- from positive and negative to multiple classes

![](https://i.wolves.top/picgo/202501201811786.png)

- Now , We need a new activation function to handle multiple classes like softmax function

#### 5.1 Softmax Function

$$
z_i = w_i \cdot x + b_i
$$

$$
a_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

- sparse categorical cross entropy function (稀疏分类交叉熵函数)
$$
loss = - \sum_{i=1}^{n} y_i \log(a_i) = - \log(a_{true})
$$

#### 5.2 Numerical RoundOff Error

- Numerical Stability:
In calculating the cross-entropy loss (Cross-Entropy Loss), directly inputting logits into the loss function rather than the probability after the sigmoid activation function can improve numerical stability. This helps avoid numerical underflow or overflow issues caused by extreme probability values (close to 0 or 1).

- Use from_logits=True in loss function
When setting from_logits=true, the loss function (such as BinaryCrossentropy or CategoricalCrossentropy) automatically applies the sigmoid or softmax activation function internally. Therefore, the last layer of the model only needs to output logits, without manually adding an activation function. This simplifies the model definition and makes the code more concise.

It is equivalent to skipping the intermediate calculation and directly calculating the final result, thereby improving numerical stability.

#### 5.3 RMSProp Optimizer

- 在深度学习中，RMSprop（Root Mean Square Propagation）是一种常用的优化算法，主要用于解决梯度下降中的学习率调整问题。它通过自适应地调整每个参数的学习率，能够有效地加速收敛并减少震荡。

$$ E[g^2]t = \beta E[g^2]{t-1} + (1 - \beta) g_t^2 $$

$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t $$

#### 5.4 Adam Optimizer (自动选择学习率)

**Adam Optimizer**

Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent, namely adaptive gradient algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). It computes adaptive learning rates for each parameter.

**Advantages of Adam Optimizer**

- **Adaptive Learning Rates**: Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

- **Efficient**: It is computationally efficient and has low memory requirements.

- **Invariance to Diagonal Rescaling of Gradients**: The algorithm is invariant to diagonal rescaling of the gradients.

- **Suitable for Non-stationary Objectives**: It works well with problems that are large in terms of data/parameters or non-stationary.

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$

$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$


$$ \hat{m}t = \frac{m_t}{1 - \beta_1^t} $$
$$ \hat{v}t = \frac{v_t}{1 - \beta_2^t} $$

$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

### 6.Some Concepts

#### 6.1 Parameter & Hyperparameter

- Parameter: The weights and biases in the model
- Hyperparameter: The learning rate, batch size, number of epochs, etc.

- In the training process, the model parameters are adjusted through backpropagation, while the hyperparameters are set manually by the user.
- Maybe we can use some methods to find the best hyperparameter in the future, but now we can only set it by experience.

#### 6.2 human brain and deep learning
There are many similarities between the human brain and deep learning. Deep learning models are inspired by the structure and function of the human brain, particularly the way neurons and synapses work. But today, deep learning models are still far from the complexity and flexibility of the human brain.

- **Neurons and Artificial Neurons**: The human brain is composed of billions of neurons connected through synapses. Similarly, deep learning models consist of many artificial neurons connected by weights.

- **Learning Process**: The human brain continuously adjusts the strength of connections between neurons through experience and learning. Deep learning models adjust weights using training data and backpropagation algorithms to minimize the loss function.

- **Hierarchical Structure**: Neurons in the human brain are distributed across different regions and layers, responsible for processing different types of information. Deep learning models also consist of multi-layer neural networks, with each layer extracting different levels of features.

Although deep learning models perform excellently on certain tasks, they are still far from the complexity and flexibility of the human brain. Future research may further draw on the principles of the human brain to enhance the performance and adaptability of deep learning models.

#### 6.3 train/dev/test sets

In the begnning of DL, we need to define the number of layers, hidden units, learning rate, activation function, etc(hyperparameter).

- train set: used for training the model
- dev set: used for tuning the model
- test set: used for testing the model

#### 6.4 vanishing/exploding gradient

- the gradient is too small, especially in the hidden layer which uses sigmoid/tanh activation function
    - the gradient is too small, so the model will not learn anything
    - the model will be stuck in the local minimum
- the gradient is too large, especially in the hidden layer which uses ReLU activation function
    - the gradient is too large, so the model will explode
    - the model will be unstable

#### 6.5 weight initialization

- random initialization
- He initialization - for ReLU activation function $$W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$$
- Xavier initialization - for sigmoid/tanh activation function $$W \sim \mathcal{N}(0, \sqrt{\frac{1}{n_{in}}})$$

<p style="color: lightblue;text-align: center">
$n_{in}$ 表示当前层的输入神经元数量
</p>

#### 6.6 mini-batch gradient descent

- mini-batch gradient descent is a variant of gradient descent that uses a small subset of the training data (mini-batch) to update the model parameters.
- It is used to speed up the training process and reduce the memory usage.

- Advantages
    - lower memory usage
    - faster convergence
    - better generalization
- eg. Data set has 10000 samples, we will use 100 of them to update the model parameters at one time

#### 6.7 Gradient Checking

- Gradient Checking is a technique used to verify the correctness of the gradient calculation in the backpropagation algorithm.
- It is used to check whether the gradient calculated by the backpropagation algorithm is correct.
- It is used to check whether the gradient calculated by the backpropagation algorithm is correct.

### 7. Bias and Variance

![](https://i.wolves.top/picgo/202501251932379.png)

- In this case
    - the left is the high bias, the right shows a low bias
    - the left is the low variance, the right shows a high variance
    - And the left is a overfitting model, the right is a underfitting model

- Bias: The difference between the expected value and the true value
- Variance: The difference between the predicted value and the expected value

### 8. Regularization

- Regularization is a technique used to prevent overfitting in deep learning models. It adds a penalty term to the loss function to encourage the model to learn simpler patterns.

- L1 Regularization: $loss = loss + \lambda ||w||_1$
- L2 Regularization: $loss = loss + \lambda \sum w^2$ 
- (L2 Regularization is also called weight decay)

- $\lambda$ is the regularization parameter, which controls the strength of the regularization.

##### 8.1 Dropout

- Dropout is a regularization technique that randomly drops out some neurons during training to prevent overfitting.

- During each training iteration, different neurons may be randomly deactivated, which enhances the learning strength of other neurons and improves the model's generalization ability.

##### 8.2 Other Regularization Techniques

- Early Stopping



### 9.Computer Vision

> From this part, we will learn how to use DL to solve some problems in computer vision.

#### 9.1 Some problems

- Instead of using 64x64x3 to represent an image, we may face 1000x1000x3 to represent an image, it means the input layer will possess a lot of parameters, which will cost a lot of time and memory in calculation.

- So we need to use some methods to reduce the dimension of the input layer.

#### 9.2 Some concepts

- Gray Image
    - Every pixel has only one channel, it just represents the brightness of every pixel
    - the value range is [0,255]
    - It`s always used in margin detection or binary classification
- RGB Image
    - Every pixel has three channels, it represents the brightness of every pixel in the red, green and blue channels
    - the value range is [0,255]
    - It`s always used in object detection or multi-class classification

### 10 Convolutional Neural Network

- Convolutional Neural Network (CNN) is a type of neural network that is commonly used in computer vision tasks.
- It is a type of feedforward neural network that is specifically designed for image processing and analysis.

- it will reduce the parameters and the calculation of the model, it also can extract the features of the image

#### 10.1 Edge detection

- use a 3x3 filter to detect the edge of the image

- filters
    - Sobel filter
        - $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$
        - $G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
    - Prewitt filter
        - $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$
        - $G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$
    - Scharr filter
        - $G_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix}$
        - $G_y = \begin{bmatrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ 3 & 10 & 3 \end{bmatrix}$

- then, we use the filter with the part of the image to calculate the convolution,we will get a new image(feature map), which is smaller than the original image

- the result of the convolution will enhance the edge of the image

#### 10.2 Padding

- Padding is a technique used to keep the size of the image after the convolution

- when we process a 6x6 image with a 3x3 filter, we will get a 4x4 feature map, because the filter can only move 4x4 times in the image
    
- the peak pixel will use a little part of the image, which will cause the loss of information

- so we need to use padding to keep the size of the image, it will add a border around the image and the peak pixel will use more to express the information

    - $n_{out} = \frac{n_{in} - n_{filter} + 2 \times padding}{stride} + 1$
    - $n_{out}$ is the output size
    - $n_{in}$ is the input size
    - $n_{filter}$ is the filter size
    - $padding$ is the padding size
    - $stride$ is the stride size

#### 10.3 Stride

- Stride is the step size of the filter when moving in the image

- when we use a 3x3 filter to process a 6x6 image, we will get a 4x4 feature map, because the filter moves 1 pixel each time

- we can also use 2 stride to reduce the size of the feature map

#### 10.4 three dimensions

- RGB image - 三层堆叠
    - 3 channels
    - 3x3x3 filter
    - 3x3x3 feature map

- The same example 6x6 image with 3 channels, the filter will be 3x3x3 which like a cube. then use the every layer of the filter to process the every layer of the image part by part, and then we will get a 4x4x3 feature map, then add them together, we will get a 4x4x1 feature map.

#### 10.5 One layer of a Convolutional Neural Network

卷积核+偏置+激活函数+(其他)

- In the previous part, we use a 3x3x3 filter to process a 6x6x3 image to get the single feature map, but in the real situation, we need to use many filters to process the image, and then we will get many feature maps and then stack them up. 

- The result is like this:
    - 6x6x3 image
    - 3x3x3 filter
    - 3x3x3 filter

    - 4x4x2 feature map
- And that is the one simple layer of a Convolutional Neural Network.

- Here is a case:
    - 6x6x3 image
    - 10 filters with 3x3x3

- the output will be:
    - 4x4x10 feature maps
    - (3x3x3 + 1)x10 parameters (We alse have a bias 'b' for each filter)

- In this case, the model will not easy to overfit, because the number of parameters is not too large, we can use more filters to process a image which is very complex and huge.

#### 10.6 Convolutional Neural Network

- we can use many layers to process the image, and then we will get a more complex and huge feature map, then unrolling them into a very long vector, use the logistic regression or softmax regression to process it which like a normal neural network, and that is the Convolutional Neural Network.

#### 10.7 Pooling layer

- Pooling layer is a layer that uses a filter to process the image for reducing the size of the feature map

- The most common pooling is max pooling, it will select the maximum value in the filter's range, and then we will get a new feature map, which is smaller than the original feature map.

- The result of the pooling is like this:
    - original 4x4x10 feature maps
    - result 2x2x10 feature maps

- Pooling is alse a filter, but it will not multiply the weight and the bias, it will just select the maximum value in the filter's range, and it will keep the same number of channels.

- There is another pooling is average pooling, it will select the average value in the filter's range.

#### 10.8 Fully Connected Layer

- The Fully Connected Layer (FC Layer) is a common type of layer in neural networks, typically used to map the output from the previous layer to the final prediction result.

- Like the normal machine learning, we can use many FC layer to process the image, and then we will get a very long vector, and then use the logistic regression or softmax regression to process it which like a normal neural network.

#### 10.9 Hyperparameter

- Hyperparameter is a parameter that is not learned from the data, it is set by the user.

- We can get the best hyperparameter by looking papers or using some tools instead of ourselves.


### 11. classic network

#### 11.1 LeNet-5

- LeNet-5 is a classic network in the field of computer vision, which is used for handwritten digit recognition.

- You can use max pooling install of average pooling.

- It has more than 60000 parameters, it is very small than the modern network.

![](https://i.wolves.top/picgo/202502111607805.png)

#### 11.2 AlexNet

- AlexNet is a classic network in the field of computer vision, which is used for image classification.

- It has more than 60m parameters

- Local Response Normalization (LRN) - normalize the feature map channel by channel

![](https://i.wolves.top/picgo/202502111609979.png)

#### 11.3 VGG

- VGG is a classic network in the field of computer vision, which is used for image classification.

- It has more than 138m parameters

- 3x3 filter with 1 stride
- 2x2 max pooling with stride 2

![](https://i.wolves.top/picgo/202502111611339.png)

#### 11.4 GoogleNet

- GoogleNet is a classic network in the field of computer vision, which is used for image classification.

- It has more than 60m parameters

- 3x3 filter with 1 stride
- 2x2 max pooling with stride 2

![](https://i.wolves.top/picgo/202502111618605.png)

- It use padding to keep the size of the feature map

- Vgg16 has more than 138m parameters

- Vgg19 has more than 140m parameters

### 12. Modern Network

#### 12.1 ResNet

- Every depp neural network are difficult to train becase of vanishing and exploding gradient.

- The function of ResNet is to use a skip connection to solve the problem of vanishing and exploding gradient, which allows you to take the activation from one layer and suddenly feed it to another layer.

#### 12.1.1 Residual Block

- It didn't follow the main path of the network, it use a \[skip connection\](shortcut) pass one or more layers for going much deeper.

- eg.

before:

$$a^{[l+2]} = g(z^{[l+2]}) = g(W^{[l+2]}a^{[l+1]} + b^{[l+2]})$$

after:

$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]}) = g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]})$$

- The normal path is $a^{[l+2]} = g(z^{[l+2]})$ and the path with the layer it skip is a Residual Block.

![](https://i.wolves.top/picgo/202502111702489.png)

#### 12.1.2 ResNet

![](https://i.wolves.top/picgo/202502111722052.png)

- The dashed lines in the figure indicate that the dimensions on both sides of skip connction are different and special processing is required

#### 12.1.3 1x1 Convolution

- 1x1 Convolution is a type of convolution that is used to change the dimension of the feature map

- It is used to reduce the number of parameters and the calculation of the model

- It likes a full connection layer, but it is used in the convolutional layer

- In practice, we will employ a pooling layer to reduce the dimensionality of the feature map, followed by a 1x1 convolution to alter the channel configuration of the feature map.
