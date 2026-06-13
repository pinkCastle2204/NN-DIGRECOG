# PAPER NOTES

## 1. Central Claim
The paper claims that multi-layer perceptron (or Neural networks) can be trained effetively by using backpropagation to compute gradients and update weights throughout the network. This is better and efficient than earler methods because backpropagation helps in determinig the contribution in error of each layer and how should be adjusted to reduce the error.

## 2. Core Algorithm
The algorithm has two passes:

* #### FORWARD PASS:
    Inputs are propagated through the layers to produce predictions and compute the loss.
    ```math
    Z = XW + b
    ```
Where:
- $Z$ = Output of the Forwar Pass
- $X$ = Input
- $W$ = Layer Weight
- $b$ = bias of the layer


* #### BACKWARD PASS:
    Gradients of the loss with respect to weights, biases, and inputs are computed using back propagation and chain rule is used to trace back to the input.

```math
W_{new} = W_{old} - \eta \frac{\partial L}{\partial W}
```
Where:
- $W_{new}$ = updated weights after gradient descent
- $W_{old}$ = weights before update
- $\eta$ = learning rate, controls step size
- $L$ = loss function (cross entropy)
- $\frac{\partial L}{\partial W}$ = gradient of loss with respect to weights

```snippet from main.py
    layer2.W -= learning_rate * layer2.dW
    layer2.b -= learning_rate * layer2.db
    layer1.W -= learning_rate * layer1.dW
    layer1.b -= learning_rate * layer1.db
```

## 3. Dataset and Evaluation
My Implementation of the paper applies the given method from the paper in Classification task i.e., Digit Recognition.

I got a a overall accuracy of 90.82% on training the Neural-Network

Networks progressively reduced loss and improved accuracy through repeated forward passed, backpropagation, and weight updates.