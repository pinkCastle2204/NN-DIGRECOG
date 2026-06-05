import numpy as np

#ReLU activation
def relu(Z):
    return np.maximum(0,Z)

def relu_derivative(Z):
    return Z > 0

#Softmax function
def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

#Loss function
def cross_entropy_loss(A2,onehot_Y):
    batch_size = A2.shape[0]
    A2_clipped = np.clip(A2,1e-7, 1-1e-7)
    loss = -np.sum(onehot_Y * np.log(A2_clipped)) / batch_size
    return loss

def onehot(Y):
    onehot_Y = np.zeros((Y.size, Y.max() + 1))
    onehot_Y[np.arange(Y.size), Y] = 1
    return onehot_Y