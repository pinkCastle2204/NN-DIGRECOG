import numpy as np

class Dense:
    def __init__(self,input_size,output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1,output_size))
    
    def forward(self,X):
        self.X = X
        return X @ self.W + self.b
    
    def backward(self,dZ):
        self.dW = self.X.T @ dZ           
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T