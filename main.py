import numpy as np
from nnfs.activations import onehot,softmax,cross_entropy_loss,relu,relu_derivative
from nnfs.layers import Dense

train_data = np.loadtxt("dataset/train/mnist_train.csv", delimiter=',', skiprows=1)
test_data = np.loadtxt("dataset/test/mnist_test.csv", delimiter=',', skiprows=1)

X_train = train_data[:, 1:] / 255.0
Y_train = train_data[:,0].astype(int)

X_test = test_data[:, 1:] / 255.0
Y_test = test_data[:,0].astype(int)

onehot_Y_train = onehot(Y_train)

learning_rate = 0.1

layer1 = Dense(784,128)
layer2 = Dense(128,10)

epochs = 500
for epoch in range(epochs):
    Z1 = layer1.forward(X_train)
    A1 = relu(Z1)
    Z2 = layer2.forward(A1)
    A2 = softmax(Z2)
    loss = cross_entropy_loss(A2,onehot_Y_train)

    dZ2 = (A2 - onehot_Y_train)/X_train.shape[0]
    dA1 = layer2.backward(dZ2)
    dZ1 = dA1*relu_derivative(Z1)
    layer1.backward(dZ1)

    layer2.W -= learning_rate * layer2.dW
    layer2.b -= learning_rate * layer2.db
    layer1.W -= learning_rate * layer1.dW
    layer1.b -= learning_rate * layer1.db

    if epoch % 10 == 0:
        predictions = np.argmax(A2, axis=1)
        accuracy = np.mean(predictions == Y_train) * 100
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

Z1_test = layer1.forward(X_test)
A1_test = relu(Z1_test)
Z2_test = layer2.forward(A1_test)
A2_test = softmax(Z2_test)

test_predictions = np.argmax(A2_test, axis=1)
test_accuracy = np.mean(test_predictions == Y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

np.save('W1.npy',layer1.W)
np.save('b1.npy',layer1.b)
np.save('W2.npy',layer2.W)
np.save('b2.npy',layer2.b)