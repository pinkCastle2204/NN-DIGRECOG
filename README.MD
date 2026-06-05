# NN-DIGRECOG

A handwritten digit recognition project built from scratch using NumPy. The project implements a simple feedforward neural network for classifying handwritten digits from the MNIST dataset.

This is my attempt to learn Neural-Networks from scratch.
Though i got ~92% accuracy while training this neural-network still not upto the par for any application. In other words use it and you'll realise how worse this is.

## Project Structure

```text
.
├── canvas.py               
├── nnfs
│   ├── activations.py      
│   ├── layers.py           
│   └── __init__.py
├── notebooks
│   └── playground.ipynb    
├── requirements.txt
├── W1.npy                  
├── W2.npy                  
├── b1.npy                  
└── b2.npy                  
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/NN-DIGRECOG.git
cd NN-DIGRECOG
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The MNIST dataset is not included in this repository because of GitHub file size limits. Hence I have removed the training code as well. I will upload it in near future


## Running the Project

Launch the drawing canvas:

```bash
python canvas.py
```

Draw a digit and let the model predict the output.

## Model Architecture

Example architecture:

```text
Input Layer (784)
      ↓
Hidden Layer
      ↓
Activation Function
      ↓
Output Layer (10 classes)
```

## Technologies Used

* Python
* NumPy

## Learning Objectives

This project was built to understand:

* Forward propagation
* Backpropagation
* Gradient descent
* Activation functions
* Neural network implementation from scratch
* Image classification fundamentals

## License

This project is open-source and available under the MIT License.
