from Layers import MaxPoolingLayer, ConvolutionLayer, DenseLayer, FlattenLayer
from Activations import ReLu, SoftMax
from CNN import trainCNN, testCNN
import numpy as np
from tensorflow.keras.datasets import mnist
from TestingTools import DigitGUI

def cross_entropy(true, pred):
    return -np.log(pred[np.arange(len(true)), true] + 1e-10)

def cross_entropy_prime(true, pred):
    y = np.zeros_like(pred)
    y[np.arange(len(pred)), true] = 1
    return (pred - y) / len(pred)

# Networks for MNIST:

# network1 reach 95% test accuracy after 20 epochs
network1 = [                              # 28 x 28 x 1
    ConvolutionLayer(3, 4, 1, 0.1, 0.9),  # 26 x 26 x 4
    ReLu(),
    MaxPoolingLayer(2),                   # 13 x 13 x 4
    FlattenLayer(),
    DenseLayer(676, 128, 0.1, 0.9),     
    ReLu(),
    DenseLayer(128, 10, 0.1, 0.9),
    SoftMax()
]

network2 = [                              # 28 x 28 x 1
    ConvolutionLayer(3, 4, 1, 0.1, 0.9),  # 26 x 26 x 4
    ReLu(),
    MaxPoolingLayer(2),                   # 13 x 13 x 4
    ReLu(),
    ConvolutionLayer(3, 5, 4, 0.1, 0.9),  # 11 x 11 x 5
    ReLu(),
    MaxPoolingLayer(2),                   # 6 x 6 x 5
    FlattenLayer(),
    DenseLayer(180, 150 , 0.1, 0.9),
    ReLu(),
    DenseLayer(150, 10, 0.1, 0.9),
    SoftMax()

]

def classifyMNIST(network):

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = xTrain.astype(np.float32) / 255.0
    xTest  = xTest.astype(np.float32) / 255.0

    xTrain = xTrain[:, :, :, None]
    xTest = xTest[:, :, :, None]

    trainedNetwork = trainCNN(network, cross_entropy, cross_entropy_prime, xTrain, yTrain, 25, 40)

    testCNN(trainedNetwork, cross_entropy, xTest, yTest)

    gui = DigitGUI(trainedNetwork)
    gui.run()


if __name__ == '__main__':
    classifyMNIST(network2)
