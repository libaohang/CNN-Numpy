from Layers import MaxPoolingLayer, ConvolutionLayer, DenseLayer, FlattenLayer
from Activations import ReLu, SoftMax
from CNN import trainCNN, testCNN
import numpy as np
from mnist import MNIST

def MSE(pred, true):
    return np.mean(np.power(true - pred, 2))

def MSE_Prime(pred, true):
    return 2 * (pred - true) / np.size(true)

def main():

    mndata = MNIST('mnist_data/')
    xTrain, yTrain = mndata.loadTraining()
    xTest, yTest = mndata.loadTesting()

    xTrain = np.array(xTrain).reshape(-1, 28, 28)
    xTest  = np.array(xTest).reshape(-1, 28, 28)

    network = [
        ConvolutionLayer(3, 4, 0.01),
        ReLu(),
        MaxPoolingLayer(2),
        ConvolutionLayer(3, 5, 0.01),
        ReLu(),
        MaxPoolingLayer(2),
        FlattenLayer(),
        DenseLayer(22*22, 128, 0.01),
        ReLu(),
        DenseLayer(128, 10, 0.01),
        SoftMax()
    ]

    trainedNetwork = trainCNN(network, MSE, MSE_Prime, xTrain, yTrain, 4)

    testCNN(trainedNetwork, MSE, xTest, yTest)

if __name__ == '__main__':
    main()