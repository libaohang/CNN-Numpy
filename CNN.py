import numpy as np

def forwardProp(network, images):
    for layer in network:
        images = layer.forward(images)
    return images

def backwardProp(network, gradient):
    for layer in reversed(network):
        gradient = layer.backward(gradient)
    return gradient

def trainCNN(network, loss, lossGradient, xTrain, yTrain, epochs = 5, batchSize = 100):
    sampleSize = xTrain.shape[0]

    for epoch in range(epochs):
        epochLoss = 0

        indices = np.random.permutation(sampleSize)
        xTrain = xTrain[indices]
        yTrain = yTrain[indices]

        for batch in range(0, sampleSize, batchSize):   
            batchX = xTrain[batch : batch + batchSize]
            batchY = yTrain[batch : batch + batchSize]

            prediction = forwardProp(network, batchX)

            epochLoss += loss(batchY, prediction)

            gradient = lossGradient(batchY, prediction)
            backwardProp(network, gradient)

        epochLoss /= (sampleSize / batchSize)
        print(f"Training error on epoch {1 + epoch} is {np.mean(epochLoss)}")
    
    return network

def testCNN(network, loss, xTest, yTest):
    prediction = forwardProp(network, xTest)

    error = loss(yTest, prediction)

    predictions = prediction.argmax(axis=1)
    accuracy = np.mean(predictions == yTest)

    print(f"Testing error is {np.mean(error)}, with accuracy of {accuracy}")