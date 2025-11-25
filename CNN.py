
def forwardProp(network, images):
    for layer in network:
        images = layer.forward(images)
    return images

def backwardProp(network, gradient):
    for layer in reversed(network):
        gradient = layer.backward(gradient)
    return gradient

def trainCNN(network, loss, lossGradient, xTrain, yTrain, epochs = 5):
    for epoch in range(epochs):     
        prediction = forwardProp(network, xTrain)
        prediction = prediction.argmax(axis=1)

        error = loss(prediction, yTrain)

        backwardProp(network, lossGradient(prediction, yTrain))

        print(f"Training error on epoch {epoch} is {error}")
    
    return network

def testCNN(network, loss, xTest, yTest):
    prediction = forwardProp(network, xTest)

    error = loss(prediction, yTest)

    print(f"Testing error is {error}")