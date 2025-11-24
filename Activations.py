import numpy as np

class ReLu:
    def __init__(self):
        pass
    
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, gradient):
        return (self.input > 0) * gradient
    
class SoftMax:
    def __init__(self):
        pass
    
    def forward(self, input):
        ex = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = ex / np.sum(ex, axis=1, keepdims=True)
        return self.output

    def backward(self, dE_dY):
        output = self.output
        dE_dX = output * (dE_dY - np.sum(output * dE_dY, axis=1, keepdims=True))
        return dE_dX