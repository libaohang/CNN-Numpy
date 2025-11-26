import numpy as np

class MaxPoolingLayer:
    def __init__(self, filterSize):
        self.filterSize = filterSize

    def forward(self, image):
        self.image = image
        # channels = numFilters of previous convolution layer
        batch, height, width, channels = image.shape
        
        # Pad images if not evenly fit by pooling size
        heightPad = height % self.filterSize
        widthPad = width % self.filterSize
        if(heightPad != 0 or widthPad != 0):
            image = np.pad(
                image,
                pad_width = ((0, 0), (heightPad, 0), (widthPad, 0), (0, 0)),
                mode = 'constant',
                constant_values = 0
            )
        
        outHeight = (height + heightPad) // self.filterSize
        outWidth = (width + widthPad) // self.filterSize

        # Organize the image into outHeight*outWidth matrix with each index being a vector of filterSize*filterSize pixels
        patches = image.reshape(batch, outHeight, self.filterSize, outWidth, self.filterSize, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch, outHeight, outWidth, self.filterSize ** 2, channels)
        self.patches = patches

        # Get the max of each of each filterSize*filterSize pixel patches
        patchesMax = patches.max(axis=3)

        return patchesMax
    
    def backward(self, dE_dY):
        batch, height, width, channels= self.image.shape
        heightPad = height % self.filterSize
        widthPad = width % self.filterSize
        outHeight = (height + heightPad) // self.filterSize
        outWidth = (width + widthPad) // self.filterSize

        # patchesMax -> (batch, outHeight, outWidth, 1, channels)
        patchesMax = self.patches.max(axis=3,keepdims=True)

        # Make a mask to select indicies where the pixel is max of that patch. 4th dimension broadcasts to filterSize*filterSize
        mask = (patchesMax == self.patches)

        # Add axis to make gradient broadcast with mask: (batch, outHeight, outWidth, numFilters) -> (batch, outHeight, outWidth, 1, numFilters)
        dE_dY = dE_dY[:, :, :, None, :]

        # For each filterSize*filterSize pixel of each patch, only the ones with max values gets the corresponding gradient of that patch
        dE_dX = dE_dY * mask

        # Reshape into original shape
        dE_dX = dE_dX.reshape(batch, outHeight, outWidth, self.filterSize, self.filterSize, channels)
        dE_dX = dE_dX.transpose(0, 1, 3, 2, 4, 5)
        dE_dX = dE_dX.reshape(batch, height + heightPad, width + widthPad, channels)

        # Remove padding
        if(heightPad != 0 or widthPad != 0):
            dE_dX = dE_dX[:, heightPad:, widthPad:, :]

        return dE_dX
    
    
class ConvolutionLayer:
    # channels need to be the numFilters of previous ConvolutionLayer if any or the original channels of sample
    def __init__(self, filterSize, numFilters, channels, lr, beta):
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.channels = channels
        self.filters = np.random.randn(numFilters, channels, filterSize, filterSize) * np.sqrt(2 / (channels * filterSize * filterSize))
        # Momentum for filters
        self.v_filters = np.zeros_like(self.filters)
        self.beta = beta
        self.lr = lr

    def forward(self, image):
        self.image = image
        filterSize = self.filterSize

        batch, height, width, channels = image.shape
        outHeight = height - filterSize + 1
        outWidth = width - filterSize + 1

        bs, hs, ws, cs = image.strides
        # Get a representation of each patch of image of stride 1 to be multiplied with filter
        self.stridedView = np.lib.stride_tricks.as_strided(
            image,
            shape=(batch, outHeight, outWidth, channels, filterSize, filterSize),
            strides=(bs, hs, ws, cs, hs, ws)
        )

        # Dot product between channels, filterSize, filterSize dimensions of the filters and patches
        # stridedView     -> (batch, outHeight, outWidth, channels, filterSize, filterSize)
        # filters         -> (numFilters, channels, filterSize, filterSize)
        convolution = np.tensordot(self.stridedView, self.filters, axes=([3,4,5],[1,2,3]))
        
        # convolution: (batch, outHeight, outWidth, numFilter)
        return convolution
    
    def backward(self, dE_dY):
        batch, outHeight, outWidth, _ = dE_dY.shape
        filterSize = self.filterSize
        height = outHeight + filterSize - 1
        width = outWidth + filterSize - 1

        # Compute gradient for filters
        # stridedView     -> (batch, outHeight, outWidth, 1, channels, filterSize, filterSize)
        # dE_dY           -> (batch, outHeight, outWidth, numFilters, 1, 1, 1)
        dE_dK = (self.stridedView[:, :, :, None, :, :, :] * dE_dY[:, :, :, :, None, None, None]).sum(axis=(0,1,2))

        self.v_filters = self.beta * self.v_filters + (1 - self.beta) * dE_dK
        self.filters -= self.v_filters * self.lr

        # Reverse the height and width dimensions of filters
        filtersFlipped = self.filters[:, :, ::-1, ::-1]

        # Calculate gradient respect to columns
        # dE_dY           -> (batch, outHeight, outWidth, numFilters)
        # filtersFlipped  -> (numFilters, channels, filterSize, filterSize)
        dE_dColumns = np.tensordot(dE_dY, filtersFlipped, axes=([3], [0]))
        # dE_dColumn: (batch, outHeight, outWidth, channels, filterSize, filterSize)
        dE_dColumns = dE_dColumns.transpose(0, 1, 2, 4, 5, 3)
        # dE_dColumn: (batch, outHeight, outWidth, filterSize, filterSize, channels)
        
        dE_dX = np.zeros((batch, height, width, self.channels))

        # Update dE_dX by adding the gradients of each patch
        for y in range(filterSize):
            for x in range(filterSize):
                dE_dX[:, y:y+outHeight, x:x+outWidth, :] += dE_dColumns[:, :, :, y, x, :]

        return dE_dX


class DenseLayer:
    def __init__(self, inputDim, outputDim, lr, beta):
        self.weights = np.random.randn(outputDim, inputDim)  * np.sqrt(2 / (inputDim))
        # Add axis so bias broadcasts in the batch dimension
        self.bias = np.random.randn(1, outputDim)

        # Momentum for weights and bias
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)

        self.beta = beta
        self.lr = lr

    
    def forward(self, input):
        self.input = input

        # Weight each input dimension by the corresponding weight dimension and add bias
        return input @ self.weights.T + self.bias
    
    def backward(self, dE_dY):
        batch = dE_dY.shape[0]

        # Gradient of weight
        dE_dW = dE_dY.T @ self.input / batch

        # Gradient of input
        dE_dX = dE_dY @ self.weights

        # Gradient of bias
        dE_dB = np.sum(dE_dY, axis=0, keepdims=True) / batch

        self.v_w = self.beta * self.v_w + (1 - self.beta) * dE_dW
        self.v_b = self.beta * self.v_b + (1 - self.beta) * dE_dB

        self.weights -= self.v_w * self.lr
        self.bias -= self.v_b * self.lr

        return dE_dX
    

class FlattenLayer:
    def forward(self, images):
        self.imageShape = images.shape
        imageSize = np.prod(images.shape[1:])
        return(images.reshape(images.shape[0], imageSize))
    
    def backward(self, dE_dY):
        return(dE_dY.reshape(*self.imageShape))





