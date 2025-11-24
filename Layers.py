import numpy as np

class MaxPoolingLayer:
    def __init__(self, filterSize):
        self.filterSize = filterSize

    def forward(self, image):
        self.image = image
        batch, height, width, numFilters = image.shape
        outHeight = height // self.filterSize
        outWidth = width // self.filterSize

        # Organize the image into outHeight*outWidth matrix with each index being a vector of filterSize*filterSize pixels
        patches = image.reshape(batch, outHeight, self.filterSize, outWidth, self.filterSize, numFilters)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch, outHeight, outWidth, self.filterSize ** 2, numFilters)
        self.patches = patches

        # Get the max of each of each filterSize*filterSize pixel patches
        patchesMax = patches.max(axis=3)

        return patchesMax
    
    def backward(self, dE_dY):
        batch, height, width, numFilters = self.image.shape
        outHeight = height // self.filterSize
        outWidth = width // self.filterSize

        # Shape of patchesMax is (batch, outHeight, outWidth, 1, numFilters)
        patchesMax = self.patches.max(axis=3,keepdims=True)

        # Make a mask to select indicies where the pixel is max of that patch. 4rd dimension broadcasts to filterSize*filterSize
        mask = (patchesMax == self.patches)

        # Add axis to make gradient broadcast with mask: (batch, outHeight, outWidth, numFilters) -> (batch, outHeight, outWidth, 1, numFilters)
        dE_dY = dE_dY[:, :, :, None, :]

        # For each filterSize*filterSize pixel of each patch, only the ones with max values gets the corresponding gradient of that patch
        dE_dX = dE_dY * mask

        # Reshape into original shape
        dE_dX = dE_dX.reshape(batch, outHeight, outWidth, self.filterSize, self.filterSize, numFilters)
        dE_dX = dE_dX.transpose(0, 1, 3, 2, 4, 5)
        dE_dX = dE_dX.reshape(batch, height, width, numFilters)

        return dE_dX
    
class ConvolutionLayer:
    def __init__(self, filterSize, numFilters):
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.filters = np.random.randn(numFilters, filterSize, filterSize) * np.sqrt(2 / (filterSize ** 2))

    def forward(self, image):
        self.image = image
        filterSize = self.filterSize

        batch, height, width = image.shape
        outHeight = height - filterSize + 1
        outWidth = width - filterSize + 1

        bs, hs, ws = image.strides
        # Get a representation of each patch of image of stride 1 to be multiplied with filter
        self.stridedView = np.lib.stride_tricks.as_strided(
            image,
            shape=(batch, outHeight, outWidth, filterSize, filterSize),
            strides=(bs, hs, ws, hs, ws)
        )

        # Dot product between filterSize, filterSize dimensions of the filters and patches
        # stridedView     -> (batch, outHeight, outWidth, filterSize, filterSize)
        # filters         -> (numFilters, filterSize, filterSize)
        convolution = np.tensordot(self.stridedView, self.filters, axes=([3,4],[1,2]))
        
        # convolution: (batch, outHeight, outWidth, numFilter)
        return convolution
    
    def backward(self, dE_dY, lr):
        batch, outHeight, outWidth, _ = dE_dY.shape
        filterSize = self.filterSize
        height = outHeight + filterSize - 1
        width = outWidth + filterSize - 1

        # Compute gradient for filters
        # stridedView     -> (batch, outHeight, outWidth, 1, filterSize, filterSize)
        # dE_dY           -> (batch, outHeight, outWidth, numFilters, 1, 1)
        dE_dK = (self.stridedView[:, :, :, None, :, :] * dE_dY[:, :, :, :, None, None]).sum(axis=(0,1,2))

        self.filters -= dE_dK * lr

        # Reverse the height and width dimensions of filters
        filtersFlipped = self.filters[:, ::-1, ::-1]

        # Calculate gradient respect to columns
        # dE_dY           -> (batch, outHeight, outWidth, numFilters)
        # filtersFlipped  -> (numFilters, filterSize, filterSize)
        dE_dColumns = np.tensordot(dE_dY, filtersFlipped, axes=([3], [0]))
        # dE_dColumn: (batch, outHeight, outWidth, filterSize, filterSize)
        
        dE_dX = np.zeros((batch, height, width))

        # Update dE_dX by adding the gradients of each patch
        for y in range(filterSize):
            for x in range(filterSize):
                dE_dX[:, y:y+outHeight, x:x+outWidth] += dE_dColumns[:, :, :, y, x]

        return dE_dX



    




