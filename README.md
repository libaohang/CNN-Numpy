
# Convolutional Neural Network with Pure Numpy Implementation
<br>

## Description and Purpose
__A convolutional neural network with all layers implemented using only Numpy__ <br>
I wanted to develop a clearer understanding of how convolution layers process data, so I decided to implement each component of it using Numpy without built-in layers from libraries.
<br>

## Classifying MNIST and CIFAR-10 Datasets
Using the layers I wrote in Numpy, I built 3 different networks of increasing complexity. I used the first 2 networks to classify MNIST, and the 3rd network to classify CIFAR-10. <br>
For context, MNIST is a dataset of grayscale images of handwritten numbers 0 to 9, and CIFAR is a dataset of colored images of 10 types of objects, such as birds, planes, trucks, etc. <br>
The details on each network and their performance on the datasets are described below: <br>
Note: errors are calculated using cross-entropy loss. <br>

### Network 1
Network 1 has a total of 1 convolution layer and 4 filters: Convo(4) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 1 on MNIST:**
<img width="1291" height="795" alt="image" src="https://github.com/user-attachments/assets/c072ca76-d054-400b-af7c-7528c5f2fd45" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 25 epochs<br>
<br>
Network 1 achieves a test error of 0.14, which is equivalent to **96% accuracy** on MNIST. This is a good result for a CNN with 1 convolution layer and 4 filters. <br>

<br>

### Network 2
Network 2 has a total of 2 convolution layers and 32 filters: Convo(16) -> ReLu -> MaxPool -> Convo(16) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 2 on MNIST:**
<img width="1296" height="815" alt="image" src="https://github.com/user-attachments/assets/42433137-5e8e-4c4b-86fb-78fb519518c6" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 30 epochs<br>
<br>
Network 2 achieves a test error of 0.07, which is equivalent to **97.8% accuracy** on MNIST. The learning curve is noticeably steeper with 2 convolution layers and more filters. With even more convolution layers, I would be able to reach 99% or more on MNIST, but I decided to move on to classifying CIFAR-10. <br>

<br>

### Network 3
Network 3 has a total of 3 convolution layers and 56 filters: Convo(8) -> ReLu -> MaxPool -> Convo(16) -> ReLu -> Convo(32) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 3 on CIFAR-10:**
<img width="1130" height="802" alt="image" src="https://github.com/user-attachments/assets/606dce1a-53a2-4dd4-889e-dab92258621d" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 25 epochs<br>
<br>
Network 3 achieves a test error of 1.16, which is equivalent to **60% accuracy** on CIFAR-10. This is actually a very good accuracy for a basic network like this without batch normalization or data augmentation.
The change from grayscale numbers in MNIST to colored objects of CIFAR-10 made learning much more difficult, as seen in how the learning curve is much flatter than the previous classifications. 
I planned to train a 4th network with even more filters and convolution layers to classify CIFAR-10, but it takes too long to run because the Numpy implementation does not support GPU acceleration. 
However, I am sure the 4th network has the potential to reach 80% accuracy with additional improvements, such as batch normalization layers.<br>

<br>

## Try For Yourself
All of the 3 networks described above and the functions used to pre-process data can be found in _main.py_. <br>
Use the function _classifyMNIST_ to classify MNIST and _classifyCIFAR10_ to classify CIFAR-10. The input to both functions is the network used for the classification, which should be a list of initialized layers. <br>
You can use the 3 networks I built or make one yourself. The layers are imported from _Layers.py_ and _Activations.py_, the test and train functions are imported from _CNN.py_. <br>
**The libraries Numpy and tensorflow.keras (for importing data) are required to be installed.** <br>

### Usage
__ConvolutionLayer(filterSize, numFilters, channels, lr, beta)__ <br>
__filterSize__: the side length of the filter/kernel, which has square dimensions and a fixed stride of 1. Input is pre-padded so no change in dimensions <br>
__numFilters__: the number of filters in this convolution layer <br>
__channels__: the number of channels the input to the layer has. If this is the first convolution layer, _channels_ is the channels of sample data (1 from MNIST and 3 for CIFAR-10); otherwise, _channels_ is the number of filters in the previous convolution layer <br>
__lr__: the learning rate of the layer (around 0.05 - 0.1 for fast learning) <br>
__beta__: the momentum of the layer (around 0.9, steepens the learning curve) <br>
<br>

__MaxPoolingLayer(filterSize)__ <br>
__filterSize__: the side length of a square filter that will apply the max pooling with _filterSize_ stride. Padding will be applied if input dimensions are not integer multiples of _filterSize_ <br>
<br>

__DenseLayer(inputDim, outputDim, lr, beta)__ <br>
__inputDim__: the size of each image after the flatten layer <br>
__outputDim__: the number of neurons in this fully connected layer. The last dense layer should have this be 10 <br>
__lr__: the learning rate of the layer (around 0.05 - 0.1 for fast learning) <br>
__beta__: the momentum of the layer (around 0.9, steepens the learning curve) <br>
<br>

__Designing a Network__ <br>
First, determine the dimensions of the data set: (28 x 28 x 1) for MNIST and (32 x 32 x 3) for CIFAR-10. <br>
The first 2 dimensions are the height and width dimensions, and the last dimension is the channel dimension. <br>
<br>
Layers are best to take these general orders: <br>
- Convolution layer needs to be upstream of any dense layer 
- Place ReLu after each convolution layer and dense layer
- Place max pooling after some of ReLu that follow convolution layer
- Place flatten before the first dense layer
- Place softmax after the last dense layer as the last layer 

Track the size of height, width, and channel dimensions through each layer using these rules: <br>
* Passing through a convolution layer does not change height and width dimensions because padding is added, but it does change the channel dimension to _numFilters_ of the layer.
* Passing through a max pooling layer divides the height and width dimensions by the _filterSize_ of the layer; division rounds up because padding is added. 
* Flatten layer reduces height, width, and channel dimensions to a single dimension with size equal to the product of the size of height, width, and channel of the input. 

Finalize parameters of layers: <br>
- The _inputDim_ parameter of the dense layer that follows flatten is the size of (height x width x channel) of the output of layer before flatten.
- The _outputDim_ parameter of the last dense layer is 10 to correspond to the 10 classes. 

<br>

## References Used
- The Independent Code, video: https://www.youtube.com/watch?v=Lakz2MoHy6o, code: https://github.com/TheIndependentCode/Neural-Network. Referenced the general structure of the convolution layer and dense layer, training and testing function.
- Riccardo Andreoni, article: https://towardsdatascience.com/build-a-convolutional-neural-network-from-scratch-using-numpy-139cbbf3c45e/, code: https://github.com/riccardoandreoni0/CNN-from-scratch. Referenced max pooling layer implementation and general network structure.
- ChatGPT, for checking the correctness of vectorization.



