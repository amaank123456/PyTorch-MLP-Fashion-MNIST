# PyTorch-MLP-Fashion-MNIST

# Description
For this problem, an end-to-end Multi-Layered Perceptron (MLP) based system was implemented to classify different articles of clothing from the Fashion-MNIST dataset. This MLP was implemented using PyTorch, and the results for all the MLPs trained were displayed via a Tensorboard. 

Experiments with different type of weight normalizations, Data Augmentation, and Dropout were ran.

# Image Pre-processing
Each image within the Fashion-MNIST dataset was sized 28 x 28 and was flattened to size 784 to allow for input into the MLP.

# Vanilla Classifier
The vanilla classifier defined for this problem is a MLP that consists of 784 neurons in the input layer, 100 neurons with ReLU activation in the hidden layer, and 10 neurons with Softmax activation in the output layer. The PyTorch cross-entropy loss criterion was used, the Stochastic Gradient Descent optimzer was used, and the ReduceLROnPlateau learning rate scheduler was used. This classifier was trained for 20 epochs, and these were its results:

![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/b0282359-e831-4eac-ab91-727e3c2e9067)
![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/d0613850-2d01-4c1f-a022-9bf60ec5a41f)

The final training loss was 0.2477 and the final test accuracy was 0.7540.

# Vanilla Classifier with Zero Initialization
This vanilla classifier initialized all the weights and biases to zero. When trained for 20 epochs, these were its results:

![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/806e3521-9c09-47c8-a4e8-b4811552c083)
![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/87e2d428-3129-42b1-bd33-2cabe3fa1a6b)

The final training loss was 2.3027 and the final test accuracy was 0.1. These results are much worse than the original initialization. The reason as to why this may be the case is because is because the ReLU layer causes all the outputs after it to be 0 as all the inputs into it are 0, and this would lead to the gradient not changing the weights at all during backpropagation as most of the gradients will be 0.

# Vanilla Classifier with Xavier Normal (XN) Initialization
This vanilla classifier used the Xavier Normal initialization for the weights and zeros for the biases. When trained for 20 epochs, these were its results:

![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/b0ce650e-0c45-479e-8fd0-76ad2ca4972d)
![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/dcd8f745-d336-41a3-8602-68c14bdedfb1)

The final training loss was 0.2433 and the final test accuracy was 0.7497. These results are comparable than the original initialization. The reason as to why this may be the case is because the Xavier Normal initialization is meant to keep the variance of activations and gradients across layers consistent and assumes that the activation functions used are linear (ReLU is a linear activation function).

# Vanilla Classifier with XN Initialization and Data-Augmentation
This vanilla classifier used the Xavier Normal initialization for the weights and zeros for the biases and Data-Augmentation. When trained for 20 epochs, these were its results:

![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/84ff09ea-598b-4f1b-86d1-475bcf5e8229)
![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/6ef6d282-8e8a-4788-83b5-e0b843bcb2f1)

The final training loss was 0.3319 and the final test accuracy was 0.8481. These results are much better than just using Xavier Normal initialization. The reason as to why this may be the case is because Data-Augmentation makes the model more robust.

# Vanilla Classifier with XN Initialization, Data-Augmentation, and Dropout
This vanilla classifier used the Xavier Normal initialization for the weights and zeros for the biases, Data-Augmentation, and a dropout rate of 0.5. When trained for 20 epochs, these were its results:

![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/6bc73229-e5a9-4e11-98e5-2875626fd2d0)
![image](https://github.com/amaank123456/PyTorch-MLP-Fashion-MNIST-/assets/149258362/14fc4e3a-415d-47a6-88c5-72e40d497427)

The final training loss was 0.5413 and the final test accuracy was 0.8289. These results are much better than just using Xavier Normal initialization and slightly worse than just using Xavier Normal initialization and data augmentation. A reason as to why the training loss is higher and test accuracy is lower for dropout is compared to just Xavier Normal initialization and Data-Augmentation is because dropout introduces more bias to the model as it is a form of regularization, preventing the model from fitting the data better.




