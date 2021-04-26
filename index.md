# Using Deep Learning for Solving Sudoku

**Alex Powell, Adit Suvarna, Wenqi Shi, GianGiacomo Navarra, Sivapriya Vellaichamy**

### INTRODUCTION

With Machine learning outperforming humans in various games like Atari and Go, we were curious to explore how big a challenge is a logic-based number game Sudoku[1,2]. Logic-based games like Sudoku have been shown to help delay neurological disorders like Alzheimer’s and dementia. However, the game is easy to learn but hard to master.

### PROBLEM DEFINITION

The object of Sudoku is to fill out each puzzle with numbers 1-9 in a way that we only use each number once in each row, column, and a grid of the puzzle. In solving Sudoku, players rely on both the temporal and position dependency between the board numbers [3]. Based on these facts, we decide to solve the Sudoku using different machine learning methods like Convolutional Neural Network (CNN) and a deep Q learning approach. The purpose of the proposal is to use Machine Learning to solve a Sudoku puzzle. The pipeline of the problem involves using an unsupervised algorithm to detect digits and further feeding the recognized digits to deep learning models, which would be trained to solve the puzzle.

![](fig_overview.PNG "Flowchart of proposed method: A handwriting recogniser (Unsupervised) feeding into a sudoku solver (Supervised).")

> Fig 1. Flowchart of proposed method: A handwriting recogniser (Unsupervised) feeding into a sudoku solver (Supervised).

### DATA COLLECTION AND PRE-PROCESSING

**Unsupervised**: The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

**Supervised**: The Kaggle dataset “1 million sudoku games” is publically  available and provided by [1 million Sudoku games](https://www.kaggle.com/bryanpark/sudoku).The dataset contains two columns. The column quizzes contain the unsolved sudoku while the column solution has respective solved games. Each game is represented by a numpy array of 81 numbers. For the one-hot encoding, each of the numbers here is considered as a category because the value of the number itself does not hold any significance i.e. predicting a 4 as 5 is as bad as predicting it as 9. For this reason we consider each as a categorical input and we use one hot encoding before feeding it into the algorithm. This would mean that each of the 81 numbers is sent as a sequence of 9 numbers. 


### METHODS

**Unsupervised:** In this part, we implement an auto-encoder as a classifier and train it on MNIST dataset. An autoencoder is a type of artificial neural network trained to learn efficient data codings by attempting to copy its input to its output. The aim of an autoencoder is to learn a representation (encoding) from the input by training the network to ignore signal noise. To be specific, the input is a 28 by 28 image of a handwriting digit and the output is the reconstruction result with predicted label. Internally, it has a hidden layer h that describes a code used to represent the input. The network can be viewed as two consistent parts: an encoder function h=f(x) and a decoder that produces a reconstruction r=g(h). In addition to the original auto-encoder-decoder structure, we add a classification model with softmax output layer to combine with the encoder for a full image classification pipeline.

![alt-text-1](fig_us_1.png "Autoencoder as a Classifier using MNIST Dataset.")

> Fig 2. Autoencoder as a Classifier using MNIST Dataset.


**Supervised:** The sudoku solver is essentially a multiclass classification task. We consider two different algorithms: 1) Multi-layer perceptron 2) Convolutional neural network. We expect our model to predict the most probable digit as we do in classification tasks. 

*Multi Layer Perceptron:* In our Multilinear Perceptron we consider the Adadelta optimizer with a learning rate 0.1. We train the network for 10 epochs  with batch size 50 and  use a final softmax layer to generate probability values. The argmax over the soft probability was taken to choose the most likely probability for a given cell. 

*Convolutional Neural Network*: For our Convolutional Neural Network the Adadelta optimizer has been used with learning rate 0.1. We train the network for 10 epochs  with batch size 50 and  use a final softmax layer to generate probability values. The argmax over the soft probability was taken to choose the most likely probability for a given cell. 

*Bidirectional Recurrent Neural Network*: For the Recurrent Neural Network we took  one Long short term memory units (LSTM) with 500 hidden units. 

**Evaluation metric:**
The choice of evaluation metric is crucial as we would want to best identify the level of current performance so as to improve it in the future. We identify three different metrics that is applicable for the scenario.

_Full accuracy:_ This is the evaluation metric that is closest to how humans are evaluated on the game. This means that you have to get all the blanks of the sudoku correct to be classified as a correct solution. It's an all-or-nothing evaluation. 

_Semi Accuracy:_ This metric looks at the percentage of cells correctly placed in the overall puzzle.

_Semi Accuracy 2:_ This metric identifies the percentage of blank cells from the unfilled Sudoku that were correctly identified.



### RESULTS AND DISCUSSION

#### Unsupervised Learning 
The best performance result is obtained for epoch=100, loss=0.113, accuracy =0.97. As we can see from the example below, the reconstruction results and prediction accuracy are pretty good. The autoencoder successfully encodes and decodes the latent space vectors with high quality. 

![](fig_us_2.png "Example of recounstruction results of auto-encoder and predicted labels.")

> Fig 3. Example of recounstruction results of auto-encoder and predicted labels.



![](fig_us_3.png "Plots of training history and accuary of unsurpervised portion.")

> Fig 4. Plots of training history and accuary of unsurpervised portion.



#### Supervised Learning
The primary metric used for both the multilayer perceptron and the convolutional neural network to evaluate the performance of the model is Semi Accuracy. In Fig.5 the loss function is represented for the MLP in the case of validation and training. 

![alt-text-1](fig_s_1_1.png "The training loss for the Multi-layer Perceptron.")
![alt-text-2](fig_s_1_2.PNG "The validation loss for the Multi-layer Perceptron.")

> Fig 5. The training and validation loss for the Multi-layer Perceptron


For the MLP we got a Semi accuracy around 80%, as shown in Fig.6. The blue line is indicative of the training while the red one is for validation. The results are then compared with a convolutional neural network  of 2 layer networks. We are planning to consider a deeper neural network with a higher number of layers  in order to optimize the best number without going overfitting.

![alt-text-1](fig_s_2.PNG)

> Fig 6. The percentage of cells correctly filled over each epoch on the training and validation data for the MLP.


In Fig.7 we represent the loss function calculated with the two layer convolutional neural network.  The Semi accuracy obtained with the CNN for training and validation is represented in Fig.8.

![alt-text-1](fig_s_3_1.PNG )
![alt-text-2](fig_s_3_2.PNG)

> Fig 7. The loss function for training and and validation data for the two layer CNN.


Both the MLP and the two layer CNN  gives an accuracy around 83%  for the training and validation. We then test the MLP in an unknown sample of sudokus, we got an accuracy of 82.9%. For the CNN we got an accuracy a little higher, 83% for testing. This gives us the possibility to exclude overfitting as both training and testing give a similar value in accuracy. We will do the same testing on CNNs with different hyperparameters, such as number of layers, and layer sizes.

![alt-text-1](fig_s_4.PNG)

> Fig 8. The percentage of cells correctly filled on the training and validation data for the CNN.


### FUTURE WORK
We are planning to consider a deeper convolutional neural network with a higher number of layers. Then we want to use a bidirectional neural network. If we have time we will consider a Deep Q Learning approach.

### REFERENCES
[1] R. ”How to Build a Vision-Based Smart Sudoku,”

[2] Syed A T, Merugu S, Kumar V. Augmented Reality on Sudoku Puzzle Using Computer Vision and Deep Learning[M]//Advances in Cybernetics, Cognition, and Machine Learning for Communication Technologies. Springer, Singapore, 2020: 567-578.

[3] Bharti S K, Babu K S, Jena S K. Parsing-based sarcasm sentiment recognition in twitter data[C]//2015 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM). IEEE, 2015: 1373-1380.
