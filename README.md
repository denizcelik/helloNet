# helloNet 

## Introduction

helloNet is an experimental DNN library that is built for tracking the mathematics behind machine learning and thus, gaining intuition about how machines learn.

The main goal of helloNet is contributing to "learning deep learning" journey of the community. Users can track all the mathematical operations behind the various concepts like activations, regularizations, optimizers, losses etc. step-by-step while they are using helloNet and optimizing some models.

All implementations in helloNet are built with functional programming approach due to the goal of simplicity and code readability. The slightly advanced OOP implemented version of helloNet will be available in future.

The code behind the study does not promise optimized performance for building a DNN model but it can promise usability and showing the building blocks of machine/deep learning by easily trackable code. This study generally uses the same methodology  (vectorized computations, model implementations etc.) from the online lecture notes of Prof. Andrew NG from DeepLearning&#46;ai and Stanford University.


## Implemented Operations

Currently, helloNet supports basic operations for ANNs. The functions Linear models and CNN module are in the To-Do list.  



**Initializers:**

* Random-Normal
* He
* Xavier 

**Activations:**
* Sigmoid
* ReLU
* Tanh (in progress)
* Leaky-ReLU (in progress)
* Softmax (in progress)

**Optimizers:**

* SGD
* Momentum
* RMSprop (in progress)
* Adam (in progress)

**Regularizations:**

* L1
* L2
* Dropout (in progress) 

**Other Solutions:**
* Batch Normalization (SGD only for now)
* Learning rate decay

**Losses:**

* Binary cross-entropy
* Categorical cross-entropy (in progress)

**Models:**

* ANNs (or Fully Connected, MLP)
* CNNs (in future)
* RNNs (in future)

**Data Preprocessing:**

* Flattening
* Normalization
* Standardization
* One-Hot Encoding

**Serialization:**

* Input Predicting
* Trained model saving
* Model reloading

## TODO List

- Unfinished functions
- Unfinished docstrings
- Comments for each statement
- Step IDs and gist references for each statement
- OOP versions of each operation
- Release versions
- License Adding
- Detailed README

## References

_Will be added soon._