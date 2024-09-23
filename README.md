Face Emotion Detection
Overview
This project focuses on detecting human emotions from facial images using deep learning techniques. The model leverages the TensorFlow and Keras libraries for building and training a convolutional neural network (CNN) to classify various emotions based on input images.

Libraries Used
The following libraries were utilized to build and optimize the model:
import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.regularizers import L2
import matplotlib.pyplot as plt
Model Architecture
The model is built using several layers, including:

Convolutional Layers: Used to detect patterns in the image, such as edges and shapes.
MaxPooling Layers: To reduce the spatial dimensions of the feature maps.
Dense Layers: Fully connected layers for classification.
Dropout: Regularization technique to prevent overfitting.
Batch Normalization: To stabilize and accelerate training.
Key Components
Loss Function: CategoricalCrossentropy to minimize the difference between predicted and actual classes.
Optimizer: Adam for adaptive learning rate and efficient training.
Metrics: TopKCategoricalAccuracy, CategoricalAccuracy for evaluating model performance.
Data Augmentation: Techniques like RandomFlip, RandomRotation, RandomContrast are used to improve the generalization of the model.
