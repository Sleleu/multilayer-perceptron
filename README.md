# Multilayer Perceptron

## Table of Contents

1. [About the Project](#1-about-the-project)  
2. [Installation](#2-installation)  
3. [Usage](#3-usage)  
    1. [Split the Dataset](#31-split-the-dataset)  
    2. [Train the Model](#32-train-the-model)  
    3. [Make Predictions](#33-make-predictions)
4. [Useful Resources](#4-useful-resources)

---

## 1. About the Project

The aim of this project is to implement a **multilayer perceptron** to predict whether a cancer diagnosis (based on a breast cancer dataset from Wisconsin) is malignant or benign.  
Dataset source: [Kaggle - Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

### What is a Multilayer Perceptron?

The **multilayer perceptron** is a feedforward network (meaning that the data flows from the input layer to the output layer) defined by the presence of one or more hidden layers as well as an interconnection of all the neurons of one layer to the next.

![Network Diagram](https://github.com/user-attachments/assets/7835eec4-730b-45e2-a892-2477ce8e9371)
**Fig 1:** Representation of a multilayer perceptron. Source: 42 subject.

The diagram above represents a network containing 4 dense layers (also called fully connected layers). Its inputs consist of 4 neurons and its output of 2. The weights of one layer to the next are represented by two dimensional matrices noted $W^{l_j l_{j+1}}$. The matrix $W^{l_0 l_1}$ is of size (3, 4) for example, as it contains the weights of the connections between the layer $l_0$ and the layer $l_1$.  
The bias is often represented as a special neuron which has no inputs and with an output always equal to 1. Like a perceptron it is connected to all the neurons of the following layer (the bias neurons are noted $b^{l_j}$ on the diagram above). The bias is generally useful as it allows to “control the behavior” of a layer.

### Composition of a perceptron

The perceptron is the type of neuron that the multilayer perceptron is composed of. They are defined by the presence of one or more input connections, an activation function and a single output. Each connection contains a weight (also called parameter) which is learned during the training phase.

![Perceptron Diagram](https://github.com/user-attachments/assets/17e44ff1-8e63-41bb-a8f9-f295caa53077)
**Fig 2:** Representation of a perceptron. Source: 42 subject.

Two steps are necessary to get the output of a neuron. 

- The first one consists in **computing the weighted sum of the outputs of the previous layer with the weights of the input connections** of the neuron, which gives  
```math
\text{weighted sum} = \sum_{k=0}^{N-1} (x_k \cdot w_k) + \text{bias}
```

- The second step consists in **applying an activation function to the weighted sum**. The activation function determines the output of the perceptron by defining a threshold above which the neuron is activated. This step introduces non-linearity into the model, enabling it to learn complex patterns. Commonly used activation functions include sigmoid (which outputs values between 0 and 1), hyperbolic tangent (tanh, which outputs values between -1 and 1), and the rectified linear unit (ReLU, which outputs the input value if positive, and 0 otherwise).

---

## 2. Installation

To run the project, ensure you have **Python**, **Git**, and **pip** installed. Follow the steps below:

```bash
# Clone the GitHub repository
git clone https://github.com/Sleleu/multilayer-perceptron.git

# Access the repository
cd multilayer-perceptron/

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## 3. Usage

The program is organized into 3 main functionalities: **split**, **train**, and **predict**. You must specify one of these actions using the `-a` or `--action` option. Although the program was designed with a specific use case in mind, the **MLP** and **Scaler** classes (for normalization/standardization) can be used flexibly in other programs. For example, a simplified demonstration using the XOR dataset as been created in `XOR_mlp/xor_mlp.py`.

### 3.1 Split the Dataset

Use the **split** functionality to preprocess raw datasets. This step divides the data into **training**, **validation**, and **test** sets based on configurable proportions. Additionally, the data is standardized (e.g., z-score normalization).

**Required Arguments:**
- `-a`| `--action`: Set to `'split'`
- `-d`| `--dataset`: Path to the raw dataset

**Example Command:**
```bash
python3 main.py -a split -d data/raw/data.csv
```

**Output Example:**

The dataset will be split into training, validation, and test sets, stored in the directory `data/processed/{train,val,test}`.

<img width="630" alt="Output example" src="https://github.com/user-attachments/assets/a13c3452-1745-467c-a33f-840052a5626b">

By default, the dataset is split into **60% training**, **20% validation**, and **20% testing**. These proportions can be adjusted in a custom implementation as follows:

```python
# Example usage
train_size = 0.7 # 70% of the dataset
val_size = 0.15 # 15% validation, leaving the remaining 15% for testing

X_train, X_val, X_test = split_features(X, train_size, val_size)
y_train, y_val, y_test = split_labels(y, train_size, val_size)
```

### 3.2 Train the Model

Use the **train** action to train the MLP model with the preprocessed datasets. You can customize the network architecture, activation functions, learning rate, and other parameters via command-line options.

**Required Arguments:**
- `-a`| `--action`: Set to `'train'`

**Optional Arguments:**
- `-e`| `--epochs`: Number of training epochs
- `-l` | `--layer`: Number and size of hidden layers
- `-b`| `--batch_size`: Batch size
- `-r` | `--learning_rate`: Learning rate
- `-l` | `--loss`: Loss function
- `-s` | `--seed`: Seed for weight initialization (for reproducibility)
- `-w` | `--weight_initializer`: Weight initialization method
- `--solver`: Optimizer type
- `-p` | `--patience`: Early stopping patience (epochs)
- `--activation` : Activation function for hidden layers
- `--output_activation` : Activation function for the output layer

For more details on available parameters, run:
```bash
python3 main.py -h
```

**Example Command:**

```bash
python3 main.py -a train
```
This runs the training phase with default settings suitable for the POC. Alternatively, you can fully customize hyperparameters:

```bash
python3 main.py -a train \
    --layer 20 20 20 \
    --epochs 150 \
    --batch_size 32 \
    --learning_rate 0.0314 \
    --activation sigmoid \
    --patience 10 \
    --solver momentum
```

In this example, the model has three hidden layers with 20 neurons each, uses the sigmoid activation function, a learning rate of 0.0314, an early stopping patience of 10 epochs, and the Momentum optimizer.

**Training Output:**

During training, the loss and accuracy are tracked for both the training and validation sets. The best epoch is saved, and the corresponding weights and biases are stored.

<img width="598" alt="output train" src="https://github.com/user-attachments/assets/3c3d48c6-f01f-461f-9e21-24a362983394">

Additionally, the following visualizations are generated:

- Loss history plot
- Accuracy history plot
- Contour plot showing model convergence

![Training figures](https://github.com/user-attachments/assets/e2e7ef22-b75f-47fd-a16e-8d7fc3867fe1)


### 3.3 Make Predictions

After training, use the **predict** action to generate predictions on a dataset by loading the trained model.

**Required Arguments:**

- `-a` | `--action`: Set to `'predict'`
- `-d` | `--dataset`: Path to the dataset to predict on
- `-m` | `--model`: Path to the saved model

**Example Command:**

```bash
python3 main.py -a predict \
    -d data/processed/test/X_test.csv \
    -m saved_model.npy
```

The predictions will be saved as a CSV file, and the terminal will display the model's accuracy.

**Output Example:**

<img width="862" alt="Predict output" src="https://github.com/user-attachments/assets/71273546-a749-4657-a96c-22931f972935">


## 4. Useful Resources

#### 1. General / MLP Fundamentals
General articles and courses
- [Alexis Nasr’s MLP Course](https://pageperso.lis-lab.fr/alexis.nasr/Ens/MASCO_AA/mlp.pdf)  
- [GeeksforGeeks - Multi-layer Perceptron Learning in TensorFlow](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)  
- [Romain Tavenard’s Deep Learning Course](https://rtavenar.github.io/deep_book/fr/content/fr/intro.html)

Youtube playlists
- [Alexandre TL - Neural Networks Playlist](https://www.youtube.com/watch?v=bkoNl7ImPBU&list=PLO5NqTx3Y6W6KkZHSzlvAQbJGQxrHErhx&index=13)  
- [Machine Learnia - MLP Theory / Implementation Playlist (Very useful to begin)](https://www.youtube.com/watch?v=XUFLq6dKQok&list=PLO_fdPEVlfKoanjvTJbIbd9V5d9Pzp8Rw) 

#### 2. Activation Functions
- [DeeplyLearning.fr - Activation Functions](https://deeplylearning.fr/cours-theoriques-deep-learning/fonction-dactivation/)
- [Machine learning mastery - Choose an activation function for deep learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

#### 3. Initialization Techniques
- [Numpy Ninja - Weight Initialization Techniques](https://www.numpyninja.com/post/weight-initialization-techniques)  
- [Data Science Stack Exchange - He and Glorot Initialization](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are)  
- [Devoteam - Advanced Deep Learning](https://france.devoteam.com/paroles-dexperts/aller-plus-loin-en-deep-learning-avec-les-reseaux-de-neurones-recurrents-rnns/)  

#### 4. Optimization Techniques
- [Machine Learning Mastery - Early Stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/)
- [Papers with code - Momentum explained](https://paperswithcode.com/method/sgd-with-momentum)

#### 5. Some debugging
- [Stack Overflow - Overflow Error in Neural Networks](https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation)  
- [Weights & Biases - Softmax Implementation](https://wandb.ai/krishamehta/softmax/reports/How-to-Implement-the-Softmax-Function-in-Python--VmlldzoxOTUwNTc)  

#### 6. Standardization and Preprocessing
- [Towards Data Science - Difference Between `fit_transform` and `transform`](https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe)  
- [ArcGIS Documentation - Standardization Methods](https://doc.arcgis.com/fr/allsource/1.0/analysis/geoprocessing-tools/data-management/standardizefield.htm)  

