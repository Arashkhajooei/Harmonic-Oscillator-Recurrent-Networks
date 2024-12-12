# HORN Implementation

## Table of Contents

1\. [Introduction](#introduction)

2\. [Getting Started](#getting-started)

3\. [Datasets](#datasets)

4\. [Model Architectures](#model-architectures)

   - [Heterogeneous HORN](#heterogeneous-horn)

   - [Homogeneous HORN](#homogeneous-horn)

   - [Multi-layer HORN](#multi-layer-horn)

5\. [Training Functions](#training-functions)

6\. [Analysis Functions](#analysis-functions)

7\. [Pretraining and Transfer Learning](#pretraining-and-transfer-learning)

8\. [Usage](#usage)

9\. [Results](#results)

10\. [Conclusion](#conclusion)

---

## Introduction

This repository implements Heterogeneous and Homogeneous Oscillatory Recurrent Networks (HORN) for solving sequential classification tasks on synthetic MNIST (sMNIST) data. The implementation includes support for pretraining with oriented line datasets, coupling delays, and analysis tools such as Kuramoto Order Parameter and Pairwise Phase Locking Value (PLV).

---

## Getting Started

### Prerequisites

- Python >= 3.8

- PyTorch >= 1.10

- NumPy

- Matplotlib

- tqdm

- seaborn

- scikit-learn

### Installation

Clone this repository:

```bash

$ git clone https://github.com/your-repo/horn-implementation.git

$ cd horn-implementation

```

Install dependencies:

```bash

$ pip install -r requirements.txt

```

---

## Datasets

### sMNIST Dataset

The sMNIST dataset is a reshaped version of the original MNIST dataset, where images are treated as sequential data with 784 time steps.

- **Attributes**:

  - `data`: MNIST pixel values flattened into vectors.

  - `targets`: MNIST labels.

### Oriented Line Dataset

A synthetic dataset containing oriented lines of specific angles (e.g., 0, 45, 90, 135 degrees) used for pretraining the HORN models.

- **Attributes**:

  - `data`: Flattened 28x28 binary images containing lines.

  - `labels`: Index of the line orientation.

#### Function: `generate_data`

- **Inputs**:

  - `num_samples`: Number of samples to generate.

  - `image_size`: Dimensions of the synthetic images (default 28x28).

  - `orientations`: List of line orientations in degrees.

- **Outputs**:

  - A dataset containing oriented lines and their respective labels.

---

## Model Architectures

### Heterogeneous HORN

Implements a network where each node has unique oscillatory dynamics, determined by parameters `omega`, `gamma`, and `alpha`, with heterogeneity introduced by random perturbations.

#### Class: `HeterogeneousHORN`

- **Attributes**:

  - `num_nodes`: Number of nodes in the network.

  - `omega`, `gamma`, `alpha`: Oscillatory parameters.

  - `sequence_length`: Length of the input sequence.

  - `W_ih`: Input-to-hidden weights.

  - `W_hh`: Hidden-to-hidden weights.

  - `readout`: Output layer weights.

- **Methods**:

  - `__init__`: Initializes network parameters.

  - `initialize_weights`: Uses Xavier initialization for weights.

  - `forward`: Simulates network dynamics over the input sequence.

#### Algorithm (Forward Pass):

1\. Initialize node states `x` (positions) and `y` (velocities).

2\. For each time step:

   - Compute external (`W_ih`) and recurrent (`W_hh`) inputs.

   - Calculate acceleration based on oscillatory equations.

   - Update `x` and `y`.

3\. Return output from the readout layer.

---

### Homogeneous HORN

Implements a network with uniform oscillatory dynamics for all nodes, sharing the same parameters `omega`, `gamma`, and `alpha`.

#### Class: `HomogeneousHORN`

- **Similar Attributes and Methods** as Heterogeneous HORN but without parameter heterogeneity.

---

### Multi-layer HORN

Extends the HORN model to multiple layers, allowing hierarchical representations.

#### Class: `MultiLayerDelayedHORN`

- **Attributes**:

  - `num_nodes_layer1`, `num_nodes_layer2`: Number of nodes in each layer.

  - Layer-specific weights: `W_ih_1`, `W_hh_1`, `W_ih_2`, `W_hh_2`.

  - `readout`: Final output layer.

- **Methods**:

  - `forward`: Processes input sequentially through both layers.

---

## Training Functions

### `train_model`

- **Inputs**:

  - `model`: HORN model to train.

  - `train_loader`, `test_loader`: DataLoaders for training and testing.

  - `epochs`: Number of training epochs.

  - `lr`: Learning rate.

- **Outputs**:

  - Trained model.

  - Training and test accuracy logs.

- **Algorithm**:

  1. Iterate over epochs.

  2. For each batch:

     - Perform forward and backward passes.

     - Update model parameters using AdamW optimizer.

  3. Evaluate accuracy on test data.

---

## Analysis Functions

### Kuramoto Order Parameter

Measures the synchronization of node oscillations.

#### Function: `kuramoto_order_parameter`

- **Inputs**:

  - `phi`: Phase angles of nodes.

- **Outputs**:

  - `r`: Mean synchronization level across nodes.

#### Algorithm:

1\. Convert phases to complex exponential form.

2\. Compute mean vector length for each time step.

### Pairwise Phase Locking Value (PLV)

Measures the consistency of phase differences between node pairs.

#### Function: `pairwise_PLV`

- **Inputs**:

  - `phi`: Phase angles of nodes.

- **Outputs**:

  - `plv`: Pairwise PLV matrix.

#### Algorithm:

1\. Compute phase differences for each pair.

2\. Calculate mean vector length over time.

### Pairwise Cross-Correlation

Measures the correlation between node activities.

#### Function: `pairwise_cross_correlation`

- **Inputs**:

  - `x`: Node positions.

- **Outputs**:

  - Cross-correlation matrix for each sample.

#### Algorithm:

1\. Calculate Pearson correlation coefficients between nodes.

2\. Average correlations across samples.

---

## Pretraining and Transfer Learning

Pretraining on oriented lines allows the network to learn canonical priors. Weights are frozen, and only the readout layer is trained on sMNIST for transfer learning.

- **Pretraining**: Train on oriented line dataset for 10 epochs.

- **Transfer Learning**: Train readout layer on sMNIST for 50 epochs.

---

## Usage

1\. Pretrain a HORN model:

```python

   train_model(hete32_pretrain, train_loader_orient, test_loader_orient, pretrain_epochs, lr_pretrain, device)
```

2\. Perform transfer learning:

```python

   train_model(hete32_pretrain, train_loader_sMNIST, test_loader_sMNIST, transfer_epochs, lr_transfer, device)
```

3\. Train from scratch for comparison:
```python

   train_model(hete32_scratch, train_loader_sMNIST, test_loader_sMNIST, transfer_epochs, lr_transfer, device)
```

---

## Results

Results include accuracy curves, confusion matrices, and distributions of synchronization metrics. Pretrained models demonstrate faster convergence and improved accuracy compared to models trained from scratch.

---

## Conclusion

This repository provides a robust implementation of HORN models for sequential tasks, highlighting the importance of learning priors. Detailed analysis tools enable the study of synchronization dynamics and performance metrics.

---

Feel free to contribute or open issues for improvements!
