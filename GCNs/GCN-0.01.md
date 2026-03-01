# GCN-0.01.ipynb Analysis

## Overview
This notebook implements a Graph Convolutional Network (GCN) for a **link prediction** task on the **WikiCS** dataset. It treats the problem as a "recommender" system where the goal is to predict missing links (edges) between nodes.

## Code Explanation

### 1. Data Preparation
- **Dataset**: WikiCS (Computer Science articles on Wikipedia).
- **Loading**: Checks for a local copy at `../WikiCS/WikiCS_data.pt`. If not found, it downloads it to `/tmp/WikiCS` and saves it locally for future use.
- **Preprocessing**:
  - `train_test_split_edges(data)` is used to split the graph's edges into training, validation, and test sets.
  - This function (deprecated in newer PyG versions) automatically creates positive edges (existing links) and negative edges (non-existing links) for supervision.

### 2. Model Architecture
The model `RecommenderGNN` is a Graph Auto-Encoder (GAE) style architecture:
- **Encoder**: A 2-layer GCN.
  - Input: Node features.
  - Hidden Layer: 128 units + ReLU.
  - Output Layer: 64 distinct units.
  - The encoder produces low-dimensional embeddings (`z`) for every node in the graph using the training edges.
- **Decoder**: A simple dot-product decoder.
  - It takes pairs of node embeddings (source and target).
  - Calculates the dot product sum to output a "score" for the likelihood of a link existing.
  - `decode_all` is capable of computing the adjacency probability matrix for all pairs.

### 3. Training Loop
- **Objective**: Binary Classification (Link vs. No-Link).
- **Loss Function**: `BCEWithLogitsLoss` (Binary Cross Entropy).
- **Optimizer**: Adam (Learning Rate: 0.01).
- **Process**:
  1.  Generate embeddings using `data.train_pos_edge_index`.
  2.  Sample negative edges (random pairs of nodes that are not connected).
  3.  Compute scores for both positive ground-truth edges and negative samples.
  4.  Backpropagate loss to update GCN weights.

### 4. Evaluation
- **Metric**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve).
- **Validation**: Evaluated every 10 epochs on a separate validation edge set.
- **Testing**: Final performance measured on the test link set.

## Results
The notebook trained for 100 epochs.

- **Training Dynamics**:
  - **Epoch 10**: Loss ~0.619, Val AUC ~0.887
  - **Epoch 100**: Loss ~0.559, Val AUC ~0.843
- **Observation**:
  - The model quickly achieves a high validation AUC (~0.88) early on (Epoch 10).
  - As training progresses to 100 epochs, the loss decreases (fitting the training data better), but the **Validation AUC decreases** (drops to ~0.84).
  - This indicates **overfitting**; the model is memorizing the specific training structure rather than learning generalizable link prediction patterns, or the embedding space is degenerating.
- **Final Test Score**:
  - **Test AUC**: **0.8407**

## Conclusion
The GCN baseline successfully learns to predict links with a respectable AUC of 0.84. However, the drop in validation performance suggests early stopping (around epoch 10-20) or stronger regularization might yield better results than training for the full 100 epochs.
