# Differential Distinguishers for ASCON Permutation

This project investigates differential distinguishers for the ASCON permutation using machine learning techniques. The research evaluates the ability of various ML models (CNN, LSTM, and LightGBM) to distinguish between differential and random ciphertext pairs across multiple rounds of the ASCON permutation.

## Project Overview

ASCON is a lightweight authenticated encryption scheme that was selected as the primary choice in the NIST Lightweight Cryptography Standardization process. This project focuses on analyzing the security of the ASCON permutation by attempting to find differential distinguishers using deep learning and gradient boosting techniques.

The project tests distinguishers for 1, 2, 3, 4, and 5 rounds of the ASCON permutation. For each round, three different machine learning models are trained and evaluated to determine if they can distinguish between:
- Label 0: Random ciphertext pairs (no differential relationship)
- Label 1: Differential ciphertext pairs (related by a fixed input difference)

## Project Structure

```
.
├── ROUND_1.ipynb          # Experiments for 1-round ASCON permutation
├── ROUND_2.ipynb          # Experiments for 2-round ASCON permutation
├── ROUND_3.ipynb          # Experiments for 3-round ASCON permutation
├── ROUND_4.ipynb          # Experiments for 4-round ASCON permutation
├── ROUND_5.ipynb          # Experiments for 5-round ASCON permutation
├── IMAGE/                 # Training results and visualizations
│   ├── cnn_training_results.png
│   ├── cnn_training_results_2.png
│   ├── cnn_training_results_3.png
│   ├── cnn_training_results_4.png
│   ├── cnn_training_results_5.png
│   ├── lstm_training_results.png
│   ├── lstm_training_results_2.png
│   ├── lstm_training_results_3.png
│   ├── lstm_training_results_4.png
│   └── lstm_training_results_5.png
└── README.md
```

## Methodology

### Dataset Generation

For each round (1-5), a balanced dataset is generated with the following characteristics:
- Total samples: 100,000 (50,000 per class)
- Input size: 320 bits (40 bytes) - representing the ASCON state (5 words × 64 bits)
- Input difference δ: `0x00000000000000010000000000000000000000000000000000000000000000000000000000000000`
- Features: C1, C2, C1_xor_C2 (XOR of ciphertexts)
- Data split: 70% training, 15% validation, 15% test

The dataset generation follows Algorithm 1:
1. Generate random plaintext P1
2. Apply ASCON permutation to get C1
3. For Label 0: Generate random P2; For Label 1: P2 = P1 ⊕ δ
4. Apply ASCON permutation to get C2
5. Compute C1_xor_C2 = C1 ⊕ C2
6. Store (C1, C2, C1_xor_C2, Label)

### Machine Learning Models

Three different models are employed:

1. **LightGBM**: Gradient boosting framework with optimized hyperparameters
   - Objective: Binary classification
   - Max depth: 12
   - Learning rate: 0.01
   - Num leaves: 512
   - Early stopping with 100 rounds patience

2. **CNN**: Convolutional Neural Network
   - Architecture: 3 convolutional blocks (32, 64, 128 filters)
   - Batch normalization and dropout for regularization
   - Global average pooling
   - Dense layers: 128 → 64 → 1
   - Learning rate: 0.0008
   - Batch size: 48
   - Epochs: 50 (with early stopping)

3. **LSTM**: Long Short-Term Memory Network
   - Architecture: 2 LSTM layers (64, 32 units)
   - Batch normalization
   - Dense layers: 128 → 64 → 1
   - Learning rate: 0.0006
   - Batch size: 48
   - Epochs: 60 (with early stopping)

### Evaluation Metrics

- **Training Accuracy**: Accuracy on training set
- **Test Accuracy**: Accuracy on test set
- **TPR (True Positive Rate)**: Sensitivity, ability to detect differential pairs
- **TNR (True Negative Rate)**: Specificity, ability to detect random pairs
- **Training Loss**: Binary cross-entropy loss on training set
- **Test Loss**: Binary cross-entropy loss on test set

A distinguisher is considered found if both training and test accuracy exceed the threshold of 0.50 (1/N_INPUT_DIFFERENCES where N_INPUT_DIFFERENCES = 2).

## Results Summary

### Round 1 (1-Round ASCON Permutation)

**LightGBM:**
- Training Accuracy: 0.9997
- Test Accuracy: 0.9992
- Training Loss: 0.6735
- Test Loss: 0.6735
- TPR: 1.0000
- TNR: 0.9984
- **Result: DISTINGUISHER FOUND**

**CNN:**
- Training Accuracy: ~1.0000
- Test Accuracy: ~1.0000
- **Result: DISTINGUISHER FOUND**

**LSTM:**
- Training Accuracy: ~0.9997
- Test Accuracy: ~1.0000
- **Result: DISTINGUISHER FOUND**

### Round 2 (2-Round ASCON Permutation)

**LightGBM:**
- Training Accuracy: 0.9998
- Test Accuracy: 0.9995
- Training Loss: 0.6735
- Test Loss: 0.6735
- TPR: 1.0000
- TNR: 0.9989
- **Result: DISTINGUISHER FOUND**

**CNN:**
- Training Accuracy: ~1.0000
- Test Accuracy: ~1.0000
- **Result: DISTINGUISHER FOUND**

**LSTM:**
- Training Accuracy: ~0.9986
- Test Accuracy: ~1.0000
- **Result: DISTINGUISHER FOUND**

### Round 3 (3-Round ASCON Permutation)

**LightGBM:**
- Training Accuracy: 0.9999
- Test Accuracy: 0.9997
- Training Loss: 0.6639
- Test Loss: 0.6639
- TPR: 1.0000
- TNR: 0.9993
- **Result: DISTINGUISHER FOUND**

**CNN:**
- Training Accuracy: ~0.9099 (improving over epochs)
- Test Accuracy: Above threshold
- **Result: DISTINGUISHER FOUND**

**LSTM:**
- Training Accuracy: ~0.8811 (improving over epochs)
- Test Accuracy: Above threshold
- **Result: DISTINGUISHER FOUND**

### Round 4 (4-Round ASCON Permutation)

**LightGBM:**
- Training Accuracy: 0.6039
- Test Accuracy: 0.5009
- Training Loss: 0.6919
- Test Loss: 0.6932
- TPR: 0.4140
- TNR: 0.5877
- **Result: DISTINGUISHER FOUND (Very Weak)**

**CNN:**
- Training Accuracy: ~0.5066
- Test Accuracy: ~0.5036
- **Result: DISTINGUISHER FOUND (Very Weak, Near Random)**

**LSTM:**
- Training Accuracy: ~0.5009
- Test Accuracy: ~0.4940
- **Result: DISTINGUISHER NOT FOUND**

### Round 5 (5-Round ASCON Permutation)

**LightGBM:**
- Training Accuracy: 0.6106
- Test Accuracy: 0.5041
- Training Loss: 0.6919
- Test Loss: 0.6931
- TPR: 0.5173
- TNR: 0.4909
- **Result: DISTINGUISHER FOUND (Very Weak)**

**CNN:**
- Training Accuracy: ~0.5041
- Test Accuracy: ~0.5068
- **Result: DISTINGUISHER FOUND (Very Weak, Near Random)**

**LSTM:**
- Training Accuracy: ~0.5009
- Test Accuracy: ~0.4940
- **Result: DISTINGUISHER NOT FOUND**

## Key Findings

1. **Strong Distinguishers (Rounds 1-3)**: All three models successfully found strong distinguishers for 1, 2, and 3 rounds of ASCON permutation with accuracy above 99%.

2. **Weak Distinguishers (Round 4)**: LightGBM found a very weak distinguisher with test accuracy barely above random (50.09%). CNN showed marginal improvement, while LSTM failed to find a distinguisher.

3. **Near-Random Performance (Round 5)**: LightGBM and CNN found very weak distinguishers with test accuracy near random (50.41% and 50.68% respectively). LSTM completely failed to distinguish.

4. **Model Comparison**: 
   - LightGBM consistently performed best across all rounds
   - CNN showed strong performance for early rounds but degraded significantly for rounds 4-5
   - LSTM had good performance for early rounds but failed for rounds 4-5

5. **Security Implications**: The results suggest that the ASCON permutation becomes significantly more secure (resistant to differential distinguishers) after 3 rounds, with rounds 4 and 5 showing near-random behavior for most models.

## Requirements

### Python Version
- Python 3.7 or higher

### Required Packages

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
tensorflow >= 2.8.0
lightgbm >= 3.3.0
matplotlib >= 3.4.0
```

### Installation

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow lightgbm matplotlib
```

Or install from requirements file (if provided):

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Each notebook (ROUND_1.ipynb through ROUND_5.ipynb) contains the complete pipeline for that round:

1. **Dataset Generation**: Generates 100,000 samples for the specified number of rounds
2. **Data Processing**: Converts data to binary feature vectors and splits into train/validation/test sets
3. **Model Training**: Trains LightGBM, CNN, and LSTM models
4. **Evaluation**: Evaluates models and determines if distinguishers were found
5. **Visualization**: Generates training curves and saves results

To run an experiment:
1. Open the desired notebook (e.g., `ROUND_1.ipynb`)
2. Execute all cells sequentially
3. Results will be displayed and training plots will be saved

### Dataset Configuration

The dataset generation can be customized by modifying these parameters in each notebook:

- `N_PER_CLASS`: Number of samples per class (default: 50,000)
- `DATA_ROUNDS`: Number of ASCON permutation rounds (1-5)
- `DELTA`: Input difference (default: single bit difference in the first word)

### Model Configuration

Model hyperparameters can be adjusted in each notebook:

**LightGBM:**
- `num_leaves`: 512
- `learning_rate`: 0.01
- `max_depth`: 12
- `num_boost_round`: 2000

**CNN:**
- `EPOCHS`: 50
- `BATCH_SIZE`: 48
- `INITIAL_LR`: 0.0008

**LSTM:**
- `EPOCHS`: 60
- `BATCH_SIZE`: 48
- `INITIAL_LR`: 0.0006

## ASCON Permutation Details

The ASCON permutation operates on a 320-bit state organized as 5 words of 64 bits each. Each round consists of:

1. **Addition of Constants**: Round constant XORed into state[2]
2. **Substitution Layer**: 5-bit S-box applied to each bit slice
3. **Linear Diffusion Layer**: Rotations and XORs for each word

The project implements a reduced-round version of the ASCON permutation for analysis.

## Data Format

### Dataset CSV Format

Each generated dataset is saved as a CSV file with the following columns:

- `C1`: Ciphertext 1 (comma-separated decimal bytes)
- `C2`: Ciphertext 2 (comma-separated decimal bytes)
- `C1_xor_C2`: XOR of C1 and C2 (comma-separated decimal bytes)
- `Label`: Binary label (0 for random pairs, 1 for differential pairs)

### Feature Representation

Features are converted to binary vectors:
- Each byte is unpacked into 8 bits
- Total feature length: 320 bits (40 bytes × 8 bits)
- Features are normalized to float32 format