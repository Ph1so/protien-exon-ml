import os
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import scipy.stats as stats
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Masking
from tensorflow.keras.models import Model

MIN_SEQUENCE_LENGTH = 0
SEQUENCE_LENGTH = 1300
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"  # 20 amino acids
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}
PADDING_INDEX = len(amino_acid_dict)  # Assign the last index for padding
amino_acid_dict["<PAD>"] = PADDING_INDEX  # Add padding as a special "amino acid"

GROUPED_AMINO_ACIDS = ["GVAR","EDSK", "LIPT", "NWQC", "NMHY"]

def load_and_preprocess_data(file_path): 
    """
    Args:
    - file_path (str): .npz file should have two columns X and Y
    
    Return
    - X (NumPy array): input for the model. shape should be (batch size, SEQUENCE_LENGTH, 20 (20 amino acids))
    - Y (NumPy array): target for the model. shape should be (batch size, SEQUENCE_LENGTH, 2) [One-hot encoded]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    
    # Print available keys in the .npz file to understand its structure
    print(f"Keys in the .npz file: {list(data.keys())}")
    
    # Load data and print shapes
    X = data['X'] 
    Y = data['Y']
    print(f"Shape of X before preprocessing: {X.shape}")
    print(f"Shape of Y before preprocessing: {Y.shape}")
    
    # Convert Y to one-hot encoded format: [0, 1] for 1, [1, 0] for 0, [0, 0] for padding
    Y_one_hot = np.zeros((Y.shape[0], Y.shape[1], 2))  # (batch_size, sequence_length, 2)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # Use np.argmax to get the index of the class in the one-hot encoded vector
            class_idx = np.argmax(Y[i, j])
            if class_idx == 0:
                Y_one_hot[i, j] = [1, 0]  # Label 0
            elif class_idx == 1:
                Y_one_hot[i, j] = [0, 1]  # Label 1
            else:
                Y_one_hot[i, j] = [0, 0]  # Padding
    
    # Print shapes after processing
    print(f"Shape of X after preprocessing: {X.shape}")
    print(f"Shape of Y after preprocessing (one-hot encoded): {Y_one_hot.shape}")
    print(f"Sample of one-hot encoded Y[0]: {Y_one_hot[0]}")
    
    return X, Y_one_hot

def print_random_samples(model, X_test, Y_test, num_samples=10):
    # Generate model predictions
    predictions = model.predict(X_test)

    # Convert predictions from one-hot encoded to class labels (argmax)
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(Y_test, axis=-1)  # Assuming Y_test is one-hot encoded
    sum = 0
    # Randomly sample indices
    random_indices = np.random.choice(len(X_test), num_samples, replace=False)

    print(f"Displaying {num_samples} random samples from the test set:\n")
    for idx in random_indices:
        print(f"Sample {idx+1}:")
        # print(f"  Predicted: {predicted_classes[idx]}")
        # print(f"  Expected: {true_classes[idx]}")
        print(f"num 1s: {np.sum(predicted_classes[idx])}")
        print("-" * 50)
        sum += np.sum(predicted_classes[idx])
    print(f"sum 1s: {sum}")

def cnn():
    pass

def lstm():
    pass

def main(args):
    print(f"{'-'*75}\nInput file: {args.input_file}\nModel architecture: {args.model_arch}")

    # Load data
    X, Y = load_and_preprocess_data(args.input_file)
    np.set_printoptions(threshold=np.inf, linewidth=200)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    # Build model
    if args.model_arch == "cnn":
        model = cnn()
    if args.model_arch == "lstm":
        model = lstm()
    model.summary()

    # Train model
    print(f"Beginning training.")
    
    model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val),
        epochs=5, batch_size=32,
    )

    # Evaluate the model
    print("\nEvaluating on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions and compare
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=-1)
    model.save("model.keras")
    print_random_samples(model, X_test, Y_test)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Sequence Classification")
    parser.add_argument(
        "--input_file", type=str, choices=["X_Y_output.npz"], default="X_Y_output.npz", help="Path to input file."
    )
    parser.add_argument(
        "--model_arch", type=str, choices=["cnn", "lstm"], default="cnn", help="Model architecture."
    )
    args = parser.parse_args()

    main(args)
