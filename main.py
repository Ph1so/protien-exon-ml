import os
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

SEQUENCE_LENGTH = 1200
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}  # 20 amino acids


def residual_block(inputs, filters, kernel_size, strides):
    """Implements the Residual Block (RB) as shown in the SpliceAI diagram."""
    # Main path
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    # Shortcut path
    shortcut = layers.Conv1D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer=initializers.GlorotUniform())(inputs)

    # Adjust shortcut if necessary to match the shape of x
    if x.shape[1] != shortcut.shape[1]:  # If the sequence lengths differ
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding="same", kernel_initializer=initializers.GlorotUniform())(shortcut)

    # Add the main path and shortcut
    x = layers.Add()([x, shortcut])

    return x

def spliceai(vocab_size=20, embedding_dim=64):  # vocab_size should be 20 (for 20 amino acids)
    """Builds the model architecture as per the SpliceAI diagram."""
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, vocab_size), name="Input_Layer")
    
    # Initial 1x1 convolution
    x1 = layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x2 = layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x3 = layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    
    # First block of residual layers (4 RBs with kernel size 11 and stride 1)
    for _ in range(4):
        x1 = residual_block(x1, filters=32, kernel_size=11, strides=1)
    for _ in range(4):
        x2 = residual_block(x2, filters=32, kernel_size=11, strides=1)
    for _ in range(4):
        x3 = residual_block(x3, filters=32, kernel_size=11, strides=1)

    # Concatenate the outputs of the 3 convolutions
    x = layers.Concatenate()([x1, x2, x3])

    # Intermediate 1x1 convolution
    x = layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    
    # Second block of residual layers (4 RBs with kernel size 11 and stride 4)
    for _ in range(4):
        x = residual_block(x, filters=32, kernel_size=11, strides=4)
    
    # Final 1x1 convolution before output
    x = layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    x = layers.Conv1D(3, kernel_size=1, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    # Output layer
    x = layers.Conv1D(1, kernel_size=1, strides=1, activation="sigmoid", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    output_layer = layers.Reshape((SEQUENCE_LENGTH,))(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # binary because its 0 or 1 - splice ai used softmax because there are 3 categories
    
    return model

def load_and_preprocess_data(file_path): 
    """
    Args:
    - file_path (str): .npz file should have two columns X and Y
    
    Return
    - X (NumPy array): input for the model. shape should be (batch size, SEQUENCE_LENGTH, 20 (20 amino acids))
    - Y (NumPy array): target for the model. shape should be (batch size, SQUENCE_LENGTH)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    
    X = data['X'] 
    Y = data['Y']  
    
    return X, Y

def deep_cnn_model(sequence_length=SEQUENCE_LENGTH, vocab_size=len(AMINO_ACID_LIST)):
    """
    Builds a CNN model optimized for sparse matrices.
    
    Args:
    - sequence_length: Length of input sequences
    - vocab_size: Number of unique features (e.g., amino acids)
    
    Returns:
    - Compiled CNN model
    """
    # Input layer for sparse data
    input_layer = layers.Input(shape=(sequence_length, vocab_size), name="Input_Layer", sparse=True)
    
    # Convert sparse input to dense (if required by Conv1D)
    x = layers.Dense(vocab_size, activation="relu")(input_layer)

    # First convolutional block
    x = layers.Conv1D(64, kernel_size=7, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.Conv1D(128, kernel_size=5, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Second convolutional block
    x = layers.Conv1D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.Conv1D(512, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global Average Pooling to reduce dimensions
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    
    # Output layer for sequence classification (binary classification for each position)
    output_layer = layers.Dense(sequence_length, activation="sigmoid")(x)  # Sigmoid for binary classification
    
    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

def main(args):
    print(f"{'-'*75}\nInput file: {args.input_file}\nModel architecture: {args.model_arch}")

    available_input_files = ["X_Y_output.npz"]
    available_model_arch = ["deep_cnn", "spliceai"]

    # Validate args
    if args.input_file not in available_input_files:
        print(f"Error: Invalid input file '{args.input_file}'. Available options are: {available_input_files}")
        sys.exit(1)
    if args.model_arch not in available_model_arch:
        print(f"Error: Invalid model architecture '{args.model_arch}'. Available options are: {available_model_arch}")
        sys.exit(1)  

    print(f"All inputs are valid. Proceeding with loading and preprocessing data from {args.input_file}")

    # Get data from a .npz file
    X, Y = load_and_preprocess_data(args.input_file)
    np.set_printoptions(threshold=np.inf, linewidth=200)

    print(f"Data processed successfully\nInput shape (X): {X.shape}\nTarget shape (Y): {Y.shape}\n{'-'*75}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    if args.model_arch == "deep_cnn":
        model = deep_cnn_model()
    elif args.model_arch == "spliceai":
        model = spliceai()
    model.summary()

    # Flatten the Y_train array to make it 1D
    Y_train_flat = Y_train.flatten()

    # Calculate dynamic class weights based on the flattened training set
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(Y_train_flat), 
        y=Y_train_flat
    )

    # Convert the class weights into a dictionary
    class_weight_dict = dict(zip(np.unique(Y_train_flat), class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Use the validation set during training
    print(f"Beginning training.")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), class_weight=class_weight_dict, batch_size=16, epochs=10)
    # model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=10)


    # Save the best model during training
    model.save("model.keras")
    
    # Evaluate on the test set
    print(f"\n{'-'*75}\nCalculating sum of predictions on all test samples...")
    predictions = model.predict(X_test)  # Predict on the entire test set
    binary_predictions = (predictions > 0.5).astype(int)  # Convert to binary
    sum_predictions = np.sum(binary_predictions)  # Sum across all samples
    print(f"Total sum of predictions: {sum_predictions}")
    
    # Example prediction on 10 samples
    # for i in range(10):
    #     X_example = X_test[i].reshape(1, SEQUENCE_LENGTH, len(AMINO_ACID_LIST))  # Ensure input shape matches
    #     prediction = model.predict(X_example)
    #     binary_output = (prediction > 0.5).astype(int)
    #     actual_value = Y_test[i]
        
    #     print(f"Sample {i + 1}:")
    #     print(f"Num 1s: {sum(binary_output.flatten())}")
    #     print('-' * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for running a machine learning model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--model_arch", type=str, required=True, help="Which model architecture to use.")
    args = parser.parse_args()
    main(args)