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
    - Y (NumPy array): target for the model. shape should be (batch size, SQUENCE_LENGTH, 2) [One-hot encoded]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    
    X = data['X'] 
    Y = data['Y']  
    
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

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.
    Args:
        alpha (float): Balancing factor for the rare class (junction spots).
        gamma (float): Focusing parameter for hard examples.
    Returns:
        A loss function to be used in model.compile().
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute the cross-entropy component
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Add the focal weight
        focal_weight = y_true * (1 - y_pred)**gamma + (1 - y_true) * y_pred**gamma
        
        # Combine alpha and focal weights
        loss = alpha * focal_weight * cross_entropy
        return tf.reduce_mean(loss)
    
    return loss_fn

def cnn_model(
    sequence_length=SEQUENCE_LENGTH,
    vocab_size=len(AMINO_ACID_LIST)+1,
    kernel_size=3,
    dropout_rate=0.3,
    learning_rate=0.001,
    output_activation="softmax",  # Change to softmax since we're doing multi-class classification
    loss_function="categorical_crossentropy",  # Change to categorical crossentropy
):
    """
    Builds an optimized hybrid CNN model for protein sequence data.

    Args:
    - sequence_length (int): Length of input sequences.
    - vocab_size (int): Number of unique features (e.g., amino acids).
    - kernel_size (int): Size of the convolutional filters.
    - dropout_rate (float): Dropout rate for regularization.
    - learning_rate (float): Learning rate for the optimizer.
    - output_activation (str): Activation function for the output layer.
    - loss_function (str): Loss function for model compilation.

    Returns:
    - tf.keras.Model: Compiled CNN model.
    """
    # Input layer
    input_layer = layers.Input(shape=(sequence_length, vocab_size), name="Input_Layer")

    # First Conv1D layer
    x = layers.Conv1D(
        filters=32,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_initializer=initializers.GlorotUniform(),
        name="Conv1D_1",
    )(input_layer)
    x = layers.BatchNormalization(name="BatchNorm_1")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_1")(x)

    # Second Conv1D layer
    x = layers.Conv1D(
        filters=64,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_initializer=initializers.GlorotUniform(),
        name="Conv1D_2",
    )(x)
    x = layers.BatchNormalization(name="BatchNorm_2")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_2")(x)

    # Third Conv1D layer
    x = layers.Conv1D(
        filters=128,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_initializer=initializers.GlorotUniform(),
        name="Conv1D_3",
    )(x)
    x = layers.BatchNormalization(name="BatchNorm_3")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_3")(x)

    # Fourth Conv1D layer
    x = layers.Conv1D(
        filters=256,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_initializer=initializers.GlorotUniform(),
        name="Conv1D_4",
    )(x)
    x = layers.BatchNormalization(name="BatchNorm_4")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_4")(x)

    # Add a Conv1D layer to adjust the number of output units to match the sequence length
    x = layers.Conv1D(
        filters=2,  # 2 classes
        kernel_size=1,  # Small kernel to keep sequence length the same
        activation="softmax",  # Softmax for multi-class classification
        padding="same",
        name="Conv1D_Output",
    )(x)

    # Output layer (no need for reshape)
    output_layer = x

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = models.Model(inputs=input_layer, outputs=output_layer, name="ProteinCNNModel")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

    return model

def lstm(input_dim, embedding_dim, lstm_units):
    """
    Builds an LSTM model for predicting exon junction locations.

    Args:
        input_dim (int): Size of the input vocabulary (e.g., amino acid tokens).
        embedding_dim (int): Dimensionality of the embedding layer.
        lstm_units (int): Number of units in the LSTM layer.

    Returns:
        tf.keras.Model: A compiled LSTM model.
    """
    # Define the input layer (variable-length protein sequences)
    inputs = Input(shape=(None, input_dim), dtype='float32', name='protein_sequence')  # Expecting sequences with shape (None, input_dim)

    # Masking layer for handling padded sequences
    masked_inputs = Masking(mask_value=0)(inputs)

    # Bidirectional LSTM layer for sequence encoding
    lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(masked_inputs)

    # TimeDistributed Dense layer for binary classification (junction or not) without mask
    outputs = TimeDistributed(Dense(2, activation='softmax'))(lstm_out)

    # Define and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main(args):
    print(f"{'-'*75}\nInput file: {args.input_file}\nModel architecture: {args.model_arch}")

    # Load data
    X, Y = load_and_preprocess_data(args.input_file)
    np.set_printoptions(threshold=np.inf, linewidth=200)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    # Build model
    if args.model_arch == "deep_cnn":
        model = cnn_model()
    if args.model_arch == "lstm":
        model = lstm(len(amino_acid_dict), 50, 128)
    model.summary()

    # Train model
    print(f"Beginning training.")
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val),
        epochs=5, batch_size=32,
        callbacks=[callback_early_stopping, callback_lr_scheduler]
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
        "--model_arch", type=str, choices=["deep_cnn", "lstm"], default="deep_cnn", help="Model architecture."
    )
    args = parser.parse_args()

    main(args)
