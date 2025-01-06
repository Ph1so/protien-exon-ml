import os
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

SEQUENCE_LENGTH = 600
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}  # 20 amino acids

GROUPED_AMINO_ACIDS = ["GVAR","EDSK", "LIPT", "NWQC", "NMHY"]

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
    
    # Convolutional layers
    x = layers.Conv1D(64, kernel_size=7, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.Conv1D(512, kernel_size=3, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    
    # Replace Flatten with GlobalAveragePooling1D to maintain vector length consistency
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    output_layer = layers.Dense(sequence_length, activation="sigmoid")(x)  # Sigmoid for binary classification
    
    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=focal_loss(alpha=0.25, gamma=2.0), metrics=["accuracy"])

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
    # add args to batch size and args 
    if args.model_arch == "deep_cnn":
        model = deep_cnn_model()
    # elif args.model_arch == "spliceai":
        # model = spliceai()
    model.summary()

    # Flatten the Y_train array to make it 1D
    # Y_train_flat = Y_train.flatten()

    # Calculate dynamic class weights based on the flattened training set
    # class_weights = class_weight.compute_class_weight(
    #     class_weight='balanced', 
    #     classes=np.unique(Y_train_flat), 
    #     y=Y_train_flat
    # )

    # # Convert the class weights into a dictionary
    # class_weight_dict = dict(zip(np.unique(Y_train_flat), class_weights))
    # print(f"Class weights: {class_weight_dict}")
    # class_weight_dict = {0: 0.1, 1: 1000}
    # Use the validation set during training

    print(f"Beginning training.")
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    model.fit(
    X_train, Y_train, 
    validation_data=(X_val, Y_val), 
    batch_size=16, 
    epochs=100, 
    callbacks=[callback_early_stopping, callback_lr_scheduler])
# learning rate, weight decay, start with 0.0001 learning rate
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
    for i in range(10):
        X_example = X_test[i].reshape(1, SEQUENCE_LENGTH, len(AMINO_ACID_LIST))
        prediction = model.predict(X_example)
        binary_output = (prediction > 0.5).astype(int)
        actual_value = Y_test[i]
        
        print(f"Sample {i + 1}:")
        print(f"Num 1s: {sum(binary_output.flatten())}")
        print(f"X: {binary_output.flatten()}")
        print(f"Y: {actual_value}")
        print('-' * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for running a machine learning model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--model_arch", type=str, required=True, help="Which model architecture to use.")
    args = parser.parse_args()
    main(args)