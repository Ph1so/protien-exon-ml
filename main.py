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

def cnn_model(sequence_length=SEQUENCE_LENGTH, vocab_size=len(AMINO_ACID_LIST)):
    """
    Builds a smaller CNN model optimized for sparse matrices and faster convergence.
    
    Args:
    - sequence_length: Length of input sequences.
    - vocab_size: Number of unique features (e.g., amino acids).
    
    Returns:
    - Compiled smaller CNN model.
    """
    # Input layer for sparse data
    input_layer = layers.Input(shape=(sequence_length, vocab_size), name="Input_Layer", sparse=True)
    
    # Convolutional layers with reduced depth
    x = layers.Conv1D(64, kernel_size=7, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    
    # GlobalAveragePooling1D to reduce the dimensions without over-parameterization
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layer with sigmoid activation for binary classification
    output_layer = layers.Dense(sequence_length, activation="sigmoid")(x)
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=focal_loss(alpha=0.25, gamma=2.0), metrics=["accuracy"])

    return model

def main(args):
    print(f"{'-'*75}\nInput file: {args.input_file}\nModel architecture: {args.model_arch}")

    # Validate args
    available_input_files = ["X_Y_output.npz"]
    available_model_arch = ["deep_cnn", "spliceai"]

    if args.input_file not in available_input_files:
        print(f"Error: Invalid input file '{args.input_file}'. Available options are: {available_input_files}")
        sys.exit(1)
    if args.model_arch not in available_model_arch:
        print(f"Error: Invalid model architecture '{args.model_arch}'. Available options are: {available_model_arch}")
        sys.exit(1)  

    print(f"All inputs are valid. Proceeding with loading and preprocessing data from {args.input_file}")

    # Load data
    X, Y = load_and_preprocess_data(args.input_file)
    np.set_printoptions(threshold=np.inf, linewidth=200)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    # Build model
    if args.model_arch == "deep_cnn":
        model = cnn_model()
    model.summary()

    # Train model
    print(f"Beginning training.")
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val), 
        class_weight = {0: 0.5, 1: 150},
        batch_size=16, 
        epochs=10, 
        callbacks=[callback_early_stopping, callback_lr_scheduler]
    )

    # Save model
    model.save("model.keras")
    
    # Evaluate on the test set
    print(f"\n{'-'*75}\nEvaluating on the test set...")

    # Predict for the entire test set in one batch
    predictions = model.predict(X_test)

    # Distribution of differences
    differences = []

    # Suppress TensorFlow logging if needed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    tf.get_logger().setLevel('ERROR')

    for i in range(len(X_test)):
        # Get prediction for a single example
        X_example = X_test[i].reshape(1, SEQUENCE_LENGTH, len(AMINO_ACID_LIST))
        prediction = predictions[i].reshape(1, -1)  # Use precomputed predictions
        actual_value = Y_test[i]

        if isinstance(actual_value, np.ndarray):
            indices_of_1 = np.where(actual_value == 1)[0]
            if len(indices_of_1) > 0:
                # Average value at indices of 1
                values_in_prediction = prediction[0][indices_of_1]
                average_of_selected = np.mean(values_in_prediction)

                # Overall average prediction
                overall_average_prediction = np.mean(prediction)

                # Difference
                difference = average_of_selected - overall_average_prediction
                differences.append(difference)

    # Analyze distribution of differences
    differences = np.array(differences)
    mean_difference = np.mean(differences)
    positive_proportion = np.sum(differences > 0) / len(differences)

    print(f"\n{'-'*75}")
    print(f"Mean difference: {mean_difference:.4f}")
    print(f"Proportion of positive differences: {positive_proportion:.4f}")

    # Perform one-sample t-test
    from scipy.stats import ttest_1samp

    t_stat, p_value = ttest_1samp(differences, popmean=0)

    # Print the results of the t-test
    print(f"\n{'-'*75}")
    print(f"T-statistic: {t_stat:.9f}")
    print(f"P-value: {p_value:.9f}")

    # Interpret the result
    if p_value < 0.05:
        print("The mean difference is statistically significant (p < 0.05).")
    else:
        print("The mean difference is not statistically significant (p >= 0.05).")


    # Optional: Plot histogram of differences
    try:
        import matplotlib.pyplot as plt
        plt.hist(differences, bins=20, edgecolor="k")
        plt.title("Distribution of Average Differences")
        plt.xlabel("Difference (Selected Avg - Overall Avg)")
        plt.ylabel("Frequency")

        # Save the histogram as an image
        output_image_path = "difference_histogram.png"
        plt.savefig(output_image_path)
        print(f"Histogram saved as {output_image_path}")
        plt.close()  # Close the figure to avoid displaying it
    except ImportError:
        print("matplotlib not available; skipping histogram.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for running a machine learning model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--model_arch", type=str, required=True, help="Which model architecture to use.")
    args = parser.parse_args()
    main(args)