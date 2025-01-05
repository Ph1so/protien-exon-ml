import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, initializers

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)

SEQUENCE_LENGTH = 1200
# Amino acid encoding dictionary (same as before)
amino_acid_dict = {char: idx for idx, char in enumerate("ARNDCEQGHILKMFPSTWYV")}  # 20 amino acids

def load_and_preprocess_data(file_path): 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    # Load the .npz file
    data = np.load(file_path)
    
    # Access X and Y from the .npz file
    X = data['X']  # X will be a NumPy array
    Y = data['Y']  # Y will be a NumPy array
    
    return X, Y

def residual_block(inputs, filters, kernel_size, strides):
    """Implements the Residual Block (RB) as shown in the diagram."""
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

def build_model(vocab_size=20, embedding_dim=64):  # vocab_size should be 20 (for 20 amino acids)
    """Builds the model architecture as per the updated diagram."""
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

def deep_cnn_model(vocab_size=20, sequence_length=SEQUENCE_LENGTH):
    """Build a deep CNN model for sequence classification."""
    input_layer = layers.Input(shape=(sequence_length, vocab_size), name="Input_Layer")
    x = layers.Conv1D(64, kernel_size=7, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x = layers.Conv1D(128, kernel_size=5, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.Conv1D(512, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(1024, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(2048, activation="relu")(x)

    output_layer = layers.Dense(sequence_length, activation="sigmoid")(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def main():
    file_path = "X_Y_output.npz"
    X, Y = load_and_preprocess_data(file_path)
    np.set_printoptions(threshold=np.inf, linewidth=200)
    print("Input shape (X):", X.shape)  
    print("Target shape (Y):", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    model = deep_cnn_model()
    model.summary()

    # Use the validation set during training
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=10)

    # Save the best model during training
    model.save("model_with_embeddings.h5")
    
    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    # Example prediction
    X_example = X_test[2].reshape(1, SEQUENCE_LENGTH, 20)  # Ensure input shape matches
    prediction = model.predict(X_example)
    binary_output = (prediction > 0.5).astype(int)
    np.set_printoptions(threshold=np.inf)
    print("Binary Output:\n", binary_output)

if __name__ == "__main__":
    main()
