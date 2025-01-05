import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Constants
SEQUENCE_LENGTH = 1200
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}

# Convert protein sequences to integer encoding
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    data = pd.read_csv(file_path)
    
    # Filter sequences with length <= SEQUENCE_LENGTH
    data = data[data["sequence"].str.len() <= SEQUENCE_LENGTH].reset_index(drop=True)
    
    # Convert sequences to integer encoding (amino acid index)
    data["int_sequence"] = data["sequence"].apply(lambda seq: [amino_acid_dict.get(char, 0) for char in seq])
    
    # Pad sequences to length SEQUENCE_LENGTH
    data["int_sequence"] = data["int_sequence"].apply(lambda x: x + [0] * (SEQUENCE_LENGTH - len(x)) if len(x) < SEQUENCE_LENGTH else x[:SEQUENCE_LENGTH])
    
    # Masks (if available) should be handled here, assuming binary masks as before.
    def process_mask(mask):
        mask = "".join(char if char in "01" else "0" for char in str(mask))
        return np.array([int(char) for char in mask])

    masks = data["mask"].apply(process_mask)
    data["mask_array"] = masks.apply(lambda x: np.pad(x, (0, SEQUENCE_LENGTH - len(x)), mode="constant"))
    
    X = np.stack(data["int_sequence"].to_numpy())
    Y = np.stack(data["mask_array"].to_numpy()).reshape(-1, SEQUENCE_LENGTH, 1)
    
    return X, Y

def main():
    file_path = "processed_file.csv"
    model_path = "protein_embeddings_model.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The file '{model_path}' does not exist.")
    
    # Load the model
    model = load_model(model_path)
    model.summary()
    
    # Load and preprocess data
    X, Y = load_and_preprocess_data(file_path)
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Select 10 random samples from the test set
    random_indices = np.random.choice(X_test.shape[0], 10, replace=False)
    X_sample = X_test[random_indices]
    Y_sample = Y_test[random_indices]
    
    # Predict the labels for the random samples
    predictions = model.predict(X_sample)
    np.set_printoptions(threshold=np.inf)
    # Print the predictions and expected values for the 10 random samples
    for i in range(10):
        binary_output = (predictions[i] > 0.5).astype(int)  # Convert to binary output (0 or 1)
        expected_output = Y_sample[i].flatten()  # Flatten the target output to match
        
        print(f"Sample {i + 1}:")
        print("Predicted Output:\n", binary_output)
        print(f"# 1s: {sum(binary_output)}\n")
        print("Expected Output:\n", expected_output)
        print("-" * 50)

if __name__ == "__main__":
    main()
