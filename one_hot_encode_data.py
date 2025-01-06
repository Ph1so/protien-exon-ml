import numpy as np
import os
import argparse
import pandas as pd

SEQUENCE_LENGTH = 1200
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}

GROUPED_AMINO_ACIDS = ["GVAR", "EDSK", "LIPT", "NWQC", "NMHY"]

def grouped_encode_sequence(sequence):
    """Encode a sequence into a grouped one-hot encoded integer array."""
    one_hot_seq = np.zeros((SEQUENCE_LENGTH, len(GROUPED_AMINO_ACIDS)), dtype=int)  # Ensure dtype=int
    for i, char in enumerate(sequence):
        for j, group in enumerate(GROUPED_AMINO_ACIDS):
            if char in group:  
                one_hot_seq[i, j] = 1  # Assign 1 to the appropriate group
                break  # Stop after the first match (one-to-one mapping)
    return one_hot_seq

def encode_sequence(sequence):
    """Encode a sequence into a one-hot encoded integer array."""
    one_hot_seq = np.zeros((SEQUENCE_LENGTH, len(amino_acid_dict)), dtype=int)  # Ensure dtype=int
    for i, char in enumerate(sequence):
        if char in amino_acid_dict:  # Ignore invalid characters
            one_hot_seq[i, amino_acid_dict[char]] = 1
    return one_hot_seq


def load_and_preprocess_data(file_path):
    """Load data, preprocess it, and save X and Y to a CSV file."""
    # Load the data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    data = pd.read_csv(file_path)
    
    # Filter data for valid sequences (length <= SEQUENCE_LENGTH and no 'U' and no '1' in mask)
    data = data[(data["sequence"].str.len() <= SEQUENCE_LENGTH) & 
                (~data["sequence"].str.contains("U")) & (data["mask"].str.contains("1"))].reset_index(drop=True)
    
    # Drop rows with invalid or missing 'mask' values
    data = data[data["mask"].notna()]  # Ensure no NaN values in 'mask'
    data = data[data["mask"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)  # Ensure 'mask' is a string

    # Preprocess X and Y with stopping condition
    X = []
    Y = []
    
    for i, row in data.iterrows():
        # Encode sequence
        seq = row["sequence"]
        one_hot_seq = grouped_encode_sequence(seq)
        X.append(one_hot_seq)
        
        # Encode mask
        mask = list(map(int, row["mask"]))  
        if len(mask) < SEQUENCE_LENGTH:
            mask = mask + [0] * (SEQUENCE_LENGTH - len(mask))  # Pad with zeros
        Y.append(mask)
    
    X = np.array(X, dtype=int)  
    Y = np.array(Y, dtype=int) 
    
    # Save to .npz file
    output_file = "X_Y_output.npz"
    np.savez(output_file, X=X, Y=Y)
    print(f"Saved X and Y to {output_file}, with {len(X)} sequences.")

def main(args):
    load_and_preprocess_data(args.csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for one hot encoding the data.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input data file.")
    args = parser.parse_args()
    main(args)
