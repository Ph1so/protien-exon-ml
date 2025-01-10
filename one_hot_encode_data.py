import numpy as np
import os
import argparse
import pandas as pd

MIN_SEQUENCE_LENGTH = 0
SEQUENCE_LENGTH = 1300
AMINO_ACID_LIST = "ARNDCEQGHILKMFPSTWYV"  # 20 amino acids
amino_acid_dict = {char: idx for idx, char in enumerate(AMINO_ACID_LIST)}
PADDING_INDEX = len(amino_acid_dict)  # Assign the last index for padding
amino_acid_dict["<PAD>"] = PADDING_INDEX  # Add padding as a special "amino acid"

# try artifically making the data easier for the model to solve


def encode_sequence(sequence):
    """Encode a sequence into a one-hot encoded integer array with a padding index."""
    # Create a zero array with an additional column for the padding index
    one_hot_seq = np.zeros((SEQUENCE_LENGTH, len(amino_acid_dict)), dtype=int)

    # Encode the sequence
    for i, char in enumerate(sequence):
        if i >= SEQUENCE_LENGTH:  # Stop if the sequence exceeds SEQUENCE_LENGTH
            break
        if char in amino_acid_dict:  # Ignore invalid characters
            one_hot_seq[i, amino_acid_dict[char]] = 1

    # Add padding indicator for remaining positions
    for i in range(len(sequence), SEQUENCE_LENGTH):
        one_hot_seq[i, PADDING_INDEX] = 1  # Mark padding in the last index

    return one_hot_seq


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

def one_hot_encode_mask(mask, sequence_length):
    """
    One-hot encodes the binary mask and differentiates padding.
    
    Args:
    - mask: List of binary values (0 or 1).
    - sequence_length: Desired sequence length for padding.
    
    Returns:
    - One-hot encoded mask with padding differentiated.
    """
    # Define one-hot encoding for binary values and padding
    one_hot_map = {
        0: [1, 0],  # Class 0
        1: [0, 1],  # Class 1
        "padding": [0, 0]  # Special value for padding
    }

    # Apply one-hot encoding to the mask
    one_hot_mask = [one_hot_map[val] for val in mask]

    # Pad with "padding" one-hot values to reach the desired sequence length
    padding = [one_hot_map["padding"]] * (sequence_length - len(mask))
    one_hot_mask += padding

    return one_hot_mask

def load_and_preprocess_data(file_path):
    """Load data, preprocess it, and save X and Y to a CSV file."""
    # Load the data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    data = pd.read_csv(file_path)
    
    # Filter data for valid sequences (length <= SEQUENCE_LENGTH and no 'U' and no '1' in mask)
    data = data[
    (data["sequence"].str.len() >= MIN_SEQUENCE_LENGTH) & 
    (data["sequence"].str.len() <= SEQUENCE_LENGTH) & 
    (~data["sequence"].str.contains("U")) & 
    (data["mask"].str.contains("1"))
].reset_index(drop=True)

    
    # Drop rows with invalid or missing 'mask' values
    data = data[data["mask"].notna()]  # Ensure no NaN values in 'mask'
    data = data[data["mask"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)  # Ensure 'mask' is a string

    # Preprocess X and Y with stopping condition
    X = []
    Y = []
    
    for i, row in data.iterrows():
        # Encode sequence
        seq = row["sequence"]
        one_hot_seq = encode_sequence(seq)
        X.append(one_hot_seq)
        
        # Encode mask
        mask = list(map(int, row["mask"]))  
        # Ensure the mask is padded and one-hot encoded
        one_hot_mask = one_hot_encode_mask(mask, SEQUENCE_LENGTH)

        # Append the one-hot encoded mask to Y
        Y.append(one_hot_mask)
            
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
