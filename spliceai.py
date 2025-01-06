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

def spliceai(vocab_size=AMINO_ACID_LIST, embedding_dim=64):  # vocab_size should be 20 (for 20 amino acids)
    """Builds the model architecture as per the SpliceAI diagram."""
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, vocab_size), name="Input_Layer")
    
    # Initial 1x1 convolution
    x1 = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x2 = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    x3 = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(input_layer)
    
    # First block of residual layers (4 RBs with kernel size 11 and stride 1)
    for _ in range(4):
        x1 = residual_block(x1, filters=32, kernel_size=11)
    for _ in range(4):
        x2 = residual_block(x2, filters=32, kernel_size=11)
    for _ in range(4):
        x3 = residual_block(x3, filters=32, kernel_size=11)

    # Concatenate the outputs of the 3 convolutions
    x = layers.Concatenate()([x1, x2, x3])

    # Intermediate 1x1 convolution
    x = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    
    # Second block of residual layers (4 RBs with kernel size 11 and stride 4)
    for _ in range(4):
        x = residual_block(x, filters=32, kernel_size=11)
    
    # Final 1x1 convolution before output
    x = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    x = layers.Conv1D(3, kernel_size=1, activation="relu", padding="same", kernel_initializer=initializers.GlorotUniform())(x)

    # Output layer
    x = layers.Conv1D(1, kernel_size=1, activation="sigmoid", padding="same", kernel_initializer=initializers.GlorotUniform())(x)
    output_layer = layers.Reshape((SEQUENCE_LENGTH,))(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # binary because its 0 or 1 - splice ai used softmax because there are 3 categories
    
    return model