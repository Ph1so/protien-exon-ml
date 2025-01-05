# Running the Machine Learning Model

**Notes:**  
This repository has been copied to the AWS EC2 instance "phi-ml." You can run the model there. Let me know if you need the `.pem` file to access it.

### Repository Structure

As of Jan 5th, the repo contains the following files:
- `main.py`
- `one_hot_encode_data.py`
- `test_model.py`
- `processed_file.csv`
- `requirements.txt`
- `documentation.md`

However, these are the only ones you need to worry about:
- `main.py`
- `one_hot_encode_data.py`
- `processed_file.csv`

---

### `main.py`

To start training the model, use the following command and specify the input data file and the model architecture.

#### Command-line Arguments:
- `input_file`: The data file (e.g., `X_Y_output.npz`).
- `model_arch`: The architecture to use, either `"deep_cnn"` or `"spliceai"`.

#### Example:
```bash
$ python3 main.py --input_file X_Y_output.npz --model_arch "deep_cnn"
```

### `one_hot_encode_data.py`

One hot encoding the data can be time consuming and use a lot of resources if done redundantly. So, just run this once to get a .npz file from the dataset. 
If you alreayd have the .npz file then theres no need to run it again. 

#### Command-line Arguments:
- `csv_file`: The origonal csv data file. This is where we use `processed_file.csv` from above.

#### Example:
```bash
$ python3 one_hot_encode_data.py --csv_file processed_file.csv
```

---
other files can be ignored
