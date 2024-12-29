# Named Entity Recognition (NER) using BERT

## Project Summary

This project trains a **BERT model** for Named Entity Recognition (NER) using a dataset generated in NLP lecture project
1. The model is trained and evaluated using our pipeline, and the trained model is later used for inference on input
sentences provided in a text file.

---

## Usage

### Step 1: Give Permissions to the Script

Before running the project, ensure the script has the necessary permissions:

```bash
chmod +x setup_and_run.sh
```

### Step 2: Run the Script

Run the script to train the model and execute the pipeline:

```bash
./setup_and_run.sh
```

---

## Important Note

- The model is trained on **GPU**, you must specify the pytorch version according to your system,
requirements.txt file doesn't include pytorch file!
- For CUDA 12.1 related command is:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
---

## Input and Output

### Input

- The input sentences for NER are provided in the `input.txt` file (one sentence per line).
- You can edit the `input.txt` file to include your own sentences for testing.

### Output

- After running the script, the model's predictions will be saved in the `output.json` file.

---

## Workflow Steps

1. **Train the Model**:
    - The script trains the BERT model using the dataset located in the `dataset` directory.

2. **Run the Pipeline**:
    - Once training is complete, the `pipeline.py` script processes the sentences in `input.txt` and generates
      predictions.

3. **View Results**:
    - The results are saved in the `output.json` file, showing the sentences and their corresponding NER predictions.

---

### Example Workflow

1. Edit `input.txt` if you want to change example sentences:
   ```
   John lives in New York.
   The Eiffel Tower is in Paris.
   ```
2. Run the script:
   ```bash
   ./setup_and_run.sh
   ```
3. Check the `output.json` file for results.

---

## Directory Structure

```
.
├── ner
│   ├── dataset
│   │   └── dataset.json      # Custom NER dataset
│   ├── train.py              # Training script
│   ├── pipeline.py           # Pipeline script for inference
│   ├── input.txt             # Input sentences for NER
│   ├── output.json           # Model's NER predictions
├── requirements.txt          # Required Python libraries
└── setup_and_run.sh          # Main script to train and run the project
