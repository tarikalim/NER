# Named Entity Recognition (NER) using BERT


This project trains a **BERT model** for Named Entity Recognition (NER) using a dataset generated in NLP lecture project 1.

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
  requirements.txt file doesn't include pytorch file! When you run the script, it will ask you the cuda version.
  You can also use cpu to train model, script will give you to select cpu option at the begining.

---

## Input and Output

### Input

- The input sentences for NER are provided in the `input.txt` file (one sentence per line).
- You can edit the `input.txt` file to include your own sentences for testing.

### Output

- After running the script, the model's predictions will be saved in the `output.json` file which located in the `ner` directory.

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

