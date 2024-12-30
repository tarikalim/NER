# Named Entity Recognition Pipeline

This project trains a **BERT model** for Named Entity Recognition (NER) using a dataset generated for NLP class project 1.

---
## Instructions

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

---

## Input and Output

### Input
- The input sentences for NER are provided in the `input.txt` file (one sentence per line).
### Output
- After running the script, the model's predictions will be saved in the `output.json` file.
  
---


