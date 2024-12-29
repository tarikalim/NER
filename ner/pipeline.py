import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_DIR = "./trained_model"
INPUT_FILE = "input.txt"  # File containing input sentences (one sentence per line!)
OUTPUT_FILE = "output.json"

# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# Initialize the pipeline for token classification
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Read input sentences from the file without specifying encoding to utf-8,
# we got an error on windows10 even default encoding set to utf-8.
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

outputs = []
for sentence in sentences:
    annotations = nlp(sentence)
    # Convert annotations to plain Python types
    clean_annotations = [
        {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in annotation.items()}
        for annotation in annotations
    ]
    outputs.append({"sentence": sentence, "annotations": clean_annotations})

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)

print(f"Annotated outputs saved to {OUTPUT_FILE}")