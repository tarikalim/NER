import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_DIR = "./trained_model"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.json"

# To convert hugging face label indexes to our labels. It makes reading easier
LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-TIME", "I-TIME",
    "B-CUR", "I-CUR",
    "B-MISC", "I-MISC"
]

# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# Initialize the pipeline for token classification
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Read input sentences
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

outputs = []
for sentence in sentences:
    annotations = nlp(sentence)
    # Convert label indices to label names and float32 to float
    clean_annotations = [
        {
            **annotation,
            "entity_group": LABEL_LIST[int(annotation["entity_group"].split("_")[-1])],
            "score": float(annotation["score"])  # Convert float32 to float
        }
        for annotation in annotations
    ]
    outputs.append({"sentence": sentence, "annotations": clean_annotations})

# Write results to JSON again use utf-8 to solve encoding issue
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)

print(f"Annotated outputs saved to {OUTPUT_FILE}")
