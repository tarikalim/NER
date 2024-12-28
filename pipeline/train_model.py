from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import json
import os
from transformers import DataCollatorForTokenClassification


os.environ["WANDB_DISABLED"] = "true"

# Constants
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./trained_model"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# Updated LABEL_LIST
LABEL_LIST = [
    "O",  # None
    "B-PER", "I-PER",  # Person
    "B-LOC", "I-LOC",  # Location
    "B-ORG", "I-ORG",  # Organization
    "B-TIME", "I-TIME",  # Time
    "B-CUR", "I-CUR"  # Currency
]
NUM_LABELS = len(LABEL_LIST)


# Load datasets
def load_ner_dataset(file_path):
    """Load and clean dataset from a JSON file in NER format"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in data:
        tokens = entry.get("tokens", [])
        ner_tags = entry.get("ner_tags", [])
        # Ensure tokens and tags have the same length
        if len(tokens) == len(ner_tags):
            # Replace invalid tags with "O"
            ner_tags = [tag if tag in LABEL_LIST else "O" for tag in ner_tags]
            cleaned_data.append({"tokens": tokens, "ner_tags": ner_tags})
    return cleaned_data


# Paths to datasets
einstein_dataset_path = "fixed_einstein_cot.json"
gpt_dataset_path = "einstein_cot.json"

# Load datasets
einstein_data = load_ner_dataset(einstein_dataset_path)
gpt_data = load_ner_dataset(gpt_dataset_path)

# Convert to Hugging Face Dataset format
einstein_dataset = Dataset.from_list(einstein_data)
gpt_dataset = Dataset.from_list(gpt_data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization and label alignment
label_map = {label: i for i, label in enumerate(LABEL_LIST)}


def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, padding="max_length")
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)  # Ignore tokens not aligned with words
            else:
                label_ids.append(label_map[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize datasets
tokenized_einstein = einstein_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_gpt = gpt_dataset.map(tokenize_and_align_labels, batched=True)

# Combine datasets into a DatasetDict
dataset = DatasetDict(
    {
        "train": tokenized_einstein,
        "validation": tokenized_gpt
    }
)

# Model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to("cuda")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to=None,  # Disable external logging
)

# Data collator

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model and tokenizer saved to {OUTPUT_DIR}")
