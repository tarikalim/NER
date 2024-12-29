import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

os.environ["WANDB_DISABLED"] = "true"

# Constants
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./trained_model"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

LABEL_LIST = [
    "O",  # None
    "B-PER", "I-PER",  # Person
    "B-LOC", "I-LOC",  # Location
    "B-ORG", "I-ORG",  # Organization
    "B-TIME", "I-TIME",  # Time
    "B-CUR", "I-CUR"  # Currency
]
NUM_LABELS = len(LABEL_LIST)
label_map = {label: i for i, label in enumerate(LABEL_LIST)}


# Load datasets
def load_ner_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned_data = []
    for entry in data:
        tokens = [token["token"] for token in entry["tokens"]]
        ner_tags = [token["tag"] for token in entry["tokens"]]
        if len(tokens) == len(ner_tags):
            ner_tags = [tag if tag in LABEL_LIST else "O" for tag in ner_tags]
            cleaned_data.append({"tokens": tokens, "ner_tags": ner_tags})
    return cleaned_data


# Paths to datasets
dataset_path = "dataset/dataset.json"
data = load_ner_dataset(dataset_path)

# Split dataset into train and validation
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Tokenization and label alignment
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"], truncation=True, is_split_into_words=True, padding="max_length"
    )
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label_map[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# Combine datasets
dataset = DatasetDict({"train": tokenized_train_dataset, "validation": tokenized_val_dataset})

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
    report_to=None,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model and tokenizer saved to {OUTPUT_DIR}")
