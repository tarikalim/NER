import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import torch

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./trained_model"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",  # Person
    "B-LOC", "I-LOC",  # Location
    "B-ORG", "I-ORG",  # Organization
    "B-TIME", "I-TIME",  # Time
    "B-CUR", "I-CUR",  # Currency
    "B-MISC", "I-MISC"  # Other
]
NUM_LABELS = len(LABEL_LIST)
label_map = {label: i for i, label in enumerate(LABEL_LIST)}

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will be performed on: {device}")


# Load datasets
def load_ner_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned_data = []
    for entry in data:
        tokens = [token["token"] for token in entry["tokens"]]
        ner_tags = [token["tag"] for token in entry["tokens"]]
        cleaned_data.append({"tokens": tokens, "ner_tags": ner_tags})
    return cleaned_data


# Paths to datasets
dataset_path = "dataset/dataset.json"
data = load_ner_dataset(dataset_path)

# Split dataset into train and validation,
# 80% of data will be used for train and 20% of data will be used for validation.
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert  data to hugging face format.
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenizer for bert
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Tokenization and label alignment (for validation and train data)
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, padding="max_length")
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

# code to train model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

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
    report_to=None,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model and tokenizer saved to {OUTPUT_DIR}")
