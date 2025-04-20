import os
os.environ["WANDB_DISABLED"] = "true"  # 🚫 Disable Weights & Biases

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

data_path = "sample_data/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv"
df = pd.read_csv(data_path)

df = df.dropna(subset=["Final Annotation"])
if "Comment" not in df.columns or "Final Annotation" not in df.columns:
    raise ValueError("Dataset must contain 'Comment' and 'Final Annotation' columns.")

# Encode labels
label_mapping = {label: i for i, label in enumerate(df["Final Annotation"].unique())}
df["label"] = df["Final Annotation"].map(label_mapping)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df[["Comment", "label"]])
test_dataset = Dataset.from_pandas(test_df[["Comment", "label"]])

MODEL_NAME = "xlm-roberta-base"  # Or use "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["Comment"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_mapping))

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

print("Accuracy:", accuracy_score(test_dataset["label"], preds))

label_names = [str(label) for label in label_mapping.keys()]
print("\nClassification Report:\n", classification_report(test_dataset["label"], preds, target_names=label_names, zero_division=0))
