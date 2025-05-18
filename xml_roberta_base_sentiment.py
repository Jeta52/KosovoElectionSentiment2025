import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)

# The different testing scenerios and cases can be found on this Google Colab link: https://colab.research.google.com/drive/1dLjLjSoWx3kQRUq9pphmkysdIv7XE9Fh?usp=sharing
data_path = "scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv"
df = pd.read_csv(data_path)

df = df[df["Final Annotation"].isin([0, 1, 2])]
df = df.dropna(subset=["Final Annotation"])
if "Comment" not in df.columns or "Final Annotation" not in df.columns:
    raise ValueError("Dataset must contain 'Comment' and 'Final Annotation' columns.")

# Encode labels
label_mapping = {label: i for i, label in enumerate(df["Final Annotation"].unique())}
df["label"] = df["Final Annotation"].map(label_mapping)


# --------------------------------------------------
# 1.  DATA
# --------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df[["Comment", "label"]])
test_dataset  = Dataset.from_pandas(test_df[["Comment", "label"]])

# --------------------------------------------------
# 2.  TOKENISATION
# --------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok(batch):
    batch["Comment"] = [str(c) for c in batch["Comment"]]
    return tokenizer(
        batch["Comment"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tok, batched=True)
test_dataset  = test_dataset.map(tok,  batched=True)

# --------------------------------------------------
# 3.  MODEL
# --------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_mapping)
)

# --------------------------------------------------
# 4.  METRICS
# --------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

# --------------------------------------------------
# 5.  TRAINING ARGUMENTS
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,

    learning_rate=1e-5,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",

    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,

    weight_decay=0.01,
    fp16=True,
    seed=42,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer)

# --------------------------------------------------
# 6.  TRAIN / EVAL
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

predictions = trainer.predict(test_dataset)
preds       = np.argmax(predictions.predictions, axis=1)

print("Accuracy:", accuracy_score(test_dataset["label"], preds))

label_names = [str(label) for label in label_mapping.keys()]
print("\nClassification Report:\n", classification_report(test_dataset["label"], preds, target_names=label_names, zero_division=0))
