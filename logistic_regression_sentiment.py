import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from collections import Counter

# Load datasets
df_train_test = pd.read_csv("scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv")
df_lemmatized = pd.read_csv("scraped_datasets/fb_comments/LEMMATIZED_SAMPLE_DATASET.csv")

# Preprocess datasets
df_train_test = df_train_test.dropna(subset=["Comment", "Final Annotation"])
df_lemmatized = df_lemmatized.dropna(subset=["Lemmatized Comment", "Label"])

df_train_test = df_train_test[df_train_test["Final Annotation"].astype(str).isin(["0", "1", "2"])]
df_lemmatized = df_lemmatized[df_lemmatized["Label"].astype(str).isin(["0", "1", "2"])]

X_train_test = df_train_test["Comment"].astype(str)
y_train_test = df_train_test["Final Annotation"].astype(int)

X_lemmatized = df_lemmatized["Lemmatized Comment"].astype(str)
y_lemmatized = df_lemmatized["Label"].astype(int)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_test)

X_train_test_seq = tokenizer.texts_to_sequences(X_train_test)
X_lemmatized_seq = tokenizer.texts_to_sequences(X_lemmatized)

max_length = 100
X_train_test_padded = pad_sequences(X_train_test_seq, maxlen=max_length, padding='post', truncating='post')
X_lemmatized_padded = pad_sequences(X_lemmatized_seq, maxlen=max_length, padding='post', truncating='post')

X_train, X_val, y_train, y_val = train_test_split(X_train_test_padded, y_train_test, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate on the original dataset
y_train_test_pred = model.predict(X_train_test_padded).argmax(axis=1)
print("Performance on SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET:")
print("Accuracy:", accuracy_score(y_train_test, y_train_test_pred))
print("Classification Report:\n", classification_report(y_train_test, y_train_test_pred))

# Analysis
print("Detailed Evaluation on Full Dataset:")
class_counts = Counter(y_train_test)
print(f"Class Distribution: {dict(class_counts)}")
print("Most common label:", class_counts.most_common(1)[0][0])
print("Predicted label distribution:", dict(Counter(y_train_test_pred)))
precision_per_class = precision_score(y_train_test, y_train_test_pred, average=None, zero_division=0)

# Evaluate on the lemmatized dataset
y_lemmatized_pred = model.predict(X_lemmatized_padded).argmax(axis=1)
print("Performance on LEMMATIZED_SAMPLE_DATASET:")
print("Accuracy:", accuracy_score(y_lemmatized, y_lemmatized_pred))
print("Classification Report:\n", classification_report(y_lemmatized, y_lemmatized_pred))

# Analysis
print("Detailed Evaluation on Lemmatized Dataset:")
class_counts_lem = Counter(y_lemmatized)
print(f"Class Distribution: {dict(class_counts_lem)}")
print("Most common label:", class_counts_lem.most_common(1)[0][0])
print("Predicted label distribution:", dict(Counter(y_lemmatized_pred)))
