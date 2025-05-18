import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

# Build the RNN model (feature extractor)
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu')  # Feature extraction layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the RNN
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Extract features from the RNN
feature_extractor = Sequential(model.layers[:-1])  # Remove the final dense layer
X_train_features = feature_extractor.predict(X_train)
X_val_features = feature_extractor.predict(X_val)
X_train_test_features = feature_extractor.predict(X_train_test_padded)
X_lemmatized_features = feature_extractor.predict(X_lemmatized_padded)

# Standardize the features for logistic regression
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)
X_train_test_features = scaler.transform(X_train_test_features)
X_lemmatized_features = scaler.transform(X_lemmatized_features)

# Train logistic regression on the extracted features
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')
logistic_model.fit(X_train_features, y_train)

# Evaluate on the original dataset
y_train_test_pred = logistic_model.predict(X_train_test_features)
print("Performance on SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET:")
print("Accuracy:", accuracy_score(y_train_test, y_train_test_pred))
print("Classification Report:\n", classification_report(y_train_test, y_train_test_pred))

# Evaluate on the lemmatized dataset
y_lemmatized_pred = logistic_model.predict(X_lemmatized_features)
print("Performance on LEMMATIZED_SAMPLE_DATASET:")
print("Accuracy:", accuracy_score(y_lemmatized, y_lemmatized_pred))
print("Classification Report:\n", classification_report(y_lemmatized, y_lemmatized_pred))
