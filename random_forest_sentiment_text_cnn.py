import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
# 1. Load and clean data
df = pd.read_csv("scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv")
df = df.dropna(subset=["Comment", "Final Annotation"])
df = df[df["Final Annotation"].astype(str).isin(["0", "1", "2"])]
X = df["Comment"].astype(str)
y = df["Final Annotation"].astype(int)
# 2. Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
max_length = 100
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
# 3. Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)
# 4. Build CNN model (TextCNN for feature extraction)
cnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu')  # Features to be extracted
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 5. Train CNN
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
# 6. Extract features
feature_extractor = Sequential(cnn_model.layers[:-1])  # Remove final dense layer
X_train_features = feature_extractor.predict(X_train)
X_val_features = feature_extractor.predict(X_val)
# 7. Standardize features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)
# 8. Train Random Forest on CNN features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_features, y_train)
# 9. Evaluate
y_pred = rf.predict(X_val_features)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
