import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Leximi i dataset-it
print("Leximi i dataset-it...")
data = pd.read_csv('scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv')

# Pastrimi i të dhënave
print("Pastrimi i të dhënave...")
data = data[['Comment', 'Final Annotation']]
data = data.dropna()

data['Final Annotation'] = pd.to_numeric(data['Final Annotation'], errors='coerce')
data = data.dropna(subset=['Final Annotation'])
data['Final Annotation'] = data['Final Annotation'].astype(int)
data = data[data['Final Annotation'].isin([0, 1, 2])]

# Tokenizimi i tekstit
print("Tokenizimi dhe Padding...")
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['Comment'])
sequences = tokenizer.texts_to_sequences(data['Comment'])
max_len = 100
X = pad_sequences(sequences, maxlen=max_len)

# Kodimi i etiketave
y = to_categorical(data['Final Annotation'])

# Ndarja në trajnime dhe testime
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ndërtimi i modelit RNN
print("Ndërtimi i modelit RNN...")
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

# Kompilimi i modelit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trajnimi i modelit
print("Trajnimi i modelit...")
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Vlerësimi i modelit
print("Vlerësimi i modelit...")
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Shfaqja e raporteve të performancës
print("Raporti i klasifikimit:\n")
print(classification_report(y_true, y_pred))
print("\nSaktësia: ", accuracy_score(y_true, y_pred))

# Matriksa e konfuzionit
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Ruajtja e modelit
model.save('rnn_sentiment_model.h5')
print("Modeli u ruajt me sukses si 'rnn_sentiment_model.h5'")

# Funksion për testim të komenteve të reja
def predict_sentiment(comment):
    model = load_model('rnn_sentiment_model.h5')
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = np.argmax(model.predict(padded), axis=1)
    sentiment = ['Neutral', 'Positive', 'Negative']
    print(f"Komenti: '{comment}' -> Sentimenti: {sentiment[prediction[0]]}")


# Interactive Prompt
while True:
    user_input = input("Shkruaj një koment për ta analizuar (ose shkruaj 'exit' për të dalë): ")
    if user_input.lower() == 'exit':
        break
    predict_sentiment(user_input)
print("Numri i rreshtave:", data.shape[0])

print("Programi perfundoi.")
