import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

df = pd.read_csv(
    "scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv",
    sep=",", 
    on_bad_lines='skip', 
    low_memory=False
)

df.columns = df.columns.str.strip().str.replace('\ufeff', '')
df = df[['Annot 1', 'Annot 2', 'Annot 3', 'Final Annotation']]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['Annot 1', 'Annot 2', 'Annot 3', 'Final Annotation'], inplace=True)

if df.empty:
    raise ValueError("Nuk ka të dhëna të mjaftueshme për trajnim!")

X = df[['Annot 1', 'Annot 2', 'Annot 3']].astype(int)
y = df['Final Annotation'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Raporti i klasifikimit:\n")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
labels = ['Neutral (0)', 'Positive (1)', 'Negative (2)']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.tight_layout()
plt.show()
