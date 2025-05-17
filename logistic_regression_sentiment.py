import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

df_train_test = pd.read_csv("scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv")
df_lemmatized = pd.read_csv("scraped_datasets/fb_comments/LEMMATIZED_SAMPLE_DATASET.csv")

df_train_test = df_train_test.dropna(subset=["Comment", "Final Annotation"])
df_lemmatized = df_lemmatized.dropna(subset=["Lemmatized Comment", "Label"])

df_train_test = df_train_test[df_train_test["Final Annotation"].astype(str).isin(["0", "1", "2"])]
df_lemmatized = df_lemmatized[df_lemmatized["Label"].astype(str).isin(["0", "1", "2"])]

X_train_test = df_train_test["Comment"].astype(str)
y_train_test = df_train_test["Final Annotation"].astype(int)

X_lemmatized = df_lemmatized["Lemmatized Comment"].astype(str)
y_lemmatized = df_lemmatized["Label"].astype(int)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_test_vect = vectorizer.fit_transform(X_train_test)
X_lemmatized_vect = vectorizer.transform(X_lemmatized)

model = LogisticRegression(max_iter=1000, class_weight='balanced')

cv_scores = cross_val_score(model, X_train_test_vect, y_train_test, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_test_vect, y_train_test)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

best_model.fit(X_train_test_vect, y_train_test)

y_train_test_pred = best_model.predict(X_train_test_vect)
print("Performance on SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET:")
print("Accuracy:", accuracy_score(y_train_test, y_train_test_pred))
print("Classification Report:\n", classification_report(y_train_test, y_train_test_pred))

y_lemmatized_pred = best_model.predict(X_lemmatized_vect)
print("Performance on LEMMATIZED_SAMPLE_DATASET:")
print("Accuracy:", accuracy_score(y_lemmatized, y_lemmatized_pred))
print("Classification Report:\n", classification_report(y_lemmatized, y_lemmatized_pred))
