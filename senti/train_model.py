import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('restaurant_reviews.csv')

le = LabelEncoder()
df['Liked'] = le.fit_transform(df['Liked'])

def preprocess_text(text):
    return text.lower()

df['Review'] = df['Review'].apply(preprocess_text)

X = df['Review']
y = df['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=1000), SVC(probability=True))
rf_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=1000), RandomForestClassifier(n_estimators=200))

ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model)], voting='soft')

parameters = {
    'svm__svc__C': [0.1, 1, 10],
    'svm__svc__kernel': ['linear', 'rbf'],
    'rf__randomforestclassifier__n_estimators': [100, 200],
    'rf__randomforestclassifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(ensemble_model, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

joblib.dump(best_model, 'model.pkl')

print("Model has been trained and saved successfully.")
