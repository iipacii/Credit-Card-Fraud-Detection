import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import os
import pickle

os.makedirs('models', exist_ok=True)

file_path = 'transactions.csv'  #loading the dataset
df = pd.read_csv(file_path)

df_filled = df.fillna(0) #filling the missing values with 0

#Encoding the categorical columns
df_filled['user_archetype_encoded'] = df_filled['user_archetype'].astype('category').cat.codes
df_filled['merchant_category_encoded'] = df_filled['merchant_category'].astype('category').cat.codes
df_filled['transaction_type_encoded'] = df_filled['transaction_type'].astype('category').cat.codes
df_filled['is_fraud_encoded'] = df_filled['is_fraud'].astype('category').cat.codes

encodings = {
    'user_archetype': df_filled['user_archetype'].astype('category').cat.categories,
    'merchant_category': df_filled['merchant_category'].astype('category').cat.categories,
    'transaction_type': df_filled['transaction_type'].astype('category').cat.categories,
    'is_fraud': df_filled['is_fraud'].astype('category').cat.categories
}
with open('models/encodings.pkl', 'wb') as f:
    pickle.dump(encodings, f)

#Feature Engieering
df_eval = df_filled[['amount', 'distance_from_home', 'auth_attempts', 'user_archetype_encoded', 'merchant_category_encoded', 'transaction_type_encoded', 'is_fraud_encoded']]

#Finding the best features using SelectKBest
X = df_eval.drop(columns=['is_fraud_encoded'])
y = df_eval['is_fraud_encoded']
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

#making sure 'user_archetype_encoded' is included in selected features due to an error
if 'user_archetype_encoded' not in selected_features:
    selected_features = selected_features.append(pd.Index(['user_archetype_encoded']))

#saving features in order so that its easy when loading the model
selected_features = selected_features.tolist()
joblib.dump(selected_features, 'models/feature_names.joblib')
print(f"Feature names saved: {selected_features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_split = pd.DataFrame(X_test, columns=selected_features)
test_split['is_fraud_encoded'] = y_test
test_split.to_csv('models/test_split.csv', index=False)

#Model training
models = {
    'RandomForest': RandomForestClassifier(random_state=2),
    'SVM': SVC(kernel='rbf'),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'NaiveBayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/{name}_model.pkl')

#Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=models['RandomForest'], param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'models/RandomForest_best_model.pkl')