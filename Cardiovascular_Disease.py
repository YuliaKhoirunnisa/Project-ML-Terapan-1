import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # << INI WAJIB!
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.stats import randint

data = pd.read_csv('cardio_train.csv', sep=';')
data.head()
data.info()
data.describe()
data['cardio'].value_counts()
data['age_years'] = (data['age'] / 365).round(1)
data.info()

plt.hist(data['age_years'], bins=20, edgecolor='k')
plt.title('Distribusi Usia Pasien')
plt.xlabel('Usia (tahun)')
plt.ylabel('Jumlah')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.show()

data = data.drop('id', axis=1)
data['age'] = (data['age'] / 365).round(1)

X = data.drop('cardio', axis=1)
y = data['cardio']

X = pd.get_dummies(X, columns=['gender', 'cholesterol', 'gluc'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
data.head()

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,              
    cv=5,                     
    n_jobs=-1,                 
    random_state=42,
    scoring='accuracy',
    verbose=2
)

rand_search.fit(X_train, y_train)
best_rf_random = rand_search.best_estimator_
print("Best Parameters dari RandomizedSearchCV:", rand_search.best_params_)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

y_pred_rf_random = best_rf_random.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_random))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, best_rf_random.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_random))
print("Classification Report:\n", classification_report(y_test, y_pred_rf_random))