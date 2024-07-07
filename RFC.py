#RANDOM FOREST CLASSIFIER WITH RANDOM SEARCH TECHNIQUE FOR PARAM SELECTION


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('diabetes.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Plotting distribution of features
df.hist(figsize=(15, 10))
plt.show()

# Split features and labels
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=84)

# Initial model (RANDOM FOREST CLASSIFIER)
rfc = RandomForestClassifier(random_state=84)
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)

# Accuracy
rfc_score = accuracy_score(y_test, rfc_prediction)
print(f"Initial Accuracy: {rfc_score:.4f}")  # acc 77%

# Confusion matrix
confmat2 = confusion_matrix(y_test, rfc_prediction)
print(confmat2)

sns.heatmap(confmat2, annot=True)
plt.show()

# Classification report
print(classification_report(y_test, rfc_prediction))



# Using Randomized Search for hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=84)

# Instantiate the randomized search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=100, cv=3, verbose=2, random_state=84, n_jobs=-1)

# Fit the randomized search to the data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Updated model with best parameters
rfc_updated = RandomForestClassifier(**best_params, random_state=84)
rfc_updated.fit(X_train, y_train)
rfc_updated_prediction = rfc_updated.predict(X_test)

rfc_updated_score = accuracy_score(y_test, rfc_updated_prediction)
print(f"Updated Model Accuracy: {rfc_updated_score:.4f}")

# Confusion matrix for updated model
confmat_updated = confusion_matrix(y_test, rfc_updated_prediction)
print(confmat_updated)

sns.heatmap(confmat_updated, annot=True)
plt.show()

# Classification report for updated model
print(classification_report(y_test, rfc_updated_prediction))

# Cross-validation scores
cv_scores = cross_val_score(rfc_updated, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")
