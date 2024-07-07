#DECISION TREE CLASSIFIER

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#load dataset
df = pd.read_csv('diabetes.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

#ploting distribution of features
df.hist(figsize=(15, 10))
plt.show()

#split features and labels
X = df.drop(columns=['Outcome'])
y = df['Outcome']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=84)

#model (DECISION TREE CLASSIFIER)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)

#Accuracy
dtc_accuracy = accuracy_score(y_test, dtc_prediction)
print(dtc_accuracy )  #69% acc

#confusion matrix
confmat = confusion_matrix(y_test,dtc_prediction)
print(confmat)

sns.heatmap(confmat,annot=True)
plt.show()


