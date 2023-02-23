from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess_message
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np


mails = pd.read_csv("mails.csv")
mails = mails.loc[:, ~mails.columns.str.contains('^Unnamed')]

mails['type'] = mails['type'].map({
    'ham' : 0,
    'spam' : 1
    })

mails['message'] = mails['message'].apply(preprocess_message)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mails['message'], mails['type'], test_size=0.2, random_state=42)

X = X_train
Y = y_test
# Create CountVectorizer object
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(type(X))
print(type(Y))

# Create Random Forest classifier
model = RandomForestClassifier(n_estimators=100)

# Train model on training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy score
print('Accuracy:', accuracy)
print("Confusion matrix:", confusion_matrix(y_test, y_pred))


#==================test the model with new data====================================#
new_msg = pd.read_csv("new_data.csv", encoding='latin-1')
new_msg = new_msg.loc[:, ~new_msg.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#


new_msg['message'] = new_msg['message'].apply(preprocess_message)

spam_features_new = vectorizer.transform(new_msg['message'])
y_pred_new = model.predict(spam_features_new)

print(y_pred_new)

#==================test the model with new data====================================#


