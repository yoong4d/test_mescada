import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# wine quality data load 
data = pd.read_csv('winequality-red.csv') # dataset available at https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data

#preprocessing steps
# x = input, y = output
x = data.drop('quality', axis=1)
y = data['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

# Split train test 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# initialize model
model = RandomForestClassifier()

# Training model
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
test_data_accuracy = accuracy_score(y_pred, y_test)

accuracy_rf = test_data_accuracy*100
print('Accuracy = ', accuracy_rf)

# Save the model
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)