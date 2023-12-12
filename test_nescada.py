import pickle
import os
import pandas as pd
import numpy as np

def get_columns_data(df):
    feature_names = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    x_arr = []
    for item in feature_names:
    
        try:
            value = df[item].to_numpy()
            x_arr.append(value)
        except:
            value = np.zeros(len(df))
            x_arr.append(value)

    return x_arr

with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


data_path = 'winequality-red.csv'

data = pd.read_csv(data_path)

data1 = get_columns_data(data)
x_in = np.vstack(data1).T

pred = model.predict(x_in)

# predict list of good and bad wines
# mapping 0 and 1 for bad and good wine
pred_mapping = {0: 'bad', 1: 'good'}
pred_list = [pred_mapping[val] for val in pred]