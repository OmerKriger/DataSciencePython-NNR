import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List


def radius_optimizer_preprocessing(distance_bank):  # preprocessing data for testing the model
    # Analyze the distance bank data
    df_distance_bank = pd.DataFrame(distance_bank)
    # calculate the minimum and maximum distances in each row and take the average of them
    d_min = df_distance_bank.min(axis=1).mean(axis=0)
    d_max = df_distance_bank.max(axis=1).mean(axis=0)
    return d_min, d_max


def DataPreprocessing(data_trn, data_vld):  # the function prepares the data for validation radius
    # NOTE: dataframe is for easy use with dataset and numpy for fast calculations
    # create dataframes from datasets
    df_trn = pd.read_csv(data_trn)
    df_vld = pd.read_csv(data_vld)
    # save classes of training and validation
    x_trn_class_np = df_trn['class'].to_numpy()
    x_vld_class_np = df_vld['class'].to_numpy()
    # create features vectors matrix and scale them
    scaler = StandardScaler()
    # scale and remove labels
    x_trn = pd.DataFrame(scaler.fit_transform(df_trn.drop(columns="class")))
    x_vld = pd.DataFrame(scaler.fit_transform(df_vld.drop(columns="class")))
    # create radius bank from vectors
    np_x_trn, np_x_vld = x_trn.to_numpy(), x_vld.to_numpy()
    distance_bank = create_distance_bank(np_x_trn, np_x_vld)
    return x_trn_class_np, x_vld_class_np, distance_bank


def euclidean_distance(v, u):  # function get two vectors (numpy array) and calculate euclidean distance
    # the distance calculation using broadcasting of numpy arrays (for faster performance)
    # vector v get another dimension (for broadcasting)
    return np.sqrt((np.square(v[:, np.newaxis]-u).sum(axis=2)))


def create_distance_bank(v_arr, u_arr):  # this function calculates distances between vectors_array.
    distance_bank = euclidean_distance(v_arr, u_arr)
    return distance_bank.T  # return the transposed matrix of distances


def prediction_with_radius(distances, x_trn_class_np, radius):
    prediction = list()
    most_freq = max(set(x_trn_class_np), key=list(x_trn_class_np).count)  # check which class most frequency
    for i in range(distances.shape[0]):  # run for each row of distances table
        distance_row = distances[i, :]  # grab the distance for this row
        inside_radius = list(x_trn_class_np[distance_row <= radius])  # check which classes are inside
        if len(inside_radius) != 0:
            prediction.append(max(set(inside_radius), key=inside_radius.count))  # check which class is the dominant
        else:
            prediction.append(most_freq)  # if nothing inside assume the class is the most frequency
    return prediction


def radius_optimization(x_trn_class_np, x_to_predict_real_class_np, distance_bank):
    d_min, d_max = radius_optimizer_preprocessing(distance_bank)  # preprocessing data
    highest_score = best_radius = index = -1  # variables for checking
    for i, radius_test in enumerate(np.linspace(d_min, d_max, num=50).round(5)):  # split the range between d_min and d_mean
        prediction = prediction_with_radius(distance_bank, x_trn_class_np, radius_test)  # prediction
        score = accuracy_score(x_to_predict_real_class_np, prediction)  # calculate accuracy score
        if score > highest_score:  # check for better accuracy
            highest_score = score
            index = i
            best_radius = radius_test
        elif i - index > 10:  # stop after best result didn't change 10 rounds
            break
    return best_radius


def predict_test(data_tst, data_trn, x_trn_class_np, radius):  # check the model on the data test
    df_tst, df_trn = pd.read_csv(data_tst), pd.read_csv(data_trn)  # read the datasets
    scaler = StandardScaler()  # scaling the datasets
    x_tst = pd.DataFrame(scaler.fit_transform(df_tst.drop(columns="class"))).to_numpy()  # scaling into numpy array
    x_trn = pd.DataFrame(scaler.fit_transform(df_trn.drop(columns="class"))).to_numpy()  # scaling into numpy array
    distances = create_distance_bank(x_trn, x_tst)  # check distances
    prediction = prediction_with_radius(distances, x_trn_class_np, radius)  # prediction for this dataset
    return prediction


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    # Prepare the classification data
    x_trn_class_np, x_vld_class_np, distance_bank = DataPreprocessing(data_trn, data_vld)
    # validate the classification radius
    radius = radius_optimization(x_trn_class_np, x_vld_class_np, distance_bank)
    # predict the classification by chosen radius for the dataset of test
    predictions = predict_test(data_tst, data_trn, x_trn_class_np, radius)
    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert (len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time() - start, 0)} sec')
