import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List


def DataPreprocessing(data_trn, data_vld, data_tst):  # the function prepares the data for model
    # NOTE: dataframe is for easy use with dataset and numpy for fast calculations
    # create dataframes from datasets
    df_trn = pd.read_csv(data_trn)
    df_vld = pd.read_csv(data_vld)
    df_tst = pd.read_csv(data_tst)
    # save classes of training and validation
    x_trn_class_np = df_trn['class'].to_numpy()
    x_vld_class_np = df_vld['class'].to_numpy()
    # create features vectors matrix and scale them
    scaler = StandardScaler()
    # scale and remove labels
    np_x_trn = scaler.fit_transform(df_trn.drop(columns="class"))  # fitting scale
    np_x_vld = scaler.transform(df_vld.drop(columns="class"))
    np_x_tst = scaler.transform(df_tst.drop(columns="class"))
    # create "distances bank" of distance between feature vectors from different datasets
    distance_bank_vld = create_distance_bank(np_x_trn, np_x_vld)  # distance between trn and vld
    distance_bank_tst = create_distance_bank(np_x_trn, np_x_tst)  # distance between trn and tst
    return x_trn_class_np, x_vld_class_np, distance_bank_vld, distance_bank_tst


def euclidean_distance(v, u):  # function get two vectors (numpy ndarray) and calculate euclidean distance.
    # the distance calculation using broadcasting of numpy ndarrays (for faster performance).
    # vector v (matrix) get another dimension (for broadcasting).
    # using euclidean distance formula.
    return np.sqrt((np.square(v[:, np.newaxis]-u).sum(axis=2)))


def create_distance_bank(v_arr, u_arr):  # this function calculates distances between vectors_array.
    distance_bank = euclidean_distance(v_arr, u_arr)
    return distance_bank.T  # return the transposed matrix of distances for work on rows and not cols


def prediction_with_radius(distances, x_trn_class_np, radius):
    prediction = list()
    for i in range(distances.shape[0]):  # run for each row of distances table
        distance_row = distances[i, :]  # take the distance of row i (distances between vector i and train vectors
        inside_radius = list(x_trn_class_np[distance_row <= radius])  # check which classes are inside the radius
        if len(inside_radius) != 0:
            prediction.append(max(set(inside_radius), key=inside_radius.count))  # check which class is the dominant
        else:
            # if no one inside, take the label of the closest vector to be the prediction for this object
            prediction.append(x_trn_class_np[np.argmin(distance_row, axis=0)])
    return prediction


def radius_optimization(x_trn_class_np, x_to_predict_real_class_np, distance_bank):
    d_min, d_max = distance_bank.min(), distance_bank.max()  # preprocessing data
    highest_score = best_radius = index = -1  # variables for checking
    num_of_samples = int(distance_bank.shape[0] / 2)  # num of splits the range between d_min and d_max
    times_to_stop_searching = int(num_of_samples / 20)  # num of times with no change in result for stop condition
    # split the range between d_min and d_max and check values for optimal radius
    for i, radius_test in enumerate(np.linspace(d_min, d_max, num=num_of_samples)):
        prediction = prediction_with_radius(distance_bank, x_trn_class_np, radius_test)  # prediction
        score = accuracy_score(x_to_predict_real_class_np, prediction)  # calculate accuracy score
        if score > highest_score:  # check for better accuracy
            best_radius, index, highest_score = radius_test, i, score
        elif i - index > times_to_stop_searching:  # stop after best result didn't change for X times
            break
    return best_radius


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    # Prepare the classification data
    x_trn_class_np, x_vld_class_np, distance_bank_vld, distance_bank_tst = DataPreprocessing(data_trn, data_vld, data_tst)
    # validate the classification radius
    radius = radius_optimization(x_trn_class_np, x_vld_class_np, distance_bank_vld)
    # predict the classification by chosen radius for the dataset of test
    predictions = prediction_with_radius(distance_bank_tst, x_trn_class_np, radius)
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
