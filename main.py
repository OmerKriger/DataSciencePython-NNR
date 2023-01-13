import time
import json
import numpy as np
import pandas as pd
from scipy.stats import mode
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List


def radius_optimizer_preprocessing(distance_bank):  # preprocessing data for function
    # Analyze the distance bank data
    df_distance_bank = pd.DataFrame(distance_bank)
    d_mean = df_distance_bank.mean(axis=1).mean(axis=0)
    d_min = df_distance_bank.min(axis=1).mean(axis=0)
    return d_mean, d_min


def DataPreprocessing(data_trn, data_vld):
    # create dataframes from datasets
    df_trn = pd.read_csv(data_trn)
    df_vld = pd.read_csv(data_vld)
    # take smaller datasets
    # df_trn = df_trn.sample(frac=1).reset_index(drop=True)
    # df_vld = df_vld.sample(frac=1).reset_index(drop=True)
    # save classes of training and validation
    x_trn_class_np = df_trn['class'].to_numpy()
    x_vld_class_np = df_vld['class'].to_numpy()
    # create features vectors matrix and scale them
    # scaler = StandardScaler()
    # x_trn = pd.DataFrame(scaler.fit_transform(df_trn.drop(columns="class")))
    # x_vld = pd.DataFrame(scaler.fit_transform(df_vld.drop(columns="class")))
    x_trn = df_trn.drop(columns="class")
    x_vld = df_vld.drop(columns="class")
    # create radius bank from vectors
    np_x_trn, np_x_vld = x_trn.to_numpy(), x_vld.to_numpy()
    distance_bank = create_distance_bank(np_x_trn, np_x_vld)

    return x_trn_class_np, x_vld_class_np, distance_bank


def euclidean_distance(v, u):  # function get two vectors (numpy array) and calculate euclidean distance
    return np.sqrt((np.square(v[:, np.newaxis]-u).sum(axis=2)))


def create_distance_bank(v_arr, u_arr):  # this function calculates distances between vectors_array.
    distance_bank = euclidean_distance(v_arr, u_arr)
    return distance_bank.T


# def find_neighbors_in_radius(distances_table, x_trn_class_np, radius):
#     inside_radius = list()
#     for i in range(distances_table.shape[0]):
#         if distances_table[i] <= radius:
#             inside_radius.append(x_trn_class_np[i])
#     return inside_radius


def prediction_with_radius(distances, x_trn_class_np, radius):
    prediction = list()
    for i in range(distances.shape[0]):
        distance_row = distances[i, :]
        inside_radius = x_trn_class_np[distance_row <= radius]
        # neighbors_classes = find_neighbors_in_radius(distances[i, :], x_trn_class_np, radius)
        if len(inside_radius) != 0:
            np.bincount(inside_radius).argmax()  # TODO: fix this figure for getting max appearance
            # prediction.append(max(set(inside_radius), key=inside_radius.count))
        else:
            prediction.append(np.NaN)
    return prediction


def radius_optimization(x_trn_class_np, x_vld_class_np, distance_bank):
    d_mean, d_min = radius_optimizer_preprocessing(distance_bank)  # preprocessing data
    highest_score = best_radius = index = -1  # variables for checking
    for i in np.arange(1, 10, 0.5):
        radius_test = (d_mean + i * d_min) / (i+1)
        prediction = prediction_with_radius(distance_bank, x_trn_class_np, radius_test)
        score = accuracy_score(x_vld_class_np, prediction)
        print(f'{prediction},\n {i}: radius: {radius_test} ,score:{score}')
        if score >= highest_score:
            highest_score = score
            index = i
            best_radius = radius_test
        print(f'(Highest score, best radius, index): {highest_score, best_radius, index}')
    return best_radius


def predict_test(data_tst, data_trn, x_trn_class_np, radius):
    df_tst, df_trn = pd.read_csv(data_tst), pd.read_csv(data_trn)  # read the datasets
    # scaler = StandardScaler()
    # x_tst = pd.DataFrame(scaler.fit_transform(df_tst.drop(columns="class"))).to_numpy()  # scaling into numpy array
    # x_trn = pd.DataFrame(scaler.fit_transform(df_trn.drop(columns="class"))).to_numpy()  # scaling into numpy array
    x_tst = df_tst.drop(columns="class").to_numpy()  # scaling into numpy array
    x_trn = df_trn.drop(columns="class").to_numpy()  # scaling into numpy array

    distances = create_distance_bank(x_trn, x_tst)
    prediction = prediction_with_radius(distances, x_trn_class_np, radius)
    return prediction


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    # Prepare the classification data
    x_trn_class_np, x_vld_class_np, distance_bank = DataPreprocessing(data_trn, data_vld)
    # validate the classification radius
    print('starting radius validation')
    start_t = time.time()
    radius = radius_optimization(x_trn_class_np, x_vld_class_np, distance_bank)
    print(f'Time for radius validation is {int((time.time() - start_t)/60)}:{int((time.time() - start_t)%60)}')
    # predict the classification by chosen radius for the dataset of test
    print('starting prediction for test dataset')
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
