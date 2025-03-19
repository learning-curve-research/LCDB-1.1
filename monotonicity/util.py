from lcdb_function.directencoder import DirectEncoder
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

def get_error_rates_from_json(json_obj):

    myEncoder = DirectEncoder()

    y_train = myEncoder.decode_label_vector_decompression(json_obj['y_train'])
    y_valid = myEncoder.decode_label_vector_decompression(json_obj['y_valid'])
    y_test = myEncoder.decode_label_vector_decompression(json_obj['y_test'])

    y_hat_train = myEncoder.decode_label_vector_decompression(json_obj['y_hat_train'])
    y_hat_valid = myEncoder.decode_label_vector_decompression(json_obj['y_hat_valid'])
    y_hat_test = myEncoder.decode_label_vector_decompression(json_obj['y_hat_test'])

    er_train = 1- accuracy_score(y_train, y_hat_train)
    er_valid = 1 - accuracy_score(y_valid, y_hat_valid)
    er_test = 1- accuracy_score(y_test, y_hat_test)

    return er_train, er_valid, er_test

def read_results_from_json_file(file_name):
    rows = []
    with open(file_name, 'r') as file:
        result_json = json.load(file)
        for row in result_json:
            
            # Check the value of the "status" key
            if row.get('status') == 'ok':

                er_train, er_valid, er_test = get_error_rates_from_json(row)
                row["er_train"] = er_train
                row["er_valid"] = er_valid
                row["er_test"] = er_test
                rows.append(row)

    return pd.DataFrame(rows)

def df_to_numpy_tensor(df):
    outer_seed = sorted(pd.unique(df))
    outer_seed = sorted(pd.unique(df))

