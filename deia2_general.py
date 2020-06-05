# we can use this to define general functions that we need in many other files
import pandas as pd
import pickle

import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score


def set_path_base(user):
    path = ""
    if user == 'Tim':
        path = 'C:/Users/s161158/Documents/Pythondingen/DEIA2_git/'
    elif user == 'Yme':
        path = 'C:/Users/Yme/Desktop/WWF Data/'
    elif user == 'Joost':
        path = 'C:/Users/s155633/Documents/aaaJADS/aaaDEiA_II/DEIA_GIT/WWF_Project'
    elif user == 'Rodger':
        path = 'D:/Downloads/DEiA/'
    elif user == 'Ellen':
        path = 'C:/Users/s153832/Desktop/JADS/Master/Jaar_1/Semester_1.2/Data_Engineer_in_Action_2/WWF/WWF_Project'
    return path


def to_dataframe(path):
    # turns the pickle at 'path' into a pandas dataframe that we can use
    with open(path, "rb") as f:  # Import data
        tile = pickle.load(f)

    reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))  # Flatten matrix
    df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',  # Transform to dataframe
                                                'AggIndex4', 'AggIndex5', 'EdgeDensity1',
                                                'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
                                                'PatchDensity1', 'PatchDensity2', 'PatchDensity3',
                                                'PatchDensity4', 'LandcoverPercentage1', 'LandcoverPercentage2',
                                                'LandcoverPercentage3', 'LandcoverPercentage4', 'ShannonDiversity',
                                                'current_deforestationDistance', 'current_degradationDistance',
                                                'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
                                                'scaledPopDensity', 'scaledASTER', 'RoadsDistance', 'UrbanicityDistance',
                                                'WaterwaysDistance', 'CoastlineDistance', 'MillDistance',
                                                'PalmOilConcession',
                                                'gradientASTER', 'LogRoadDistance', 'Vegetype', 'CurrentMonth', 'y_center',
                                                'x_center', 'time', 'size'])
    return df


def xg_boost(x_train_param, y_train_param, x_test_param, y_test_param, model_name: str):
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=10, n_jobs=-1, random_state=47)
    xgb_model.fit(x_train_param, y_train_param)
    print("Model has been fitted")
    xgb_model.save_model('trained_model_{}'.format(model_name))
    y_pred = xgb_model.predict(x_test_param)
    conf_matrix = confusion_matrix(y_test_param, y_pred)
    accuracy = accuracy_score(y_test_param, y_pred, normalize=True)
    print('Accuracy own xgb: {} '.format(accuracy_score(y_test_param, y_pred)))
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print('Sensitivity own xgb: {}'.format(sensitivity))
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    print('Specificity own xgb: {}'.format(specificity))
    print(conf_matrix)
    return conf_matrix, xgb_model, sensitivity, specificity, accuracy





