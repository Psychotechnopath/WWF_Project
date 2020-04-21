# we can use this to define general functions that we need in many other files
import pandas as pd
import pickle

def set_path_base(user):
    path = ""
    if user == 'Tim':
        path = 'C:/Users/s161158/Documents/Pythondingen/DEIA2_git/'
    elif user == 'Yme':
        path = 'C:/Users/Yme/Desktop/WWF Data/'
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