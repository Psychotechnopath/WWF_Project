### Purpose:
# Using the entire pickle kept crashing my laptop, and also took very look each run
# For rapid prototyping I'm taking just a selection of the data.
# Once the framework of ADASYN stands, it can run on the entire pickle
# (deel ontbossing + deel niet-ontbossing, voor verhouding 1:1000)

import pandas as pd
import pickle

# Import data
with open("D:\Downloads\DEiA\TilePickle_25.pkl", "rb") as f:
    tile = pickle.load(f)

    # Flatten matrix
    reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))
    tile = 0

    # Transform to dataframe
    df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',
                                                'AggIndex4', 'AggIndex5', 'EdgeDensity1',
                                                'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
                                                'PatchDensity1', 'PatchDensity2', 'PatchDensity3',
                                                'PatchDensity4', 'LandcoverPercentage1', 'LandcoverPercentage2',
                                                'LandcoverPercentage3', 'LandcoverPercentage4', 'ShannonDiversity',
                                                'current_deforestationDistance', 'current_degradationDistance',
                                                'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
                                                'scaledPopDensity', 'scaledASTER', 'RoadsDistance',
                                                'UrbanicityDistance',
                                                'WaterwaysDistance', 'CoastlineDistance', 'MillDistance',
                                                'PalmOilConcession',
                                                'gradientASTER', 'LogRoadDistance', 'Vegetype', 'CurrentMonth',
                                                'y_center',
                                                'x_center', 'time', 'size'])
    reshaped_tile = 0

    # Select all 1000 cases uit 27174 where y is positive
    sample_pos = df[df['future_deforestation'] == 1].sample(n = 1000, random_state=47, replace=False)
    # And for the 1000:1 ratio, select 1000x as many cases where y is negative
    sample_neg = df[df['future_deforestation'] == 0].sample(n = 1000000, random_state=47, replace=False)
    # Add these two parts together in a new df
    df = pd.concat([sample_pos, sample_neg])

    # And store that df as a new pickle, so it can be used in other files :)
    with open('./partial_pickle_25.pkl', 'wb+') as g:
        pickle.dump(df, g)