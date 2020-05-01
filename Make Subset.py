from WWF_Project.deia2_general import set_path_base, to_dataframe
from collections import Counter

import pandas as pd
import pickle


#%%


# Deze aanpassen naar welke files je op je laptoppie heb staan
files_on_my_laptop = [8, 9, 15, 16, 17, 24, 25, 26]
files = ['TilePickle_{}.pkl'.format(file) for file in files_on_my_laptop]

#How many pixels per pickle should there be in the subset?
PIXELS_PER_PICKLE = 4000
#%%
##now determining the split of deforestation and non-deforestation

# Two functions which are needed to construct the subset
def amounts(df):
    forest, deforestation = list(Counter(df['future_deforestation']).values())[0], list(Counter(df['future_deforestation']).values())[1]

    perc_forest = forest/(forest+deforestation)
    amount_forest = round(PIXELS_PER_PICKLE * perc_forest)

    perc_deforest = deforestation/(forest+deforestation)
    amount_deforest = round(PIXELS_PER_PICKLE*perc_deforest)

    return amount_forest, amount_deforest

def subset_pickle(amount_deforest, amount_forest, df, seed=47):
    # Select all needed cases of deforestation
    sample_pos = df[df['future_deforestation'] == 1].sample(n=amount_deforest, random_state=seed, replace=False)
    # Select all needed cases of forest
    sample_neg = df[df['future_deforestation'] == 0].sample(n=amount_forest, random_state=seed, replace=False)
    # Add these two parts together in a new df
    df = pd.concat([sample_pos, sample_neg])
    return df


#%%

#creating an empty dataframe to fill
subset = pd.DataFrame(columns=['AggIndex1', 'AggIndex2', 'AggIndex3', 'AggIndex4', 'AggIndex5',
       'EdgeDensity1', 'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
       'PatchDensity1', 'PatchDensity2', 'PatchDensity3', 'PatchDensity4',
       'LandcoverPercentage1', 'LandcoverPercentage2', 'LandcoverPercentage3',
       'LandcoverPercentage4', 'ShannonDiversity',
       'current_deforestationDistance', 'current_degradationDistance',
       'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
       'scaledPopDensity', 'scaledASTER', 'RoadsDistance',
       'UrbanicityDistance', 'WaterwaysDistance', 'CoastlineDistance',
       'MillDistance', 'PalmOilConcession', 'gradientASTER', 'LogRoadDistance',
       'Vegetype', 'CurrentMonth', 'y_center', 'x_center', 'time', 'size'])


#this for loop constructs the actual subset (as a dataframe)
for file in files:
    print("new iteration started now handling file:", file)
    df = to_dataframe("{}/{}".format(set_path_base("Joost"), file))
    print("{}/{}".format(set_path_base("Joost"), file))
    amount_forest, amount_deforest = amounts(df)
    to_append = subset_pickle(amount_deforest, amount_forest, df)
    subset = pd.concat([subset, to_append])
    print("shape of subset is now:", subset.shape, "\ndataframe is updated with pickle", file, "\n")

#%%
# Overview and saving on your system
forest, deforestation = list(Counter(subset['future_deforestation']).values())[0], list(Counter(subset['future_deforestation']).values())[1]
print("there are {} forest tiles in this pickle and {} deforested tiles in this subset".format(forest, deforestation))
subset.to_pickle("WWF_Project/subset.pkl")

