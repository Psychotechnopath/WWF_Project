from deia2_general import set_path_base, to_dataframe
import pandas as pd
from sklearn.model_selection import train_test_split

path = set_path_base("Yme")
df = to_dataframe(f"{path}/TilePickle_25.pkl")

# Deze aanpassen naar welke files je op je laptoppie heb staan
files_on_my_laptop = [8, 9, 15, 16, 17, 24, 25, 26]
paths = [f'{path}TilePickle_{file}.pkl' for file in files_on_my_laptop]

#this for loop constructs the actual subset (as a dataframe)
subset_x = pd.DataFrame()
subset_y = pd.DataFrame()

for index, pkl in enumerate(paths):
    #For every pickle in the path list make a Dataframe
    df = to_dataframe(pkl)
    df.dropna(inplace=True)
    y = df['future_deforestation']
    df.drop(columns=['future_deforestation'], inplace=True)
    X = df
    # Abuse train_test_split somewhat to generate stratified subsamples of data
    x_outsample, x_insample, y_outsample, y_insample = train_test_split(X, y, test_size=0.0003, stratify=y, random_state=47)
    # Appending sub-samples to a complete dataframe which is a subset over all pickles.
    subset_x = subset_x.append(x_insample)
    subset_y = subset_y.append(y_insample.to_frame())

#Reset indices so we have an increasing index again. We don't want the old indices so we drop them
subset_x.reset_index(inplace=True, drop=True)
subset_y.reset_index(inplace=True, drop=True)

subset_x.to_pickle("subset_x.pkl", protocol=3)
subset_y.to_pickle("subset_y.pkl", protocol=3)

# Overview
forest, deforestation = subset_y['future_deforestation'].value_counts()[0], subset_y['future_deforestation'].value_counts()[1]
print(f"there are {forest} forest tiles in this pickle and {deforestation} deforested tiles in this subset")
