from general_functions import set_path_base, to_dataframe
import pandas as pd
from sklearn.model_selection import train_test_split

path = set_path_base("Rodger")
files_on_my_laptop = list(range(0, 31))
paths = [f'{path}TilePickle_{file}.pkl' for file in files_on_my_laptop]

#this for loop constructs the actual subset (as a dataframe)
subset_x = pd.DataFrame()
subset_y = pd.DataFrame()

for index, pkl in enumerate(paths):
    print(index, pkl)
    #For every pickle in the path list make a Dataframe
    df = to_dataframe(pkl)
    print(len(df))
    if len(df.dropna()) > 5000:
        print("niet lege df")
        df.dropna(inplace=True)
        print(len(df))
        y = df['future_deforestation']
        # This is our X
        # To not brick my laptop, I'm not gonna save it twice
        df.drop(columns=['future_deforestation'], inplace=True)
        # X = df
        print(len(y), len(df))
        # Abuse train_test_split somewhat to generate stratified subsamples of data
        x_outsample, x_insample, y_outsample, y_insample = train_test_split(df, y, train_size=0.0003, test_size=0.0003,
                                                                            stratify=y, random_state=47)
        df = 0
        y = 0
        # Appending sub-samples to a complete dataframe which is a subset over all pickles.
        subset_x = subset_x.append(x_insample)
        subset_y = subset_y.append(y_insample.to_frame())
    else:
        print("lege of bijna lege df")
        df.dropna(inplace=True)
        print(len(df))


#Reset indices so we have an increasing index again. We don't want the old indices so we drop them
subset_x.reset_index(inplace=True, drop=True)
subset_y.reset_index(inplace=True, drop=True)

subset_x.to_pickle("subset_x.pkl", protocol=3)
subset_y.to_pickle("subset_y.pkl", protocol=3)

# Overview
forest, deforestation = subset_y['future_deforestation'].value_counts()[0], subset_y['future_deforestation'].value_counts()[1]
print(f"there are {forest} forest tiles in this pickle and {deforestation} deforested tiles in this subset")