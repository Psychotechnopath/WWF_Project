import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

#Import data
with open("TilePickle_28.pkl", "rb") as f:
    tile = pickle.load(f)

# Flatten matrix
reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))

# Transform to dataframe
df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',
                                            'AggIndex4', 'AggIndex5', 'EdgeDensity1',
                                            'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
                                            'PatchDensity1', 'PatchDensity2', 'PatchDensity3',
                                            'PatchDensity4', 'LandcoverPercentage1', 'LandcoverPercentage2',
                                            'LandcoverPercentage3', 'LandcoverPercentage4', 'ShannonDiversity ',
                                            'current_deforestationDistance ', 'current_degradationDistance',
                                            'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
                                            'scaledPopDensity', 'scaledASTER', 'RoadsDistance', 'UrbanicityDistance',
                                            'WaterwaysDistance', 'CoastlineDistance', 'MillDistance',
                                            'PalmOilConcession ',
                                            'gradientASTER', 'LogRoadDistance', 'Vegetype', 'CurrentMonth', 'y center',
                                            'x center', 'time', 'size '])

print(df['future_deforestation'].describe())

df_subset = df[:130000]
Y = df_subset['future_deforestation'].to_numpy()
del df_subset['future_deforestation']
X = df_subset.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=47)

xgb_model = xgb.XGBClassifier(objective='reg:linear', random_state=47)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
print(y_pred.max())

score = accuracy_score(y_test, y_pred, normalize=True)
conf_matrix = confusion_matrix(y_test, y_pred)

#Always 0
print(score)
print(conf_matrix)
