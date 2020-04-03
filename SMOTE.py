from collections import Counter
import pandas as pd
import pickle
from numpy import where
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb


# Comment for discussion to be pushed:
# We should use SMOTENC to denote which features are nominal/categorical,
# otherwise the algorithm is going to interpolate between months, where month = 6.4 does not make any sense


# Import data
with open("C:/Users/Yme/Desktop/WWF Data/TilePickle_25.pkl", "rb") as f:
    tile = pickle.load(f)

# Flatten matrix
reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))

# Transform to dataframe
df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',
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
X = df.copy(deep=True)
X.drop(columns=['future_deforestation'], inplace=True)
y = df['future_deforestation']

over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.33)

steps = [ ('o', over), ('u', under)]
pipeline = Pipeline(steps)

X, y = pipeline.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)

xgb_model = xgb.XGBClassifier(objective='reg:linear', random_state=47)
xgb_model.fit(x_train, y_train)
xgb_model.save_model('trained_model')

y_pred = xgb_model.predict(x_test)
print(y_pred.max())

score = accuracy_score(y_test, y_pred, normalize=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Always 0
print(score)
print(conf_matrix)

