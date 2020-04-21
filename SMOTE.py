import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from deia2_general import set_path_base, to_dataframe

#%%
# Comment for discussion to be pushed:
# We should use SMOTENC to denote which features are nominal/categorical,
# otherwise the algorithm is going to interpolate between months, where month = 6.4 does not make any sense

path_yme = set_path_base("Yme")
df = to_dataframe("{}/TilePickle_25.pkl".format(path_yme))

#Set correct predicted and predictor variables
y = df['future_deforestation']
df.drop(columns=['future_deforestation'], inplace=True)
X = df

#%%
#Initialize a SMOTE sampler with
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.33)

steps = [ ('o', over), ('u', under)]
pipeline = Pipeline(steps)

X, y = pipeline.fit_resample(X, y)

hi = y.value_counts()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)
#%%

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

