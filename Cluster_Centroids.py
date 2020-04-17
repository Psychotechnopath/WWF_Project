#%%
from imblearn.pipeline import pipeline
from deia2_general import to_dataframe, set_path_base
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

#%%

base_path = set_path_base("Yme")

print(base_path)


df = to_dataframe("{}TilePickle_25.pkl".format(base_path))


#Make a copy of the original df, drop future_df column and create X and y variable
#TODO change this so a copy is no longer required, as pkl is 1.5gb and copying is a waste of RAM
X = df.copy(deep=True)
X.drop(columns=['future_deforestation'], inplace=True)
y = df['future_deforestation']
# #%%
# X, y = pipeline.fit_resample(X, y)

# %%

from imblearn.under_sampling import ClusterCentroids

#%%

cc = ClusterCentroids(sampling_strategy={1: 400, 0: 9600})
X_res, y_res = cc.fit_sample(X, y)

#%%
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=47)

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
