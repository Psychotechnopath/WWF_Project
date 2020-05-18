#%%
from imblearn.pipeline import pipeline
from deia2_general import to_dataframe, set_path_base, xg_boost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

base_path = set_path_base("Yme")
df = to_dataframe("{}TilePickle_25.pkl".format(base_path))

#Make a copy of the original df, drop future_df column and create X and y variable
X = df.copy(deep=True)
X.drop(columns=['future_deforestation'], inplace=True)
y = df['future_deforestation']


from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(sampling_strategy={1: 400, 0: 9600})
X_res, y_res = cc.fit_sample(X, y)

x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=47)
xgb = xg_boost(x_train, y_train, x_test, y_test, "SMOTE_10%")