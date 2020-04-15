import pandas as pd
import deia2_general as d2g
import pickle
import imblearn as imb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

path = d2g.set_path_base('Tim') + 'TilePickle_25.pkl'
df = d2g.to_dataframe(path)

if False:  # Rodger z'n df downsize
    # Select all 1000 cases uit 27174 where y is positive
    sample_pos = df[df['future_deforestation'] == 1].sample(n = 10, random_state=47, replace=False)
    # And for the 1000:1 ratio, select 1000x as many cases where y is negative
    sample_neg = df[df['future_deforestation'] == 0].sample(n = 10000, random_state=47, replace=False)
    # Add these two parts together in a new df
    df = pd.concat([sample_pos, sample_neg])


X = df.copy(deep=True)
X.drop(columns=['future_deforestation'], inplace=True)
y = df['future_deforestation']

over = imb.over_sampling.RandomOverSampler(random_state=42)
under = imb.under_sampling.RandomUnderSampler(sampling_strategy=0.33)
steps = [('o', over), ('u', under)]
pipeline = imb.pipeline.Pipeline(steps)
X, y = pipeline.fit_resample(X, y)
x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=47)

xgb_model = XGBClassifier(objective='reg:squarederror', random_state=47)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
print(y_pred.max())

score = accuracy_score(y_test, y_pred, normalize=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(score)
print(conf_matrix)
