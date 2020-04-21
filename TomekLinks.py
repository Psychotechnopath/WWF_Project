import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
import imblearn.combine
import imblearn.over_sampling as os
from deia2_general import set_path_base, to_dataframe

#%%
path_yme = set_path_base("Yme")
df = to_dataframe("{}/TilePickle_25.pkl".format(path_yme))

#Set correct predicted and predictor variables
y = df['future_deforestation']
df.drop(columns=['future_deforestation'], inplace=True)
X = df

#%%
# we take a random train test split of 25%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)

print(y_train.mean()) # what percentage of training data has deforestation
print(y_test.mean()) # and what percentage of testing data has deforestation (should be rel. close)
print(y_test.value_counts())

def train_xgb(x_resampled, y_resampled):
    print(y_resampled.value_counts())
    xgb_model = xgb.XGBClassifier(objective='reg:squarederror', random_state=47)
    xgb_model.fit(x_resampled, y_resampled)
    #xgb_model.save_model('trained_model')
    y_pred = xgb_model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print('The geometric mean is {}'.format(geometric_mean_score(
        y_test,
        y_pred)))


# Third pipeline SMOTE + Tomek links
print("SMOTE + Tomek")
over_3 = imblearn.combine.SMOTETomek(sampling_strategy=1)
steps_3 = [ ('o', over_3)]
pipeline_3 = Pipeline(steps_3)
x_r_3, y_r_3 = pipeline_3.fit_resample(x_train, y_train)
train_xgb(x_r_3, y_r_3)
